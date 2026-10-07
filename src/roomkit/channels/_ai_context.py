"""AIChannel mixin for building the AI provider context from room events."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from roomkit.channels._ai_cuts import CUT_MARK, cut_answer_ids, cut_records
from roomkit.channels._ai_policy import policy_check
from roomkit.channels._dangling_recovery import patch_dangling_tool_calls
from roomkit.channels._instruction import instruction_fingerprint, is_standalone, mark_instruction
from roomkit.channels._skill_constants import (
    SKILLS_NO_SCRIPTS_NOTE as _SKILLS_NO_SCRIPTS_NOTE,
)
from roomkit.channels._skill_constants import (
    SKILLS_PREAMBLE as _SKILLS_PREAMBLE,
)
from roomkit.channels._skill_constants import TOOL_RUN_SCRIPT
from roomkit.channels._task_planner import TaskPlanner
from roomkit.channels._tasks_note import render_tasks_note
from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels._tool_search import search_tool_defs, should_activate_tool_search
from roomkit.channels._tool_search_constants import TOOL_SEARCH_PREAMBLE
from roomkit.channels._turn_budget import TurnBudget, turn_budget
from roomkit.channels._turn_notes import turn_input, turn_notes, with_turn_notes
from roomkit.channels._user_text import with_leading_text
from roomkit.core.visibility import visible_events
from roomkit.memory.base import MemoryResult
from roomkit.memory.token_estimator import estimate_tokens, estimate_tool_tokens
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.delivery import SUPERSEDED
from roomkit.models.enums import ChannelCategory, ChannelMediaType, EventType
from roomkit.models.event import CompositeContent, MediaContent, TextContent
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AITool,
)
from roomkit.sandbox.tools import SANDBOX_PREAMBLE as _SANDBOX_PREAMBLE
from roomkit.sandbox.tools import SANDBOX_TOOL_PREFIX as _SANDBOX_TOOL_PREFIX
from roomkit.tools.context import TurnFootprint

if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import RoomTasksLoader, ToolUsageLoader
    from roomkit.channels._skill_activation import SkillActivationMemory
    from roomkit.channels._tool_registry import ChannelRegistry
    from roomkit.channels._tool_usage import ToolUsageMemory
    from roomkit.channels._turn_config import AIChannelTurnConfig, ConfigProvider
    from roomkit.channels.ai import EmptyEventDescriber, _ContentPart
    from roomkit.memory.base import MemoryProvider
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.providers.ai.base import AIProvider
    from roomkit.sandbox.executor import SandboxExecutor
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry
    from roomkit.tools._human_input_channel import ChannelHumanInput
    from roomkit.tools.context import _ToolLoopContext
    from roomkit.tools.policy import ToolPolicy

if TYPE_CHECKING:
    from roomkit.channels._ai_contract import _AIChannelContract
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")


# The settings a turn resolves binding metadata > config provider > channel
# default, each read from ``_<name>`` on the channel. ``tools`` resolves apart.
# The levels, most specific first.
_BINDING_LEVEL, _TURN_LEVEL, _CHANNEL_LEVEL = 0, 1, 2
_TURN_SETTINGS = (
    "system_prompt",
    "temperature",
    "max_tokens",
    "thinking_budget",
    "enable_thinking",
    "reasoning_effort",
    "response_schema",
)

# Injected once per turn when the history window holds several speakers: the
# model must read the "Name:" prefixes as transcript metadata, and not start
# prefixing its own replies with one.
_SPEAKER_ATTRIBUTION_NOTE = (
    "Several people take part in this conversation. Their messages are "
    'prefixed with the sender\'s name ("Name: message"). The prefix is '
    "transcript metadata, not text they typed: rely on it to know who said "
    "what, and never prefix your own replies with a name."
)


class AIContextMixin(_AIChannelContract):
    """Builds the AIContext passed to the provider from room state and events.

    What it calls on the other mixins is declared once, in
    :class:`~roomkit.channels._ai_contract._AIChannelContract`, which it
    derives from for the type checker only.
    """

    _provider: AIProvider
    _fallback_provider: AIProvider | None
    _system_prompt: str | None
    _temperature: float
    _max_tokens: int | None
    _thinking_budget: int | None
    _enable_thinking: bool | None
    _reasoning_effort: str | None
    _response_schema: dict[str, Any] | None
    _turn_budget_tokens: int | None
    _turn_budget_usd: float | None
    _skills: SkillRegistry | None
    _skills_in_prompt: bool
    _script_executor: ScriptExecutor | None
    _sandbox: SandboxExecutor | None
    _human_input: ChannelHumanInput
    _memory: MemoryProvider
    _describe_empty_event: EmptyEventDescriber | None
    _eviction: ToolEviction
    _tool_usage: ToolUsageMemory
    _tool_usage_loader: ToolUsageLoader | None
    _room_tasks_loader: RoomTasksLoader | None
    _skill_activation: SkillActivationMemory
    _planner: TaskPlanner | None
    _user_tools: list[AITool]
    _config_provider: ConfigProvider | None
    _tool_search: bool | None
    _tool_search_pinned: set[str]
    _tool_search_threshold: int
    _tool_search_threshold_pct: float
    _tool_search_threshold_tokens: int | None
    _registry: ChannelRegistry
    _tool_policy: ToolPolicy | None
    channel_id: str

    async def _build_context(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> AIContext:
        """Build AI context from room events.

        Config precedence per field:
        1. ``binding.metadata`` explicit overrides (system_prompt,
           temperature, max_tokens, thinking_budget, enable_thinking,
           reasoning_effort, response_schema, turn_budget_tokens,
           turn_budget_usd) — per-room operator intent, wins whenever it
           sets a value; a ``null`` defers like an absent key.
        2. The channel's ``config_provider`` result, resolved fresh at the
           start of every turn (see channels/_turn_config.py).
        3. The channel's constructor defaults.

        ``tools`` is the exception: when a ``config_provider`` is set, its
        toolset REPLACES ``binding.metadata["tools"]`` — that metadata key
        is an attach-time snapshot, and the whole point of the provider is
        that snapshots go stale. Without a provider, the metadata toolset
        is used.
        """
        turn, settings = await self._resolve_turn(binding, context)
        # The prompt grows below (skills, sandbox, planner, notes); the other
        # settings reach the context as resolved.
        system_prompt = settings.pop("system_prompt")

        tools = self._turn_base_tools(turn, binding)

        loop_ctx = self._get_loop_ctx()
        loop_ctx.turn_budget = self._turn_budget(binding, turn)

        # A standalone instruction reads nothing of the room (RFC §10.1.1 step
        # 7): no history below, and none of the room's working memories either
        # — active skill bodies, the plan, the tool-usage digest (which quotes
        # earlier tool results) and sticky tools are the room's past in
        # another form. The channel's own prompt, tools and catalogue stay.
        standalone = is_standalone(event)
        loop_ctx.standalone = standalone

        system_prompt = await self._add_channel_features(
            tools, system_prompt, event, binding, context, loop_ctx, standalone=standalone
        )

        # Store unfiltered tool list for re-application after skill activation
        self._stamp_toolset(loop_ctx, list(tools))

        # A human-input tool the turn never offers is a wiring mistake that
        # only shows up at runtime, and quietly: the model is not told the tool
        # exists, so it never calls it — and a scripted provider that does is
        # failed closed at dispatch. ``tool_names`` gates interception; what
        # offers the tool is ``tool_definitions`` on the handler, or the
        # channel's own ``tools`` / binding metadata. Warn once per channel.
        self._human_input.warn_unoffered(tools, self.channel_id)

        # Tool policy + skill gating. Tool Search's collapse is applied to what
        # BEFORE_AI_GENERATION leaves, so the hook sees the whole catalogue it
        # may withdraw from (RFC §6.4).
        tools = self._reachable_tools(tools)

        own_notes = self._channel_notes(loop_ctx, standalone=standalone)
        own_notes += await self._tasks_notes(loop_ctx, standalone=standalone)
        self._measure_turn(loop_ctx, system_prompt, own_notes, settings.get("max_tokens"))
        messages = await self._turn_conversation(event, context, loop_ctx, standalone, own_notes)
        loop_ctx.turn_input = turn_input(messages)

        target_media, target_caps = self._target_capabilities(context)

        return AIContext(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            room=context,
            target_capabilities=target_caps,
            target_media_types=target_media,
            # The turn's record, not a fresh dict: hooks and tool handlers write
            # into what the loop context already holds (by identity — see the type).
            response_metadata=loop_ctx.response_metadata,
            **settings,
        )

    def _target_capabilities(
        self, context: RoomContext
    ) -> tuple[list[ChannelMediaType], ChannelCapabilities | None]:
        """What every transport the answer reaches can carry: the media types
        they share, and their capabilities intersected (``None`` for none)."""
        transports = [
            b.capabilities
            for b in context.bindings
            if b.category == ChannelCategory.TRANSPORT and b.channel_id != self.channel_id
        ]
        if not transports:
            return [], None
        shared = _intersected(transports)
        return list(shared.media_types), shared

    async def _visible_memory(
        self, event: RoomEvent, context: RoomContext, standalone: bool
    ) -> MemoryResult:
        """What this channel's memory provider returns of the room as this
        channel sees it; nothing for a standalone instruction."""
        # Retrieve memory from this channel's view of the room, never the
        # room's whole timeline (RFC §7.5 rule 8): an event visibility kept
        # from this channel at broadcast must not reach the model as history
        # one turn later. Filtered here rather than inside the providers —
        # one shared ``RoomContext`` serves every channel of a broadcast, so
        # the filter can only run where the reader is known, and running it on
        # the way *in* is what stops a summarizing provider from re-emitting
        # hidden content as a summary.
        #
        # A standalone instruction's provider is not asked at all: an empty
        # view is not a blank page, since a provider may return messages of
        # its own (a summary, a minimum it always keeps).
        if standalone:
            return MemoryResult()
        return await self._memory.retrieve(
            event.room_id,
            event,
            context.model_copy(update={"recent_events": visible_events(context, self.channel_id)}),
            channel_id=self.channel_id,
        )

    async def _turn_conversation(
        self,
        event: RoomEvent,
        context: RoomContext,
        loop_ctx: _ToolLoopContext,
        standalone: bool,
        own_notes: list[str],
    ) -> list[AIMessage]:
        """The conversation the model reads this turn: the history this channel
        sees, then the input carrying the turn's notes, the channel's own
        (*own_notes*) after what the memory retrieved."""
        memory_result = await self._visible_memory(event, context, standalone)
        messages, attribute_speakers = self._turn_messages(event, context, memory_result, loop_ctx)
        notes = self._turn_notes(
            own_notes, speakers=attribute_speakers, retrieved=memory_result.notes
        )
        messages = with_turn_notes(messages, notes)
        return messages

    def _turn_base_tools(
        self, turn: AIChannelTurnConfig | None, binding: ChannelBinding
    ) -> list[AITool]:
        """The turn's tools before the channel's own features add theirs."""
        if turn is not None and turn.tools is not None:
            tools = list(turn.tools)
        else:
            # A null declares no toolset, like an absent key (RFC Appendix A.9).
            raw_tools = binding.metadata.get("tools") or []
            # Convert raw tool dicts to AITool instances
            tools = [
                AITool(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("parameters", {}),
                    tags=t.get("tags", []) or [],
                )
                for t in raw_tools
            ]

        # The host's tools, each name declared once and none the channel or
        # orchestration serves in this room; then the ones orchestration set up
        # for the room. The channel's own tools are added below (RFC §21.1).
        tools.extend(self._user_tools)
        tools = self._declared_once(tools, binding.room_id)
        tools.extend(self._orchestration_tools(binding.room_id))

        # Inject human-input tool definitions (e.g. AskUserQuestion)
        tools.extend(self._human_input.definitions)
        return tools

    async def _add_channel_features(
        self,
        tools: list[AITool],
        system_prompt: str | None,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
        loop_ctx: _ToolLoopContext,
        *,
        standalone: bool,
    ) -> str | None:
        """The turn's system prompt with what the channel's own features add to
        it: skills, sandbox, planner, Tool Search, the large-result re-read and
        the channel's identity, in that order. Their tools join *tools* in place.

        What changes from turn to turn is not here but in ``_turn_notes``: the
        system prompt stays the same between turns (RFC §6.4).
        """
        # Skill activation is keyed on the tool loop's room — the very id
        # ``activate_skill`` will write under (``handle_event`` stamps it on this
        # ctx, and the loop's child ctx inherits it), so what is written and what
        # is rendered can never drift apart. It is the event's room in practice;
        # reading it from the ctx is what makes that a fact rather than a hope.
        activation_room = loop_ctx.room_id

        # Rebuild the room-scoped working memories from persisted history the
        # first time this process serves the room (see _hydrate_room_memories).
        # Runs BEFORE the skills block: the active-skill bodies it may restore
        # are rendered into the prompt just below.
        await self._hydrate_room_memories(event.room_id, activation_room)

        system_prompt = self._add_skills(
            tools, system_prompt, activation_room, standalone=standalone
        )
        system_prompt = self._add_sandbox(tools, system_prompt)

        # The planning tool; the plan itself travels with the turn's input
        # (``_turn_notes``), since it changes from one turn to the next.
        if self._planner is not None:
            tools.append(TaskPlanner.tool_definition())

        system_prompt = self._collapse_behind_tool_search(
            tools, system_prompt, loop_ctx, binding, event, standalone
        )
        # The generation hook sees the re-read tool once the room holds a
        # stored result, and may withdraw it. The rounds declare it from the
        # first one either way (``_prepare_round_context``, RFC §6.4).
        if self._eviction.has_evicted:
            tools.append(ToolEviction.tool_definition())
        return (system_prompt or "") + (self._prompt_identity(context) or "") or None

    def _turn_messages(
        self,
        event: RoomEvent,
        context: RoomContext,
        memory_result: MemoryResult,
        loop_ctx: _ToolLoopContext,
    ) -> tuple[list[AIMessage], bool]:
        """The turn's messages, its history then its input, and whether several
        speakers are named in them.

        ``_determine_role`` flattens every non-self event into one "user"
        stream, which erases who said what in a room where several people
        speak — the model can only guess the addressee, and it guesses wrong.
        The speaker is a fact of the event (``metadata["sender_name"]``,
        stamped at ingress by hosts and transport providers), so when the
        window holds two or more distinct speakers each user turn carries its
        speaker's name. A single-speaker room (a 1:1 DM) is left untouched.
        """
        past_turns = self._past_turns(memory_result, context)
        current_content, current_speaker = self._turn_input(event, context, loop_ctx)
        speakers = {speaker for _, _, speaker in past_turns if speaker}
        if current_content and current_speaker:
            speakers.add(current_speaker)
        attribute_speakers = len(speakers) >= 2

        # Pre-built messages from memory (e.g. summaries)
        memory = list(memory_result.messages)
        messages: list[AIMessage] = list(memory)
        for role, content, speaker in past_turns:
            if attribute_speakers and speaker:
                content = _with_speaker_prefix(content, speaker)
            messages.append(AIMessage(role=role, content=content))

        # Patch orphaned tool calls from interrupted tool loops (barge-in)
        messages = patch_dangling_tool_calls(messages)

        if current_content:
            content = current_content
            if attribute_speakers and current_speaker:
                content = _with_speaker_prefix(content, current_speaker)
            messages.append(AIMessage(role="user", content=content))
        return _after_memory(memory, messages[len(memory) :]), attribute_speakers

    def _past_turns(
        self, memory_result: MemoryResult, context: RoomContext
    ) -> list[tuple[str, str | list[_ContentPart], str | None]]:
        """The history's turns as (role, content, speaker), a user turn's speaker named."""
        past_turns: list[tuple[str, str | list[_ContentPart], str | None]] = []
        # An answer cut off by a barge-in reads as cut, not as heard whole (§6.4).
        records = cut_records(context, self.channel_id)
        cut_ids = cut_answer_ids(memory_result.events, records, self.channel_id)
        for past_event in memory_result.events:
            if past_event.metadata.get("cancellation_reason") == SUPERSEDED:
                # A response nobody heard: the user continued the turn first,
                # and replaying it would answer what they never heard (§12.3.12).
                continue
            role = self._determine_role(past_event)
            content = self._transcript_content(past_event)
            if content and past_event.id in cut_ids:
                content = _with_cut_mark(content)
            if content:
                speaker = event_speaker(past_event, context) if role == "user" else None
                past_turns.append((role, content, speaker))
        return past_turns

    def _turn_input(
        self, event: RoomEvent, context: RoomContext, loop_ctx: _ToolLoopContext
    ) -> tuple[str | list[_ContentPart] | None, str | None]:
        """The turn's input and its speaker; an instruction marked as the
        application's, with no speaker."""
        current_content = self._transcript_content(event)
        current_speaker = event_speaker(event, context)
        if event.type == EventType.INSTRUCTION:
            # The application's direction for this one turn (RFC §10.1.1). It
            # is the turn's input — a system-role message after the history is
            # refused or silently re-roled by several model APIs — marked so the
            # model never reads it as a participant's words, and recorded on the
            # turn so every reply it produces says why the agent spoke — as a
            # fingerprint, never the text: the metadata rides on every reply
            # and segment, and a copy would store (and deliver to every
            # transport) what the room never stores.
            instruction = event.content.body if isinstance(event.content, TextContent) else ""
            current_content = mark_instruction(instruction) if instruction else None
            current_speaker = None
            loop_ctx.response_metadata["instruction"] = instruction_fingerprint(instruction)
        return current_content, current_speaker

    @staticmethod
    def _turn_notes(own: list[str], *, speakers: bool, retrieved: list[str]) -> str | None:
        """What changes from one turn to the next, as the notes the turn's
        input carries (RFC §6.4): how speakers are named when several speak,
        what the memory retrieved for this turn, then the channel's *own*."""
        blocks = [_SPEAKER_ATTRIBUTION_NOTE] if speakers else []
        return turn_notes([*blocks, *retrieved, *own])

    def _channel_notes(self, loop_ctx: _ToolLoopContext, *, standalone: bool) -> list[str]:
        """The notes the channel adds to the turn's input from the room's
        working memories: the room's plan and the tools already used here.

        A standalone turn reads none of them (RFC §10.1.1). Each is read under
        the tool loop's room, as its writer keys it.
        """
        room_id = loop_ctx.room_id
        if standalone or room_id is None:
            return []
        blocks: list[str] = []
        plan = self._planner.plan_for(room_id) if self._planner is not None else None
        if plan:
            blocks.append(TaskPlanner.format_plan_prompt(plan))
        # "Tools you've already used" digest — the rebuilt context drops
        # tool-call events, so without this the model forgets, across turns,
        # which tools/source it used (it would re-ask the user). Injected for
        # every model, not just small ones — the loss is provider-agnostic.
        digest = self._tool_usage.render_digest(room_id)
        if digest:
            blocks.append(digest)
        return blocks

    async def _tasks_notes(self, loop_ctx: _ToolLoopContext, *, standalone: bool) -> list[str]:
        """The room's background tasks for the turn's notes (RFC §23.4), read
        from the StatusBus as the turn is built; a standalone turn reads none."""
        room_id = loop_ctx.room_id
        if standalone or room_id is None or self._room_tasks_loader is None:
            return []
        try:
            tasks = await self._room_tasks_loader(room_id)
        except Exception:
            # A remote bus that fails costs the turn its tasks' note, not the turn.
            logger.warning("Could not read room %s's tasks for its turn", room_id, exc_info=True)
            return []
        note = render_tasks_note(tasks, now=datetime.now(UTC))
        return [note] if note else []

    def _add_skills(
        self,
        tools: list[AITool],
        system_prompt: str | None,
        activation_room: str | None,
        *,
        standalone: bool,
    ) -> str | None:
        """The system prompt with the skills' manifest and the bodies of the
        skills active in *activation_room*; the skill tools join *tools*
        (infra tools here, gated tools later)."""
        if not self._skills or not self._skills.has_entries:
            return system_prompt
        tools.extend(self._skill_tools())
        # The manifest block is skipped when the host renders its own skills
        # manifest inside ``system_prompt`` (``skills_in_prompt=False``).
        if self._skills_in_prompt:
            preamble = _SKILLS_PREAMBLE
            # No executor, or a policy that denies the tool: either way
            # the model must not be told it can run a skill's scripts.
            if not self._script_executor or not self._policy_allows(TOOL_RUN_SCRIPT):
                preamble += _SKILLS_NO_SCRIPTS_NOTE
            skills_xml = self._skills.to_prompt_xml()
            skill_block = f"\n\n{preamble}\n\n{skills_xml}"
            system_prompt = (system_prompt or "") + skill_block
        # Bodies of the skills activated in this room. Unlike the manifest
        # above, this is RUNTIME state, not the catalogue: a host that
        # renders its own manifest (``skills_in_prompt=False``) still cannot
        # know what the model activated mid-conversation, so this block is
        # injected either way, like the Tool Search and sandbox preambles. It
        # stays in the system prompt although it changes when a skill is
        # activated: it is instructions, not notes (RFC §24.4, §6.4).
        # This is what makes ``activate_skill``'s later ACKs safe: the rules
        # are in front of the model without the body being re-sent.
        active_skills = (
            None
            if standalone
            else self._skill_activation.render_prompt(activation_room, self._skills)
        )
        if active_skills:
            system_prompt = (system_prompt or "") + f"\n\n{active_skills}"

        return system_prompt

    def _add_sandbox(self, tools: list[AITool], system_prompt: str | None) -> str | None:
        """The system prompt with the sandbox preamble when the tool policy
        allows one of its tools; the sandbox tools join *tools*."""
        if self._sandbox is None:
            return system_prompt
        sandbox_allowed = False
        for tdef in self._sandbox.tool_definitions():
            name = tdef["name"]
            if not name.startswith(_SANDBOX_TOOL_PREFIX):
                logger.warning(
                    "Sandbox tool %r does not start with %r — skipping",
                    name,
                    _SANDBOX_TOOL_PREFIX,
                )
                continue
            sandbox_allowed = sandbox_allowed or self._policy_allows(name)
            tools.append(
                AITool(
                    name=name,
                    description=tdef.get("description", ""),
                    parameters=tdef.get("parameters", {}),
                )
            )
        # The preamble describes tools the policy may deny (RFC §21.1): with
        # none of them allowed it would promise what the model cannot call.
        if sandbox_allowed:
            system_prompt = (system_prompt or "") + f"\n\n{_SANDBOX_PREAMBLE}"

        return system_prompt

    def _never_hidden(self, room_id: str | None) -> set[str]:
        """What Tool Search never hides in *room_id*: what orchestration
        injected, the channel's own tools and the person's, as a realtime
        session declares them. It stays declared, outside the catalogue whose
        size decides the collapse (RFC §9.3, §21.1)."""
        own = self._registry.names(room_id, lambda traits: not traits.deferrable)
        return self._orchestration_tool_names(room_id) | own | self._human_input.declared_names

    def _collapse_behind_tool_search(
        self,
        tools: list[AITool],
        system_prompt: str | None,
        loop_ctx: _ToolLoopContext,
        binding: ChannelBinding,
        event: RoomEvent,
        standalone: bool,
    ) -> str | None:
        """Tool Search for the turn, applied to *tools* in place; the system
        prompt, with the Tool Search preamble when it hides the catalogue.

        When the catalogue is large, hide it behind the two discovery tools
        and let the model reveal what it needs via find_tools.
        Decided once here on the REAL catalogue (before the infra tools are
        added) and recorded on the loop ctx so every round's re-filter agrees.
        Unlike realtime, no provider.reconfigure is needed: the tool loop
        re-sends its (re-filtered) tool list every round.
        """
        window = self._provider.context_window
        never = self._never_hidden(binding.room_id)
        loop_ctx.tool_search_active = should_activate_tool_search(
            mode=self._tool_search,
            catalogue=[t for t in tools if t.name not in never],
            pinned=self._tool_search_pinned,
            window=window,
            threshold_pct=self._tool_search_threshold_pct,
            threshold_count=self._tool_search_threshold,
            threshold_tokens=self._tool_search_threshold_tokens,
        )
        if loop_ctx.tool_search_active:
            catalogue_names = {t.name for t in tools}
            # Parity with the realtime channel's Tool Search log: make the
            # deferral observable (the text path is otherwise silent about it).
            logger.info(
                "Tool Search active: %d tools deferred behind find_tools/list_tools "
                "(pinned=%d, window=%s)",
                len(tools),
                len((self._tool_search_pinned | never) & catalogue_names),
                window if window else "unknown",
            )
            tools.extend(t for t in search_tool_defs() if t.name not in catalogue_names)
            system_prompt = (system_prompt or "") + f"\n\n{TOOL_SEARCH_PREAMBLE}"
            # Re-reveal tools the agent already called this conversation so one it
            # used once stays callable even though Tool Search re-hides the
            # catalogue each turn. Seeded on ``sticky_tools`` (NOT ``revealed_tools``)
            # because the per-round re-filter runs under the for_loop CHILD ctx,
            # which inherits sticky_tools but resets revealed_tools — seeding the
            # latter here would be dropped at round 0. Intersected with the live
            # catalogue so a tool that has since disappeared (e.g. an edge device
            # unbound) is never surfaced as a phantom.
            if not standalone:
                loop_ctx.sticky_tools |= (
                    self._tool_usage.tool_names(event.room_id) & catalogue_names
                )
        return system_prompt

    async def _resolve_turn(
        self, binding: ChannelBinding, context: RoomContext
    ) -> tuple[AIChannelTurnConfig | None, dict[str, Any]]:
        """A turn's config provider result and its resolved per-turn settings.

        Each setting from the binding metadata, else the config provider,
        else the channel default (:meth:`_turn_settings`). The system prompt
        resolved here is the one the turn starts from, before the channel's
        own blocks (skills, sandbox, Tool Search).
        """
        turn = None
        if self._config_provider is not None:
            turn = await self._config_provider(binding, context)
        return turn, self._turn_settings(binding, turn)

    def _driven_turn(
        self,
        binding: ChannelBinding,
        loop_ctx: _ToolLoopContext,
        messages: list[AIMessage],
        tools: list[AITool],
    ) -> AIContext:
        """The context of a turn another component drives with a conversation
        of its own (a reasoning backend's, RFC §12.4.1): the channel's settings
        and turn budget as a turn with no room override starts from, the
        conversation's orphaned calls answered as a room turn's are, and
        *tools* as the turn's resolved toolset."""
        loop_ctx.turn_budget = self._turn_budget(binding, None)
        self._stamp_toolset(loop_ctx, tools)
        settings = self._turn_settings(binding, None)
        return AIContext(messages=patch_dangling_tool_calls(messages), tools=tools, **settings)

    def _prompt_identity(self, context: RoomContext) -> str | None:
        """What the channel says of itself at the end of the system prompt;
        nothing for a plain AI channel (an :class:`Agent` says its identity)."""
        return None

    def _measure_turn(
        self,
        loop_ctx: _ToolLoopContext,
        system_prompt: str | None,
        own_notes: list[str],
        max_tokens: int | None,
    ) -> None:
        """Measure what the turn takes of the window besides its history, as
        round 0 will send it, before the memory reads the room (RFC §20): the
        system prompt, the tools declared (Tool Search's collapse and the
        re-read tool included), the channel's notes with room for the speaker
        attribution the history may call for, and the reply budget."""
        declared = self._eviction.with_reread_tool(
            self._apply_tool_filters(list(loop_ctx.all_context_tools or []))
        )
        notes = turn_notes([_SPEAKER_ATTRIBUTION_NOTE, *own_notes]) or ""
        loop_ctx.turn_footprint = TurnFootprint(
            input_tokens=estimate_tokens(system_prompt or "")
            + sum(estimate_tool_tokens(tool) for tool in declared)
            + estimate_tokens(notes),
            reply_tokens=max_tokens or 0,
        )

    def _stamp_toolset(self, loop_ctx: _ToolLoopContext, tools: list[AITool]) -> None:
        """Make *tools* the turn's resolved toolset, with what the channel's
        policy, resolved for the turn's participant, admits of it: what
        ``current_tool_allowed_names()`` answers (RFC §21.4)."""
        loop_ctx.all_context_tools = tools
        role = loop_ctx.current_participant_role
        policy = None if self._tool_policy is None else self._tool_policy.resolve(role)
        loop_ctx.admits = policy_check(policy, self._exempt_tool_names)

    def _turn_settings(
        self, binding: ChannelBinding, turn: AIChannelTurnConfig | None
    ) -> dict[str, Any]:
        """Each per-turn setting from the binding metadata, else the config
        provider's result, else the channel default."""
        settings = {key: self._turn_value(key, binding, turn) for key in _TURN_SETTINGS}
        if self._thinking_turned_off(binding, turn):
            settings["thinking_budget"] = None
        return settings

    def _thinking_turned_off(
        self, binding: ChannelBinding, turn: AIChannelTurnConfig | None
    ) -> bool:
        """Whether an ``enable_thinking: false`` sits above the level that set
        the thinking budget (RFC §6.7, Appendix A.9).

        The budget states the switch before ``enable_thinking``, so a room
        that says off on a channel built with a budget would think anyway: a
        level's off turns off a budget a less specific level set, and a
        budget set at the same level or a more specific one still decides.
        """
        off_level, enabled = self._turn_source("enable_thinking", binding, turn)
        budget_level, budget = self._turn_source("thinking_budget", binding, turn)
        return enabled is False and budget is not None and off_level < budget_level

    def _turn_value(
        self, key: str, binding: ChannelBinding, turn: AIChannelTurnConfig | None
    ) -> Any:
        """The turn's *key* from the first level that sets it."""
        return self._turn_source(key, binding, turn)[1]

    def _turn_source(
        self, key: str, binding: ChannelBinding, turn: AIChannelTurnConfig | None
    ) -> tuple[int, Any]:
        """The first level that sets the turn's *key*, and its value: the
        binding metadata, the config provider's result, the channel default.

        ``None`` is "not set here" at every level, an explicit ``null`` in the
        binding metadata included (RFC Appendix A.9): a host that serializes
        an empty field never clears the channel's prompt or lifts its budget.
        """
        value = binding.metadata.get(key)
        if value is not None:
            return _BINDING_LEVEL, value
        value = getattr(turn, key) if turn is not None else None
        if value is not None:
            return _TURN_LEVEL, value
        return _CHANNEL_LEVEL, getattr(self, f"_{key}")

    def _turn_budget(
        self, binding: ChannelBinding, turn: AIChannelTurnConfig | None
    ) -> TurnBudget | None:
        """What this turn may spend, each budget resolved like the turn's
        other settings (RFC §6.4)."""
        tokens = self._turn_value("turn_budget_tokens", binding, turn)
        usd = self._turn_value("turn_budget_usd", binding, turn)
        return turn_budget(tokens, usd, self._provider, self._fallback_provider)

    async def _hydrate_room_memories(
        self, usage_room_id: str, activation_room_id: str | None
    ) -> None:
        """Rebuild the room-scoped working memories from persisted history.

        Both memories live on the channel object, which dies (process restart,
        cache expiry, the object swapped when another room attaches the same
        agent) while conversations outlive it. Without this the agent restarts
        amnesic mid-conversation: re-``find_tools``, re-fetches of data it
        already had, and a re-activation of every skill it was already running
        under. One fetch feeds both — they read different rows of the same
        ``TOOL_CALL_END`` history — and each keeps its own one-shot flag, so a
        memory already hydrated is never re-seeded from stale history.

        Each memory is keyed the way its own writer keys it: the usage digest
        on the event's room, the activation record on the tool loop's. The two
        are the same room in production — the ``RoomContext`` is built for the
        event — so this is one fetch, not two.
        """
        if self._tool_usage_loader is None:
            return
        usage_needs = self._tool_usage.needs_hydration(usage_room_id)
        skills_needs = self._skills is not None and self._skill_activation.needs_hydration(
            activation_room_id
        )
        if not usage_needs and not skills_needs:
            return
        try:
            past_calls = await self._tool_usage_loader(usage_room_id)
        except Exception:
            logger.exception("Tool-call hydration failed for room %s", usage_room_id)
            past_calls = []
        if usage_needs:
            self._tool_usage.seed(usage_room_id, past_calls)
        if skills_needs:
            self._skill_activation.seed(activation_room_id, past_calls)

    def _determine_role(self, event: RoomEvent) -> str:
        if event.source.channel_id == self.channel_id:
            return "assistant"
        return "user"

    def _transcript_content(self, event: RoomEvent) -> str | list[_ContentPart]:
        """What *event* says in the turn's transcript, history and input alike:
        its content, or, when that extracts to nothing, what the channel's
        ``describe_empty_event`` says of it. Empty omits the event."""
        content = self._extract_content(event)
        if content or self._describe_empty_event is None:
            return content
        return self._describe_empty_event(event) or ""

    def _extract_content(
        self,
        event: RoomEvent,
    ) -> str | list[_ContentPart]:
        """Extract content, including images if provider supports vision."""
        content = event.content

        if not self._provider.supports_vision:
            # Text-only fallback (existing behavior)
            return self._extract_text(event)

        # Build multimodal content
        if isinstance(content, TextContent):
            return content.body  # Simple case: just text

        if isinstance(content, MediaContent):
            parts: list[_ContentPart] = []
            if content.caption:
                parts.append(AITextPart(text=content.caption))
            parts.append(AIImagePart(url=content.url, mime_type=content.mime_type))
            return parts

        if isinstance(content, CompositeContent):
            cparts: list[_ContentPart] = []
            for part in content.parts:
                if isinstance(part, TextContent):
                    cparts.append(AITextPart(text=part.body))
                elif isinstance(part, MediaContent):
                    if part.caption:
                        cparts.append(AITextPart(text=part.caption))
                    cparts.append(AIImagePart(url=part.url, mime_type=part.mime_type))
            return cparts if cparts else ""

        # Fallback for other types
        return self._extract_text(event)

    def _extract_text(self, event: RoomEvent) -> str:
        if isinstance(event.content, TextContent):
            return event.content.body
        return ""


def event_speaker(event: RoomEvent, context: RoomContext) -> str | None:
    """Display name of whoever is behind an event, or ``None``.

    ``metadata["sender_name"]`` is the stamp transports and hosts write at
    ingress (the Teams/WhatsApp providers do, and so does a host's session
    ingress); the room's participant record is the fallback for transports
    that register named participants without stamping events.
    """
    name = event.metadata.get("sender_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    participant_id = event.source.participant_id
    if participant_id:
        for participant in context.participants:
            if participant.id == participant_id and participant.display_name:
                return participant.display_name
    return None


def _with_cut_mark(content: str | list[_ContentPart]) -> str | list[_ContentPart]:
    """An answer cut off by a barge-in, marked as such (RFC §6.4)."""
    if isinstance(content, str):
        return f"{content}\n{CUT_MARK}"
    return [*content, AITextPart(text=CUT_MARK)]


def _with_speaker_prefix(content: str | list[_ContentPart], name: str) -> str | list[_ContentPart]:
    """Carry the speaker on a user turn: ``"Name: text"``; parts get a lead part."""
    if isinstance(content, str):
        return f"{name}: {content}"
    return [AITextPart(text=f"{name}:"), *content]


def _after_memory(memory: list[AIMessage], rest: list[AIMessage]) -> list[AIMessage]:
    """The messages a memory provider built (a summary, say), then *rest*: a
    user text the provider ends on joins the user message that follows it
    rather than forming a second user message in a row."""
    last = memory[-1] if memory else None
    if last is None or last.role != "user" or not isinstance(last.content, str):
        return [*memory, *rest]
    return [*memory[:-1], *with_leading_text(last.content, rest)]


# What an answer may use only where every transport it reaches has it, and the
# limits the strictest of them sets.
_SHARED_FLAGS = (
    "supports_threading",
    "supports_reactions",
    "supports_edit",
    "supports_delete",
    "supports_read_receipts",
    "supports_typing",
    "supports_templates",
    "supports_rich_text",
    "supports_buttons",
    "supports_cards",
    "supports_quick_replies",
    "supports_media",
    "supports_audio",
    "supports_video",
)
_SMALLEST_LIMITS = (
    "max_length",
    "max_buttons",
    "max_media_size_bytes",
    "max_audio_duration_seconds",
    "max_video_duration_seconds",
)


def _intersected(capabilities: list[ChannelCapabilities]) -> ChannelCapabilities:
    """What all of *capabilities* allow: the media types they share, a flag
    set only where every one sets it, each limit the smallest one set."""
    merged = capabilities[0].model_dump()
    merged["media_types"] = list(set.intersection(*(set(c.media_types) for c in capabilities)))
    for other in capabilities[1:]:
        for name in _SHARED_FLAGS:
            merged[name] = merged[name] and getattr(other, name)
        for name in _SMALLEST_LIMITS:
            merged[name] = _smallest(merged[name], getattr(other, name))
    return ChannelCapabilities(**merged)


def _smallest(a: int | float | None, b: int | float | None) -> int | float | None:
    """The smaller of two limits, a limit of ``None`` being no limit."""
    if a is None or b is None:
        return b if a is None else a
    return min(a, b)
