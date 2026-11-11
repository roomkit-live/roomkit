"""AIChannel mixin for building the AI provider context from room events."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from roomkit._media import content_text, media_label, text_form
from roomkit.channels._ai_cuts import CUT_NOTE, cut_answers, cut_mark, cut_records
from roomkit.channels._ai_policy import policy_check
from roomkit.channels._ai_router import messages_request, turn_request
from roomkit.channels._author_images import (
    AuthorImages,
    RefusedImages,
    author_images,
    unseen_images,
    with_attached_media,
)
from roomkit.channels._dangling_recovery import patch_dangling_tool_calls
from roomkit.channels._discussion_turn import (
    discussion_handles,
    discussion_notes,
    discussion_turn,
)
from roomkit.channels._instruction import is_standalone, mark_instruction
from roomkit.channels._mark_copies import (
    compile_mark_patterns,
    content_without_mark_copies,
    cut_split_copies,
    without_mark_copies,
    without_split_blocks,
)
from roomkit.channels._runtime_record import written_by_runtime
from roomkit.channels._server_instructions import with_server_instructions
from roomkit.channels._skill_constants import (
    SERVED_SKILLS_PREAMBLE as _SERVED_SKILLS_PREAMBLE,
)
from roomkit.channels._skill_constants import (
    SKILLS_NO_SCRIPTS_NOTE as _SKILLS_NO_SCRIPTS_NOTE,
)
from roomkit.channels._skill_constants import (
    SKILLS_PREAMBLE as _SKILLS_PREAMBLE,
)
from roomkit.channels._skill_constants import TOOL_RUN_SCRIPT
from roomkit.channels._speaker import (
    SPEAKER_ATTRIBUTION_NOTE,
    SPEAKER_KEY,
    TurnSpeaker,
    labelled_lines,
    people_note,
    people_taking_part,
    several_speakers,
    turn_speakers,
)
from roomkit.channels._task_planner import TaskPlanner
from roomkit.channels._tasks_note import render_tasks_note
from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels._tool_history import (
    EarlierTurns,
    kept_instructions,
    tool_rounds,
    with_tool_rounds,
)
from roomkit.channels._tool_search import search_tool_defs, should_activate_tool_search
from roomkit.channels._tool_search_constants import TOOL_SEARCH_PREAMBLE
from roomkit.channels._turn_budget import TurnBudget, turn_budget
from roomkit.channels._turn_notes import (
    conversation_without_header_copies,
    turn_input,
    turn_notes,
    with_turn_images,
    with_turn_notes,
)
from roomkit.channels._user_text import with_leading_text
from roomkit.core.visibility import visible_events
from roomkit.memory.base import MemoryResult, _without_current_event
from roomkit.memory.token_estimator import (
    estimate_message_tokens,
    estimate_tokens,
    estimate_tool_tokens,
)
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
    from roomkit.channels._ai_callbacks import (
        RoomTasksLoader,
        RoomVisionLoader,
        ToolHistoryLoader,
        ToolUsageLoader,
    )
    from roomkit.channels._author_images import AttachedMedia, MediaResolver
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
    from roomkit.tools.timeout import ToolTimeouts

if TYPE_CHECKING:
    from roomkit.channels._ai_contract import _AIChannelContract
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")


TURN_INSTRUCTION = "_turn_instruction"
"""The binding metadata key of an instruction a strategy gives one turn
(a supervisor's task-formulation pass, RFC §19.7.3)."""


def with_turn_instruction(prompt: str | None, binding: ChannelBinding) -> str | None:
    """*prompt* followed by the instruction *binding* carries for its turn."""
    instruction = binding.metadata.get(TURN_INSTRUCTION)
    if not isinstance(instruction, str) or not instruction:
        return prompt
    return f"{prompt}\n\n{instruction}" if prompt else instruction


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


class AIContextMixin(_AIChannelContract):
    """Builds the AIContext passed to the provider from room state and events.

    What it calls on the other mixins is declared once, in
    :class:`~roomkit.channels._ai_contract._AIChannelContract`, which it
    derives from for the type checker only.
    """

    _tool_call_intents: bool
    _system_prompt: str | None
    _server_instructions: str | None
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
    _tool_history_loader: ToolHistoryLoader | None
    _room_tasks_loader: RoomTasksLoader | None
    _room_vision_loader: RoomVisionLoader | None
    _media_resolver: MediaResolver | None
    _attached_media: AttachedMedia | None
    _max_author_images: int | None
    _refused_images: RefusedImages
    _skill_activation: SkillActivationMemory
    _planner: TaskPlanner | None
    _user_tools: list[AITool]
    _config_provider: ConfigProvider | None
    _tool_timeouts: ToolTimeouts
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
        await compile_mark_patterns()
        binding = _without_former_vision(binding)
        turn, settings = await self._resolve_turn(binding, context)
        # The prompt grows below (skills, sandbox, planner, notes); the other
        # settings reach the context as resolved. An instruction a strategy
        # gives one turn follows the prompt the turn resolved (RFC §19.7.3).
        system_prompt = with_turn_instruction(settings.pop("system_prompt"), binding)

        tools = self._turn_base_tools(turn, binding)

        loop_ctx = self._get_loop_ctx()
        # The context is built for the provider the turn runs on (RFC §6.4).
        provider = await self._route_turn(loop_ctx, turn_request(event, binding.room_id))
        loop_ctx.turn_budget = self._turn_budget(binding, turn, provider)
        loop_ctx.tool_handler = turn.tool_handler if turn is not None else None
        loop_ctx.skills = self._given_turn_skills(turn)
        loop_ctx.tool_timeouts = self._given_turn_timeouts(turn)

        # A standalone instruction reads nothing of the room (RFC §10.1.1 step
        # 7): no history below, and none of the room's working memories either
        # — active skill bodies, the plan, the earlier rounds of tool calls
        # and sticky tools are the room's past in another form. The channel's
        # own prompt, tools and catalogue stay.
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

        own_notes = await self._own_notes(loop_ctx, context, standalone=standalone)
        own_notes += discussion_notes(event)
        earlier = await self._earlier_turns(event, context, standalone)
        self._measure_turn(
            loop_ctx,
            context,
            system_prompt,
            own_notes,
            settings.get("max_tokens"),
            earlier=earlier,
        )
        messages = await self._turn_conversation(
            event, context, loop_ctx, standalone, own_notes, earlier
        )
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
        they share, and their capabilities intersected (``None`` for none).
        A transport that does not send, the host channel among them, is never
        delivered to (RFC §4.2), so it narrows nothing."""
        transports = [
            b.capabilities
            for b in context.bindings
            if b.category == ChannelCategory.TRANSPORT
            and b.sends
            and b.channel_id != self.channel_id
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
        # Each event as the model reads it, the media the application keeps
        # beside its content among its parts (RFC §6.4): the provider measures
        # and summarizes what the turn will send.
        visible = [
            with_attached_media(past, self._attached_media)
            for past in visible_events(context, self.channel_id)
        ]
        result = await self._memory.retrieve(
            event.room_id,
            event,
            context.model_copy(update={"recent_events": visible}),
            channel_id=self.channel_id,
        )
        return _without_current_event(result, event)

    async def _turn_conversation(
        self,
        event: RoomEvent,
        context: RoomContext,
        loop_ctx: _ToolLoopContext,
        standalone: bool,
        own_notes: list[str],
        earlier: EarlierTurns,
    ) -> list[AIMessage]:
        """The conversation the model reads this turn: the history this channel
        sees, what it replays of its *earlier* turns (their rounds of tool
        calls, the instructions they read) at their place in it, then the
        input carrying the turn's notes, the channel's own (*own_notes*) after
        what the memory retrieved. The notes' header is the
        channel's alone: a copy the conversation holds is replaced (RFC §6.4).
        The event is read with the media its application keeps beside its
        content, as the history is."""
        event = with_attached_media(event, self._attached_media)
        memory_result = await self._visible_memory(event, context, standalone)
        loop_ctx.author_images = await self._author_images(event, memory_result)
        people = self._people(context, standalone)
        messages, attribute_speakers, requester = self._turn_messages(
            event, context, memory_result, loop_ctx, people, earlier
        )
        answered = next(
            (i for i, message in enumerate(messages) if message is loop_ctx.answered_input), None
        )
        # Who a task this turn delegates names as having asked (RFC §19.7).
        loop_ctx.requester = requester
        messages = conversation_without_header_copies(messages)
        notes = self._turn_notes(
            own_notes,
            speakers=attribute_speakers,
            people=people,
            cut=bool(self._cut_answers(context, memory_result)),
            retrieved=memory_result.notes,
        )
        messages = with_turn_notes(messages, notes)
        # The latest images the room's tools returned, after the notes: a
        # replayed result only says what they showed in its tool's words.
        messages = with_turn_images(messages, self._turn_images(loop_ctx))
        if answered is not None:
            loop_ctx.answered_input = messages[answered]
        return messages

    async def _author_images(self, event: RoomEvent, memory_result: MemoryResult) -> AuthorImages:
        """What the images the turn's history and input carry become (RFC §6.4);
        for a model that does not see, a mention that it does not see them."""
        if not self._turn_provider().supports_vision:
            return unseen_images([*memory_result.events, event])
        return await author_images(
            [*memory_result.events, event],
            room_id=event.room_id,
            resolver=self._media_resolver,
            limit=self._max_author_images,
            refused=self._refused_images.of(event.room_id),
        )

    def _turn_images(self, loop_ctx: _ToolLoopContext) -> list[AITextPart | AIImagePart]:
        """The images this channel's tool calls in the room returned that ride
        the turn's input, their labels quoting the calls with each copy of a
        runtime mark replaced (RFC §6.4); none for a standalone turn or a
        model that does not see."""
        if loop_ctx.standalone or not self._turn_provider().supports_vision:
            return []
        return [
            AITextPart(text=without_mark_copies(part.text))
            if isinstance(part, AITextPart)
            else part
            for part in self._tool_usage.render_images(loop_ctx.room_id)
        ]

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
        it: the MCP servers' instructions, skills, sandbox, planner, Tool Search,
        the large-result re-read and the channel's identity, in that order. Their
        tools join *tools* in place. The servers' instructions are the same on
        every turn: ahead of the skills, whose active bodies change, they stay in
        the prefix a provider caches (RFC §21.2).

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

        system_prompt = with_server_instructions(system_prompt, self._server_instructions)
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

    def _people(self, context: RoomContext, standalone: bool) -> tuple[str, ...]:
        """Who takes part besides the agent (RFC §6.4); no one for a standalone
        turn, which reads nothing of the room (RFC §10.1.1)."""
        return () if standalone else people_taking_part(context, self.channel_id)

    def _turn_messages(
        self,
        event: RoomEvent,
        context: RoomContext,
        memory_result: MemoryResult,
        loop_ctx: _ToolLoopContext,
        people: tuple[str, ...],
        earlier: EarlierTurns,
    ) -> tuple[list[AIMessage], bool, str | None]:
        """The turn's messages, its history with what it replays of its
        *earlier* turns at their place, then its input, whether several
        speakers are labelled in them, and the label of the turn's author when
        they are.

        ``_determine_role`` flattens every non-self event into one "user"
        stream, which erases who said what in a room where several people
        speak — the model can only guess the addressee, and it guesses wrong.
        The speaker is a fact of the event (``metadata["sender_name"]``,
        stamped at ingress by hosts and transport providers, or the channel
        of a sender without a name), so when the window holds two or more
        distinct speakers, or two *people* or more take part, silent ones
        included, each user turn opens with its label
        (:func:`~roomkit.channels._speaker.turn_labels`). A single-speaker room (a 1:1 DM) is left
        untouched, except in a thinker's context, which names its one speaker too
        (``loop_ctx.names_every_speaker``, RFC §6.4).
        """
        speakers = turn_speakers([*memory_result.events, event], context)
        past_turns, past_indexes = self._past_turns(memory_result, context, speakers, loop_ctx)
        current_content, current_speaker = self._turn_input(event, loop_ctx, speakers)
        said = [speaker for _, _, speaker in past_turns if speaker is not None]
        if current_content and current_speaker is not None:
            said.append(current_speaker)
        attribute_speakers = _labels_turns(said, memory_result.speakers, people, loop_ctx)
        current_label = (
            current_speaker.label if attribute_speakers and current_speaker is not None else None
        )

        # Pre-built messages from memory (e.g. summaries)
        memory = [_runtime_or_cleaned(message) for message in memory_result.messages]
        past = [
            _turn_message(role, content, speaker.label if attribute_speakers and speaker else None)
            for role, content, speaker in past_turns
        ]
        past, past_indexes = with_tool_rounds(
            past,
            past_indexes,
            earlier.rounds,
            before=_rounds_before(event),
            # An event that reads as nothing opens no history for a round.
            answered=_answered_index(event) if current_content else None,
            said=self._own_turns(memory_result),
            instructions=earlier.instructions,
        )
        messages: list[AIMessage] = [*memory, *past]

        current = (
            _turn_message("user", current_content, current_label) if current_content else None
        )
        # A discussion's turn answers an event others may have followed: it
        # reads at its place, before them, not as the last message (§19.7.5).
        at = _answered_position(event, past_indexes)
        if current is not None and at is not None:
            messages.insert(len(memory) + at, current)
            loop_ctx.answered_input = current
            current = None

        # Patch orphaned tool calls from interrupted tool loops (barge-in)
        messages = patch_dangling_tool_calls(messages)

        if current is not None:
            messages.append(current)
        conversation = _conversation(messages, len(memory), loop_ctx)
        return conversation, attribute_speakers, current_label

    def _own_turns(self, memory_result: MemoryResult) -> dict[int, str]:
        """The turn each of this channel's own events in the history belongs
        to, by its index: the text a round's model wrote with its calls."""
        return {
            past.index: past.correlation_id
            for past in memory_result.events
            if past.source.channel_id == self.channel_id and past.correlation_id is not None
        }

    def _past_turns(
        self,
        memory_result: MemoryResult,
        context: RoomContext,
        speakers: dict[str, TurnSpeaker | None],
        loop_ctx: _ToolLoopContext,
    ) -> tuple[list[tuple[str, str | list[_ContentPart], TurnSpeaker | None]], list[int | None]]:
        """The history's turns as (role, content, speaker), a user turn naming
        who said it, and the room index of the event each turn is."""
        past_turns: list[tuple[str, str | list[_ContentPart], TurnSpeaker | None]] = []
        indexes: list[int | None] = []
        # An answer cut off by a barge-in reads as cut, not as heard whole (§6.4).
        cuts = self._cut_answers(context, memory_result)
        images = loop_ctx.author_images or _NO_IMAGES
        for past_event in memory_result.events:
            if past_event.metadata.get("cancellation_reason") == SUPERSEDED:
                # A response nobody heard: the user continued the turn first,
                # and replaying it would answer what they never heard (§12.3.12).
                continue
            role = self._determine_role(past_event)
            content = self._transcript_content(past_event, images)
            if content and past_event.id in cuts:
                content = _with_cut_mark(
                    content, cut_mark(cuts[past_event.id], without_mark_copies)
                )
            if content:
                speaker = speakers.get(past_event.id) if role == "user" else None
                past_turns.append((role, content, speaker))
                indexes.append(past_event.index)
        return past_turns, indexes

    def _turn_input(
        self, event: RoomEvent, loop_ctx: _ToolLoopContext, speakers: dict[str, TurnSpeaker | None]
    ) -> tuple[str | list[_ContentPart] | None, TurnSpeaker | None]:
        """The turn's input and its speaker; an instruction marked as the
        application's, with none."""
        current_content = self._transcript_content(event, loop_ctx.author_images or _NO_IMAGES)
        current_speaker = speakers.get(event.id)
        if event.type == EventType.INSTRUCTION:
            # The application's direction for this one turn (RFC §10.1.1),
            # the text its content reads as (§6.6). It is the turn's input — a
            # system-role message after the history is refused or silently
            # re-roled by several model APIs — marked so the model never reads
            # it as a participant's words. The pipeline records on each reply
            # which instruction produced it (step 6).
            current_content = _instruction_input(text_form(event.content))
            current_speaker = None
        return current_content, current_speaker

    def _cut_answers(
        self, context: RoomContext, memory_result: MemoryResult
    ) -> dict[str, RoomEvent]:
        """The cut record of each of this channel's answers the history holds
        cut off by a barge-in, by the id of the event that ends the answer."""
        records = cut_records(context, self.channel_id)
        return cut_answers(memory_result.events, records, self.channel_id)

    @staticmethod
    def _turn_notes(
        own: list[str],
        *,
        speakers: bool,
        people: tuple[str, ...],
        retrieved: list[str],
        cut: bool = False,
    ) -> str | None:
        """What changes from one turn to the next, as the notes the turn's
        input carries (RFC §6.4): how speakers are named when several speak,
        who takes part when several people do, how a cut answer reads when
        the history holds one, what the memory retrieved for this turn, then
        the channel's *own*.
        What the memory retrieved and the channel's own blocks quote others
        (a passage, a tool's result, a task, what a camera read, a plan): a
        copy of a runtime mark in them is replaced; the speaker and people
        notes are the runtime's."""
        blocks = [SPEAKER_ATTRIBUTION_NOTE] if speakers else []
        if note := people_note(people):
            blocks.append(note)
        if cut:
            blocks.append(CUT_NOTE)
        quoting = [without_mark_copies(block) for block in (*retrieved, *own)]
        return turn_notes([*blocks, *without_split_blocks(quoting)])

    def _channel_notes(self, loop_ctx: _ToolLoopContext, *, standalone: bool) -> list[str]:
        """The notes the channel adds to the turn's input from the room's
        working memories: the room's plan. The tools already used ride the
        history instead (``_replayed_rounds``).

        A standalone turn reads none of them (RFC §10.1.1). Each is read under
        the tool loop's room, as its writer keys it.
        """
        room_id = loop_ctx.room_id
        if standalone or room_id is None or self._planner is None:
            return []
        plan = self._planner.plan_for(room_id)
        return [TaskPlanner.format_plan_prompt(plan)] if plan else []

    async def _earlier_turns(
        self, event: RoomEvent, context: RoomContext, standalone: bool
    ) -> EarlierTurns:
        """What the turn's history replays of this channel's earlier turns in
        the room, at their place: their rounds of tool calls (RFC §6.4) and the
        instructions they read (RFC §10.1.1); nothing for a standalone turn,
        which reads nothing of the room.

        Read from the store every turn, the channel's own rows only: what a
        call asked and returned outlives the channel object, and a model reads
        it the way the turn that made the call did. Every row from the room's
        message before its context, or before the event a discussion's turn
        answers: the history the memory reads lies within it, and the
        instruction that opened its first turn just before.
        """
        if standalone or self._tool_history_loader is None:
            return EarlierTurns()
        first = history_start(event, context)
        if first is None:
            return EarlierTurns()
        try:
            rows = await self._tool_history_loader(event.room_id, first)
        except Exception:
            # A store that fails costs the turn its earlier calls, not the turn.
            logger.warning(
                "Could not read room %s's tool calls for its turn", event.room_id, exc_info=True
            )
            return EarlierTurns()
        rounds = tool_rounds(rows, resolves=self._eviction.holds, intents=self._tool_call_intents)
        return EarlierTurns(rounds, kept_instructions(rows, _instruction_input))

    async def _tasks_notes(
        self, loop_ctx: _ToolLoopContext, context: RoomContext, *, standalone: bool
    ) -> list[str]:
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
        handles = discussion_handles(context.room.metadata)
        note = render_tasks_note(tasks, now=datetime.now(UTC), handles=handles)
        return [note] if note else []

    async def _vision_notes(self, context: RoomContext, *, standalone: bool) -> list[str]:
        """What the room's video last showed, for the turn's notes (RFC
        §12.8.7), fenced as data; a standalone turn reads none."""
        if standalone or self._room_vision_loader is None:
            return []
        try:
            note = await self._room_vision_loader(context)
        except Exception:
            # An owner that cannot answer costs the turn its view, not the turn.
            logger.warning(
                "Could not read room %s's view for its turn", context.room.id, exc_info=True
            )
            return []
        return [note] if note else []

    async def _own_notes(
        self, loop_ctx: _ToolLoopContext, context: RoomContext, *, standalone: bool
    ) -> list[str]:
        """The notes the channel adds to the turn's input: the room's plan, its
        background tasks, what its video last showed."""
        notes = self._channel_notes(loop_ctx, standalone=standalone)
        notes += await self._tasks_notes(loop_ctx, context, standalone=standalone)
        return notes + await self._vision_notes(context, standalone=standalone)

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
        skills = self._turn_skills()
        if not skills or not skills.has_entries:
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
            if skills.has_served:
                preamble += _SERVED_SKILLS_PREAMBLE
            skills_xml = skills.to_prompt_xml()
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
            None if standalone else self._skill_activation.render_prompt(activation_room, skills)
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
        window = self._turn_provider().context_window
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

    def _given_turn_timeouts(self, turn: AIChannelTurnConfig | None) -> ToolTimeouts | None:
        """The bounds of *turn*'s calls (RFC §21.6): the channel's under the
        ones its configuration gives, ``None`` when it gives none. A bound
        that is not positive fails the turn, before its model is asked."""
        given = turn.tool_timeouts if turn is not None else None
        if given is None:
            return None
        return self._tool_timeouts.overridden(given)

    def _given_turn_skills(self, turn: AIChannelTurnConfig | None) -> SkillRegistry | None:
        """The skill registry *turn*'s configuration gives it (RFC §24.3),
        read only on a channel that serves skills: one built without a
        registry declares no skill tools for a turn's to use."""
        given = turn.skills if turn is not None else None
        if given is not None and self._skills is None:
            logger.warning(
                "Channel %s was built without skills: the turn's skill registry is "
                "ignored (build the channel with a registry, an empty one will do)",
                self.channel_id,
            )
            return None
        return given

    async def _driven_turn(
        self,
        binding: ChannelBinding,
        loop_ctx: _ToolLoopContext,
        messages: list[AIMessage],
        tools: list[AITool],
    ) -> AIContext:
        """The context of a turn another component drives with a conversation
        of its own (a reasoning backend's, RFC §12.4.1): the channel's settings
        and turn budget as a turn with no room override starts from, the
        conversation's orphaned calls answered as a room turn's are, its MCP
        servers' instructions after its prompt (RFC §21.2), and *tools* as the
        turn's resolved toolset. The turn is routed on its last user message
        first (RFC §6.4), so its budget is priced on the provider it runs on."""
        provider = await self._route_turn(
            loop_ctx, messages_request(messages, binding.room_id or None)
        )
        loop_ctx.turn_budget = self._turn_budget(binding, None, provider)
        self._stamp_toolset(loop_ctx, tools)
        settings = self._turn_settings(binding, None)
        prompt = settings["system_prompt"]
        settings["system_prompt"] = with_server_instructions(prompt, self._server_instructions)
        return AIContext(messages=patch_dangling_tool_calls(messages), tools=tools, **settings)

    def _prompt_identity(self, context: RoomContext) -> str | None:
        """What the channel says of itself at the end of the system prompt;
        nothing for a plain AI channel (an :class:`Agent` says its identity)."""
        return None

    def _measure_turn(
        self,
        loop_ctx: _ToolLoopContext,
        context: RoomContext,
        system_prompt: str | None,
        own_notes: list[str],
        max_tokens: object,
        *,
        earlier: EarlierTurns,
    ) -> None:
        """Measure what the turn takes of the window besides its history, as
        round 0 will send it, before the memory reads the room (RFC §20): the
        system prompt, the tools declared (Tool Search's collapse and the
        re-read tool included), the channel's notes with room for the speaker
        attribution, the people taking part and the cut answer's note the
        history may call for, the images the input replays, what the history
        replays of the *earlier* turns, their rounds of tool calls and the
        instructions they read (all of them: the window may drop some, never
        add one), and the reply budget."""
        declared = self._eviction.with_reread_tool(
            self._apply_tool_filters(list(loop_ctx.all_context_tools or []))
        )
        people = self._people(context, loop_ctx.standalone)
        notes = self._turn_notes(own_notes, speakers=True, people=people, retrieved=[], cut=True)
        notes = notes or ""
        images = self._turn_images(loop_ctx)
        loop_ctx.turn_footprint = TurnFootprint(
            input_tokens=estimate_tokens(system_prompt or "")
            + sum(estimate_tool_tokens(tool) for tool in declared)
            + estimate_tokens(notes)
            + (estimate_message_tokens(AIMessage(role="user", content=images)) if images else 0),
            reply_tokens=_reply_tokens(max_tokens),
            replayed=_replayed_costs(earlier),
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
        prompt = settings["system_prompt"]
        if prompt is not None and not isinstance(prompt, str):
            # Read as AIContext reads it, before the turn reads it as a text:
            # a value it refuses fails the turn, every refused setting named.
            settings["system_prompt"] = AIContext(**settings).system_prompt
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
        return _reads_off(enabled) and budget is not None and off_level < budget_level

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
        self, binding: ChannelBinding, turn: AIChannelTurnConfig | None, provider: AIProvider
    ) -> TurnBudget | None:
        """What this turn may spend, each budget resolved like the turn's
        other settings, priced on *provider*, the one the turn starts on
        (RFC §6.4)."""
        tokens = self._turn_value("turn_budget_tokens", binding, turn)
        usd = self._turn_value("turn_budget_usd", binding, turn)
        return turn_budget(tokens, usd, provider)

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

        Each memory is keyed the way its own writer keys it: the tool usage
        on the event's room, the activation record on the tool loop's. The two
        are the same room in production — the ``RoomContext`` is built for the
        event — so this is one fetch, not two.
        """
        if self._tool_usage_loader is None:
            return
        usage_needs = self._tool_usage.needs_hydration(usage_room_id)
        skills = self._turn_skills()
        skills_needs = skills is not None and self._skill_activation.needs_hydration(
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
            skip = skills.is_served if skills is not None else None
            self._skill_activation.seed(activation_room_id, past_calls, skip=skip)

    def _determine_role(self, event: RoomEvent) -> str:
        if event.source.channel_id == self.channel_id:
            return "assistant"
        return "user"

    def _transcript_content(
        self, event: RoomEvent, images: AuthorImages
    ) -> str | list[_ContentPart]:
        """What *event* says in the turn's transcript, history and input alike:
        its content, or, when that extracts to nothing, what the channel's
        ``describe_empty_event`` says of it. Empty omits the event. A copy of a
        runtime mark in it is replaced here, before the runtime places its own
        marks (an instruction's, a cut answer's) after it (RFC §6.4)."""
        content = self._extract_content(event, images)
        if not content and self._describe_empty_event is not None:
            content = self._describe_empty_event(event) or ""
        if not content and isinstance(event.content, MediaContent):
            # A media the host does not describe reads as its label (RFC §6.6).
            content = media_label(event.content)
        if written_by_runtime(event.metadata):
            # The runtime's own record (a handoff's): its marks are its own.
            return content
        return content_without_mark_copies(content)

    def _extract_content(self, event: RoomEvent, images: AuthorImages) -> str | list[_ContentPart]:
        """What *event* says to the model: its text, and for a model that sees,
        each media as *images* decided (its image, else a mention)."""
        content = event.content
        if not self._turn_provider().supports_vision:
            return self._extract_text(event, images)
        if isinstance(content, TextContent):
            return content.body
        if isinstance(content, MediaContent):
            return list(images.parts(event.id, 0, content))
        if isinstance(content, CompositeContent):
            parts: list[_ContentPart] = []
            for index, part in enumerate(content.parts):
                if isinstance(part, MediaContent):
                    parts.extend(images.parts(event.id, index, part))
                elif text := content_text(part):
                    # Any other part as a text-only reader reads it (RFC §6.6).
                    parts.append(AITextPart(text=text))
            return parts or ""
        return self._extract_text(event, images)

    @staticmethod
    def _extract_text(event: RoomEvent, images: AuthorImages) -> str:
        """What *event* says to a model that does not see: its text, an image
        as *images* decided (a mention that the model does not see it), any
        other media as its label (RFC §6.6), never bytes; nothing for a lone
        media without a caption that is no image, which the host's describer
        has the word on first."""
        content = event.content
        if isinstance(content, MediaContent):
            if not content.caption and (event.id, 0) not in images.decided:
                return ""
            return _text_of(images.parts(event.id, 0, content))
        if isinstance(content, CompositeContent):
            texts = (
                _text_of(images.parts(event.id, index, part))
                if isinstance(part, MediaContent)
                else content_text(part)
                for index, part in enumerate(content.parts)
            )
            return "\n".join(text for text in texts if text)
        return content_text(content) or ""


def _text_of(parts: list[AITextPart | AIImagePart]) -> str:
    """The text parts of *parts*, one per line."""
    return "\n".join(part.text for part in parts if isinstance(part, AITextPart))


_NO_IMAGES = AuthorImages()
"""A turn whose authors' images nothing decided: each media reads as its label."""

_FORMER_VISION_BASE = "_base_system_prompt"
"""Where RoomKit's vision path up to 0.95 kept a binding's own prompt when it
wrote what a camera saw into ``system_prompt``."""


def _without_former_vision(binding: ChannelBinding) -> ChannelBinding:
    """*binding* with its own prompt back where RoomKit's vision path up to 0.95
    wrote what a camera saw into it (RFC §12.8.7): a persisted binding would
    otherwise keep the last text a camera read in its prompt for good."""
    if _FORMER_VISION_BASE not in binding.metadata:
        return binding
    metadata = {k: v for k, v in binding.metadata.items() if k != _FORMER_VISION_BASE}
    metadata["system_prompt"] = binding.metadata[_FORMER_VISION_BASE] or None
    return binding.model_copy(update={"metadata": metadata})


def _reads_off(enabled: object) -> bool:
    """Whether *enabled* reads as false, as ``AIContext`` reads its
    ``enable_thinking`` (RFC Appendix A.9): ``"off"`` and ``0`` as ``False``.
    A value it refuses is no off: it fails the turn when the context is built."""
    try:
        return AIContext.model_validate({"enable_thinking": enabled}).enable_thinking is False
    except ValidationError:
        return False


def _reply_tokens(max_tokens: object) -> int:
    """The reply budget a turn's resolved *max_tokens* reserves, read as
    AIContext reads it (RFC §6.4, Appendix A.9): ``"100"`` is 100. 0 for no
    budget, and for a value the context refuses, which then fails the turn
    when its context is built, every refused setting named."""
    if max_tokens is None:
        return 0
    try:
        return AIContext.model_validate({"max_tokens": max_tokens}).max_tokens or 0
    except ValidationError:
        return 0


def _replayed_costs(earlier: EarlierTurns) -> tuple[tuple[int, str | None, int], ...]:
    """What each round and instruction the history replays costs, by its
    timeline index and its turn, for the memory to charge to the event it
    reads with (RFC §20)."""
    rounds = (
        (r.index, r.turn, estimate_message_tokens(r.calls) + estimate_message_tokens(r.results))
        for r in earlier.rounds
    )
    instructions = (
        (kept.index, kept.turn, estimate_message_tokens(kept.message))
        for kept in earlier.instructions
    )
    return (*rounds, *instructions)


def _instruction_input(text: str) -> str | None:
    """The turn's input an instruction's *text* reads as, marked as the
    application's, its copies of the runtime's marks replaced; ``None`` for a
    text that reads as nothing. One form for the turn the instruction opens
    and for the later turns that replay it (RFC §10.1.1)."""
    return mark_instruction(without_mark_copies(text)) if text else None


def _rounds_before(event: RoomEvent) -> int | None:
    """Where the earlier rounds the turn replays end: before its event, which
    reads last. A discussion's turn reads its event at its place, before the
    turns that followed it, rounds of tool calls included (§19.7.5). Index 0
    is an event the store never numbered: nothing came before it."""
    if discussion_turn(event) is not None:
        return None
    return event.index or None


def history_start(event: RoomEvent, context: RoomContext) -> int | None:
    """The index a turn's history starts at: its oldest event, or the event a
    discussion's turn answers; ``None`` for a history with neither."""
    answered = _answered_index(event)
    known = [past.index for past in context.recent_events]
    known += [] if answered is None else [answered]
    return min(known) if known else None


def _answered_index(event: RoomEvent) -> int | None:
    """What a round must come after to read after the event a discussion's turn
    answers, read at its place among the history's turns (§19.7.5): that event's
    index, one less for an instruction, which, never stored, holds the index of
    what followed it, a round among them; None for any other turn, or an event
    the store never numbered."""
    if discussion_turn(event) is None or not event.index:
        return None
    return event.index - 1 if event.type == EventType.INSTRUCTION else event.index


def _answered_position(event: RoomEvent, past_indexes: list[int | None]) -> int | None:
    """Where a discussion's turn reads the event it answers among the history's
    turns: before the first turn that came after it; None when it reads last
    (no discussion gave the turn, the event is unstored, or nothing followed)."""
    if discussion_turn(event) is None or event.index is None:
        return None
    before = sum(1 for index in past_indexes if index is not None and index < event.index)
    return before if before < len(past_indexes) else None


def _with_cut_mark(content: str | list[_ContentPart], mark: str) -> str | list[_ContentPart]:
    """An answer cut off by a barge-in, followed by *mark* (RFC §6.4)."""
    if isinstance(content, str):
        return f"{content}\n{mark}"
    return [*content, AITextPart(text=mark)]


def _labels_turns(
    said: list[TurnSpeaker],
    remembered: list[str],
    people: tuple[str, ...],
    loop_ctx: _ToolLoopContext,
) -> bool:
    """Whether the turn's messages open with their speakers' labels (RFC
    §6.4): several speakers counted by source, or several people taking
    part; the people a summary named count too, since a summary that names
    Alice leaves no unlabelled "Alice: ..." turn after it. A thinker's
    context names its one speaker too."""
    return several_speakers(said, people, remembered) or (
        loop_ctx.names_every_speaker and bool(said or remembered)
    )


def _turn_message(role: str, content: str | list[_ContentPart], speaker: str | None) -> AIMessage:
    """A turn's message; a named *speaker*'s carries their name before their
    words, and under :data:`SPEAKER_KEY`, where a transcript reads it."""
    if not speaker:
        return AIMessage(role=role, content=content)
    return AIMessage(
        role=role,
        content=_with_speaker_prefix(content, speaker),
        metadata={SPEAKER_KEY: speaker},
    )


def _with_speaker_prefix(content: str | list[_ContentPart], name: str) -> str | list[_ContentPart]:
    """Carry the speaker on a user turn: ``Name: "line"`` for each line, the
    line a JSON string (:func:`~roomkit.channels._speaker.labelled_lines`);
    each line of a text part gets its label, and parts that open with an
    image a lead part."""
    if isinstance(content, str):
        return labelled_lines(content, name)
    parts = [
        part.model_copy(update={"text": labelled_lines(part.text, name)})
        if isinstance(part, AITextPart)
        else part
        for part in content
    ]
    if parts and isinstance(parts[0], AITextPart) and parts[0].text.strip():
        return parts
    return [AITextPart(text=f"{name}:"), *parts]


def _runtime_or_cleaned(message: AIMessage) -> AIMessage:
    """A message a memory built: a copy of a runtime mark in it replaced, unless
    a memory of the runtime built it (a summary, the context handed on), whose
    marks are its own (RFC §6.4)."""
    if written_by_runtime(message.metadata):
        return message
    cleaned = content_without_mark_copies(message.content)
    return (
        message if cleaned == message.content else message.model_copy(update={"content": cleaned})
    )


def _conversation(
    messages: list[AIMessage], built: int, loop_ctx: _ToolLoopContext
) -> list[AIMessage]:
    """*messages*, the first *built* of them a memory's, as the model reads
    them: a copy of a mark split over consecutive user messages cut, the
    memory's last message and the turn after it included, then the memory's
    user text joined to that turn (RFC §6.4)."""
    answered = next(
        (i for i, message in enumerate(messages) if message is loop_ctx.answered_input), None
    )
    cut, emptied = cut_split_copies(messages)
    kept = [message for i, message in enumerate(cut) if i not in emptied]
    built -= sum(1 for i in emptied if i < built)
    conversation = _after_memory(kept[:built], kept[built:])
    if answered is not None:
        at = answered - sum(i < answered for i in emptied) + len(conversation) - len(kept)
        loop_ctx.answered_input = None if answered in emptied else conversation[at]
    return conversation


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
