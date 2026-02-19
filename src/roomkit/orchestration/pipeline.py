"""Pipeline helper for sequential agent workflows.

ConversationPipeline generates RoutingRules for structured multi-agent
workflows with optional loops (e.g., coder <-> reviewer).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.orchestration.handoff import HandoffHandler, build_handoff_tool, setup_handoff
from roomkit.orchestration.router import ConversationRouter, RoutingConditions, RoutingRule

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.orchestration")


class PipelineStage(BaseModel):
    """A stage in a conversation pipeline."""

    phase: str
    agent_id: str
    next: str | None = None
    can_return_to: set[str] = Field(default_factory=set)
    description: str | None = None


class ConversationPipeline:
    """Generates routing rules for sequential agent workflows.

    Example::

        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="analysis", agent_id="agent-discuss", next="coding"),
                PipelineStage(phase="coding", agent_id="agent-coder", next="review"),
                PipelineStage(phase="review", agent_id="agent-reviewer",
                              next="report", can_return_to={"coding"}),
                PipelineStage(phase="report", agent_id="agent-writer", next=None),
            ],
            supervisor_id="agent-supervisor",
        )
        router = pipeline.to_router()
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        default_phase: str | None = None,
        supervisor_id: str | None = None,
    ) -> None:
        self._stages = stages
        self._default_phase = default_phase or (stages[0].phase if stages else None)
        self._supervisor_id = supervisor_id
        self._stage_map = {s.phase: s for s in stages}
        self._validate()

    def _validate(self) -> None:
        """Validate pipeline graph consistency."""
        phase_names = {s.phase for s in self._stages}
        for stage in self._stages:
            if stage.next and stage.next not in phase_names:
                msg = f"Stage '{stage.phase}' has next='{stage.next}' which is not a valid phase"
                raise ValueError(msg)
            for ret in stage.can_return_to:
                if ret not in phase_names:
                    msg = (
                        f"Stage '{stage.phase}' has can_return_to='{ret}' "
                        f"which is not a valid phase"
                    )
                    raise ValueError(msg)

    @property
    def stages(self) -> list[PipelineStage]:
        """The pipeline stages."""
        return list(self._stages)

    def to_router(self) -> ConversationRouter:
        """Generate a ConversationRouter from this pipeline."""
        rules = []
        for i, stage in enumerate(self._stages):
            rules.append(
                RoutingRule(
                    agent_id=stage.agent_id,
                    conditions=RoutingConditions(phases={stage.phase}),
                    priority=i,
                )
            )

        default_stage = self._stage_map.get(self._default_phase or "")
        default_agent = default_stage.agent_id if default_stage else None

        return ConversationRouter(
            rules=rules,
            default_agent_id=default_agent,
            supervisor_id=self._supervisor_id,
        )

    def get_phase_map(self) -> dict[str, str]:
        """Return agent_id -> default phase mapping for HandoffHandler."""
        return {s.agent_id: s.phase for s in self._stages}

    def get_allowed_transitions(self) -> dict[str, set[str]]:
        """Return phase -> allowed next phases for validation."""
        transitions: dict[str, set[str]] = {}
        for stage in self._stages:
            allowed: set[str] = set()
            if stage.next:
                allowed.add(stage.next)
            allowed.update(stage.can_return_to)
            transitions[stage.phase] = allowed
        return transitions

    def install(
        self,
        kit: RoomKit,
        agents: list[Agent],
        *,
        agent_aliases: dict[str, str] | None = None,
        hook_priority: int = -100,
        greet_on_handoff: bool = False,
        voice_channel_id: str | None = None,
        greeting_prompt: str | None = None,
    ) -> tuple[ConversationRouter, HandoffHandler]:
        """Wire routing and handoff in one call.

        Creates a router from this pipeline, registers it as a
        ``BEFORE_BROADCAST`` sync hook, builds a ``HandoffHandler``
        with the pipeline's phase map and transition constraints,
        and calls ``setup_handoff`` on every agent.

        When *greet_on_handoff* is ``True``, two extra hooks are
        registered:

        - **ON_HANDOFF** (async): blocks the old agent's farewell via
          a ``BEFORE_TTS`` flag, then sends a synthetic inbound message
          on *voice_channel_id* to prompt the new agent to greet.
        - **BEFORE_TTS** (sync): blocks TTS while a handoff is pending.

        Returns ``(router, handler)`` for further customisation.
        """
        if greet_on_handoff and not voice_channel_id:
            raise ValueError("greet_on_handoff=True requires voice_channel_id")

        router = self.to_router()

        kit.hook(
            HookTrigger.BEFORE_BROADCAST,
            execution=HookExecution.SYNC,
            priority=hook_priority,
        )(router.as_hook())

        # Detect speech-to-speech mode
        is_realtime = False
        if voice_channel_id:
            from roomkit.channels.realtime_voice import RealtimeVoiceChannel

            is_realtime = isinstance(kit._channels.get(voice_channel_id), RealtimeVoiceChannel)

        handler = HandoffHandler(
            kit=kit,
            router=router,
            agent_aliases=agent_aliases,
            phase_map=self.get_phase_map(),
            allowed_transitions=self.get_allowed_transitions(),
            known_agents=({a.channel_id for a in agents} if is_realtime else None),
            event_channel_id=(voice_channel_id if is_realtime else None),
        )

        # Populate agent metadata from Agent fields
        handler._greeting_map = {
            a.channel_id: g for a in agents if (g := getattr(a, "greeting", None)) is not None
        }
        handler._agents = {a.channel_id: a for a in agents}

        if is_realtime:
            self._wire_realtime(
                kit,
                agents,
                handler,
                voice_channel_id,  # type: ignore[arg-type]
                greet_on_handoff=greet_on_handoff,
                greeting_prompt=greeting_prompt,
            )
        else:
            self._wire_handoff(agents, handler)

            if voice_channel_id:
                self._wire_voice_map(kit, agents, voice_channel_id)

            if greet_on_handoff:
                self._register_greet_hooks(
                    kit,
                    voice_channel_id=voice_channel_id,  # type: ignore[arg-type]
                    greeting_prompt=greeting_prompt,
                    hook_priority=hook_priority,
                )

        return router, handler

    def _wire_handoff(
        self,
        agents: list[Agent],
        handler: HandoffHandler,
    ) -> None:
        """Set up per-agent handoff tools with constrained target enums.

        Each agent's handoff tool ``target`` parameter is restricted to
        only the reachable agent IDs (with descriptions derived from
        ``Agent.description`` or ``PipelineStage.description``).
        """
        agent_map: dict[str, Agent] = {a.channel_id: a for a in agents}
        stage_by_agent: dict[str, PipelineStage] = {s.agent_id: s for s in self._stages}

        for agent in agents:
            stage = stage_by_agent.get(agent.channel_id)
            if stage is None:
                # Agent not in pipeline — generic tool
                setup_handoff(agent, handler)
                continue

            # Compute reachable targets
            reachable_phases: set[str] = set()
            if stage.next:
                reachable_phases.add(stage.next)
            reachable_phases.update(stage.can_return_to)

            targets: list[tuple[str, str | None]] = []
            for s in self._stages:
                if s.phase in reachable_phases and s.agent_id != agent.channel_id:
                    # Prefer Agent.description, fall back to PipelineStage.description
                    target_agent = agent_map.get(s.agent_id)
                    desc = target_agent.description if target_agent else None
                    if desc is None:
                        desc = s.description
                    targets.append((s.agent_id, desc))

            tool = build_handoff_tool(targets)
            setup_handoff(agent, handler, tool=tool)

    def _wire_voice_map(
        self,
        kit: RoomKit,
        agents: list[Agent],
        voice_channel_id: str,
    ) -> None:
        """Auto-wire voice map entries from Agent.voice fields."""
        from roomkit.channels.voice import VoiceChannel

        auto_map: dict[str, str] = {a.channel_id: a.voice for a in agents if a.voice is not None}
        if not auto_map:
            return

        vc = kit._channels.get(voice_channel_id)
        if isinstance(vc, VoiceChannel):
            vc.update_voice_map(auto_map)
        else:
            logger.warning(
                "voice_channel_id=%r does not point to a VoiceChannel; "
                "skipping voice map auto-wiring",
                voice_channel_id,
            )

    def _wire_realtime(
        self,
        kit: RoomKit,
        agents: list[Agent],
        handler: HandoffHandler,
        rtv_channel_id: str,
        *,
        greet_on_handoff: bool = False,
        greeting_prompt: str | None = None,
    ) -> None:
        """Wire speech-to-speech orchestration on a RealtimeVoiceChannel.

        Builds per-agent configurations (system prompt with identity block,
        voice, handoff tools), sets the initial agent config, and installs
        a tool handler that intercepts ``handoff_conversation`` calls and
        reconfigures the provider session on handoff.
        """
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel
        from roomkit.orchestration.state import get_conversation_state

        rtv = kit._channels.get(rtv_channel_id)
        if not isinstance(rtv, RealtimeVoiceChannel):
            msg = f"voice_channel_id={rtv_channel_id!r} does not point to a RealtimeVoiceChannel"
            raise TypeError(msg)

        agent_map: dict[str, Agent] = {a.channel_id: a for a in agents}
        stage_by_agent: dict[str, PipelineStage] = {s.agent_id: s for s in self._stages}

        # Build per-agent configurations
        agent_configs: dict[str, dict[str, Any]] = {}
        for agent in agents:
            prompt = agent._system_prompt or ""
            identity = agent._build_identity_block()
            if identity:
                prompt = prompt + identity

            # Build handoff tool with enum-constrained targets
            stage = stage_by_agent.get(agent.channel_id)
            if stage:
                reachable: set[str] = set()
                if stage.next:
                    reachable.add(stage.next)
                reachable.update(stage.can_return_to)

                targets: list[tuple[str, str | None]] = []
                for s in self._stages:
                    if s.phase in reachable and s.agent_id != agent.channel_id:
                        ta = agent_map.get(s.agent_id)
                        desc = ta.description if ta else None
                        if desc is None:
                            desc = s.description
                        targets.append((s.agent_id, desc))
                tool = build_handoff_tool(targets)
            else:
                tool = build_handoff_tool([])

            agent_configs[agent.channel_id] = {
                "system_prompt": prompt or None,
                "voice": agent.voice,
                "tools": [tool.model_dump()],
            }

        # Set initial agent config on the RealtimeVoiceChannel
        default_stage = self._stage_map.get(self._default_phase or "")
        default_agent_id = default_stage.agent_id if default_stage else agents[0].channel_id
        if default_agent_id in agent_configs:
            initial = agent_configs[default_agent_id]
            rtv._system_prompt = initial["system_prompt"]
            rtv._voice = initial["voice"]
            rtv._tools = initial["tools"]

        # Per-agent greeting builder (identity + language aware)
        default_greet = (
            "Handoff complete. You are now the active agent. "
            "Please introduce yourself briefly to the caller."
        )

        def _build_greet(agent_id: str, language: str | None = None) -> str:
            """Build a greeting message for the given agent and language."""
            if greeting_prompt:
                msg = greeting_prompt
            else:
                target = agent_map.get(agent_id)
                role = target.role if target else None
                if role:
                    msg = (
                        f"Handoff complete. You are now the {role}. "
                        f"Your previous identity in this conversation no longer "
                        f"applies — introduce yourself in your new role."
                    )
                else:
                    msg = default_greet
            lang = language
            if not lang and agent_id in agent_map:
                lang = getattr(agent_map[agent_id], "language", None)
            if lang:
                msg = f"[Respond in {lang}] {msg}"
            return msg

        # Install tool handler that intercepts handoff_conversation
        original_handler = rtv._tool_handler

        async def _realtime_tool_handler(
            session: Any,
            name: str,
            arguments: dict[str, Any],
        ) -> dict[str, Any] | str:
            if name != "handoff_conversation":
                if original_handler:
                    return await original_handler(
                        session,
                        name,
                        arguments,
                    )
                return {"error": f"Unknown tool: {name}"}

            room_id = rtv._session_rooms.get(session.id)
            if not room_id:
                return {"error": "No room context for this session"}

            room = await kit.get_room(room_id)
            state = get_conversation_state(room)
            calling_agent = state.active_agent_id or default_agent_id

            result = await handler.handle(
                room_id=room_id,
                calling_agent_id=calling_agent,
                arguments=arguments,
            )

            output = result.model_dump()
            if result.accepted and greet_on_handoff:
                target = arguments.get("target", "")
                # Re-read room for current language
                room = await kit.get_room(room_id)
                lang = handler._get_room_language(room, target)
                output["message"] = _build_greet(target, language=lang)
            return output

        rtv._tool_handler = _realtime_tool_handler

        # on_handoff_complete: reconfigure the realtime session
        async def _on_complete(room_id: str, result: Any) -> None:
            new_id = result.new_agent_id
            if not new_id or new_id not in agent_configs:
                return
            config = agent_configs[new_id]

            # Check for per-room language override
            room = await kit.get_room(room_id)
            lang = handler._get_room_language(room, new_id)

            # Rebuild prompt with language if needed
            prompt = config["system_prompt"]
            if lang:
                agent = agent_map.get(new_id)
                if agent is not None:
                    base = getattr(agent, "_system_prompt", None) or ""
                    identity = agent._build_identity_block(language=lang)
                    prompt = (base + identity) if identity else prompt

            for session in rtv._get_room_sessions(room_id):
                await rtv.reconfigure_session(
                    session,
                    system_prompt=prompt,
                    voice=config["voice"],
                    tools=config["tools"],
                )

                if greet_on_handoff:
                    # Session resumption doesn't preserve pending function-
                    # call state, so the tool result alone won't trigger a
                    # response.  Inject a language-aware message to give
                    # the new agent a turn to speak in its new role.
                    msg = _build_greet(new_id, language=lang)
                    await rtv._provider.inject_text(
                        session,
                        msg,
                        role="user",
                    )

        handler._on_handoff_complete = _on_complete

        logger.info(
            "Wired speech-to-speech orchestration: %d agents on %s",
            len(agents),
            rtv_channel_id,
        )

    def _register_greet_hooks(
        self,
        kit: RoomKit,
        *,
        voice_channel_id: str,
        greeting_prompt: str | None,
        hook_priority: int,
    ) -> None:
        """Register ON_HANDOFF + BEFORE_TTS hooks for handoff greeting."""
        from roomkit.models.context import RoomContext
        from roomkit.models.delivery import InboundMessage
        from roomkit.models.event import RoomEvent, TextContent
        from roomkit.models.hook import HookResult

        prompt = greeting_prompt or (
            "[The caller has just been transferred to you — please introduce yourself briefly]"
        )

        # Rooms where a handoff is in flight — TTS blocked until greeting fires
        _handoff_pending: set[str] = set()

        @kit.hook(HookTrigger.ON_HANDOFF, execution=HookExecution.ASYNC)
        async def _on_handoff(event: RoomEvent, _ctx: RoomContext) -> None:
            meta = event.metadata or {}
            to_agent = meta.get("to_agent", "")
            from_agent = meta.get("from_agent", "")
            logger.info("greet_on_handoff: %s → %s", from_agent, to_agent)

            _handoff_pending.add(event.room_id)

            async def _trigger_greeting() -> None:
                try:
                    await kit.process_inbound(
                        InboundMessage(
                            channel_id=voice_channel_id,
                            sender_id="system",
                            content=TextContent(body=prompt),
                        ),
                        room_id=event.room_id,
                    )
                except Exception:
                    logger.exception("Handoff greeting failed for room %s", event.room_id)
                finally:
                    _handoff_pending.discard(event.room_id)

            loop = asyncio.get_running_loop()
            loop.create_task(_trigger_greeting(), context=contextvars.Context())

        @kit.hook(
            HookTrigger.BEFORE_TTS,
            execution=HookExecution.SYNC,
            priority=hook_priority - 1,
        )
        async def _block_farewell(
            text: str,
            ctx: RoomContext,  # noqa: ARG001
        ) -> HookResult:
            if ctx.room.id in _handoff_pending:
                return HookResult.block("handoff_transition")
            return HookResult.allow()
