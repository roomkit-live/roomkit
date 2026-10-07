"""A reconnect a tool call's own handler caused does not abandon that call (RMK-280).

Gemini Live applies a reconfiguration by reconnecting, and a reconnect orphans
every call the old socket issued, the one whose handler asked for it included.
That handler is not interrupted, its result stays off the wire (the new socket
never issued the id), and its outcome is reported as usual. Every other call
the reconnect orphaned is abandoned as before (RFC §9.3).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from roomkit import HookExecution, HookResult, HookTrigger, RoomContext, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.exceptions import ToolRefusedError
from roomkit.models.tool_call import ToolCallEvent
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


class ReconnectingProvider(MockRealtimeProvider):
    """Orphans a session's outstanding calls on reconfigure, as Gemini Live does.

    Like the real socket, it suspends after the report (the new connection's
    setup), where a cancelled handler would stop. The new connection's receive
    loop starts inside the reconfiguring task, as Gemini's does, so it
    inherits that task's context.
    """

    def __init__(self) -> None:
        super().__init__()
        self.outstanding: dict[str, set[str]] = {}
        self.reconfigured: list[str] = []
        self.inbox: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self.receive_loops: list[asyncio.Task[None]] = []

    async def reconfigure(self, session: VoiceSession, **kwargs: Any) -> None:
        self.reconfigured.append(session.id)
        orphaned = sorted(self.outstanding.pop(session.id, set()))
        await self.simulate_tool_call_cancellation(session, orphaned)
        await asyncio.sleep(0)
        self.receive_loops.append(asyncio.create_task(self._receive_loop(session)))

    async def _receive_loop(self, session: VoiceSession) -> None:
        """What the new connection delivers: a call, or its own drop."""
        while True:
            kind, call = await self.inbox.get()
            if kind == "call":
                await self.simulate_tool_call(session, *call)
            else:  # the connection dropped and came back on its own
                orphaned = sorted(self.outstanding.pop(session.id, set()))
                await self.simulate_tool_call_cancellation(session, orphaned)

    async def simulate_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self.outstanding.setdefault(session.id, set()).add(call_id)
        await super().simulate_tool_call(session, call_id, name, arguments)

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        self.outstanding.get(session.id, set()).discard(call_id)
        await super().submit_tool_result(session, call_id, result)

    def injected(self, session: VoiceSession) -> list[str]:
        return [
            c.args["text"]
            for c in self.calls
            if c.method == "inject_text" and c.args.get("session_id") == session.id
        ]


_TOOLS = [
    {"name": name, "description": "d", "parameters": {"type": "object", "properties": {}}}
    for name in ("switch_agent", "lookup")
]


async def _channel(
    provider: ReconnectingProvider, handler: Any, sessions: int = 1
) -> tuple[RealtimeVoiceChannel, list[VoiceSession], list[ToolCallEvent]]:
    ch = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=_TOOLS,
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(ch)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt")
    started = [await ch.start_session(room.id, f"u{i}", "ws") for i in range(sessions)]
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="observe")
    async def observe(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        observed.append(event)
        return HookResult.allow()

    return ch, started, observed


class TestTheCallWhoseHandlerReconnected:
    async def test_it_runs_to_its_end_and_is_reported_served(self) -> None:
        provider = ReconnectingProvider()
        holder: dict[str, Any] = {}

        async def switch_agent(name: str, arguments: dict[str, Any]) -> str:
            ch, session = holder["ch"], holder["session"]
            await ch.reconfigure_session(session, system_prompt="You are the new agent.")
            await provider.inject_text(session, "Introduce yourself", role="system")
            return '{"accepted": true}'

        ch, [session], observed = await _channel(provider, switch_agent)
        holder.update(ch=ch, session=session)

        await provider.simulate_tool_call(session, "h1", "switch_agent", {})
        await asyncio.sleep(0.1)

        assert provider.reconfigured == [session.id]
        assert provider.injected(session) == ["Introduce yourself"]
        # The new socket never issued h1: nothing goes out for it.
        assert provider.tool_results == []
        assert [(e.tool_call_id, e.cancelled, e.is_error) for e in observed] == [
            ("h1", False, False)
        ]
        assert json.loads(observed[0].result) == {"accepted": True}
        # No provider output is owed for a result that was not sent.
        assert ch._tool_calls.settled(session.id)
        assert not ch._tool_calls.busy(session.id)

    async def test_a_refusal_after_the_reconnect_is_reported_refused(self) -> None:
        provider = ReconnectingProvider()
        holder: dict[str, Any] = {}

        async def switch_agent(name: str, arguments: dict[str, Any]) -> str:
            await holder["ch"].reconfigure_session(holder["session"], system_prompt="New.")
            raise ToolRefusedError("The target agent is not available.")

        ch, [session], observed = await _channel(provider, switch_agent)
        holder.update(ch=ch, session=session)

        await provider.simulate_tool_call(session, "h1", "switch_agent", {})
        await asyncio.sleep(0.1)

        assert provider.tool_results == []
        assert [(e.tool_call_id, e.cancelled, e.is_error) for e in observed] == [
            ("h1", False, True)
        ]

    async def test_a_reconfigure_it_gathers_spares_it(self) -> None:
        provider = ReconnectingProvider()
        holder: dict[str, Any] = {}

        async def switch_agent(name: str, arguments: dict[str, Any]) -> str:
            ch = holder["ch"]
            await asyncio.gather(
                *(ch.reconfigure_session(s, system_prompt="New.") for s in holder["sessions"])
            )
            return '{"accepted": true}'

        ch, sessions, observed = await _channel(provider, switch_agent, sessions=2)
        holder.update(ch=ch, sessions=sessions)

        await provider.simulate_tool_call(sessions[0], "h1", "switch_agent", {})
        await asyncio.sleep(0.1)

        assert sorted(provider.reconfigured) == sorted(s.id for s in sessions)
        assert [(e.tool_call_id, e.cancelled) for e in observed] == [("h1", False)]
        assert provider.tool_results == []


class TestTheOtherCallsTheReconnectOrphaned:
    async def test_a_call_pending_beside_it_is_still_abandoned(self) -> None:
        provider = ReconnectingProvider()
        holder: dict[str, Any] = {}
        lookup_started = asyncio.Event()
        lookup_interrupted = asyncio.Event()

        async def handler(name: str, arguments: dict[str, Any]) -> str:
            if name == "lookup":
                lookup_started.set()
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    lookup_interrupted.set()
                    raise
                return "too late"
            await holder["ch"].reconfigure_session(holder["session"], system_prompt="New.")
            return '{"accepted": true}'

        ch, [session], observed = await _channel(provider, handler)
        holder.update(ch=ch, session=session)
        await provider.simulate_tool_call(session, "c1", "lookup", {})
        await asyncio.wait_for(lookup_started.wait(), 1)

        await provider.simulate_tool_call(session, "h1", "switch_agent", {})
        await asyncio.wait_for(lookup_interrupted.wait(), 1)
        await asyncio.sleep(0.1)

        assert provider.tool_results == []
        assert sorted((e.tool_call_id, e.cancelled) for e in observed) == [
            ("c1", True),
            ("h1", False),
        ]

    async def test_a_reconnect_from_elsewhere_abandons_the_call(self) -> None:
        provider = ReconnectingProvider()
        started = asyncio.Event()
        interrupted = asyncio.Event()

        async def lookup(name: str, arguments: dict[str, Any]) -> str:
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                interrupted.set()
                raise
            return "too late"

        ch, [session], observed = await _channel(provider, lookup)
        await provider.simulate_tool_call(session, "c1", "lookup", {})
        await asyncio.wait_for(started.wait(), 1)

        # The application reconfigures the session, outside any handler
        await ch.reconfigure_session(session, system_prompt="New.")
        await asyncio.wait_for(interrupted.wait(), 1)
        await asyncio.sleep(0.05)

        assert provider.tool_results == []
        assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c1", True)]

    async def test_another_sessions_call_with_the_same_id_is_abandoned(self) -> None:
        provider = ReconnectingProvider()
        holder: dict[str, Any] = {}
        started = asyncio.Event()

        async def handler(name: str, arguments: dict[str, Any]) -> str:
            if name == "lookup":
                started.set()
                await asyncio.sleep(30)
                return "too late"
            await holder["ch"].reconfigure_session(holder["other"], system_prompt="New.")
            return '{"accepted": true}'

        ch, [session, other], observed = await _channel(provider, handler, sessions=2)
        holder.update(ch=ch, other=other)
        await provider.simulate_tool_call(other, "c", "lookup", {})
        await asyncio.wait_for(started.wait(), 1)

        await provider.simulate_tool_call(session, "c", "switch_agent", {})
        await asyncio.sleep(0.1)

        assert sorted((e.session.id == session.id, e.cancelled) for e in observed) == [
            (False, True),
            (True, False),
        ]

    async def test_a_later_call_reusing_the_id_is_abandoned(self) -> None:
        """The new receive loop inherits the handler's context and outlives it."""
        provider = ReconnectingProvider()
        holder: dict[str, Any] = {}
        started = asyncio.Event()

        async def handler(name: str, arguments: dict[str, Any]) -> str:
            if name == "switch_agent":
                await holder["ch"].reconfigure_session(holder["session"], system_prompt="New.")
                return '{"accepted": true}'
            started.set()
            await asyncio.sleep(30)
            return "too late"

        ch, [session], observed = await _channel(provider, handler)
        holder.update(ch=ch, session=session)
        await provider.simulate_tool_call(session, "c1", "switch_agent", {})
        await asyncio.sleep(0.1)
        # The new connection issues the same id, then drops while it runs
        await provider.inbox.put(("call", ("c1", "lookup", {})))
        await asyncio.wait_for(started.wait(), 1)
        await provider.inbox.put(("drop", None))
        await asyncio.sleep(0.1)
        for loop in provider.receive_loops:
            loop.cancel()

        assert [(e.name, e.cancelled) for e in observed] == [
            ("switch_agent", False),
            ("lookup", True),
        ]
        assert provider.tool_results == []


class TestIdleAfterACallThatOwesNothing:
    """RMK-288: ``wait_idle`` opens once a call whose result stays off the wire
    ends, with no response to wait for."""

    async def test_a_spared_call_leaves_the_session_idle(self) -> None:
        provider = ReconnectingProvider()
        holder: dict[str, Any] = {}

        async def switch_agent(name: str, arguments: dict[str, Any]) -> str:
            await holder["ch"].reconfigure_session(holder["session"], system_prompt="New.")
            return '{"accepted": true}'

        ch, [session], observed = await _channel(provider, switch_agent)
        holder.update(ch=ch, session=session)
        await provider.simulate_tool_call(session, "h1", "switch_agent", {})
        await asyncio.sleep(0.1)
        assert [(e.tool_call_id, e.cancelled) for e in observed] == [("h1", False)]

        await ch.wait_idle(session.room_id, timeout=0.5)

    async def test_a_cancelled_call_leaves_the_session_idle(self) -> None:
        provider = ReconnectingProvider()
        started = asyncio.Event()

        async def lookup(name: str, arguments: dict[str, Any]) -> str:
            started.set()
            await asyncio.sleep(30)
            return "too late"

        ch, [session], observed = await _channel(provider, lookup)
        await provider.simulate_tool_call(session, "c1", "lookup", {})
        await asyncio.wait_for(started.wait(), 1)
        await ch.reconfigure_session(session, system_prompt="New.")
        await asyncio.sleep(0.05)
        assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c1", True)]

        await ch.wait_idle(session.room_id, timeout=0.5)


class TestSpeechToSpeechHandoff:
    async def test_every_session_of_the_room_takes_the_new_agent(self) -> None:
        provider = ReconnectingProvider()
        ch = RealtimeVoiceChannel("rtv", provider=provider, transport=MockRealtimeTransport())
        kit = RoomKit()
        kit.register_channel(ch)
        triage = Agent("agent-triage", role="Triage", voice="v-t", system_prompt="Hi.")
        advisor = Agent("agent-advisor", role="Advisor", voice="v-a", system_prompt="Help.")
        pipeline = ConversationPipeline(
            stages=[
                PipelineStage(phase="triage", agent_id="agent-triage", next="handling"),
                PipelineStage(phase="handling", agent_id="agent-advisor", next=None),
            ],
        )
        pipeline.install(kit, [triage, advisor], voice_channel_id="rtv", greet_on_handoff=True)
        room = await kit.create_room()
        state = ConversationState(active_agent_id="agent-triage", phase="triage")
        await kit.store.update_room(set_conversation_state(room, state))
        await kit.attach_channel(room.id, "rtv")
        caller = await ch.start_session(room.id, "u1", "ws")
        listener = await ch.start_session(room.id, "u2", "ws")
        observed: list[ToolCallEvent] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="observe")
        async def observe(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            observed.append(event)
            return HookResult.allow()

        await provider.simulate_tool_call(
            caller,
            "h1",
            "handoff_conversation",
            {"target": "agent-advisor", "reason": "help", "summary": "ctx"},
        )
        await asyncio.sleep(0.2)

        assert sorted(provider.reconfigured) == sorted([caller.id, listener.id])
        assert len(provider.injected(caller)) == 1
        assert len(provider.injected(listener)) == 1
        assert [(e.tool_call_id, e.cancelled) for e in observed] == [("h1", False)]
        assert provider.tool_results == []
