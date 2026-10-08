"""Integrator-side reasoning delegation is served by the channel (RFC §12.4.1).

A full-duplex model hands work over with no task text. The channel keeps the
transcript ledger from the provider's partials, hands the backend what was
said since the previous delegation, relays every output as spoken or silent
context, answers a silent or failing backend with one spoken fallback, and
runs the backend's tool calls through the same gate as any realtime tool.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels._realtime_delegation import (
    FALLBACK_FAILED,
    FALLBACK_NO_BACKEND,
    FALLBACK_NO_OUTPUT,
    FALLBACK_TIMEOUT,
)
from roomkit.channels._tool_registry import orchestration_tool
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import (
    AIResponse,
    AITool,
    AIToolCall,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.mock import MockTelemetryProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import (
    AgentReasoningBackend,
    AIProviderReasoningBackend,
    ReasoningBackend,
    ReasoningCutShortError,
    ReasoningOutput,
    ReasoningRequest,
    ToolCallResult,
    TranscriptLine,
    render_transcript_request,
)

LOOKUP = {
    "name": "lookup",
    "description": "Look up a flight",
    "parameters": {
        "type": "object",
        "properties": {"flight": {"type": "string"}},
        "required": ["flight"],
    },
}


class _ScriptedBackend(ReasoningBackend):
    """Yields a fixed script; can call tools, stall, or fail on request."""

    def __init__(
        self,
        outputs: list[ReasoningOutput] | None = None,
        *,
        tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.outputs = outputs or []
        self.tool_calls = tool_calls or []
        self.delay = delay
        self.error = error
        self.requests: list[ReasoningRequest] = []
        self.tool_results: list[str] = []
        self.ended: list[str] = []
        self.closed = False

    async def run(self, request: ReasoningRequest) -> AsyncIterator[ReasoningOutput]:
        self.requests.append(request)
        for name, arguments in self.tool_calls:
            assert request.execute_tool is not None
            self.tool_results.append(await request.execute_tool(name, arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        for output in self.outputs:
            yield output

    async def session_ended(self, session_id: str) -> None:
        self.ended.append(session_id)

    async def close(self) -> None:
        self.closed = True


async def _channel(
    backend: ReasoningBackend | None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_handler: Any = None,
    timeout: float = 1.0,
    full_duplex: bool = True,
) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeProvider, VoiceSession]:
    provider = MockRealtimeProvider(full_duplex=full_duplex)
    channel = RealtimeVoiceChannel(
        "rt-1",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=tools,
        tool_handler=tool_handler,
        reasoning_backend=backend,
        reasoning_timeout_s=timeout,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt-1")
    session = await channel.start_session("r1", "user-1", "fake-ws")
    return kit, channel, provider, session


async def _partials(
    provider: MockRealtimeProvider, session: VoiceSession, *fragments: tuple[str, str]
) -> None:
    for role, text in fragments:
        await provider.simulate_transcription(session, text, role, False)


async def _settle(seconds: float = 0.05) -> None:
    await asyncio.sleep(seconds)


class TestConfiguration:
    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="reasoning_timeout_s"):
            RealtimeVoiceChannel(
                "rt-bad",
                provider=MockRealtimeProvider(full_duplex=True),
                transport=MockRealtimeTransport(),
                reasoning_timeout_s=0,
            )

    async def test_half_duplex_provider_feeds_no_ledger(self) -> None:
        backend = _ScriptedBackend()
        _, channel, provider, session = await _channel(backend, full_duplex=False)
        await _partials(provider, session, ("user", "hello"))
        assert channel._transcript_ledger == {}  # noqa: SLF001


class TestLedger:
    async def test_transcript_since_the_previous_delegation(self) -> None:
        backend = _ScriptedBackend([ReasoningOutput("done", is_final=True)])
        _, _, provider, session = await _channel(backend)

        await _partials(
            provider,
            session,
            ("assistant", "How can"),
            ("assistant", " I help?"),
            ("user", "Is my"),
            ("user", " flight late?"),
        )
        # The final repeats the deltas and must not be recorded twice.
        await provider.simulate_transcription(session, "How can I help?", "assistant", True)
        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle()

        assert len(backend.requests) == 1
        first = backend.requests[0]
        assert first.first is True
        assert first.transcript == [
            TranscriptLine("assistant", "How can I help?"),
            TranscriptLine("user", "Is my flight late?"),
        ]

        await _partials(provider, session, ("user", "And tomorrow?"))
        await provider.simulate_delegation(session, "d2", "integrator")
        await _settle()

        second = backend.requests[1]
        assert second.first is False
        assert second.transcript == [TranscriptLine("user", "And tomorrow?")]
        assert second.delegation_id == "d2"

    async def test_request_carries_the_declared_catalogue(self) -> None:
        backend = _ScriptedBackend([ReasoningOutput("ok", is_final=True)])
        _, _, provider, session = await _channel(backend, tools=[LOOKUP])
        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle()
        assert [t["name"] for t in backend.requests[0].tools] == ["lookup"]
        assert backend.requests[0].execute_tool is not None


class TestRelay:
    async def test_every_output_reaches_the_model_with_its_flag(self) -> None:
        backend = _ScriptedBackend(
            [
                ReasoningOutput("Checking the flight.", spoken=False),
                ReasoningOutput("UA482 is cancelled.", spoken=True, is_final=True),
            ]
        )
        _, _, provider, session = await _channel(backend)

        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle()

        assert provider.delegation_outputs == [
            (session.id, "d1", "Checking the flight.", False),
            (session.id, "d1", "UA482 is cancelled.", True),
        ]

    async def test_hosted_delegations_are_not_served(self) -> None:
        backend = _ScriptedBackend([ReasoningOutput("never", is_final=True)])
        _, _, provider, session = await _channel(backend)
        await provider.simulate_delegation(session, "d1", "hosted")
        await _settle()
        assert backend.requests == []
        assert provider.delegation_outputs == []

    async def test_hook_fires_while_the_backend_is_served(self) -> None:
        backend = _ScriptedBackend([ReasoningOutput("ok", is_final=True)])
        kit, _, provider, session = await _channel(backend)
        seen: list[Any] = []

        @kit.hook(HookTrigger.ON_REALTIME_DELEGATION, HookExecution.ASYNC)
        async def on_delegation(event, ctx) -> None:  # noqa: ANN001
            seen.append((event.delegation_id, event.target))

        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle()
        assert seen == [("d1", "integrator")]
        assert provider.delegation_outputs[-1][2] == "ok"


class TestFallbacks:
    async def test_no_backend_declines_aloud(self) -> None:
        _, _, provider, session = await _channel(None)
        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle()
        assert provider.delegation_outputs == [(session.id, "d1", FALLBACK_NO_BACKEND, True)]

    async def test_silent_backend_gets_a_spoken_fallback(self) -> None:
        _, _, provider, session = await _channel(_ScriptedBackend([]))
        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle()
        assert provider.delegation_outputs == [(session.id, "d1", FALLBACK_NO_OUTPUT, True)]

    async def test_blank_outputs_count_as_silence(self) -> None:
        _, _, provider, session = await _channel(_ScriptedBackend([ReasoningOutput("   ")]))
        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle()
        assert provider.delegation_outputs == [(session.id, "d1", FALLBACK_NO_OUTPUT, True)]

    async def test_failing_backend_gets_a_spoken_fallback(self) -> None:
        _, _, provider, session = await _channel(_ScriptedBackend(error=RuntimeError("boom")))
        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle()
        assert provider.delegation_outputs == [(session.id, "d1", FALLBACK_FAILED, True)]

    @pytest.mark.parametrize(("status", "level"), [(None, logging.WARNING), (503, logging.ERROR)])
    async def test_a_backend_provider_error_is_one_line_without_a_traceback(
        self, caplog: pytest.LogCaptureFixture, status: int | None, level: int
    ) -> None:
        """The reasoning backend's turn raises and logs nothing: the delegation
        that catches its provider error writes the one line, at its level."""
        exc = ProviderError("upstream down", provider="mock", status_code=status)
        _, _, provider, session = await _channel(_ScriptedBackend(error=exc))
        with caplog.at_level(logging.DEBUG):
            await provider.simulate_delegation(session, "d1", "integrator")
            await _settle()

        records = [
            r
            for r in caplog.records
            if r.name.startswith("roomkit") and r.levelno >= logging.WARNING
        ]
        assert [r.levelno for r in records] == [level]
        assert records[0].exc_info is None
        assert provider.delegation_outputs == [(session.id, "d1", FALLBACK_FAILED, True)]

    async def test_slow_backend_is_abandoned(self) -> None:
        backend = _ScriptedBackend([ReasoningOutput("late", is_final=True)], delay=0.5)
        _, _, provider, session = await _channel(backend, timeout=0.05)
        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle(0.15)
        assert provider.delegation_outputs == [(session.id, "d1", FALLBACK_TIMEOUT, True)]


class TestIdleAndTeardown:
    async def test_a_running_delegation_keeps_the_session_busy(self) -> None:
        backend = _ScriptedBackend([ReasoningOutput("ok", is_final=True)], delay=0.1)
        _, channel, provider, session = await _channel(backend)

        await provider.simulate_delegation(session, "d1", "integrator")
        await asyncio.sleep(0)
        assert not channel._idle_events[session.id].is_set()  # noqa: SLF001

        await _settle(0.15)
        assert provider.delegation_outputs[-1][2] == "ok"
        with pytest.raises(TimeoutError):
            await channel.wait_idle("r1", timeout=0.01)
        await provider.simulate_response_start(session)
        await provider.simulate_response_end(session)
        await channel.wait_idle("r1", timeout=1.0)
        assert channel._idle_events[session.id].is_set()  # noqa: SLF001

    async def test_end_session_cancels_the_run_and_releases_the_backend(self) -> None:
        backend = _ScriptedBackend([ReasoningOutput("late", is_final=True)], delay=5.0)
        _, channel, provider, session = await _channel(backend)

        await provider.simulate_delegation(session, "d1", "integrator")
        await asyncio.sleep(0)
        await channel.end_session(session)
        await _settle()

        assert backend.ended == [session.id]
        assert provider.delegation_outputs == []
        assert session.state == VoiceSessionState.ENDED

    async def test_close_closes_the_backend(self) -> None:
        backend = _ScriptedBackend()
        _, channel, _, _ = await _channel(backend)
        await channel.close()
        assert backend.closed


class TestBackendToolGate:
    async def test_declared_tools_run_through_the_handler(self) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        async def handler(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            calls.append((name, arguments))
            return {"status": "cancelled"}

        backend = _ScriptedBackend(
            [ReasoningOutput("done", is_final=True)],
            tool_calls=[
                ("lookup", {"flight": "UA482"}),
                ("undeclared", {}),
                ("lookup", {}),  # schema: flight is required
            ],
        )
        kit, _, provider, session = await _channel(backend, tools=[LOOKUP], tool_handler=handler)
        observed: list[ToolCallEvent] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, HookExecution.SYNC)
        async def observe(event: ToolCallEvent, ctx: Any) -> HookResult:
            observed.append(event)
            return HookResult.allow()

        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle()

        ok, undeclared, invalid = backend.tool_results
        assert json.loads(ok) == {"status": "cancelled"}
        assert "error" in json.loads(undeclared)
        assert "error" in json.loads(invalid)
        assert calls == [("lookup", {"flight": "UA482"})], "denied calls never reach the handler"
        assert [e.name for e in observed] == ["lookup"]
        assert observed[0].session is session
        assert observed[0].tool_call_id.startswith("d1:")

    async def test_before_tool_use_denies_a_backend_call(self) -> None:
        calls: list[str] = []

        async def handler(name: str, arguments: dict[str, Any]) -> str:
            calls.append(name)
            return "ok"

        backend = _ScriptedBackend(
            [ReasoningOutput("done", is_final=True)], tool_calls=[("lookup", {"flight": "UA482"})]
        )
        kit, _, provider, session = await _channel(backend, tools=[LOOKUP], tool_handler=handler)

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, HookExecution.SYNC)
        async def deny(event: ToolCallEvent, ctx: Any) -> HookResult:
            return HookResult.block("outside business hours")

        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle()

        assert calls == []
        assert "outside business hours" in backend.tool_results[0]

    async def test_a_handler_error_returns_an_error_result(self) -> None:
        async def handler(name: str, arguments: dict[str, Any]) -> str:
            raise RuntimeError("db down")

        backend = _ScriptedBackend(
            [ReasoningOutput("done", is_final=True)], tool_calls=[("lookup", {"flight": "UA482"})]
        )
        _, _, provider, session = await _channel(backend, tools=[LOOKUP], tool_handler=handler)
        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle()
        assert json.loads(backend.tool_results[0]) == {
            "error": "Tool 'lookup' failed (RuntimeError)"
        }
        assert provider.delegation_outputs[-1][2] == "done"


class TestAIProviderReasoningBackend:
    def _session(self) -> VoiceSession:
        return VoiceSession(
            id="s1",
            room_id="r1",
            participant_id="u1",
            channel_id="rt-1",
            state=VoiceSessionState.ACTIVE,
        )

    async def test_a_call_replays_with_its_metadata(self) -> None:
        """A call's provider metadata (a Gemini thought signature) goes back
        with it on the next round, as on the AI channel's loop (RMK-308)."""
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    tool_calls=[
                        AIToolCall(
                            id="c1",
                            name="lookup",
                            arguments={"flight": "UA482"},
                            metadata={"thought_signature": "TS"},
                        )
                    ],
                ),
                AIResponse(content="UA482 is cancelled."),
            ]
        )
        backend = AIProviderReasoningBackend(provider)

        async def execute(name: str, arguments: dict[str, Any]) -> str:
            return '{"status": "cancelled"}'

        request = ReasoningRequest(
            session=self._session(),
            delegation_id="d1",
            transcript=[TranscriptLine("user", "Is UA482 running?")],
            first=True,
            tools=[LOOKUP],
            execute_tool=execute,
        )
        _ = [o async for o in backend.run(request)]

        replayed = [
            part
            for message in provider.calls[-1].messages
            if isinstance(message.content, list)
            for part in message.content
            if isinstance(part, AIToolCallPart)
        ]
        assert [part.metadata for part in replayed] == [{"thought_signature": "TS"}]

    async def test_tool_round_then_answer(self) -> None:
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="Let me check.",
                    tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "UA482"})],
                ),
                AIResponse(content="UA482 is cancelled."),
            ]
        )
        backend = AIProviderReasoningBackend(provider, system_prompt="backend rules")
        executed: list[tuple[str, dict[str, Any]]] = []

        async def execute(name: str, arguments: dict[str, Any]) -> str:
            executed.append((name, arguments))
            return '{"status": "cancelled"}'

        request = ReasoningRequest(
            session=self._session(),
            delegation_id="d1",
            transcript=[TranscriptLine("user", "Is UA482 running?")],
            first=True,
            tools=[LOOKUP],
            execute_tool=execute,
        )
        outputs = [o async for o in backend.run(request)]

        assert outputs == [
            ReasoningOutput("Let me check.", spoken=False, is_final=False),
            ReasoningOutput("UA482 is cancelled.", spoken=True, is_final=True),
        ]
        assert executed == [("lookup", {"flight": "UA482"})]
        first_call, second_call = provider.calls
        assert first_call.system_prompt == "backend rules"
        # The re-read of a stored result is declared, as on any agent turn.
        assert [t.name for t in first_call.tools] == ["lookup", "read_stored_result"]
        assert "USER: “Is UA482 running?”" in first_call.messages[0].content
        assert "Voice conversation so far:" in first_call.messages[0].content
        assert [m.role for m in second_call.messages] == ["user", "assistant", "tool"]
        assert second_call.messages[2].content[0].result == '{"status": "cancelled"}'

    @pytest.mark.parametrize(
        ("garbled", "error"),
        [(False, "Tool call cut off"), (True, "Tool call arguments unreadable")],
        ids=["cut", "garbled"],
    )
    async def test_a_partial_call_is_answered_without_running(
        self, garbled: bool, error: str
    ) -> None:
        """A call whose arguments do not read never runs; the model reads
        whether it was cut or written unreadable (RFC §6.4)."""
        call = AIToolCall(
            id="c1",
            name="lookup",
            arguments={"raw": '{"flight": "UA'},
            partial=True,
            garbled=garbled,
        )
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(content="", finish_reason="length", tool_calls=[call]),
                AIResponse(content="Let me try again."),
            ]
        )
        backend = AIProviderReasoningBackend(provider)
        executed: list[str] = []

        async def execute(name: str, arguments: dict[str, Any]) -> str:
            executed.append(name)
            return "ran"

        request = ReasoningRequest(
            session=self._session(),
            delegation_id="d1",
            transcript=[TranscriptLine("user", "Is UA482 running?")],
            first=True,
            tools=[LOOKUP],
            execute_tool=execute,
        )
        _ = [o async for o in backend.run(request)]

        assert executed == []
        [answer] = provider.calls[1].messages[-1].content
        assert json.loads(answer.result)["error"] == error
        assert answer.outcome == "refused"

    async def test_history_persists_across_delegations_until_the_session_ends(self) -> None:
        provider = MockAIProvider(responses=["one", "two"])
        backend = AIProviderReasoningBackend(provider)
        session = self._session()

        first = ReasoningRequest(session, "d1", [TranscriptLine("user", "a")], True)
        second = ReasoningRequest(session, "d2", [TranscriptLine("user", "b")], False)
        _ = [o async for o in backend.run(first)]
        _ = [o async for o in backend.run(second)]

        messages = provider.calls[1].messages
        assert [m.role for m in messages] == ["user", "assistant", "user"]
        assert "since the previous delegation" in messages[2].content

        await backend.session_ended(session.id)
        _ = [o async for o in backend.run(first)]
        assert len(provider.calls[2].messages) == 1

    async def test_spoken_progress_and_round_cap(self) -> None:
        looping = AIResponse(
            content="Still working.",
            tool_calls=[AIToolCall(id="c", name="lookup", arguments={"flight": "X"})],
        )
        provider = MockAIProvider(ai_responses=[looping])
        backend = AIProviderReasoningBackend(provider, max_tool_rounds=1, spoken_progress=True)

        async def execute(name: str, arguments: dict[str, Any]) -> str:
            return "{}"

        request = ReasoningRequest(self._session(), "d1", [], True, [LOOKUP], execute)
        outputs: list[ReasoningOutput] = []
        with pytest.raises(ReasoningCutShortError) as cut:
            async for output in backend.run(request):
                outputs.append(output)

        # A turn the round cap cut has no answer: its narration stays progress
        # (RMK-396, RFC §6.4), and the run fails for the channel to answer.
        assert cut.value.reason == "max_rounds"
        assert outputs == [ReasoningOutput("Still working.", spoken=True, is_final=False)]

    async def test_missing_executor_answers_tool_calls_with_an_error(self) -> None:
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})], content=""
                ),
                AIResponse(content="gave up"),
            ]
        )
        backend = AIProviderReasoningBackend(provider)
        request = ReasoningRequest(self._session(), "d1", [], True, [LOOKUP], None)
        outputs = [o async for o in backend.run(request)]
        assert outputs == [ReasoningOutput("gave up", spoken=True, is_final=True)]
        assert "error" in json.loads(provider.calls[1].messages[2].content[0].result)

    def test_rejects_negative_round_cap(self) -> None:
        with pytest.raises(ValueError, match="max_tool_rounds"):
            AIProviderReasoningBackend(MockAIProvider(), max_tool_rounds=-1)


class TestAgentReasoningBackend:
    """The backend is an agent like any other, on the AI channel's tool loop
    (RMK-396, RFC §6.4 and §12.4.1)."""

    def _request(self, execute: Any = None, delegation_id: str = "d1") -> ReasoningRequest:
        async def served(name: str, arguments: dict[str, Any]) -> ToolCallResult:
            return ToolCallResult('{"status": "on time"}')

        return ReasoningRequest(
            session=VoiceSession(
                id="s1",
                room_id="r1",
                participant_id="u1",
                channel_id="rt-1",
                state=VoiceSessionState.ACTIVE,
            ),
            delegation_id=delegation_id,
            transcript=[TranscriptLine("user", "Is UA482 running?")],
            first=True,
            tools=[LOOKUP],
            execute_tool_call=execute or served,
        )

    async def test_an_agent_serves_with_its_own_settings(self) -> None:
        provider = MockAIProvider(ai_responses=[AIResponse(content="UA482 is on time.")])
        backend = AgentReasoningBackend(
            Agent("reasoner", provider=provider, system_prompt="agent rules", temperature=0.2)
        )

        outputs = [o async for o in backend.run(self._request())]

        assert outputs == [ReasoningOutput("UA482 is on time.", spoken=True, is_final=True)]
        [call] = provider.calls
        assert call.system_prompt == "agent rules"
        assert call.temperature == 0.2
        assert [t.name for t in call.tools] == ["lookup", "read_stored_result"]

    @pytest.mark.parametrize(
        ("own", "named"),
        [
            ({"tools": [AITool(name="own", description="own")]}, "tools"),
            ({"enable_planning": True}, "planning"),
            ({"skills": SkillRegistry()}, "skills"),
            ({"sandbox": MagicMock()}, "a sandbox"),
            ({"external_tool_handler": MagicMock()}, "an external tool handler"),
            ({"human_input_handler": MagicMock()}, "a human-input handler"),
        ],
        ids=["tools", "planning", "skills", "sandbox", "external", "human-input"],
    )
    def test_an_agent_with_tools_of_its_own_is_refused(
        self, own: dict[str, Any], named: str
    ) -> None:
        """Its own tools would run outside the voice channel's gate."""
        with pytest.raises(ValueError, match=named):
            AgentReasoningBackend(AIChannel("reasoner", provider=MockAIProvider(), **own))

    async def test_a_call_the_provider_could_not_parse_is_tried_again(self) -> None:
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(content="", finish_reason="MALFORMED_FUNCTION_CALL"),
                AIResponse(content="Your flight is AF123."),
            ]
        )
        backend = AIProviderReasoningBackend(provider)

        outputs = [o async for o in backend.run(self._request())]

        assert outputs == [ReasoningOutput("Your flight is AF123.", spoken=True, is_final=True)]
        assert len(provider.calls) == 2

    async def test_an_empty_answer_after_a_round_is_asked_again(self) -> None:
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})],
                ),
                AIResponse(content=""),
                AIResponse(content="It leaves at 9."),
            ]
        )
        backend = AIProviderReasoningBackend(provider)

        outputs = [o async for o in backend.run(self._request())]

        assert outputs == [ReasoningOutput("It leaves at 9.", spoken=True, is_final=True)]

    async def test_the_next_delegation_reads_the_tool_rounds(self) -> None:
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})],
                ),
                AIResponse(content="first"),
                AIResponse(content="second"),
            ]
        )
        backend = AIProviderReasoningBackend(provider)

        _ = [o async for o in backend.run(self._request())]
        _ = [o async for o in backend.run(self._request(delegation_id="d2"))]

        roles = [m.role for m in provider.calls[-1].messages]
        assert roles == ["user", "assistant", "tool", "assistant", "user"]

    async def test_a_refused_call_reads_as_one(self) -> None:
        async def refused(name: str, arguments: dict[str, Any]) -> ToolCallResult:
            return ToolCallResult('{"error": "outside business hours"}', is_error=True)

        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})],
                ),
                AIResponse(content="It is closed."),
            ]
        )
        backend = AIProviderReasoningBackend(provider)

        _ = [o async for o in backend.run(self._request(refused))]

        [result] = provider.calls[1].messages[-1].content
        assert result.is_error
        assert "outside business hours" in result.result

    async def test_a_turn_cut_by_its_deadline_has_no_answer(self) -> None:
        """The loop's own bounds hold: its deadline ends the turn, which
        fails for the channel to answer (RFC §6.4)."""

        async def slow(name: str, arguments: dict[str, Any]) -> ToolCallResult:
            await asyncio.sleep(0.05)
            return ToolCallResult("{}")

        looping = AIResponse(
            content="Checking.",
            tool_calls=[AIToolCall(id="c", name="lookup", arguments={"flight": "X"})],
        )
        agent = AIChannel(
            "reasoner",
            provider=MockAIProvider(ai_responses=[looping]),
            tool_loop_timeout_seconds=0.01,
        )
        backend = AgentReasoningBackend(agent)

        with pytest.raises(ReasoningCutShortError) as cut:
            _ = [o async for o in backend.run(self._request(slow))]
        assert cut.value.reason == "timeout"


class TestAgentBackendAsAnAgent:
    """What the agent's own configuration brings to a delegation (RMK-396)."""

    def _request(self, delegation_id: str = "d1") -> ReasoningRequest:
        async def served(name: str, arguments: dict[str, Any]) -> ToolCallResult:
            return ToolCallResult("x" * 30000 if name == "lookup" else "{}")

        return ReasoningRequest(
            session=VoiceSession(
                id="s1",
                room_id="r1",
                participant_id="u1",
                channel_id="rt-1",
                state=VoiceSessionState.ACTIVE,
            ),
            delegation_id=delegation_id,
            transcript=[TranscriptLine("user", "Is X running?")],
            first=True,
            tools=[LOOKUP],
            execute_tool_call=served,
        )

    async def test_its_turn_budget_holds(self) -> None:
        rounds = [
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id=c, name="lookup", arguments={"flight": "X"})],
                usage={"input_tokens": 1000, "output_tokens": 10},
            )
            for c in ("a", "b", "c")
        ]
        provider = MockAIProvider(ai_responses=[*rounds, AIResponse(content="done")])
        backend = AgentReasoningBackend(
            Agent("reasoner", provider=provider, turn_budget_tokens=500)
        )

        with pytest.raises(ReasoningCutShortError) as cut:
            _ = [o async for o in backend.run(self._request())]

        assert cut.value.reason == "budget_exceeded"

    async def test_a_result_it_stores_can_be_read_back(self) -> None:
        """A result past the agent's eviction threshold is stored, and the
        re-read it points to is declared (RFC §21.5)."""
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})],
                ),
                AIResponse(content="done"),
            ]
        )
        backend = AgentReasoningBackend(Agent("reasoner", provider=provider))

        _ = [o async for o in backend.run(self._request())]

        [part] = provider.calls[1].messages[-1].content
        assert "read_stored_result" in str(part.result)
        assert "read_stored_result" in [t.name for t in provider.calls[1].tools]

    async def test_an_answer_its_policy_continues_is_progress_not_run_on(
        self, streaming: bool
    ) -> None:
        """The announcement the agent's continuation policy goes on is its own
        segment: said as progress, then the answer, never the two run together
        (RFC §6.4)."""
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(content="I will check the run.", finish_reason="stop"),
                AIResponse(content="X is running.", finish_reason="stop"),
            ],
            streaming=streaming,
        )
        agent = Agent(
            "reasoner",
            provider=provider,
            continuation=lambda text: "Go on." if text.startswith("I will") else None,
        )

        outputs = [o async for o in AgentReasoningBackend(agent).run(self._request())]

        assert outputs == [
            ReasoningOutput("I will check the run.", spoken=False),
            ReasoningOutput("X is running.", spoken=True, is_final=True),
        ]

    async def test_a_registered_agent_is_refused(self) -> None:
        """A kit's hooks would judge each call a second time."""
        agent = Agent("reasoner", provider=MockAIProvider())
        RoomKit().register_channel(agent)

        with pytest.raises(ValueError, match="registered with a kit"):
            AgentReasoningBackend(agent)

    async def test_an_agent_registered_afterwards_is_refused_at_its_run(self) -> None:
        agent = Agent("reasoner", provider=MockAIProvider(responses=["ok"]))
        backend = AgentReasoningBackend(agent)
        RoomKit().register_channel(agent)

        with pytest.raises(ValueError, match="registered with a kit"):
            _ = [o async for o in backend.run(self._request())]

    async def test_a_sessions_delegations_run_one_at_a_time(self) -> None:
        """Two at once: the second sees the first's exchange, none is lost."""
        release = asyncio.Event()

        async def held(name: str, arguments: dict[str, Any]) -> ToolCallResult:
            await release.wait()
            return ToolCallResult('{"status": "on time"}')

        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})],
                ),
                AIResponse(content="first"),
                AIResponse(content="second"),
            ]
        )
        backend = AIProviderReasoningBackend(provider)
        first = self._request()
        first = ReasoningRequest(**{**first.__dict__, "execute_tool_call": held})

        async def drain(request: ReasoningRequest) -> list[ReasoningOutput]:
            return [o async for o in backend.run(request)]

        tasks = [
            asyncio.create_task(drain(first)),
            asyncio.create_task(drain(self._request("d2"))),
        ]
        await asyncio.sleep(0.05)
        release.set()
        outputs = await asyncio.gather(*tasks)

        assert [o[-1].text for o in outputs] == ["first", "second"]
        roles = [m.role for m in provider.calls[-1].messages]
        assert roles == ["user", "assistant", "tool", "assistant", "user"]


class TestAgentBackendOnTheChannel:
    """What the voice channel observes of an agent backend's turn."""

    async def _served(
        self, responses: list[AIResponse], *, deny: bool = False, **backend: Any
    ) -> tuple[RoomKit, MockRealtimeProvider, list[ToolCallEvent], MockTelemetryProvider]:
        async def handler(name: str, arguments: dict[str, Any]) -> str:
            return '{"status": "on time"}'

        telemetry = MockTelemetryProvider()
        provider = MockRealtimeProvider(full_duplex=True)
        channel = RealtimeVoiceChannel(
            "rt-1",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=[LOOKUP],
            tool_handler=handler,
            reasoning_backend=AIProviderReasoningBackend(
                MockAIProvider(ai_responses=responses), **backend
            ),
        )
        kit = RoomKit(telemetry=telemetry)
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "rt-1")
        observed: list[ToolCallEvent] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)
        async def observe(event: ToolCallEvent, ctx: Any) -> None:
            observed.append(event)

        if deny:

            @kit.hook(HookTrigger.BEFORE_TOOL_USE, HookExecution.SYNC)
            async def refuse(event: ToolCallEvent, ctx: Any) -> HookResult:
                return HookResult.block("outside business hours")

        session = await channel.start_session("r1", "user-1", "fake-ws")
        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle(0.1)
        return kit, provider, observed, telemetry

    async def test_a_served_call_is_reported_once(self) -> None:
        _, provider, observed, _ = await self._served(
            [
                AIResponse(
                    content="",
                    tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})],
                ),
                AIResponse(content="On time."),
            ]
        )

        assert [(e.name, e.is_error) for e in observed] == [("lookup", False)]
        assert provider.delegation_outputs[-1][2] == "On time."

    async def test_a_call_the_gate_refuses_is_reported_once_and_read_as_refused(
        self,
    ) -> None:
        """The gate reports its refusal; the loop, which reads it as one,
        reports it no second time (RFC §9.3)."""
        call = AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})
        _, _, observed, _ = await self._served(
            [AIResponse(content="", tool_calls=[call]), AIResponse(content="Sorry.")], deny=True
        )

        [event] = observed
        assert (event.name, event.is_error) == ("lookup", True)
        assert "outside business hours" in str(event.result)

    async def test_a_partial_call_reaches_the_observers(self) -> None:
        cut = AIToolCall(id="c1", name="lookup", arguments={"raw": '{"fl'}, partial=True)
        _, _, observed, _ = await self._served(
            [
                AIResponse(content="", finish_reason="length", tool_calls=[cut]),
                AIResponse(content="ok"),
            ]
        )

        [event] = observed
        assert (event.name, event.is_error) == ("lookup", True)
        assert json.loads(str(event.result))["error"] == "Tool call cut off"

    async def test_a_round_capped_turn_is_answered_by_the_fallback(self) -> None:
        looping = AIResponse(
            content="Still working.",
            tool_calls=[AIToolCall(id="c", name="lookup", arguments={"flight": "X"})],
        )
        _, provider, _, _ = await self._served([looping], max_tool_rounds=1)

        said = [(text, spoken) for _, _, text, spoken in provider.delegation_outputs]
        assert said == [("Still working.", False), (FALLBACK_FAILED, True)]

    async def test_the_turn_has_a_span_with_its_usage(self) -> None:
        answer = AIResponse(content="On time.", usage={"input_tokens": 120, "output_tokens": 8})
        _, _, _, telemetry = await self._served([answer])

        [span] = telemetry.get_spans(SpanKind.LLM_GENERATE)
        assert span.status == "ok"
        assert span.attributes[Attr.LLM_INPUT_TOKENS] == 120
        assert span.attributes[Attr.LLM_OUTPUT_TOKENS] == 8


class TestAgentBackendCutOnTheChannel:
    """What a delegation the channel cuts leaves (RMK-396)."""

    async def _served(
        self, responses: list[AIResponse], *, timeout: float = 120.0
    ) -> tuple[
        RealtimeVoiceChannel,
        MockRealtimeProvider,
        VoiceSession,
        list[ToolCallEvent],
        MockAIProvider,
        MockTelemetryProvider,
    ]:
        async def slow(name: str, arguments: dict[str, Any]) -> str:
            await asyncio.sleep(5)
            return "{}"

        model = MockAIProvider(ai_responses=responses)
        telemetry = MockTelemetryProvider()
        provider = MockRealtimeProvider(full_duplex=True)
        channel = RealtimeVoiceChannel(
            "rt-1",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=[LOOKUP],
            tool_handler=slow,
            reasoning_backend=AIProviderReasoningBackend(model, spoken_progress=True),
            reasoning_timeout_s=timeout,
        )
        kit = RoomKit(telemetry=telemetry)
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "rt-1")
        observed: list[ToolCallEvent] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)
        async def observe(event: ToolCallEvent, ctx: Any) -> None:
            observed.append(event)

        session = await channel.start_session("r1", "user-1", "fake-ws")
        return channel, provider, session, observed, model, telemetry

    @staticmethod
    def _call() -> AIResponse:
        return AIResponse(
            content="",
            tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})],
        )

    async def test_a_call_its_timeout_cut_is_reported_once_cancelled(self) -> None:
        _, provider, session, observed, _, _ = await self._served(
            [self._call(), AIResponse(content="done")], timeout=0.2
        )

        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle(0.5)

        [event] = observed
        assert (event.name, event.cancelled) == ("lookup", True)
        assert "The delegation ended" in str(event.result)
        assert provider.delegation_outputs[-1][2] == FALLBACK_TIMEOUT

    async def test_a_call_the_sessions_end_cut_is_reported_once(self) -> None:
        channel, provider, session, observed, _, _ = await self._served(
            [self._call(), AIResponse(content="done")]
        )

        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle(0.2)
        await channel.end_session(session)
        await _settle(0.2)

        [event] = observed
        assert (event.name, event.cancelled) == ("lookup", True)
        assert "The session ended" in str(event.result)

    async def test_the_next_delegation_sends_no_call_left_unanswered(self) -> None:
        _, provider, session, _, model, _ = await self._served(
            [self._call(), AIResponse(content="second answer")], timeout=0.2
        )

        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle(0.5)
        await provider.simulate_delegation(session, "d2", "integrator")
        await _settle(0.3)

        sent = model.calls[-1].messages
        parts = [p for m in sent if isinstance(m.content, list) for p in m.content]
        calls = {p.id for p in parts if isinstance(p, AIToolCallPart)}
        answered = {p.tool_call_id for p in parts if isinstance(p, AIToolResultPart)}
        assert calls and calls <= answered

    async def test_a_delegation_left_while_it_speaks_ends_cleanly(self) -> None:
        """Its generator is finalised in another context; nothing fails there."""
        failures: list[Any] = []
        asyncio.get_running_loop().set_exception_handler(lambda _l, ctx: failures.append(ctx))
        _, provider, session, _, _, _ = await self._served(
            [
                AIResponse(
                    content="Let me check.",
                    tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})],
                ),
                AIResponse(content="answer"),
            ]
        )
        submit = provider.submit_delegation_output

        async def end_while_speaking(
            sess: VoiceSession, did: str, text: str, *, spoken: bool = True
        ) -> None:
            await submit(sess, did, text, spoken=spoken)
            session.state = VoiceSessionState.ENDED

        provider.submit_delegation_output = end_while_speaking  # type: ignore[method-assign]
        await provider.simulate_delegation(session, "d1", "integrator")
        for _ in range(5):
            await _settle()
            gc.collect()

        assert failures == []

    async def test_the_backends_turn_is_traced_under_the_session(self) -> None:
        channel, provider, session, _, _, telemetry = await self._served(
            [AIResponse(content="ok")]
        )

        await provider.simulate_delegation(session, "d1", "integrator")
        await _settle(0.1)

        [span] = telemetry.get_spans(SpanKind.LLM_GENERATE)
        assert span.parent_id is not None
        assert span.parent_id == channel._session_spans.get(session.id)


class TestBackendCallBound:
    """A backend's call is bounded by the voice channel's gate, as every call
    of the session, not by the backend agent's own bound (RMK-417, RFC §21.6)."""

    async def _served(self, *, agent_bound: float, channel_bound: float, waits: bool) -> Any:
        async def slow(name: str, arguments: dict[str, Any]) -> str:
            await asyncio.sleep(0.3)
            return "slow result"

        model = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"flight": "X"})],
                ),
                AIResponse(content="done"),
            ]
        )
        agent = Agent("reasoner", provider=model, tool_timeout_seconds=agent_bound)
        provider = MockRealtimeProvider(full_duplex=True)
        channel = RealtimeVoiceChannel(
            "rt-1",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=[] if waits else [LOOKUP],
            tool_handler=None if waits else slow,
            tool_timeout_seconds=channel_bound,
            reasoning_backend=AgentReasoningBackend(agent),
        )
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "rt-1")
        if waits:
            lookup = AITool(name="lookup", description="d", parameters=LOOKUP["parameters"])
            channel._registry.register(
                orchestration_tool(
                    lookup, lambda arguments: slow("lookup", arguments), waits=True
                ),
                room_id="r1",
                owner=object(),
            )
        observed: list[ToolCallEvent] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)
        async def observe(event: ToolCallEvent, ctx: Any) -> None:
            observed.append(event)

        session = await channel.start_session("r1", "user-1", "fake-ws")
        await provider.simulate_delegation(session, "d1", "integrator")
        for _ in range(100):
            if len(model.calls) == 2:
                break
            await _settle()
        [result] = next(m for m in model.calls[-1].messages if m.role == "tool").content
        await kit.close()
        return result, observed

    async def test_the_voice_channels_bound_holds_not_the_agents(self) -> None:
        result, observed = await self._served(agent_bound=0.1, channel_bound=5.0, waits=False)

        assert (result.result, result.is_error) == ("slow result", False)
        assert [(e.name, e.is_error, e.cancelled) for e in observed] == [("lookup", False, False)]

    async def test_the_voice_channels_bound_still_cuts_the_call(self) -> None:
        result, observed = await self._served(agent_bound=5.0, channel_bound=0.1, waits=False)

        assert result.is_error and "ToolTimeoutError" in result.result
        assert [(e.name, e.is_error, e.cancelled) for e in observed] == [("lookup", True, False)]

    async def test_a_tool_that_waits_by_design_is_not_bounded(self) -> None:
        result, observed = await self._served(agent_bound=0.1, channel_bound=0.1, waits=True)

        assert (result.result, result.is_error) == ("slow result", False)
        assert [(e.is_error, e.cancelled) for e in observed] == [(False, False)]


class TestRendering:
    def test_first_and_later_requests_read_differently(self) -> None:
        lines = [TranscriptLine("user", "hi"), TranscriptLine("assistant", "hello")]
        first = render_transcript_request(lines, first=True)
        later = render_transcript_request(lines, first=False)
        assert first.startswith("Voice conversation so far:\nUSER: “hi”\nASSISTANT: “hello”\n")
        assert later.startswith("Voice conversation since the previous delegation:")
        assert first.endswith("Act on the user's most recent request in the conversation above.")

    def test_empty_transcript_is_just_the_instruction(self) -> None:
        assert render_transcript_request([], first=True, instruction="Go.") == "Go."
