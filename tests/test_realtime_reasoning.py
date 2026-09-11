"""Integrator-side reasoning delegation is served by the channel (RFC §12.4.1).

A full-duplex model hands work over with no task text. The channel keeps the
transcript ledger from the provider's partials, hands the backend what was
said since the previous delegation, relays every output as spoken or silent
context, answers a silent or failing backend with one spoken fallback, and
runs the backend's tool calls through the same gate as any realtime tool.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels._realtime_delegation import (
    FALLBACK_FAILED,
    FALLBACK_NO_BACKEND,
    FALLBACK_NO_OUTPUT,
    FALLBACK_TIMEOUT,
)
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import (
    AIProviderReasoningBackend,
    ReasoningBackend,
    ReasoningOutput,
    ReasoningRequest,
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

        await channel.wait_idle("r1", timeout=1.0)
        assert channel._idle_events[session.id].is_set()  # noqa: SLF001
        assert provider.delegation_outputs[-1][2] == "ok"

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
        assert json.loads(backend.tool_results[0])["tool"] == "lookup"
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
        assert [t.name for t in first_call.tools] == ["lookup"]
        assert "USER: Is UA482 running?" in first_call.messages[0].content
        assert "Voice conversation so far:" in first_call.messages[0].content
        assert [m.role for m in second_call.messages] == ["user", "assistant", "tool"]
        assert second_call.messages[2].content[0].result == '{"status": "cancelled"}'

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
        outputs = [o async for o in backend.run(request)]

        assert outputs == [
            ReasoningOutput("Still working.", spoken=True, is_final=False),
            ReasoningOutput("Still working.", spoken=True, is_final=True),
        ]
        assert len(provider.calls) == 2

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


class TestRendering:
    def test_first_and_later_requests_read_differently(self) -> None:
        lines = [TranscriptLine("user", "hi"), TranscriptLine("assistant", "hello")]
        first = render_transcript_request(lines, first=True)
        later = render_transcript_request(lines, first=False)
        assert first.startswith("Voice conversation so far:\nUSER: hi\nASSISTANT: hello\n")
        assert later.startswith("Voice conversation since the previous delegation:")
        assert first.endswith("Act on the user's most recent request in the conversation above.")

    def test_empty_transcript_is_just_the_instruction(self) -> None:
        assert render_transcript_request([], first=True, instruction="Go.") == "Go."
