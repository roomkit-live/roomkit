"""A full-duplex provider leaves interruption to the model (RFC §12.4.1).

GPT-Live listens and speaks at once and handles being talked over itself. On
such a session the channel must not run its barge-in path — no playback flush,
no provider interrupt or truncation, no gating of provider audio on user
speech — while the observation side of speech (hooks, client indicator) keeps
working. The channel also announces every reasoning delegation to
ON_REALTIME_DELEGATION, whichever backend it went to.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.vad.mock import MockVADProvider
from roomkit.voice.realtime.events import RealtimeDelegationEvent
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

AUDIO = b"\x01\x00" * 480  # 20 ms of PCM16 at 24 kHz


async def _session(
    provider: MockRealtimeProvider, *, pipeline: AudioPipelineConfig | None = None
) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeTransport, VoiceSession]:
    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel(
        "rt-1",
        provider=provider,
        transport=transport,
        input_sample_rate=24000,
        output_sample_rate=24000,
        pipeline=pipeline,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt-1")
    session = await channel.start_session("r1", "user-1", "fake-ws")
    return kit, channel, transport, session


def _prime_playback(channel: RealtimeVoiceChannel, session: VoiceSession) -> None:
    """Make the session look mid-response, so a barge-in would have work to do."""
    channel._provider_idle[session.id] = False  # noqa: SLF001
    channel._playback_started_at[session.id] = 1.0  # noqa: SLF001
    channel._playback_position_ms[session.id] = 320.0  # noqa: SLF001


class TestProviderContract:
    def test_half_duplex_is_the_default(self) -> None:
        assert MockRealtimeProvider().full_duplex is False
        assert MockRealtimeProvider(full_duplex=True).full_duplex is True

    def test_trigger_value(self) -> None:
        assert HookTrigger.ON_REALTIME_DELEGATION == "on_realtime_delegation"

    async def test_base_provider_has_no_delegation_output(self) -> None:
        """A provider that never delegates has nothing to answer with."""
        provider = MockRealtimeProvider()
        session = VoiceSession(
            id="s1",
            room_id="r1",
            participant_id="u1",
            channel_id="rt-1",
            state=VoiceSessionState.ACTIVE,
        )
        with pytest.raises(NotImplementedError, match="reasoning delegation"):
            await RealtimeVoiceProvider.submit_delegation_output(
                provider, session, "d1", "text", spoken=True
            )


class TestInterruptionBelongsToTheModel:
    async def test_provider_audio_keeps_flowing_while_the_user_speaks(self) -> None:
        provider = MockRealtimeProvider(full_duplex=True)
        _, channel, transport, session = await _session(provider)
        await provider.simulate_response_start(session)
        await provider.simulate_speech_start(session)

        await provider.simulate_audio(session, AUDIO)
        await asyncio.sleep(0.3)

        assert transport.sent_audio, "full-duplex audio must not be gated on user speech"
        assert channel._user_speaking.get(session.id) is not True  # noqa: SLF001

    async def test_half_duplex_still_drops_audio_during_barge_in(self) -> None:
        """Regression guard: the existing barge-in path is untouched."""
        provider = MockRealtimeProvider()
        _, _, transport, session = await _session(provider)
        await provider.simulate_response_start(session)
        await provider.simulate_speech_start(session)

        await provider.simulate_audio(session, AUDIO)
        await asyncio.sleep(0.3)

        assert transport.sent_audio == []

    async def test_provider_speech_start_flushes_nothing(self) -> None:
        provider = MockRealtimeProvider(full_duplex=True)
        _, channel, transport, session = await _session(provider)
        transport.interrupt = MagicMock()  # type: ignore[method-assign]
        _prime_playback(channel, session)

        await provider.simulate_speech_start(session)
        await asyncio.sleep(0.05)

        transport.interrupt.assert_not_called()
        assert not [c for c in provider.calls if c.method in {"interrupt", "truncate_audio"}]
        assert not [m for _, m in transport.sent_messages if m.get("type") == "clear_audio"]

    async def test_speech_hooks_fire_and_barge_in_does_not(self) -> None:
        provider = MockRealtimeProvider(full_duplex=True)
        kit, channel, _, session = await _session(provider)
        _prime_playback(channel, session)
        seen: list[str] = []

        @kit.hook(HookTrigger.ON_SPEECH_START, HookExecution.ASYNC)
        async def on_start(event, ctx) -> None:  # noqa: ANN001
            seen.append("start")

        @kit.hook(HookTrigger.ON_SPEECH_END, HookExecution.ASYNC)
        async def on_end(event, ctx) -> None:  # noqa: ANN001
            seen.append("end")

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, ctx) -> None:  # noqa: ANN001
            seen.append("barge_in")

        await provider.simulate_speech_start(session)
        await provider.simulate_speech_end(session)
        await asyncio.sleep(0.1)

        assert seen.count("start") == 1
        assert seen.count("end") == 1
        assert "barge_in" not in seen

    async def test_pipeline_vad_is_observation_only(self) -> None:
        provider = MockRealtimeProvider(full_duplex=True)
        kit, channel, transport, session = await _session(
            provider, pipeline=AudioPipelineConfig(vad=MockVADProvider())
        )
        transport.interrupt = MagicMock()  # type: ignore[method-assign]
        _prime_playback(channel, session)
        seen: list[str] = []

        @kit.hook(HookTrigger.ON_SPEECH_START, HookExecution.ASYNC)
        async def on_start(event, ctx) -> None:  # noqa: ANN001
            seen.append("start")

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, ctx) -> None:  # noqa: ANN001
            seen.append("barge_in")

        channel._on_pipeline_speech_start(session)  # noqa: SLF001
        channel._on_pipeline_speech_end(session)  # noqa: SLF001
        await asyncio.sleep(0.1)

        assert seen == ["start"]
        transport.interrupt.assert_not_called()
        methods = {c.method for c in provider.calls}
        assert not methods & {
            "interrupt",
            "truncate_audio",
            "send_activity_start",
            "send_activity_end",
        }
        connect = next(c for c in provider.calls if c.method == "connect")
        assert connect.args["server_vad"] is True

    async def test_half_duplex_pipeline_vad_still_signals_activity(self) -> None:
        """Regression guard: the manual-VAD contract for Gemini-style providers."""
        provider = MockRealtimeProvider()
        _, channel, _, session = await _session(
            provider, pipeline=AudioPipelineConfig(vad=MockVADProvider())
        )

        channel._on_pipeline_speech_start(session)  # noqa: SLF001
        channel._on_pipeline_speech_end(session)  # noqa: SLF001
        await asyncio.sleep(0.05)

        methods = [c.method for c in provider.calls]
        assert "send_activity_start" in methods
        assert "send_activity_end" in methods
        connect = next(c for c in provider.calls if c.method == "connect")
        assert connect.args["server_vad"] is False


class TestDelegationHook:
    async def test_fires_for_both_targets(self) -> None:
        provider = MockRealtimeProvider(full_duplex=True)
        kit, _, _, session = await _session(provider)
        seen: list[RealtimeDelegationEvent] = []

        @kit.hook(HookTrigger.ON_REALTIME_DELEGATION, HookExecution.ASYNC)
        async def on_delegation(event, ctx) -> None:  # noqa: ANN001
            seen.append(event)

        await provider.simulate_delegation(session, "d-hosted", "hosted")
        await provider.simulate_delegation(session, "d-client", "integrator")
        await asyncio.sleep(0.1)

        assert {(e.delegation_id, e.target) for e in seen} == {
            ("d-hosted", "hosted"),
            ("d-client", "integrator"),
        }
        assert all(e.session is session for e in seen)

    async def test_no_hook_registered_is_quiet(self) -> None:
        provider = MockRealtimeProvider(full_duplex=True)
        _, _, _, session = await _session(provider)

        await provider.simulate_delegation(session, "d1", "hosted")
        await asyncio.sleep(0.05)

    async def test_ended_session_is_ignored(self) -> None:
        provider = MockRealtimeProvider(full_duplex=True)
        kit, channel, _, session = await _session(provider)
        seen: list[RealtimeDelegationEvent] = []

        @kit.hook(HookTrigger.ON_REALTIME_DELEGATION, HookExecution.ASYNC)
        async def on_delegation(event, ctx) -> None:  # noqa: ANN001
            seen.append(event)

        await channel.end_session(session)
        await provider.simulate_delegation(session, "d1", "integrator")
        await asyncio.sleep(0.05)

        assert seen == []
