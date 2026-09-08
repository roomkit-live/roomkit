"""Transport drain and SIP playback are distinct from model generation end."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.backends._sip_types import SIPSessionState
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.sip_transport import SIPRealtimeTransport
from tests.test_voice_session_lifecycle import sip  # noqa: F401


@pytest.mark.parametrize("next_response", [False, True])
async def test_idle_follows_audio_drain_for_the_latest_response(next_response: bool) -> None:
    provider, transport = MockRealtimeProvider(), MockRealtimeTransport()
    entered = [asyncio.Event(), asyncio.Event()]
    resume = [asyncio.Event(), asyncio.Event()]
    delivered: list[bytes] = []
    calls = 0

    async def send(session, audio):
        nonlocal calls
        index = calls
        calls += 1
        entered[index].set()
        await resume[index].wait()
        delivered.append(audio)

    transport.send_audio = send
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=transport)
    active = await channel.start_session("r", "p", object())
    try:
        await provider.simulate_response_start(active)
        await provider.simulate_audio(active, b"\x01\x00" * 480)
        await asyncio.wait_for(entered[0].wait(), 1)
        await provider.simulate_response_end(active)
        with pytest.raises(TimeoutError):
            await channel.wait_idle("r", timeout=0.01)
        if next_response:
            await provider.simulate_response_start(active)
            await provider.simulate_audio(active, b"\x02\x00" * 480)
            await provider.simulate_response_end(active)
        resume[0].set()
        if next_response:
            await asyncio.wait_for(entered[1].wait(), 1)
            with pytest.raises(TimeoutError):
                await channel.wait_idle("r", timeout=0.01)
            resume[1].set()
        await channel.wait_idle("r", timeout=1)
        assert len(delivered) == (2 if next_response else 1)
    finally:
        for gate in resume:
            gate.set()
        await channel.close()


@pytest.mark.parametrize("rate", [8000, 16000])
@pytest.mark.parametrize("generation_ended", [False, True])
@pytest.mark.parametrize("local_vad", [False, True])
async def test_sip_barge_in_uses_paced_position(sip, rate, generation_ended, local_vad) -> None:  # noqa: F811
    backend, media = sip
    carrier = VoiceSession(id="sip", room_id="r", participant_id="p", channel_id="sip")
    state = SIPSessionState(session=carrier, call_session=media, codec_rate=rate, clock_rate=8000)
    backend._session_states[carrier.id] = state
    provider = MockRealtimeProvider()
    truncated, sent = asyncio.Event(), asyncio.Event()
    provider.truncate_audio = AsyncMock(side_effect=lambda *a: truncated.set())
    provider.interrupt = AsyncMock()
    frames = 0

    def packet(*args):
        nonlocal frames
        frames += 1
        if frames >= 10:
            sent.set()

    media.send_audio_pcm.side_effect = packet
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=SIPRealtimeTransport(backend),
        input_sample_rate=rate,
        output_sample_rate=rate,
    )
    active = await channel.start_session("r", "p", carrier)
    try:
        # No task drain between these callbacks: start bookkeeping must not
        # erase an audio chunk accepted before the indicator task gets CPU.
        await provider.simulate_response_start(active)
        await provider.simulate_audio(active, b"\x01\x00" * rate)
        if generation_ended:
            await provider.simulate_response_end(active)
        await asyncio.wait_for(sent.wait(), 1)
        assert backend.is_playing(carrier)
        transmitted_ms = frames * 20
        if local_vad:
            channel._on_pipeline_speech_start(active)
        else:
            await provider.simulate_speech_start(active)
        await asyncio.wait_for(truncated.wait(), 1)
        position = provider.truncate_audio.await_args.args[1]
        # Excludes the known RTP lead, and never truncates to future audio.
        assert 0 < position < transmitted_ms < 1000
        assert provider.interrupt.await_count == int(local_vad and not generation_ended)
        await asyncio.sleep(0.03)
        assert not backend.is_playing(carrier)
    finally:
        await channel.close()
        await backend.close()


async def test_sip_drain_completes_partial_packet_and_ends_playback(sip) -> None:  # noqa: F811
    backend, media = sip
    carrier = VoiceSession(id="sip", room_id="r", participant_id="p", channel_id="sip")
    state = SIPSessionState(session=carrier, call_session=media)
    backend._session_states[carrier.id] = state
    provider = MockRealtimeProvider()
    provider.truncate_audio = AsyncMock()
    transport = SIPRealtimeTransport(backend)
    played = []
    transport.on_audio_played(lambda s, frame: played.append((s.id, frame)))
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=transport,
        input_sample_rate=8000,
        output_sample_rate=8000,
    )
    active = await channel.start_session("r", "p", carrier)
    try:
        await provider.simulate_response_start(active)
        await provider.simulate_audio(active, b"\x01\x00" * 240)  # 30 ms -> two RTP packets
        await provider.simulate_response_end(active)
        await channel.wait_idle("r", timeout=1)
        assert state.pacer is not None
        await asyncio.wait_for(state.pacer.wait_for_response_done(), 1)
        assert media.send_audio_pcm.call_count == 2
        assert not state.send_buffer
        assert not backend.is_playing(carrier)
        assert sum(f.metadata["played_bytes"] for _, f in played) == 480
        assert played[-1][1].metadata["playback_ended"]
        assert {sid for sid, _ in played} == {active.id}
        assert active.id not in channel._playback_started_at
        await provider.simulate_speech_start(active)
        await asyncio.sleep(0)
        provider.truncate_audio.assert_not_awaited()
    finally:
        await channel.close()
        await backend.close()


async def test_sip_idle_silence_does_not_create_assistant_playback(sip) -> None:  # noqa: F811
    backend, _ = sip
    carrier = VoiceSession(id="sip", room_id="r", participant_id="p", channel_id="sip")
    state = SIPSessionState(session=carrier)
    backend._session_states[carrier.id] = state
    provider = MockRealtimeProvider()
    provider.truncate_audio = AsyncMock()
    channel = RealtimeVoiceChannel(
        "rt", provider=provider, transport=SIPRealtimeTransport(backend)
    )
    active = await channel.start_session("r", "p", carrier)
    try:
        backend._notify_sip_playback(carrier, b"\x00\x00" * 160, silence=True)
        assert active.id not in channel._playback_started_at
        await provider.simulate_speech_start(active)
        await asyncio.sleep(0)
        provider.truncate_audio.assert_not_awaited()
    finally:
        await channel.close()
        await backend.close()


async def test_pacer_old_drain_cannot_complete_a_new_response() -> None:
    from roomkit.voice.realtime.pacer import OutboundAudioPacer

    drain_entered, drain_resume = asyncio.Event(), asyncio.Event()
    send_entered, send_resume = asyncio.Event(), asyncio.Event()
    calls = 0

    async def send(audio):
        nonlocal calls
        calls += 1
        if calls == 2:
            send_entered.set()
            await send_resume.wait()

    async def drain():
        if calls == 1:
            drain_entered.set()
            await drain_resume.wait()

    pacer = OutboundAudioPacer(send, sample_rate=8000, prebuffer_ms=0)
    pacer._set_playback_observer(lambda data, silence: None, drain)
    await pacer.start()
    try:
        pacer.push(b"\x01\x00" * 160)
        pacer.end_of_response()
        await asyncio.wait_for(drain_entered.wait(), 1)
        pacer.push(b"\x02\x00" * 160)
        pacer.end_of_response()
        drain_resume.set()
        await asyncio.wait_for(send_entered.wait(), 1)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(pacer.wait_for_response_done(), 0.01)
        send_resume.set()
        await asyncio.wait_for(pacer.wait_for_response_done(), 1)
    finally:
        drain_resume.set()
        send_resume.set()
        await pacer.stop()


async def test_sip_playback_feeds_aec_once_for_paced_and_bridged_audio(sip) -> None:  # noqa: F811
    from roomkit.voice.base import AudioChunk
    from roomkit.voice.pipeline import AudioPipelineConfig, MockAECProvider

    backend, media = sip
    carrier = VoiceSession(id="sip", room_id="r", participant_id="p", channel_id="sip")
    state = SIPSessionState(session=carrier, call_session=media)
    backend._session_states[carrier.id] = state
    provider, aec = MockRealtimeProvider(), MockAECProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=SIPRealtimeTransport(backend),
        input_sample_rate=8000,
        output_sample_rate=8000,
        pipeline=AudioPipelineConfig(aec=aec),
    )
    active = await channel.start_session("r", "p", carrier)
    try:
        await provider.simulate_response_start(active)
        await provider.simulate_audio(active, b"\x01\x00" * 320)
        await provider.simulate_response_end(active)
        await channel.wait_idle("r", timeout=1)
        await asyncio.wait_for(state.pacer.wait_for_response_done(), 1)
        assert sum(len(frame.data) for frame in aec.reference_frames) == 640
        backend.send_audio_sync(carrier, AudioChunk(data=b"\x02\x00" * 160, sample_rate=8000))
        assert sum(len(frame.data) for frame in aec.reference_frames) == 960
    finally:
        await channel.close()
        await backend.close()


async def test_sip_failed_final_packet_releases_drain_and_preserves_sender(sip) -> None:  # noqa: F811
    backend, media = sip
    carrier = VoiceSession(id="sip", room_id="r", participant_id="p", channel_id="sip")
    state = SIPSessionState(session=carrier, call_session=media)
    backend._session_states[carrier.id] = state
    media.send_audio_pcm.side_effect = [None, ConnectionError("media failed"), None, None]
    try:
        await backend.send_audio(carrier, b"\x01\x00" * 240)
        backend.end_of_response(carrier)
        await asyncio.wait_for(state.pacer.wait_for_response_done(), 1)
        assert not state.is_playing
        assert not state.send_buffer
        await backend.send_audio(carrier, b"\x02\x00" * 320)
        backend.end_of_response(carrier)
        await asyncio.wait_for(state.pacer.wait_for_response_done(), 1)
        assert media.send_audio_pcm.call_count == 4
        assert not state.is_playing
    finally:
        await backend.close()
