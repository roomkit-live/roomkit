"""Tests for turn detection integration in VoiceChannel."""

from __future__ import annotations

import asyncio

from roomkit.channels.voice import VoiceChannel
from roomkit.models.enums import HookTrigger
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.turn.base import TurnDecision
from roomkit.voice.pipeline.turn.mock import MockTurnDetector
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.pipeline.vad.mock import MockVADProvider


class _MockSTT:
    """Minimal mock STT that returns preconfigured transcripts."""

    name = "mock_stt"

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = transcripts
        self._index = 0

    async def transcribe(self, frame: AudioFrame):
        from roomkit.voice.base import TranscriptionResult

        if self._index < len(self._transcripts):
            text = self._transcripts[self._index]
            self._index += 1
            return TranscriptionResult(text=text)
        return TranscriptionResult(text="")

    async def close(self) -> None:
        pass


class _MockBackend:
    """Minimal mock backend for wiring."""

    name = "mock_backend"

    from roomkit.voice.base import VoiceCapability

    _caps = VoiceCapability.NONE
    _sessions: dict = {}
    _barge_in_cbs: list = []
    _audio_cbs: list = []

    @property
    def capabilities(self):
        return self._caps

    @property
    def feeds_aec_reference(self):
        return False

    def on_audio_received(self, cb):
        self._audio_cbs.append(cb)

    def on_barge_in(self, cb):
        self._barge_in_cbs.append(cb)

    async def send_transcription(self, session, text, role):
        pass

    def get_session(self, sid):
        return None

    def list_sessions(self, room_id):
        return []

    async def cancel_audio(self, session):
        pass

    async def close(self):
        pass

    async def simulate_audio(self, session, frame):
        for cb in self._audio_cbs:
            cb(session, frame)


class TestTurnDetection:
    async def test_complete_turn_routes_immediately(self):
        """When turn detector says complete, text is routed and ON_TURN_COMPLETE fires."""
        from unittest.mock import AsyncMock

        from roomkit.voice.base import VoiceSession

        # Turn detector returns complete
        detector = MockTurnDetector(
            decisions=[TurnDecision(is_complete=True, confidence=0.9, reason="done")]
        )
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio"),
            ]
        )
        config = AudioPipelineConfig(vad=vad, turn_detector=detector)
        stt = _MockSTT(transcripts=["Hello world"])
        backend = _MockBackend()

        channel = VoiceChannel("ch1", stt=stt, backend=backend, pipeline=config)

        # Mock framework
        mock_fw = AsyncMock()
        mock_fw._build_context = AsyncMock(return_value=AsyncMock())
        mock_fw.hook_engine.run_async_hooks = AsyncMock()
        mock_fw.hook_engine.run_sync_hooks = AsyncMock(
            return_value=AsyncMock(allowed=True, event="Hello world")
        )
        mock_fw.process_inbound = AsyncMock()
        channel.set_framework(mock_fw)

        session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="ch1")
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        binding = ChannelBinding(room_id="r1", channel_id="ch1", channel_type=ChannelType.VOICE)
        channel.bind_session(session, "r1", binding)

        # Simulate audio -> speech end -> transcribe -> turn detect
        await backend.simulate_audio(session, AudioFrame(data=b"\x00"))
        await asyncio.sleep(0.15)

        # Check ON_TURN_COMPLETE was fired
        turn_complete_calls = [
            c
            for c in mock_fw.hook_engine.run_async_hooks.call_args_list
            if c.args[1] == HookTrigger.ON_TURN_COMPLETE
        ]
        assert len(turn_complete_calls) == 1

        # Check text was routed
        assert mock_fw.process_inbound.called

    async def test_incomplete_turn_accumulates(self):
        """When turn detector says incomplete, text accumulates."""
        from unittest.mock import AsyncMock

        from roomkit.voice.base import VoiceSession

        # First call: incomplete. Second call: complete.
        detector = MockTurnDetector(
            decisions=[
                TurnDecision(is_complete=False, confidence=0.6, reason="trailing"),
                TurnDecision(is_complete=True, confidence=0.95, reason="done"),
            ]
        )
        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio1"),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio2"),
            ]
        )
        config = AudioPipelineConfig(vad=vad, turn_detector=detector)
        stt = _MockSTT(transcripts=["Hello", "world"])
        backend = _MockBackend()

        channel = VoiceChannel("ch1", stt=stt, backend=backend, pipeline=config)

        mock_fw = AsyncMock()
        mock_fw._build_context = AsyncMock(return_value=AsyncMock())
        mock_fw.hook_engine.run_async_hooks = AsyncMock()
        mock_fw.hook_engine.run_sync_hooks = AsyncMock(
            side_effect=[
                AsyncMock(allowed=True, event="Hello"),
                AsyncMock(allowed=True, event="world"),
            ]
        )
        mock_fw.process_inbound = AsyncMock()
        channel.set_framework(mock_fw)

        session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="ch1")
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        binding = ChannelBinding(room_id="r1", channel_id="ch1", channel_type=ChannelType.VOICE)
        channel.bind_session(session, "r1", binding)

        # First utterance — incomplete, should not route
        await backend.simulate_audio(session, AudioFrame(data=b"\x00"))
        await asyncio.sleep(0.15)

        incomplete_calls = [
            c
            for c in mock_fw.hook_engine.run_async_hooks.call_args_list
            if c.args[1] == HookTrigger.ON_TURN_INCOMPLETE
        ]
        assert len(incomplete_calls) == 1
        assert not mock_fw.process_inbound.called

        # Second utterance — complete, should route combined text
        await backend.simulate_audio(session, AudioFrame(data=b"\x01"))
        await asyncio.sleep(0.15)

        complete_calls = [
            c
            for c in mock_fw.hook_engine.run_async_hooks.call_args_list
            if c.args[1] == HookTrigger.ON_TURN_COMPLETE
        ]
        assert len(complete_calls) == 1
        assert mock_fw.process_inbound.called

        # The routed text should be the combined "Hello world"
        inbound_call = mock_fw.process_inbound.call_args
        assert "Hello world" in str(inbound_call)

    async def test_no_turn_detector_routes_immediately(self):
        """Without a turn detector, text is routed immediately (existing behaviour)."""
        from unittest.mock import AsyncMock

        from roomkit.voice.base import VoiceSession

        vad = MockVADProvider(
            events=[
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio"),
            ]
        )
        config = AudioPipelineConfig(vad=vad)  # No turn_detector
        stt = _MockSTT(transcripts=["Hello"])
        backend = _MockBackend()

        channel = VoiceChannel("ch1", stt=stt, backend=backend, pipeline=config)

        mock_fw = AsyncMock()
        mock_fw._build_context = AsyncMock(return_value=AsyncMock())
        mock_fw.hook_engine.run_async_hooks = AsyncMock()
        mock_fw.hook_engine.run_sync_hooks = AsyncMock(
            return_value=AsyncMock(allowed=True, event="Hello")
        )
        mock_fw.process_inbound = AsyncMock()
        channel.set_framework(mock_fw)

        session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="ch1")
        from roomkit.models.channel import ChannelBinding
        from roomkit.models.enums import ChannelType

        binding = ChannelBinding(room_id="r1", channel_id="ch1", channel_type=ChannelType.VOICE)
        channel.bind_session(session, "r1", binding)

        await backend.simulate_audio(session, AudioFrame(data=b"\x00"))
        await asyncio.sleep(0.15)

        # Should route directly without turn hooks
        assert mock_fw.process_inbound.called
        turn_calls = [
            c
            for c in mock_fw.hook_engine.run_async_hooks.call_args_list
            if c.args[1] in (HookTrigger.ON_TURN_COMPLETE, HookTrigger.ON_TURN_INCOMPLETE)
        ]
        assert len(turn_calls) == 0
