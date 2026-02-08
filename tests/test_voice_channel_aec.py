"""VoiceChannel AEC integration tests.

Verifies that VoiceChannel correctly wires AEC through the pipeline:
- _wrap_outbound routes TTS through process_outbound (feeding AEC reference)
- _setup_pipeline passes feeds_aec_reference from the backend
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from roomkit.voice.backends.base import VoiceBackend
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.pipeline.aec.mock import MockAECProvider
from roomkit.voice.pipeline.config import AudioPipelineConfig


def _session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(id=sid, room_id="r1", participant_id="p1", channel_id="c1")


class _StubBackend(VoiceBackend):
    """Minimal backend for testing VoiceChannel pipeline wiring."""

    def __init__(self, *, feeds_aec: bool = False) -> None:
        self._feeds_aec = feeds_aec
        self._audio_cb = None

    @property
    def name(self) -> str:
        return "stub"

    @property
    def feeds_aec_reference(self) -> bool:
        return self._feeds_aec

    def on_audio_received(self, callback):
        self._audio_cb = callback

    async def connect(self, room_id, participant_id, channel_id, *, metadata=None):
        return _session()

    async def disconnect(self, session):
        pass

    async def send_audio(self, session, audio):
        pass


class TestWrapOutbound:
    """Test that _wrap_outbound routes TTS through the pipeline outbound path."""

    async def test_wrap_outbound_feeds_aec_reference(self):
        """_wrap_outbound should feed AEC reference via process_outbound."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        backend = _StubBackend()

        from roomkit.channels.voice import VoiceChannel

        channel = VoiceChannel("voice", backend=backend, pipeline=config)

        session = _session()

        async def _chunks() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=b"\x01\x00" * 10, sample_rate=16000)
            yield AudioChunk(data=b"\x02\x00" * 10, sample_rate=16000)

        results = []
        async for chunk in channel._wrap_outbound(session, _chunks()):
            results.append(chunk)

        assert len(results) == 2
        assert len(aec.reference_frames) == 2

    async def test_wrap_outbound_passes_through_without_pipeline(self):
        """Without a pipeline, _wrap_outbound yields chunks unchanged."""
        from roomkit.channels.voice import VoiceChannel

        channel = VoiceChannel("voice")

        session = _session()

        async def _chunks() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=b"\x01\x00", sample_rate=16000)

        results = []
        async for chunk in channel._wrap_outbound(session, _chunks()):
            results.append(chunk)

        assert len(results) == 1
        assert results[0].data == b"\x01\x00"


class TestSetupPipelineFeedsAEC:
    """Test that _setup_pipeline passes feeds_aec_reference from the backend."""

    def test_feeds_aec_reference_true(self):
        """Backend with feeds_aec_reference=True sets pipeline flag."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        backend = _StubBackend(feeds_aec=True)

        from roomkit.channels.voice import VoiceChannel

        channel = VoiceChannel("voice", backend=backend, pipeline=config)

        assert channel._pipeline is not None
        assert channel._pipeline._backend_feeds_aec_ref is True

    def test_feeds_aec_reference_false(self):
        """Backend with feeds_aec_reference=False sets pipeline flag to False."""
        aec = MockAECProvider()
        config = AudioPipelineConfig(aec=aec)
        backend = _StubBackend(feeds_aec=False)

        from roomkit.channels.voice import VoiceChannel

        channel = VoiceChannel("voice", backend=backend, pipeline=config)

        assert channel._pipeline is not None
        assert channel._pipeline._backend_feeds_aec_ref is False
