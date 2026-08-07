"""VoiceChannel AEC integration tests.

Verifies that VoiceChannel correctly wires AEC through the pipeline:
- _wrap_outbound routes TTS through process_outbound (feeding AEC reference)
- _setup_pipeline passes feeds_aec_reference from the backend
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.base import VoiceBackend
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession
from roomkit.voice.pipeline.aec.mock import MockAECProvider
from roomkit.voice.pipeline.config import AudioPipelineConfig


def _session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(id=sid, room_id="r1", participant_id="p1", channel_id="c1")


class _StubBackend(VoiceBackend):
    """Minimal backend for testing VoiceChannel pipeline wiring."""

    def __init__(
        self,
        *,
        feeds_aec: bool = False,
        supports_playback: bool = False,
        capabilities: VoiceCapability = VoiceCapability.NONE,
    ) -> None:
        self._feeds_aec = feeds_aec
        self._supports_playback = supports_playback
        self._capabilities = capabilities
        self._audio_cb = None
        self.audio_played_callbacks = []

    @property
    def name(self) -> str:
        return "stub"

    @property
    def feeds_aec_reference(self) -> bool:
        return self._feeds_aec

    @property
    def supports_playback_callback(self) -> bool:
        return self._supports_playback

    @property
    def capabilities(self) -> VoiceCapability:
        return self._capabilities

    def on_audio_received(self, callback):
        self._audio_cb = callback

    def on_audio_played(self, callback):
        self.audio_played_callbacks.append(callback)

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

    def test_native_aec_does_not_wire_pipeline_reference_callback(self):
        """NATIVE_AEC owns both directions; pipeline AEC stays untouched."""
        aec = MockAECProvider()
        backend = _StubBackend(
            supports_playback=True,
            capabilities=VoiceCapability.NATIVE_AEC,
        )

        from roomkit.channels.voice import VoiceChannel

        channel = VoiceChannel("voice", backend=backend, pipeline=AudioPipelineConfig(aec=aec))

        assert channel._pipeline is not None
        assert channel._pipeline._playback_aec_wired is False

    def test_playback_end_bypasses_pipeline_aec_after_final_reference(self):
        """A playback callback bypasses AEC without resetting its filter."""
        aec = MockAECProvider()
        backend = _StubBackend(supports_playback=True)

        from roomkit.channels.voice import VoiceChannel

        channel = VoiceChannel(
            "voice",
            backend=backend,
            pipeline=AudioPipelineConfig(aec=aec),
        )
        assert channel._pipeline is not None
        channel._pipeline.set_aec_active("s1", True)
        final_frame = AudioFrame(
            data=b"\x00\x00" * 160,
            sample_rate=16000,
            metadata={"playback_ended": True},
        )

        for callback in backend.audio_played_callbacks:
            callback(_session(), final_frame)

        assert aec.reference_streams == ["s1"]
        assert aec.active_changes == [("s1", True), ("s1", False)]
        assert aec.reset_streams == []


class TestBridgeAECActivity:
    """Bridge playback participates in per-session AEC lifecycle."""

    def test_bridge_keeps_aec_active_until_last_peer_leaves(self):
        """Forwarded audio activates AEC and a one-party bridge deactivates it."""
        aec = MockAECProvider()
        backend = _StubBackend()

        from roomkit.channels.voice import VoiceChannel

        channel = VoiceChannel(
            "voice",
            backend=backend,
            bridge=True,
            pipeline=AudioPipelineConfig(aec=aec),
        )
        alice = _session("alice")
        bob = _session("bob")
        binding = ChannelBinding(
            channel_id="voice",
            room_id="r1",
            channel_type=ChannelType.VOICE,
        )
        channel.bind_session(alice, "r1", binding)
        channel.bind_session(bob, "r1", binding)

        channel._process_bridge_outbound(
            bob,
            AudioFrame(data=b"\x01\x00" * 160, sample_rate=16000),
        )

        assert aec.active_changes == [("bob", True)]
        assert aec.reference_streams[-1] == "bob"
        aec.reset_streams.clear()

        channel.unbind_session(alice)

        assert aec.active_changes == [("bob", True), ("bob", False)]
        assert "bob" not in aec.reset_streams


class TestPlaybackReferencePausesWithCapture:
    """The pipeline-AEC reference honours the backend's capture_paused flag.

    While capture is paused (session mute, gating, half-duplex) the mic
    thread drops frames; feeding the reference alone desyncs AEC3's
    render/capture alignment by the pause's full duration — measured live
    as a 6 s mute leaving the filter cancelling against audio the capture
    never saw, then a false barge-in on unmute.
    """

    def _wired_channel(self):
        from roomkit.channels.voice import VoiceChannel

        aec = MockAECProvider()
        backend = _StubBackend(supports_playback=True)
        channel = VoiceChannel(
            "voice",
            backend=backend,
            pipeline=AudioPipelineConfig(aec=aec),
        )
        assert backend.audio_played_callbacks, "playback→AEC wiring did not register"
        return aec, backend, channel

    def _played(self, paused: bool) -> AudioFrame:
        return AudioFrame(
            data=b"\x01\x00" * 240,
            sample_rate=24000,
            channels=1,
            sample_width=2,
            metadata={"playback_ended": False, "capture_paused": paused},
        )

    def test_reference_flows_while_capture_runs(self):
        aec, backend, _channel = self._wired_channel()
        backend.audio_played_callbacks[0](_session(), self._played(paused=False))
        assert len(aec.reference_frames) == 1

    def test_reference_pauses_while_capture_is_paused(self):
        aec, backend, _channel = self._wired_channel()
        cb = backend.audio_played_callbacks[0]
        cb(_session(), self._played(paused=False))
        cb(_session(), self._played(paused=True))
        cb(_session(), self._played(paused=True))
        cb(_session(), self._played(paused=False))
        assert len(aec.reference_frames) == 2  # the two paused blocks were held

    def test_playback_ended_still_deactivates_during_pause(self):
        aec, backend, channel = self._wired_channel()
        channel._pipeline.set_aec_active(_session().id, True)
        frame = self._played(paused=True)
        frame.metadata["playback_ended"] = True
        backend.audio_played_callbacks[0](_session(), frame)
        assert aec.reference_frames == []  # paused: no reference fed
        assert (_session().id, False) in aec.active_changes  # ...but still deactivated
