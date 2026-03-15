"""Tests for AnamRealtimeProvider."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

from roomkit.providers.anam.config import AnamConfig
from roomkit.voice.base import VoiceSession, VoiceSessionState

# ---------------------------------------------------------------------------
# Mock anam SDK
# ---------------------------------------------------------------------------


class _MockPersonaConfig:
    """Simulates anam.PersonaConfig."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


class _MockAnamSession:
    """Simulates an Anam WebRTC session (anam.Session)."""

    def __init__(self) -> None:
        self.closed = False
        self.interrupted = False
        self.sent_audio: list[tuple[bytes, int, int]] = []
        self.sent_messages: list[str] = []
        self._audio_frames: list[Any] = []
        self._video_frames: list[Any] = []

    def send_user_audio(
        self,
        audio: bytes,
        sample_rate: int,
        num_channels: int,
    ) -> None:
        self.sent_audio.append((audio, sample_rate, num_channels))

    def send_message(self, content: str) -> None:
        self.sent_messages.append(content)

    def talk(self, content: str) -> None:
        self.sent_messages.append(content)

    def interrupt(self) -> None:
        self.interrupted = True

    def close(self) -> None:
        self.closed = True

    async def audio_frames(self) -> Any:
        for frame in self._audio_frames:
            yield frame

    async def video_frames(self) -> Any:
        for frame in self._video_frames:
            yield frame


class _MockAnamConnectCtx:
    """Async context manager returned by AnamClient.connect()."""

    def __init__(self, session: _MockAnamSession) -> None:
        self._session = session

    async def __aenter__(self) -> _MockAnamSession:
        return self._session

    async def __aexit__(self, *exc: Any) -> None:
        self._session.closed = True


class _MockAnamClient:
    """Simulates the anam.AnamClient."""

    def __init__(
        self,
        api_key: str,
        persona_id: str | None = None,
        persona_config: Any = None,
        options: Any = None,
    ) -> None:
        self.api_key = api_key
        self.persona_id = persona_id
        self.persona_config = persona_config
        self.session = _MockAnamSession()

    def connect(self) -> _MockAnamConnectCtx:
        return _MockAnamConnectCtx(self.session)


@dataclass
class _FakeAVAudioFrame:
    """Minimal PyAV AudioFrame stand-in."""

    _data: Any = None

    def to_ndarray(self) -> Any:
        import numpy as np

        if self._data is not None:
            return self._data
        return np.zeros(160, dtype=np.float32)


@dataclass
class _FakeAVVideoFrame:
    """Minimal PyAV VideoFrame stand-in."""

    width: int = 320
    height: int = 240
    _rgb_data: Any = None

    def to_ndarray(self, format: str = "rgb24") -> Any:
        import numpy as np

        if self._rgb_data is not None:
            return self._rgb_data
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


def _make_mock_anam_module() -> ModuleType:
    """Create a mock 'anam' module."""
    mod = ModuleType("anam")
    mod.AnamClient = _MockAnamClient  # type: ignore[attr-defined]
    mod.PersonaConfig = _MockPersonaConfig  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anam_module() -> ModuleType:
    return _make_mock_anam_module()


@pytest.fixture
def config() -> AnamConfig:
    return AnamConfig(api_key="ak-test", persona_id="persona-1")


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(
        id="sess-1",
        room_id="room-1",
        participant_id="part-1",
        channel_id="ch-1",
        state=VoiceSessionState.CONNECTING,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnamConfig:
    def test_defaults(self) -> None:
        cfg = AnamConfig(api_key="ak-test")
        assert cfg.language_code == "en"
        assert cfg.timeout == 30.0
        assert cfg.enable_audio_passthrough is False

    def test_inline_persona(self) -> None:
        cfg = AnamConfig(
            api_key="ak-test",
            avatar_id="av-1",
            voice_id="v-1",
            llm_id="llm-1",
            system_prompt="Hello",
        )
        assert cfg.persona_id is None
        assert cfg.avatar_id == "av-1"


class TestAnamRealtimeProviderConnect:
    async def test_connect_disconnect(
        self,
        anam_module: ModuleType,
        config: AnamConfig,
        session: VoiceSession,
    ) -> None:
        with patch.dict(sys.modules, {"anam": anam_module}):
            # Reset lazy-loaded modules
            import roomkit.providers.anam.realtime as _mod
            from roomkit.providers.anam.realtime import AnamRealtimeProvider

            _mod._anam_mod = None
            _mod._np = None

            provider = AnamRealtimeProvider(config)

            audio_received: list[bytes] = []
            provider.on_audio(lambda s, a: audio_received.append(a))

            await provider.connect(session)

            assert session.state == VoiceSessionState.ACTIVE
            assert session.id in provider._states

            await provider.disconnect(session)
            assert session.state == VoiceSessionState.ENDED
            assert session.id not in provider._states

    async def test_send_audio(
        self,
        anam_module: ModuleType,
        config: AnamConfig,
        session: VoiceSession,
    ) -> None:
        with patch.dict(sys.modules, {"anam": anam_module}):
            import roomkit.providers.anam.realtime as _mod

            _mod._anam_mod = None
            _mod._np = None

            provider = _mod.AnamRealtimeProvider(config)
            await provider.connect(session)

            state = provider._states[session.id]
            await provider.send_audio(session, b"\x00\x01\x02")
            assert len(state.anam_session.sent_audio) == 1
            audio, rate, channels = state.anam_session.sent_audio[0]
            assert audio == b"\x00\x01\x02"
            assert rate == 16000
            assert channels == 1

            await provider.disconnect(session)

    async def test_inject_text(
        self,
        anam_module: ModuleType,
        config: AnamConfig,
        session: VoiceSession,
    ) -> None:
        with patch.dict(sys.modules, {"anam": anam_module}):
            import roomkit.providers.anam.realtime as _mod

            _mod._anam_mod = None
            _mod._np = None

            provider = _mod.AnamRealtimeProvider(config)
            await provider.connect(session)

            state = provider._states[session.id]
            await provider.inject_text(session, "Hello")
            assert state.anam_session.sent_messages == ["Hello"]

            await provider.disconnect(session)

    async def test_interrupt(
        self,
        anam_module: ModuleType,
        config: AnamConfig,
        session: VoiceSession,
    ) -> None:
        with patch.dict(sys.modules, {"anam": anam_module}):
            import roomkit.providers.anam.realtime as _mod

            _mod._anam_mod = None
            _mod._np = None

            provider = _mod.AnamRealtimeProvider(config)
            await provider.connect(session)

            state = provider._states[session.id]
            await provider.interrupt(session)
            assert state.anam_session.interrupted is True

            await provider.disconnect(session)


class TestAnamConsumeLoops:
    async def test_audio_consume_fires_callbacks(
        self,
        anam_module: ModuleType,
        config: AnamConfig,
        session: VoiceSession,
    ) -> None:
        pytest.importorskip("numpy")
        with patch.dict(sys.modules, {"anam": anam_module}):
            import roomkit.providers.anam.realtime as _mod

            _mod._anam_mod = None
            _mod._np = None

            provider = _mod.AnamRealtimeProvider(config)

            audio_received: list[bytes] = []
            response_started = []
            provider.on_audio(lambda s, a: audio_received.append(a))
            provider.on_response_start(lambda s: response_started.append(True))

            await provider.connect(session)

            # Inject fake audio frames into the session
            state = provider._states[session.id]
            state.anam_session._audio_frames = [_FakeAVAudioFrame()]

            # Wait for the audio consume task to process
            await asyncio.sleep(0.1)

            await provider.disconnect(session)

            assert len(audio_received) == 1
            assert len(response_started) == 1

    async def test_video_consume_fires_callbacks(
        self,
        anam_module: ModuleType,
        config: AnamConfig,
        session: VoiceSession,
    ) -> None:
        pytest.importorskip("numpy")
        with patch.dict(sys.modules, {"anam": anam_module}):
            import roomkit.providers.anam.realtime as _mod

            _mod._anam_mod = None
            _mod._np = None

            provider = _mod.AnamRealtimeProvider(config)

            video_received: list[Any] = []
            provider.on_video(lambda s, f: video_received.append(f))

            await provider.connect(session)

            # Inject fake video frames
            state = provider._states[session.id]
            state.anam_session._video_frames = [_FakeAVVideoFrame()]

            await asyncio.sleep(0.1)

            await provider.disconnect(session)

            assert len(video_received) == 1
            frame = video_received[0]
            assert frame.codec == "raw_rgb24"
            assert frame.width == 320
            assert frame.height == 240


class TestAnamFormatConversion:
    def test_av_audio_to_pcm(self) -> None:
        import numpy as np

        pytest.importorskip("numpy")
        with patch.dict(sys.modules, {"anam": _make_mock_anam_module()}):
            import roomkit.providers.anam.realtime as _mod

            _mod._anam_mod = None
            _mod._np = None
            _ensure_np()

            # Mono float32 → int16
            from roomkit.providers.anam.realtime import AnamRealtimeProvider

            frame = _FakeAVAudioFrame(_data=np.array([0.5, -0.5], dtype=np.float32))
            pcm = AnamRealtimeProvider._av_audio_to_pcm(frame)
            assert isinstance(pcm, bytes)
            assert len(pcm) == 4  # 2 samples × 2 bytes

    def test_av_video_to_frame(self) -> None:
        pytest.importorskip("numpy")
        with patch.dict(sys.modules, {"anam": _make_mock_anam_module()}):
            import roomkit.providers.anam.realtime as _mod

            _mod._anam_mod = None
            _mod._np = None
            _ensure_np()

            from roomkit.providers.anam.realtime import AnamRealtimeProvider

            frame = _FakeAVVideoFrame(width=640, height=480)
            vf = AnamRealtimeProvider._av_video_to_frame(frame, sequence=5)
            assert vf.codec == "raw_rgb24"
            assert vf.width == 640
            assert vf.height == 480
            assert vf.sequence == 5


class TestAnamProviderClose:
    async def test_close_disconnects_all(
        self,
        anam_module: ModuleType,
        config: AnamConfig,
    ) -> None:
        with patch.dict(sys.modules, {"anam": anam_module}):
            import roomkit.providers.anam.realtime as _mod

            _mod._anam_mod = None
            _mod._np = None

            provider = _mod.AnamRealtimeProvider(config)

            s1 = VoiceSession(
                id="s1",
                room_id="r1",
                participant_id="p1",
                channel_id="c1",
                state=VoiceSessionState.CONNECTING,
            )
            s2 = VoiceSession(
                id="s2",
                room_id="r1",
                participant_id="p2",
                channel_id="c1",
                state=VoiceSessionState.CONNECTING,
            )

            await provider.connect(s1)
            await provider.connect(s2)
            assert len(provider._states) == 2

            await provider.close()
            assert len(provider._states) == 0
            assert s1.state == VoiceSessionState.ENDED
            assert s2.state == VoiceSessionState.ENDED


def _ensure_np() -> None:
    """Force numpy lazy-load in the anam module."""
    import numpy as np

    import roomkit.providers.anam.realtime as _mod

    _mod._np = np
