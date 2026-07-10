"""Tests for VoiceChannel(close_providers=...) lifecycle control."""

from __future__ import annotations

from roomkit import VoiceChannel
from roomkit.voice.backends.mock import MockVoiceBackend


class _FakeClosable:
    """Minimal stand-in for an STT/TTS provider that records close()."""

    supports_streaming = False  # read by pipeline setup when used as STT

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _backend_closed(backend: MockVoiceBackend) -> bool:
    return any(call.method == "close" for call in backend.calls)


class TestVoiceChannelCloseProviders:
    async def test_default_closes_stt_and_tts(self) -> None:
        stt, tts, backend = _FakeClosable(), _FakeClosable(), MockVoiceBackend()
        channel = VoiceChannel("v", stt=stt, tts=tts, backend=backend)

        await channel.close()

        assert stt.closed is True
        assert tts.closed is True
        assert _backend_closed(backend) is True

    async def test_close_providers_false_leaves_stt_tts_open(self) -> None:
        """The caller owns the injected providers; only the backend is closed."""
        stt, tts, backend = _FakeClosable(), _FakeClosable(), MockVoiceBackend()
        channel = VoiceChannel("v", stt=stt, tts=tts, backend=backend, close_providers=False)

        await channel.close()

        assert stt.closed is False
        assert tts.closed is False
        assert _backend_closed(backend) is True
