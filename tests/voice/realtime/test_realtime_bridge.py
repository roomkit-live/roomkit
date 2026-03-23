"""Tests for RealtimeAVBridge (voice/realtime/bridge.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.realtime.mock import MockRealtimeProvider


class TestResamplePcm:
    def test_same_rate_passthrough(self) -> None:
        from roomkit.voice.realtime.bridge import resample_pcm

        pcm = np.zeros(160, dtype=np.int16).tobytes()
        result = resample_pcm(pcm, 16000, 16000)
        assert result == pcm

    def test_upsample(self) -> None:
        from roomkit.voice.realtime.bridge import resample_pcm

        # 160 samples at 16kHz -> should become ~480 samples at 48kHz
        pcm = np.zeros(160, dtype=np.int16).tobytes()
        result = resample_pcm(pcm, 16000, 48000)
        # Each sample is 2 bytes (int16)
        result_samples = len(result) // 2
        assert result_samples == 480

    def test_downsample(self) -> None:
        from roomkit.voice.realtime.bridge import resample_pcm

        # 480 samples at 48kHz -> should become ~160 samples at 16kHz
        pcm = np.zeros(480, dtype=np.int16).tobytes()
        result = resample_pcm(pcm, 48000, 16000)
        result_samples = len(result) // 2
        assert result_samples == 160

    def test_empty_input(self) -> None:
        from roomkit.voice.realtime.bridge import resample_pcm

        result = resample_pcm(b"", 16000, 48000)
        assert result == b""


class TestRealtimeAVBridge:
    def test_constructor(self) -> None:
        from roomkit.voice.realtime.bridge import RealtimeAVBridge

        provider = MockRealtimeProvider()
        backend = MockVoiceBackend()
        bridge = RealtimeAVBridge(provider, backend)
        assert bridge._provider is provider
        assert bridge._backend is backend
        assert len(bridge._calls) == 0

    def test_add_video_tap(self) -> None:
        from roomkit.voice.realtime.bridge import RealtimeAVBridge

        provider = MockRealtimeProvider()
        backend = MockVoiceBackend()
        bridge = RealtimeAVBridge(provider, backend)
        tap = MagicMock()
        bridge.add_video_tap(tap)
        assert tap in bridge._video_taps

    async def test_close_empty(self) -> None:
        from roomkit.voice.realtime.bridge import RealtimeAVBridge

        provider = MockRealtimeProvider()
        backend = MockVoiceBackend()
        bridge = RealtimeAVBridge(provider, backend)
        await bridge.close()
        # Provider should be closed
        assert any(c.method == "close" for c in provider.calls)
