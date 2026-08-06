"""Tests for continuous WebRTC noise suppression."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from roomkit import WebRTCNoiseSuppressorProvider as RootWebRTCNoiseSuppressorProvider
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.denoiser import (
    WebRTCNoiseSuppressorProvider as ExportedWebRTCNoiseSuppressorProvider,
)


def _mock_module() -> tuple[SimpleNamespace, MagicMock]:
    processors: list[MagicMock] = []

    def make_processor(**_kwargs: object) -> MagicMock:
        processor = MagicMock()
        processor.process_stream.side_effect = lambda data: bytes(value ^ 0xFF for value in data)
        processors.append(processor)
        return processor

    constructor = MagicMock(side_effect=make_processor)
    return SimpleNamespace(AudioProcessor=constructor), constructor


def _provider(module: SimpleNamespace, **kwargs: object):
    with patch.dict(sys.modules, {"aec_audio_processing": module}):
        import roomkit.voice.pipeline.denoiser.webrtc as webrtc_mod

        importlib.reload(webrtc_mod)
        return webrtc_mod.WebRTCNoiseSuppressorProvider(**kwargs)


def _frame(data: bytes, *, sample_rate: int = 16000) -> AudioFrame:
    return AudioFrame(
        data=data,
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
    )


class TestWebRTCNoiseSuppressorProvider:
    def test_exported_from_root_package(self) -> None:
        assert RootWebRTCNoiseSuppressorProvider is ExportedWebRTCNoiseSuppressorProvider

    def test_processes_noise_while_no_aec_playback_exists(self) -> None:
        module, constructor = _mock_module()
        provider = _provider(module)
        source = bytes(range(64)) * 5

        result = provider.process(_frame(source), "alice")

        constructor.assert_called_once_with(enable_aec=False, enable_ns=True, enable_agc=False)
        assert result.data == bytes(value ^ 0xFF for value in source)
        assert result.metadata["noise_suppressed"] is True

    def test_irregular_chunks_keep_exact_length_without_reemitting_raw_audio(self) -> None:
        module, _ = _mock_module()
        provider = _provider(module)
        first_data = b"\x11" * 100
        second_data = b"\x22" * 300

        first = provider.process(_frame(first_data), "alice")
        second = provider.process(_frame(second_data), "alice")

        assert len(first.data) == len(first_data)
        assert len(second.data) == len(second_data)
        assert first.data == b"\x00" * 100
        assert second.data[:220] == b"\x00" * 220
        assert second.data[220:] == b"\xee" * 80

    def test_streams_have_independent_processors(self) -> None:
        module, constructor = _mock_module()
        provider = _provider(module)

        provider.process(_frame(b"\x01\x00" * 160), "alice")
        provider.process(_frame(b"\x01\x00" * 160), "bob")

        assert constructor.call_count == 2
        assert set(provider._streams) == {"alice", "bob"}

    def test_mismatched_format_is_bypassed_without_state(self) -> None:
        module, constructor = _mock_module()
        provider = _provider(module, sample_rate=16000)
        frame = _frame(b"\x01\x00" * 160, sample_rate=8000)

        assert provider.process(frame, "alice") is frame
        constructor.assert_not_called()
        assert provider._streams == {}

    def test_close_prevents_state_resurrection(self) -> None:
        module, constructor = _mock_module()
        provider = _provider(module)
        provider.close()
        frame = _frame(b"\x01\x00" * 160)

        assert provider.process(frame, "alice") is frame
        constructor.assert_not_called()
        assert provider._streams == {}

    def test_invalid_native_output_length_fails_open_and_clears_chunking(self) -> None:
        module, _ = _mock_module()
        provider = _provider(module)
        frame = _frame(b"\x01\x00" * 160)
        state = provider._state_for("alice")
        assert state is not None
        state.processor.process_stream.return_value = b""
        state.processor.process_stream.side_effect = None

        result = provider.process(frame, "alice")

        assert result is frame
        assert state.input_buffer == bytearray()
        assert state.output_buffer == bytearray()
        assert state.chunking is False

    def test_native_initialization_failure_fails_open(self) -> None:
        module, constructor = _mock_module()
        constructor.side_effect = RuntimeError("native init failed")
        provider = _provider(module)
        frame = _frame(b"\x01\x00" * 160)

        assert provider.process(frame, "alice") is frame
        assert provider._streams == {}
