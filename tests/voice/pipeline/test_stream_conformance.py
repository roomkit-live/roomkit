"""Every pipeline stage in the repo must keep its state per stream.

One check, applied to every stage implementation: a stream's output sequence
must not change when a second stream is interleaved with it.  See
``stream_conformance.assert_stage_keeps_state_per_stream`` for the contract.

The resamplers are in here with the rest.  They are stage 1 of the inbound
pipeline and bound by the same contract, and their entry point taking a target
format rather than only a frame is a reason for an adapter, not an exemption:
a resampler that buffers a frame for look-ahead and keys that buffer on format
alone hands one participant's audio to the next stream that asks.

The native providers are driven through **stateful** fakes rather than no-op
mocks.  A stub that ignores its arguments would return the same bytes whether
or not the state is shared, so the check would pass on a provider that mixes
streams — which is the one thing it exists to catch.  Each fake therefore
derives its output from the history of the state object it was handed.
"""

from __future__ import annotations

import ctypes
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.aec.mock import MockAECProvider
from roomkit.voice.pipeline.agc.mock import MockAGCProvider
from roomkit.voice.pipeline.agc.simple import SimpleAGCProvider
from roomkit.voice.pipeline.denoiser.mock import MockDenoiserProvider
from roomkit.voice.pipeline.diarization.base import DiarizationResult
from roomkit.voice.pipeline.diarization.mock import MockDiarizationProvider
from roomkit.voice.pipeline.dtmf.base import DTMFEvent
from roomkit.voice.pipeline.dtmf.mock import MockDTMFDetector
from roomkit.voice.pipeline.resampler.base import ResamplerProvider
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.pipeline.resampler.mock import MockResamplerProvider
from roomkit.voice.pipeline.resampler.numpy import NumpyResamplerProvider
from roomkit.voice.pipeline.resampler.sinc import SincResamplerProvider
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider
from roomkit.voice.pipeline.vad.mock import MockVADProvider
from tests.voice.pipeline.stream_conformance import assert_stage_keeps_state_per_stream

# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------


def _frame(n_bytes: int, value: int, sample_rate: int = 16000) -> AudioFrame:
    return AudioFrame(
        data=value.to_bytes(2, "little", signed=True) * (n_bytes // 2),
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
    )


def _plain_frame(i: int) -> AudioFrame:
    """320 bytes — 160 samples, valid for every stage's chunk size."""
    return _frame(320, 100 + i)


def _speech_pattern(i: int) -> AudioFrame:
    """Alternating loud and silent frames, so a VAD actually transitions."""
    return _frame(320, 4000 if i % 2 == 0 else 0)


def _speex_frame(i: int) -> AudioFrame:
    """640 bytes — SpeexAECProvider's default 320-sample frame."""
    return _frame(640, 100 + i)


def _diarization_frame(i: int) -> AudioFrame:
    """250 ms of speech, ending an utterance every third frame."""
    frame = _frame(8000, 100 + i)
    frame.metadata["vad_is_speech"] = True
    frame.metadata["vad_speech_end"] = i % 3 == 2
    return frame


# ---------------------------------------------------------------------------
# Stateful fakes for the native / optional-dependency providers
# ---------------------------------------------------------------------------


def _fake_librnnoise():
    """librnnoise whose output depends on how often each state was used."""
    handles = {"next": 0xB000}
    calls: dict[int, int] = {}

    def create(_model):
        handles["next"] += 0x10
        calls[handles["next"]] = 0
        return handles["next"]

    def process_frame(state, out_buf, in_buf):
        key = state.value if isinstance(state, ctypes.c_void_p) else state
        calls[key] = calls.get(key, 0) + 1
        for i in range(480):
            out_buf[i] = float(in_buf[i] + calls[key])
        return ctypes.c_float(0.9)

    lib = MagicMock(spec=ctypes.CDLL)
    lib.rnnoise_get_frame_size = MagicMock(return_value=480)
    lib.rnnoise_create = MagicMock(side_effect=create)
    lib.rnnoise_destroy = MagicMock()
    lib.rnnoise_process_frame = MagicMock(side_effect=process_frame)
    return lib


def _make_rnnoise():
    lib = _fake_librnnoise()
    with (
        patch("ctypes.util.find_library", return_value="/fake/librnnoise.so"),
        patch("ctypes.CDLL", return_value=lib),
    ):
        import roomkit.voice.pipeline.denoiser.rnnoise as mod

        mod._lib = None
        importlib.reload(mod)
        return mod.RNNoiseDenoiserProvider(sample_rate=16000)


def _fake_libspeexdsp():
    """libspeexdsp whose cancelled output depends on each state's history."""
    handles = {"next": 0xC000}
    calls: dict[int, int] = {}

    def state_init(_frame_size, _filter_length):
        handles["next"] += 0x10
        calls[handles["next"]] = 0
        return handles["next"]

    def capture(state, rec, out):
        key = state.value if isinstance(state, ctypes.c_void_p) else state
        calls[key] = calls.get(key, 0) + 1
        for i in range(320):
            out[i] = (rec[i] + calls[key]) % 32767

    lib = MagicMock(spec=ctypes.CDLL)
    lib.speex_echo_state_init = MagicMock(side_effect=state_init)
    lib.speex_echo_state_destroy = MagicMock()
    lib.speex_echo_state_reset = MagicMock()
    lib.speex_echo_playback = MagicMock()
    lib.speex_echo_capture = MagicMock(side_effect=capture)
    lib.speex_echo_cancellation = MagicMock()
    lib.speex_echo_ctl = MagicMock(return_value=0)
    return lib


def _make_speex():
    lib = _fake_libspeexdsp()
    with (
        patch("ctypes.util.find_library", return_value="/fake/libspeexdsp.so"),
        patch("ctypes.CDLL", return_value=lib),
    ):
        import roomkit.voice.pipeline.aec.speex as mod

        mod._lib = None
        importlib.reload(mod)
        return mod.SpeexAECProvider()


class _FakeAudioProcessor:
    """WebRTC AP whose output depends on its own call history."""

    def __init__(self, **_kwargs) -> None:
        self._calls = 0

    def set_stream_format(self, *_a) -> None: ...

    def set_reverse_stream_format(self, *_a) -> None: ...

    def set_stream_delay(self, *_a) -> None: ...

    def process_reverse_stream(self, _chunk: bytes) -> None:
        self._calls += 1

    def process_stream(self, chunk: bytes) -> bytes:
        self._calls += 1
        shift = self._calls % 128
        return bytes((b + shift) % 256 for b in chunk)


def _make_webrtc():
    mod = SimpleNamespace(AudioProcessor=_FakeAudioProcessor)
    with patch.dict(sys.modules, {"aec_audio_processing": mod}):
        import roomkit.voice.pipeline.aec.webrtc as webrtc_mod

        importlib.reload(webrtc_mod)
        provider = webrtc_mod.WebRTCAECProvider()
        provider.set_active(True)  # bypass is on until playback starts
        return provider


def _make_webrtc_denoiser():
    mod = SimpleNamespace(AudioProcessor=_FakeAudioProcessor)
    with patch.dict(sys.modules, {"aec_audio_processing": mod}):
        import roomkit.voice.pipeline.denoiser.webrtc as webrtc_mod

        importlib.reload(webrtc_mod)
        return webrtc_mod.WebRTCNoiseSuppressorProvider()


class _FakeSherpaVAD:
    """Speech is reported once this detector alone has seen enough audio."""

    def __init__(self, *_a, **_kw) -> None:
        self._fed = 0

    def accept_waveform(self, _samples) -> None:
        self._fed += 1

    def empty(self) -> bool:
        return True

    def pop(self) -> None: ...

    def is_speech_detected(self) -> bool:
        return (self._fed % 4) in (1, 2)

    def reset(self) -> None:
        self._fed = 0

    def flush(self) -> None: ...


def _make_sherpa_vad():
    mod = SimpleNamespace(
        VadModelConfig=lambda: SimpleNamespace(
            silero_vad=SimpleNamespace(),
            ten_vad=SimpleNamespace(),
            sample_rate=16000,
            num_threads=1,
            provider="cpu",
        ),
        VoiceActivityDetector=_FakeSherpaVAD,
    )
    with patch.dict(sys.modules, {"sherpa_onnx": mod}):
        import roomkit.voice.pipeline.vad.sherpa_onnx as vad_mod

        importlib.reload(vad_mod)
        return vad_mod.SherpaOnnxVADProvider(
            vad_mod.SherpaOnnxVADConfig(model="/fake.onnx", min_speech_duration_ms=0)
        )


class _FakeSherpaDenoiser:
    """Output length tracks the context window this instance was given."""

    sample_rate = 16000

    def __init__(self, *_a, **_kw) -> None:
        self._calls = 0

    def run(self, samples, _rate):
        self._calls += 1
        shift = self._calls / 1000.0
        return SimpleNamespace(samples=[float(s) + shift for s in samples])


def _make_sherpa_denoiser():
    mod = SimpleNamespace(
        OfflineSpeechDenoiserGtcrnModelConfig=MagicMock(),
        OfflineSpeechDenoiserModelConfig=MagicMock(),
        OfflineSpeechDenoiserConfig=MagicMock(),
        OfflineSpeechDenoiser=_FakeSherpaDenoiser,
    )
    with patch.dict(sys.modules, {"sherpa_onnx": mod}):
        import roomkit.voice.pipeline.denoiser.sherpa_onnx as dn_mod

        importlib.reload(dn_mod)
        return dn_mod.SherpaOnnxDenoiserProvider(
            dn_mod.SherpaOnnxDenoiserConfig(model="/fake.onnx", silence_threshold=0)
        )


class _FakeQuailProcessor:
    def __init__(self, *_a, **_kw) -> None:
        self._calls = 0

    def context(self):
        return SimpleNamespace(set_parameter=lambda *_a: None)

    def process(self, samples_2d):
        self._calls += 1
        shift = self._calls / 1000.0
        return [[float(s) + shift for s in samples_2d[0]]]


def _make_aicoustics():
    mod = SimpleNamespace(
        Model=SimpleNamespace(download=MagicMock(return_value="/fake/model.onnx")),
        ProcessorConfig=SimpleNamespace(
            optimal=MagicMock(return_value=SimpleNamespace(num_frames=160))
        ),
        Processor=_FakeQuailProcessor,
    )
    with patch.dict(sys.modules, {"aic_sdk": mod}):
        import roomkit.voice.pipeline.denoiser.aicoustics as aic_mod

        importlib.reload(aic_mod)
        return aic_mod.AICousticsDenoiserProvider()


def _make_sherpa_diarization():
    """The registry is deliberately shared, so it answers the same every time.

    The per-stream signal is therefore ``is_new_speaker`` — which reads the
    stream's own ``last_speaker_id`` — and *when* an extraction fires, which
    depends on the stream's own accumulated buffer.
    """
    extractor = MagicMock()
    extractor.dim = 192
    extractor.is_ready = MagicMock(return_value=True)
    extractor.compute = MagicMock(return_value=[0.1] * 192)

    manager = MagicMock()
    manager.all_speakers = []
    manager.search = MagicMock(return_value="alice")

    mod = SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=MagicMock(),
        SpeakerEmbeddingExtractor=MagicMock(return_value=extractor),
        SpeakerEmbeddingManager=MagicMock(return_value=manager),
    )
    with patch.dict(sys.modules, {"sherpa_onnx": mod}):
        import roomkit.voice.pipeline.diarization.sherpa_onnx as diar_mod

        importlib.reload(diar_mod)
        return diar_mod.SherpaOnnxDiarizationProvider(
            diar_mod.SherpaOnnxDiarizationConfig(model="/fake.onnx", min_speech_ms=500)
        )


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _mock_vad():
    return MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"a", duration_ms=20.0),
            None,
            VADEvent(type=VADEventType.SPEECH_START),
            None,
        ]
    )


def _mock_dtmf():
    return MockDTMFDetector(
        events=[DTMFEvent(digit=d, duration_ms=40.0) for d in "12345"] + [None]
    )


def _mock_diarization():
    return MockDiarizationProvider(
        results=[
            DiarizationResult(speaker_id=f"s{i}", confidence=0.9, is_new_speaker=i == 0)
            for i in range(6)
        ]
    )


class _ResamplerStage:
    """A resampler behind the ``process(frame, stream)`` surface the check drives.

    The contract is the stage contract; only the call shape differs, because a
    resampler is told what format to produce. Converting 48 kHz down to the
    16 kHz internal format is what engages a delay line at all — asking for the
    format the frame already has returns it untouched and would test nothing.
    """

    def __init__(self, provider: ResamplerProvider) -> None:
        self._provider = provider

    @property
    def name(self) -> str:
        return self._provider.name

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        return self._provider.resample(frame, 16000, 1, 2, stream)

    def reset(self, stream: str) -> None:
        self._provider.reset(stream)


def _resampler_frame(i: int) -> AudioFrame:
    """160 samples at 48 kHz, each frame carrying a different value.

    Wrapped into int16 range: the check feeds noise frames from a much higher
    index than the stream under test, and a raw multiple would not fit.
    """
    return _frame(320, (i * 37) % 20000 - 10000, sample_rate=48000)


_STAGES = [
    # (id, factory, frame builder)
    ("vad/energy", lambda: EnergyVADProvider(min_speech_duration_ms=0), _speech_pattern),
    ("vad/mock", _mock_vad, _plain_frame),
    ("vad/sherpa_onnx", _make_sherpa_vad, _speech_pattern),
    ("denoiser/mock", MockDenoiserProvider, _plain_frame),
    ("denoiser/rnnoise", _make_rnnoise, _plain_frame),
    ("denoiser/sherpa_onnx", _make_sherpa_denoiser, _plain_frame),
    ("denoiser/aicoustics", _make_aicoustics, _plain_frame),
    ("denoiser/webrtc", _make_webrtc_denoiser, _plain_frame),
    ("aec/mock", MockAECProvider, _plain_frame),
    ("aec/speex", _make_speex, _speex_frame),
    ("aec/webrtc", _make_webrtc, _plain_frame),
    ("agc/mock", MockAGCProvider, _plain_frame),
    ("agc/simple", SimpleAGCProvider, _plain_frame),
    ("dtmf/mock", _mock_dtmf, _plain_frame),
    ("diarization/mock", _mock_diarization, _plain_frame),
    ("diarization/sherpa_onnx", _make_sherpa_diarization, _diarization_frame),
    ("resampler/linear", lambda: _ResamplerStage(LinearResamplerProvider()), _resampler_frame),
    ("resampler/numpy", lambda: _ResamplerStage(NumpyResamplerProvider()), _resampler_frame),
    ("resampler/sinc", lambda: _ResamplerStage(SincResamplerProvider()), _resampler_frame),
    ("resampler/mock", lambda: _ResamplerStage(MockResamplerProvider()), _resampler_frame),
]


@pytest.mark.parametrize(
    ("factory", "make_frame"),
    [pytest.param(f, mf, id=name) for name, f, mf in _STAGES],
)
def test_stage_keeps_state_per_stream(factory, make_frame) -> None:
    assert_stage_keeps_state_per_stream(factory, make_frame)


def test_the_check_rejects_a_stage_that_ignores_the_key() -> None:
    """The net must have teeth: a stage that accepts `stream` and drops it fails.

    Without this, a vacuous check would pass every provider and the whole
    suite would be decoration.
    """

    class SharedStateVAD:
        """Exactly the defect this ticket removes: one counter for all streams."""

        def __init__(self) -> None:
            self._index = 0

        def process(self, frame, stream):  # noqa: ARG002 — the point is it ignores it
            self._index += 1
            return VADEvent(type=VADEventType.SPEECH_START, duration_ms=float(self._index))

        def reset(self, stream) -> None:  # noqa: ARG002
            self._index = 0

    with pytest.raises(AssertionError, match="shares state between streams"):
        assert_stage_keeps_state_per_stream(SharedStateVAD, _plain_frame)


def test_every_stage_implementation_is_covered() -> None:
    """Guard against a new stage landing without a conformance entry."""
    import pkgutil

    import roomkit.voice.pipeline as pipeline_pkg

    # "resampler" belongs here: it is stage 1 of the inbound pipeline, and
    # leaving it out is what let a stateful one mix two participants' audio
    # while this suite reported full coverage.
    stage_dirs = {"vad", "denoiser", "aec", "agc", "dtmf", "diarization", "resampler"}
    found = set()
    for stage in stage_dirs:
        pkg = importlib.import_module(f"roomkit.voice.pipeline.{stage}")
        for mod in pkgutil.iter_modules(pkg.__path__):
            if mod.name != "base":
                found.add(f"{stage}/{mod.name}")

    covered = {name for name, _, _ in _STAGES}
    assert found == covered, (
        f"stage implementations without a conformance entry: {sorted(found - covered)}; "
        f"entries with no implementation: {sorted(covered - found)}"
    )
    assert pipeline_pkg is not None
