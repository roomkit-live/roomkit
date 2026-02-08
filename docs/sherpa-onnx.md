# sherpa-onnx Providers

RoomKit integrates with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for neural audio processing. All sherpa-onnx providers are pure-Python — no system libraries required.

```bash
pip install roomkit[sherpa-onnx]
```

---

## GPU Acceleration (CUDA)

All sherpa-onnx providers support NVIDIA GPU acceleration via CUDA for faster inference.

> **Important:** The default `sherpa-onnx` pip package is **CPU-only**. You must install
> the CUDA-specific wheel to enable GPU acceleration.

### Prerequisites

- NVIDIA GPU with compute capability 6.0+
- NVIDIA driver (verify with `nvidia-smi`)
- **cuDNN 9** system library (see installation below)

### Step 1 — Install cuDNN 9 (system-wide)

sherpa-onnx CUDA wheels link against cuDNN 9. This is a system library, not a Python package.

**Ubuntu/Debian:**

```bash
# Add NVIDIA package repository (if not already configured)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install cuDNN 9 for CUDA 12
sudo apt-get -y install cudnn9-cuda-12
```

For Ubuntu 22.04, replace `ubuntu2404` with `ubuntu2204` in the URL above.

Verify:

```bash
ldconfig -p | grep cudnn   # should list libcudnn.so.9
```

### Step 2 — Install the sherpa-onnx CUDA wheel

The CUDA wheels are hosted on a separate index (not on PyPI).

**With uv (recommended):**

```bash
uv pip install sherpa-onnx==1.12.23+cuda12.cudnn9 \
    -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
```

**With pip:**

```bash
pip install sherpa-onnx==1.12.23+cuda12.cudnn9 --no-index \
    -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
```

> **CUDA 11 vs 12:** If your system has CUDA 11.x, use `sherpa-onnx==1.12.23+cuda`
> instead. Check with `nvcc --version`.

### Step 3 — Verify

```bash
python -c "
import sherpa_onnx
print('sherpa-onnx version:', sherpa_onnx.__version__)
print('file:', sherpa_onnx.__file__)
"
```

When you set `provider="cuda"` in a config and CUDA is available, you should see
GPU memory usage increase in `nvidia-smi`.

### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `libcublasLt.so.11: cannot open shared object file` | sherpa-onnx CUDA 11 wheel installed on CUDA 12 system | Install the `+cuda12.cudnn9` wheel instead |
| `libcublasLt.so.12: cannot open shared object file` | CUDA toolkit not installed | `sudo apt install cuda-toolkit-12` |
| `libonnxruntime_providers_cuda.so: Failed to load` | cuDNN not installed | Install cuDNN 9 (Step 1 above) |
| `libcudnn.so.9: cannot open shared object file` | cuDNN 9 missing | `sudo apt install cudnn9-cuda-12` |
| `Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON` | CPU-only sherpa-onnx wheel installed | Install the CUDA wheel (Step 2 above) |
| CUDA requested but falls back to CPU silently | Wrong wheel or missing cuDNN | Check `nvidia-smi` for GPU memory usage |

### Configuration

Set `provider="cuda"` on any sherpa-onnx config dataclass:

| Provider | Config class | Recommended provider | Why |
|---|---|---|---|
| STT | `SherpaOnnxSTTConfig` | `"cuda"` | Batch inference on larger models — benefits from GPU |
| TTS | `SherpaOnnxTTSConfig` | `"cuda"` | Synthesis is compute-heavy — benefits from GPU |
| VAD | `SherpaOnnxVADConfig` | `"cpu"` | Runs per-frame (every 20ms), tiny model — GPU transfer overhead makes CUDA slower |
| Denoiser | `SherpaOnnxDenoiserConfig` | `"cpu"` | Runs per-frame (every 20ms), tiny model — GPU transfer overhead makes CUDA slower |

> **Best practice:** Use CUDA for STT and TTS only. VAD and denoiser process every
> 20ms audio frame with tiny models (2-5MB). The GPU memory transfer overhead per
> frame exceeds the computation time, making CUDA slower than CPU for these providers.
> Worse, it can cause 100% CPU usage and system hangs.

### Example

```python
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
    SherpaOnnxDenoiserConfig,
    SherpaOnnxDenoiserProvider,
)
from roomkit.voice.pipeline.vad.sherpa_onnx import (
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
)
from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider

# STT + TTS on GPU, VAD + denoiser on CPU (recommended)
denoiser = SherpaOnnxDenoiserProvider(
    SherpaOnnxDenoiserConfig(model="gtcrn_simple.onnx", provider="cpu")
)
vad = SherpaOnnxVADProvider(
    SherpaOnnxVADConfig(model="ten-vad.onnx", provider="cpu")
)
stt = SherpaOnnxSTTProvider(
    SherpaOnnxSTTConfig(
        tokens="tokens.txt",
        encoder="encoder.onnx",
        decoder="decoder.onnx",
        joiner="joiner.onnx",
        provider="cuda",
    )
)
tts = SherpaOnnxTTSProvider(
    SherpaOnnxTTSConfig(model="vits.onnx", tokens="tokens.txt", provider="cuda")
)

pipeline_config = AudioPipelineConfig(denoiser=denoiser, vad=vad)
```

If CUDA is unavailable at runtime, ONNX Runtime falls back to CPU automatically.

---

## SherpaOnnxDenoiserProvider

Neural speech enhancement using the GTCRN model. Removes background noise from microphone audio before it reaches VAD and STT.

Unlike `RNNoiseDenoiserProvider` (which requires `librnnoise` installed via apt/brew), the sherpa-onnx denoiser is a pure-Python dependency.

### Quick start

```python
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
    SherpaOnnxDenoiserConfig,
    SherpaOnnxDenoiserProvider,
)

denoiser = SherpaOnnxDenoiserProvider(
    SherpaOnnxDenoiserConfig(model="path/to/gtcrn_simple.onnx")
)

config = AudioPipelineConfig(denoiser=denoiser)
```

### Download the model

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
```

### Configuration

All parameters are set via `SherpaOnnxDenoiserConfig`:

| Parameter | Default | Description |
|---|---|---|
| `model` | `""` | Path to the `gtcrn_simple.onnx` model file. |
| `num_threads` | `1` | CPU threads for inference. |
| `provider` | `"cpu"` | ONNX execution provider (`"cpu"` or `"cuda"`). |
| `context_frames` | `10` | Number of preceding frames for temporal context (see below). |

### How it works

GTCRN is an offline model — it needs temporal context to distinguish speech from noise. Processing isolated 20ms frames produces poor quality and boundary artifacts. The provider uses a **sliding context window** to give the model enough context:

1. Each audio frame is converted from int16 PCM to float32 samples in [-1, 1].
2. The samples are appended to a sliding context buffer (up to `context_frames` preceding frames).
3. `OfflineSpeechDenoiser.run(context_buffer, sample_rate)` processes the full context window.
4. Only the **last frame's portion** of the denoised output is extracted — no latency is added.
5. The denoised float32 samples are converted back to int16 PCM.
6. A new `AudioFrame` is returned with the cleaned audio, preserving all metadata.
7. On any error, the original frame is returned unchanged (pass-through).

The default `context_frames=10` (200ms at 16kHz) provides ~6x better signal-to-noise ratio compared to frame-by-frame processing, at ~5ms per frame (well within the 20ms real-time budget).

### Lazy initialization

The denoiser is created lazily on the first call to `process()`, not at construction time. This means:

- Constructing `SherpaOnnxDenoiserProvider` is fast.
- Model loading happens only when audio actually arrives.
- `sherpa-onnx` must be installed at construction time (the import check happens in `__init__`).

### Lazy loader

```python
from roomkit.voice import get_sherpa_onnx_denoiser_provider, get_sherpa_onnx_denoiser_config

SherpaOnnxDenoiserProvider = get_sherpa_onnx_denoiser_provider()
SherpaOnnxDenoiserConfig = get_sherpa_onnx_denoiser_config()

denoiser = SherpaOnnxDenoiserProvider(
    SherpaOnnxDenoiserConfig(model="gtcrn_simple.onnx")
)
```

### Denoiser + VAD together

```python
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
    SherpaOnnxDenoiserConfig,
    SherpaOnnxDenoiserProvider,
)
from roomkit.voice.pipeline.vad.sherpa_onnx import (
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
)

denoiser = SherpaOnnxDenoiserProvider(
    SherpaOnnxDenoiserConfig(model="gtcrn_simple.onnx")
)
vad = SherpaOnnxVADProvider(
    SherpaOnnxVADConfig(model="ten-vad.onnx")
)

config = AudioPipelineConfig(denoiser=denoiser, vad=vad)
```

---

## SherpaOnnxVADProvider

Neural voice activity detection using [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Supports **TEN-VAD** and **Silero VAD** models for production-quality speech detection that works reliably in noisy environments.

Unlike `EnergyVADProvider` (simple RMS thresholding), the neural VAD can distinguish speech from loud non-speech noise like keyboard typing, music, or HVAC.

### Quick start

```python
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider

vad = SherpaOnnxVADProvider(
    SherpaOnnxVADConfig(
        model="path/to/ten-vad.onnx",
        model_type="ten",
    )
)

config = AudioPipelineConfig(vad=vad)
```

### Download a model

```bash
# TEN-VAD (recommended — fast, low latency)
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx

# Silero VAD
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/silero_vad.onnx
```

### Configuration

All parameters are set via `SherpaOnnxVADConfig`:

| Parameter | Default | Description |
|---|---|---|
| `model` | `""` | Path to the `.onnx` model file. |
| `model_type` | `"ten"` | Model architecture: `"ten"` or `"silero"`. |
| `threshold` | `0.5` | Speech probability threshold (0–1). Lower = more sensitive. |
| `silence_threshold_ms` | `500` | Consecutive silence in ms to trigger `SPEECH_END`. |
| `min_speech_duration_ms` | `250` | Minimum speech duration to emit. Shorter segments are discarded. |
| `speech_pad_ms` | `300` | Pre-roll buffer — audio before speech onset included in the segment. |
| `max_speech_duration` | `20.0` | Maximum speech segment length in seconds (sherpa-internal). |
| `sample_rate` | `16000` | Expected audio sample rate. |
| `num_threads` | `1` | CPU threads for inference. |
| `provider` | `"cpu"` | ONNX execution provider (`"cpu"` or `"cuda"`). |

### How it works

The provider uses sherpa-onnx's `VoiceActivityDetector` with frame-level `is_speech_detected()` plus its own state machine:

1. Each audio frame is converted from int16 PCM to float32 and fed to the detector.
2. `is_speech_detected()` returns whether the current frame contains speech.
3. The state machine tracks idle/speaking transitions:
   - **Idle → Speaking**: emits `SPEECH_START` immediately (no delay).
   - **Speaking → Idle**: after `silence_threshold_ms` of consecutive non-speech, emits `SPEECH_END` with accumulated `audio_bytes`.
4. A pre-roll buffer keeps recent frames so speech onset isn't clipped.
5. Segments shorter than `min_speech_duration_ms` are silently discarded.

This gives instant `SPEECH_START` events (not delayed until a segment completes) and full control over timing parameters.

### TEN-VAD vs Silero VAD

| | TEN-VAD | Silero VAD |
|---|---|---|
| Latency | Very low | Low |
| Model size | ~2 MB | ~2 MB |
| Accuracy | Excellent | Excellent |
| Use case | Real-time voice assistants | General purpose |

Both models work well. TEN-VAD is recommended for real-time applications where latency matters.

### Lazy initialization

The sherpa-onnx detector is created lazily on the first call to `process()`, not at construction time. This means:

- Importing and constructing `SherpaOnnxVADProvider` is fast.
- Model loading happens only when audio actually arrives.
- `sherpa-onnx` must be installed at construction time (the import check happens in `__init__`).

### Fallback pattern

Use neural VAD when available, fall back to energy-based:

```python
import os

from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider

vad_model = os.environ.get("VAD_MODEL", "")
if vad_model:
    from roomkit.voice.pipeline.vad.sherpa_onnx import (
        SherpaOnnxVADConfig,
        SherpaOnnxVADProvider,
    )

    vad = SherpaOnnxVADProvider(
        SherpaOnnxVADConfig(
            model=vad_model,
            model_type=os.environ.get("VAD_MODEL_TYPE", "ten"),
            threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
        )
    )
else:
    vad = EnergyVADProvider(energy_threshold=300.0)

config = AudioPipelineConfig(vad=vad)
```

### Lazy loader

If you want to avoid importing sherpa-onnx at module level:

```python
from roomkit.voice import get_sherpa_onnx_vad_provider, get_sherpa_onnx_vad_config

SherpaOnnxVADProvider = get_sherpa_onnx_vad_provider()
SherpaOnnxVADConfig = get_sherpa_onnx_vad_config()

vad = SherpaOnnxVADProvider(SherpaOnnxVADConfig(model="ten-vad.onnx"))
```

---

## Examples

- [`examples/voice_local_onnx_vllm.py`](../examples/voice_local_onnx_vllm.py) — fully local voice assistant with sherpa-onnx STT/TTS/VAD + Ollama/vLLM, CUDA support.
- [`examples/voice_sherpa_onnx_vad.py`](../examples/voice_sherpa_onnx_vad.py) — standalone local mic demo with neural VAD + optional denoiser.
- [`examples/voice_cloud.py`](../examples/voice_cloud.py) — cloud voice assistant with Deepgram STT + ElevenLabs TTS + Claude (set `VAD_MODEL` and/or `DENOISE_MODEL` env vars).
