# WavFileRecorder

A debug audio recorder that writes pipeline audio to `.wav` files on disk using Python's stdlib `wave` module. Zero external dependencies.

Use it to inspect exactly what the audio pipeline sees — useful for verifying AEC effectiveness, denoiser quality, AGC levels, or diagnosing audio issues in general.

## Quick start

```python
from roomkit.voice.pipeline import AudioPipelineConfig, WavFileRecorder
from roomkit.voice.pipeline.recorder import RecordingConfig

config = AudioPipelineConfig(
    recorder=WavFileRecorder(),
    recording_config=RecordingConfig(storage="./recordings"),
    # ... other providers (vad, denoiser, etc.)
)
```

Recording starts automatically when a voice session becomes active and stops when the session ends. Output files appear in the `storage` directory as `{session_id}_{timestamp}.wav`.

## Channel modes

Configure via `RecordingConfig.channels`:

| Mode | Output | Description |
|---|---|---|
| `MIXED` (default) | Single mono `.wav` | Inbound + outbound averaged into one channel. |
| `SEPARATE` | Two files: `*_inbound.wav`, `*_outbound.wav` | Each direction in its own file. Best for comparing mic vs speaker audio side by side. |
| `STEREO` | Single stereo `.wav` | Inbound on the left channel, outbound on the right. Open in any audio editor and solo L/R to inspect each direction. |

```python
from roomkit.voice.pipeline.recorder import RecordingChannelMode, RecordingConfig

# Stereo: inbound=left, outbound=right
config = RecordingConfig(
    storage="./recordings",
    channels=RecordingChannelMode.STEREO,
)

# Separate files per direction
config = RecordingConfig(
    storage="./recordings",
    channels=RecordingChannelMode.SEPARATE,
)
```

## Recording modes

Configure via `RecordingConfig.mode`:

| Mode | Behavior |
|---|---|
| `BOTH` (default) | Record both inbound (mic) and outbound (TTS/speaker) audio. |
| `INBOUND_ONLY` | Record only inbound audio. Outbound taps are ignored. |
| `OUTBOUND_ONLY` | Record only outbound audio. Inbound taps are ignored. |

```python
from roomkit.voice.pipeline.recorder import RecordingConfig, RecordingMode

# Only capture what the mic picks up
config = RecordingConfig(
    storage="./recordings",
    mode=RecordingMode.INBOUND_ONLY,
)
```

## Recording trigger

`RecordingTrigger.ALWAYS` is the only supported trigger. The recorder taps run before VAD in the pipeline, so speech boundaries are not available at recording time. If `SPEECH_ONLY` is configured, the recorder logs a warning and falls back to `ALWAYS`.

## Output directory

- If `RecordingConfig.storage` is set, files are written there (directories created automatically).
- If `storage` is empty, files go to the system temp directory (`tempfile.gettempdir()`).

## File naming

Files are named `{session_id}_{timestamp}.wav` where timestamp is `YYYYMMDDTHHMMSS` in UTC.

In `SEPARATE` mode, two files are created: `{session_id}_{timestamp}_inbound.wav` and `{session_id}_{timestamp}_outbound.wav`.

## How mixing works

For `MIXED` and `STEREO` modes, inbound and outbound audio is buffered in memory during the session. On `stop()`:

- The shorter buffer is padded with silence to match the longer one.
- **MIXED**: samples are averaged `(inbound + outbound) / 2` into a mono signal.
- **STEREO**: samples are interleaved as left (inbound) / right (outbound).

For `SEPARATE` mode, audio is written directly to disk via `wave.Wave_write` — no buffering needed.

## Pipeline position

The recorder taps are positioned early in the pipeline:

- **Inbound tap**: after resampling, before AEC/AGC/denoiser/VAD. You hear the raw mic signal (at pipeline sample rate).
- **Outbound tap**: after post-processors, before AEC reference feed. You hear the final TTS output.

## Example

See [`examples/wav_recorder.py`](../examples/wav_recorder.py) for a complete runnable example using mock providers.
