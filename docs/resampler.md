# Resampler

A pluggable audio resampler that converts between transport and internal pipeline formats. The pipeline uses it in two directions:

- **Inbound**: transport format (e.g. 48kHz stereo from WebRTC) to internal format (e.g. 16kHz mono for VAD/STT)
- **Outbound**: internal format back to transport format

## Quick start

```python
from roomkit.voice.pipeline import AudioPipelineConfig, AudioPipelineContract, AudioFormat

# Declare formats — the contract is the single source of truth
contract = AudioPipelineContract(
    transport_inbound_format=AudioFormat(sample_rate=48000, channels=2),
    transport_outbound_format=AudioFormat(sample_rate=48000, channels=2),
    internal_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
)

config = AudioPipelineConfig(
    contract=contract,
    # ... other providers (vad, denoiser, etc.)
)
```

When `contract` is set but no explicit `resampler` is provided, the pipeline auto-creates a `LinearResamplerProvider`. No extra configuration needed for the common case.

## Built-in providers

### LinearResamplerProvider

Pure-Python resampler using linear interpolation. Handles channel conversion (mono/stereo), sample rate conversion, and sample width conversion. Zero external dependencies.

```python
from roomkit.voice.pipeline import AudioPipelineConfig, LinearResamplerProvider

config = AudioPipelineConfig(
    resampler=LinearResamplerProvider(),
    contract=contract,
)
```

This is the same algorithm that was previously hardcoded in the pipeline engine. It's suitable for development and testing. For production use with high sample-rate ratios, consider writing a custom provider backed by a higher-quality library.

### MockResamplerProvider

Passes frames through unchanged and records all calls. Useful for testing.

```python
from roomkit.voice.pipeline.resampler import MockResamplerProvider

resampler = MockResamplerProvider()
# ... run pipeline ...
assert len(resampler.calls) == 1
assert resampler.calls[0].target_rate == 16000
```

## Custom providers

Implement `ResamplerProvider` to plug in a higher-quality algorithm:

```python
from roomkit.voice.pipeline.resampler import ResamplerProvider
from roomkit.voice.audio_frame import AudioFrame


class SoxrResamplerProvider(ResamplerProvider):
    """High-quality resampler using libsoxr."""

    @property
    def name(self) -> str:
        return "soxr"

    def resample(
        self,
        frame: AudioFrame,
        target_rate: int,
        target_channels: int,
        target_width: int,
    ) -> AudioFrame:
        if (
            frame.sample_rate == target_rate
            and frame.channels == target_channels
            and frame.sample_width == target_width
        ):
            return frame
        # ... your soxr/libsamplerate/scipy implementation ...

    def reset(self) -> None:
        """Reset any internal state between sessions."""

    def close(self) -> None:
        """Release native resources."""
```

The `resample()` method receives target format parameters (not fixed at construction) because the pipeline calls it with different targets for inbound vs outbound.

## Auto-default behavior

| `resampler` | `contract` | Behavior |
|---|---|---|
| not set | not set | No resampling. Frames pass through unchanged. |
| not set | set | Auto-creates `LinearResamplerProvider`. |
| set | set | Uses the explicit provider. |
| set | not set | Resampler is stored but inbound resampling is skipped (no target format). Outbound resampling is also skipped. |

## Pipeline position

- **Inbound**: first stage (before recorder tap, AEC, denoiser, VAD)
- **Outbound**: last stage (after post-processors, recorder tap, AEC reference feed)

## Lifecycle

The pipeline calls `reset()` on session start and `close()` on pipeline shutdown, following the same pattern as all other providers.
