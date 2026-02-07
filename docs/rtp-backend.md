# RTP Voice Backend

A voice backend that sends and receives audio over RTP, for integration with PBX/SIP gateways or any system that speaks RTP. Uses the `aiortp` library for packet handling, codec encoding/decoding, and RFC 4733 DTMF.

## Quick start

```python
from roomkit.voice.backends.rtp import RTPVoiceBackend
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider
from roomkit import VoiceChannel

backend = RTPVoiceBackend(
    local_addr=("0.0.0.0", 10000),
    remote_addr=("192.168.1.100", 20000),
)

pipeline = AudioPipelineConfig(vad=vad)
voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend, pipeline=pipeline)

session = await backend.connect("room-1", "user-1", "voice")
# ... pipeline processes inbound RTP audio, AI responds, TTS sends outbound RTP ...
await backend.disconnect(session)
```

Install with:

```
pip install roomkit[rtp]
```

## Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `local_addr` | `(str, int)` | `("0.0.0.0", 0)` | Host and port to bind RTP. Use port `0` for OS-assigned. |
| `remote_addr` | `(str, int) \| None` | `None` | Host and port to send RTP to. Can be overridden per-session. |
| `payload_type` | `int` | `0` | RTP payload type number (see codec table below). |
| `clock_rate` | `int` | `8000` | Clock rate in Hz. Must match the payload type. |
| `dtmf_payload_type` | `int` | `101` | RTP payload type for RFC 4733 DTMF events. |
| `rtcp_interval` | `float` | `5.0` | Seconds between RTCP sender reports. |

## Codec selection

Set `payload_type` and `clock_rate` to match the codec negotiated with the remote endpoint:

| Codec | `payload_type` | `clock_rate` | Notes |
|---|---|---|---|
| PCMU (G.711 mu-law) | `0` | `8000` | Default. Standard telephony codec. |
| PCMA (G.711 A-law) | `8` | `8000` | Common in European telephony. |
| L16 (uncompressed) | `11` | `44100` | Linear 16-bit PCM. High bandwidth. |

```python
# G.711 A-law example
backend = RTPVoiceBackend(
    local_addr=("0.0.0.0", 10000),
    remote_addr=("pbx.local", 20000),
    payload_type=8,
    clock_rate=8000,
)
```

## Per-session remote address

When connecting to multiple endpoints (e.g. a multi-tenant PBX), you can omit `remote_addr` from the constructor and supply it per-session via `metadata["remote_addr"]`:

```python
backend = RTPVoiceBackend(local_addr=("0.0.0.0", 10000))

# Each session gets its own remote endpoint
session1 = await backend.connect(
    "room-1", "caller-1", "voice",
    metadata={"remote_addr": ("10.0.0.1", 20000)},
)
session2 = await backend.connect(
    "room-2", "caller-2", "voice",
    metadata={"remote_addr": ("10.0.0.2", 20000)},
)
```

If neither the constructor nor metadata provides a remote address, `connect()` raises `ValueError`.

## DTMF

The backend receives DTMF digits out-of-band via RFC 4733 (telephone-event payload type). Inbound DTMF events are delivered as `DTMFEvent` objects containing `digit` (str) and `duration_ms` (float).

DTMF integrates with the hook system via `HookTrigger.ON_DTMF`:

```python
from roomkit import HookTrigger, HookExecution

@kit.hook(HookTrigger.ON_DTMF, execution=HookExecution.ASYNC)
async def on_dtmf(event, ctx):
    print(f"DTMF digit: {event.digit}, duration: {event.duration_ms}ms")
```

You can also add an in-band `MockDTMFDetector` to the pipeline for testing:

```python
from roomkit.voice.pipeline.dtmf import MockDTMFDetector

pipeline = AudioPipelineConfig(
    vad=vad,
    dtmf=MockDTMFDetector(),
)
```

## Capabilities

The RTP backend declares two capabilities:

| Capability | Description |
|---|---|
| `DTMF_SIGNALING` | DTMF digits are received out-of-band via RFC 4733. The pipeline skips in-band DTMF detection when this is set. |
| `INTERRUPTION` | Outbound audio playback can be cancelled mid-stream (for barge-in). |

## Audio flow

### Inbound

```
Remote endpoint → RTP packets → aiortp decode → PCM bytes
  → AudioFrame(sample_rate=clock_rate, channels=1, sample_width=2)
  → on_audio_received callback → AudioPipeline inbound chain
```

aiortp handles codec decoding (G.711/L16) and delivers raw PCM-16 LE. The backend wraps each decoded buffer in an `AudioFrame` and passes it to the pipeline.

### Outbound

```
TTS → AudioChunk stream or bytes → PCM-16 LE
  → 20ms RTP frames (clock_rate / 50 samples each)
  → aiortp encode → RTP packets → remote endpoint
```

The backend accepts either a complete `bytes` buffer or an `AsyncIterator[AudioChunk]`. PCM data is split into 20ms frames and sent with incrementing RTP timestamps.

## SIP integration

The RTP backend handles only the media (RTP) plane. For SIP signaling (INVITE, BYE, codec negotiation via SDP), pair it with a SIP library such as `aiosip` or `pjsip`. The SIP layer negotiates the codec and provides the remote RTP address, which you pass to `RTPVoiceBackend` via `remote_addr` or `metadata["remote_addr"]`.

## Example

See [`examples/voice_rtp.py`](../examples/voice_rtp.py) for a complete runnable example with mock providers, DTMF handling, and session lifecycle.
