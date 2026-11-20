# Installation

## Basic Install

```bash
pip install roomkit
# or with uv
uv add roomkit
```

The core library has two dependencies: `pydantic>=2.9`, and `pyyaml>=6`, which reads skill frontmatter.

## Optional Extras

RoomKit uses optional extras for provider-specific dependencies. Install only what you need:

### AI Providers

```bash
pip install roomkit[anthropic]    # Anthropic (Claude)
pip install roomkit[openai]       # OpenAI (GPT), on the openai SDK 3.x
pip install roomkit[gemini]       # Google Gemini
pip install roomkit[mistral]      # Mistral AI
pip install roomkit[vllm]         # vLLM local inference (uses openai SDK)
pip install roomkit[azure]        # Azure OpenAI
```

### Voice — Speech-to-Text

```bash
pip install roomkit[deepgram]     # Deepgram STT (cloud)
pip install roomkit[sherpa-onnx]  # SherpaOnnx STT (local, offline)
pip install roomkit[gradium]      # Gradium STT
pip install roomkit[qwen-asr]     # Qwen3 ASR
pip install roomkit[mistral-voice] # Voxtral 3 STT (Mistral, file + realtime)
```

### Voice — Text-to-Speech

```bash
pip install roomkit[elevenlabs]   # ElevenLabs TTS (cloud)
pip install roomkit[sherpa-onnx]  # SherpaOnnx TTS (local, offline)
pip install roomkit[gradium]      # Gradium TTS
pip install roomkit[qwen-tts]     # Qwen3 TTS
pip install roomkit[neutts]       # NeuTTS
pip install roomkit[fish-audio]   # Fish Audio TTS (Drama 3, S2)
pip install roomkit[phonon]       # Phonon TTS (Gradium, local CPU)
pip install roomkit[mistral-voice] # Voxtral 3 TTS (Mistral)
```

### Voice — Backends

```bash
pip install roomkit[local-audio]  # Local mic/speaker (sounddevice + numpy)
pip install roomkit[fastrtc]      # FastRTC WebRTC backend
pip install roomkit[rtp]          # RTP backend
pip install roomkit[sip]          # SIP backend
pip install roomkit[webtransport] # WebTransport backend
```

### Voice — Pipeline

```bash
pip install roomkit[webrtc-aec]   # WebRTC echo cancellation
pip install roomkit[aicoustics]   # ai|coustics denoiser
pip install roomkit[rnnoise]      # librnnoise for RNNoise, when the system has none
pip install roomkit[smart-turn]   # ML-based turn detection
```

### Realtime Voice (Speech-to-Speech)

```bash
pip install roomkit[realtime-openai]   # OpenAI Realtime API
pip install roomkit[realtime-gemini]   # Google Gemini Live API
```

### Messaging Providers

```bash
pip install roomkit[twilio]            # Twilio SMS/RCS
pip install roomkit[telegram]          # Telegram Bot API
pip install roomkit[teams]             # Microsoft Teams (Bot Framework)
pip install roomkit[whatsapp-personal] # WhatsApp Personal (neonize)
pip install roomkit[websocket]         # WebSocket source
pip install roomkit[sse]               # Server-Sent Events source
```

### Storage & Infrastructure

```bash
pip install roomkit[postgres]          # PostgreSQL persistence (asyncpg)
pip install roomkit[mcp]               # Model Context Protocol tools
pip install roomkit[opentelemetry]     # OpenTelemetry tracing
```

### Meta Extras

```bash
pip install roomkit[providers]  # All AI + transport providers
pip install roomkit[sources]    # All event-driven sources
pip install roomkit[all]        # Everything installable together
pip install roomkit[dev]        # Development (test + lint + type check)
```

### Incompatible Extras

`qwen-tts`, `qwen-asr` and `neutts` each pin an exact `transformers` version
(4.57.3, 4.57.6 and 5.1.0), so they cannot be installed together. They are also
incompatible with `smart-turn`, whose floor is `transformers>=5.5` — the first
version without GHSA-fgcw-684q-jj6r — and therefore with `all`, which includes
`smart-turn`.

```bash
pip install roomkit[all]              # includes smart-turn
pip install roomkit[qwen-tts]         # in its own environment
```

`pyproject.toml` declares these conflicts, so an unsatisfiable combination fails
at resolution naming the two extras rather than backtracking onto an older
`transformers`.

## Environment Variables

Provider-specific API keys are passed via configuration objects, not environment variables. Example:

```python
from roomkit.providers.anthropic import AnthropicConfig

config = AnthropicConfig(api_key="sk-ant-...", model="claude-opus-5")
```

Voice providers are imported from their package; a provider's module, and the
extra it needs, load on first use of its name:

```python
from roomkit.voice.stt import DeepgramConfig, DeepgramSTTProvider

stt = DeepgramSTTProvider(DeepgramConfig(api_key="..."))
```

## Development Setup

```bash
git clone https://github.com/roomkit-live/roomkit
cd roomkit
uv sync --extra dev    # Install all dev dependencies
make all               # Run lint + typecheck + security + tests
```
