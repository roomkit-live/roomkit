"""RoomKit -- Voice assistant over FastRTC (WebSocket transport).

A browser-to-AI voice assistant using FastRTCVoiceBackend for the
traditional STT/TTS pipeline. Audio flows from the browser via
FastRTC's WebSocket transport through the full voice pipeline:

  Browser mic → FastRTC WebSocket → [Resampler] → [AEC] → [Denoiser]
    → VAD → Deepgram STT → Claude AI → ElevenLabs TTS → mu-law → Browser

The browser connects to the /voice WebSocket endpoint (or /voice/webrtc/offer
for WebRTC). Use the companion ``fastrtc_client.html`` page as the frontend,
or FastRTC's built-in Gradio UI at /voice/ui.

Requirements:
    pip install roomkit[fastrtc,anthropic] fastapi uvicorn

Run with:
    ANTHROPIC_API_KEY=... \\
    DEEPGRAM_API_KEY=... \\
    ELEVENLABS_API_KEY=... \\
    uv run uvicorn examples.voice_fastrtc:app

Environment variables:
    ANTHROPIC_API_KEY   (required) Anthropic API key
    DEEPGRAM_API_KEY    (required) Deepgram API key
    STT_LANGUAGE        STT language code (default: en)
    ELEVENLABS_API_KEY  (required) ElevenLabs API key
    ELEVENLABS_VOICE_ID Voice ID (default: Rachel)
    SYSTEM_PROMPT       Custom system prompt for Claude
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from roomkit import (
    AIChannel,
    AnthropicAIProvider,
    AnthropicConfig,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend, mount_fastrtc_voice
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_fastrtc")

kit = RoomKit()

# --- Audio settings ---
INPUT_SAMPLE_RATE = 48000  # Browser mic (FastRTC default)
OUTPUT_SAMPLE_RATE = 24000  # ElevenLabs native rate

# --- Backend: FastRTC WebSocket transport ---
backend = FastRTCVoiceBackend(
    input_sample_rate=INPUT_SAMPLE_RATE,
    output_sample_rate=OUTPUT_SAMPLE_RATE,
)

# --- VAD ---
vad = EnergyVADProvider(
    energy_threshold=300.0,
    silence_threshold_ms=600,
    min_speech_duration_ms=200,
)

# --- Pipeline ---
pipeline = AudioPipelineConfig(vad=vad)

# --- Deepgram STT ---
stt_language = os.environ.get("STT_LANGUAGE", "en")
stt = DeepgramSTTProvider(
    config=DeepgramConfig(
        api_key=os.environ.get("DEEPGRAM_API_KEY", ""),
        model="nova-3",
        language=stt_language,
        punctuate=True,
        smart_format=True,
        endpointing=300,
    )
)

# --- ElevenLabs TTS ---
tts = ElevenLabsTTSProvider(
    config=ElevenLabsConfig(
        api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
        voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        model_id="eleven_multilingual_v2",
        output_format=f"pcm_{OUTPUT_SAMPLE_RATE}",
        optimize_streaming_latency=3,
    )
)

# --- Claude AI ---
ai_provider = AnthropicAIProvider(
    AnthropicConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.7,
    )
)

system_prompt = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a friendly voice assistant. Keep your responses "
    "short and conversational — one or two sentences at most.",
)

# --- Channels ---
voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend, pipeline=pipeline)
kit.register_channel(voice)

ai = AIChannel("ai", provider=ai_provider, system_prompt=system_prompt)
kit.register_channel(ai)


# --- Session factory: auto-create room + session on WebSocket connect ---
async def session_factory(websocket_id: str):
    """Create a room and voice session when a browser connects."""
    room = await kit.create_room()
    binding = await kit.attach_channel(room.id, "voice")
    await kit.attach_channel(room.id, "ai", category=ChannelCategory.INTELLIGENCE)
    session = await backend.connect(room.id, "browser-user", "voice")
    session.metadata["websocket_id"] = websocket_id
    voice.bind_session(session, room.id, binding)
    logger.info("Session created: session=%s, room=%s", session.id, room.id)
    return session


# --- Hooks ---
@kit.hook(HookTrigger.ON_TRANSCRIPTION)
async def on_transcription(event, ctx):
    logger.info("User: %s", event.text)
    return HookResult.allow()


@kit.hook(HookTrigger.BEFORE_TTS)
async def before_tts(text, ctx):
    logger.info("Assistant: %s", text)
    return HookResult.allow()


@kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
async def on_speech_start(session, ctx):
    logger.info("Speech started")


@kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
async def on_speech_end(session, ctx):
    logger.info("Speech ended")


# --- Optional: auth callback ---
async def authenticate(websocket) -> dict[str, object] | None:
    """Accept all connections. In production, validate tokens here."""
    return {"authenticated": True}


# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    mount_fastrtc_voice(
        app,
        backend,
        path="/voice",
        session_factory=session_factory,
        auth=authenticate,
    )
    logger.info("FastRTC voice backend ready at /voice")
    logger.info("Open http://localhost:8000 for the browser client")
    yield
    await kit.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    """Serve a minimal browser client for testing."""
    return HTMLResponse(BROWSER_CLIENT_HTML)


@app.get("/health")
async def health():
    return {"status": "ok", "transport": "fastrtc-websocket"}


# ---------------------------------------------------------------------------
# Inline browser client (minimal, for quick testing)
# ---------------------------------------------------------------------------
BROWSER_CLIENT_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RoomKit Voice (FastRTC)</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px;
         margin: 2rem auto; padding: 0 1rem; }
  button { padding: 0.5rem 1.5rem; font-size: 1rem; cursor: pointer;
           border-radius: 6px; border: 1px solid #ccc; }
  button:disabled { opacity: 0.5; cursor: default; }
  #start { background: #4CAF50; color: white; border-color: #4CAF50; }
  #stop { background: #f44336; color: white; border-color: #f44336; }
  #log { margin-top: 1rem; padding: 1rem; background: #f5f5f5; border-radius: 6px;
         max-height: 400px; overflow-y: auto; font-size: 0.9rem; white-space: pre-wrap; }
  .user { color: #1976D2; }
  .assistant { color: #388E3C; }
  .system { color: #777; font-style: italic; }
</style>
</head>
<body>
<h2>RoomKit Voice Assistant</h2>
<p>Uses FastRTC WebSocket transport with Deepgram STT + Claude + ElevenLabs TTS.</p>
<div>
  <button id="start" onclick="startVoice()">Start</button>
  <button id="stop" onclick="stopVoice()" disabled>Stop</button>
  <span id="status" style="margin-left:1rem" class="system">Disconnected</span>
</div>
<div id="log"></div>

<script>
let ws, audioCtx, processor, micStream, nextPlayTime = 0;

function log(text, cls) {
  const el = document.getElementById('log');
  const span = document.createElement('span');
  span.className = cls || '';
  span.textContent = text + '\\n';
  el.appendChild(span);
  el.scrollTop = el.scrollHeight;
}

function setStatus(text) { document.getElementById('status').textContent = text; }

// mu-law encoding table (ITU-T G.711)
const MULAW_BIAS = 0x84, MULAW_CLIP = 32635;
function encodeMulaw(sample) {
  let sign = (sample >= 0) ? 0x80 : 0x00;
  let mag = Math.min(Math.abs(sample), MULAW_CLIP) + MULAW_BIAS;
  let exp = 7, mask = 0x4000;
  while (exp > 0 && !(mag & mask)) { exp--; mask >>= 1; }
  let mantissa = (mag >> (exp + 3)) & 0x0F;
  return ~((exp << 4) | mantissa) & 0x7F | sign;
}

function pcm16ToMulaw(pcmData) {
  const out = new Uint8Array(pcmData.length);
  for (let i = 0; i < pcmData.length; i++) {
    out[i] = encodeMulaw(pcmData[i]);
  }
  return out;
}

// mu-law decoding table
const MULAW_DECODE = new Int16Array(256);
(function buildDecodeTable() {
  for (let i = 0; i < 256; i++) {
    let v = ~i;
    let sign = v & 0x80;
    let exp = (v >> 4) & 0x07;
    let mantissa = v & 0x0F;
    let sample = ((mantissa << 3) + MULAW_BIAS) << exp;
    sample -= MULAW_BIAS;
    MULAW_DECODE[i] = sign ? sample : -sample;
  }
})();

function decodeMulaw(mulawBytes) {
  const pcm = new Int16Array(mulawBytes.length);
  for (let i = 0; i < mulawBytes.length; i++) {
    pcm[i] = MULAW_DECODE[mulawBytes[i]];
  }
  return pcm;
}

async function startVoice() {
  document.getElementById('start').disabled = true;
  document.getElementById('stop').disabled = false;
  setStatus('Connecting...');
  log('Connecting to FastRTC WebSocket...', 'system');

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsId = Math.random().toString(36).substring(2, 10);
  ws = new WebSocket(`${proto}//${location.host}/voice/websocket/offer`);

  audioCtx = new AudioContext({ sampleRate: 48000 });
  nextPlayTime = 0;

  ws.onopen = async () => {
    ws.send(JSON.stringify({ event: 'start', websocket_id: wsId }));
    setStatus('Connected');
    log('Connected (id=' + wsId + ')', 'system');

    // Capture mic audio
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = audioCtx.createMediaStreamSource(micStream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const float32 = e.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        pcm16[i] = Math.max(-32768, Math.min(32767, Math.round(float32[i] * 32767)));
      }
      const mulaw = pcm16ToMulaw(pcm16);
      const b64 = btoa(String.fromCharCode.apply(null, mulaw));
      ws.send(JSON.stringify({ event: 'media', media: { payload: b64 } }));
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.event === 'media' && data.media && data.media.payload) {
      // Decode mu-law audio and play sequentially (queued)
      const raw = atob(data.media.payload);
      const mulaw = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) mulaw[i] = raw.charCodeAt(i);
      const pcm = decodeMulaw(mulaw);
      const float32 = new Float32Array(pcm.length);
      for (let i = 0; i < pcm.length; i++) float32[i] = pcm[i] / 32768;

      const sampleRate = 24000;
      const buffer = audioCtx.createBuffer(1, float32.length, sampleRate);
      buffer.getChannelData(0).set(float32);
      const src = audioCtx.createBufferSource();
      src.buffer = buffer;
      src.connect(audioCtx.destination);

      // Schedule chunk after previous one finishes (no overlap)
      const now = audioCtx.currentTime;
      if (nextPlayTime < now) nextPlayTime = now;
      src.start(nextPlayTime);
      nextPlayTime += buffer.duration;
    }

    if (data.type === 'transcription') {
      const role = data.data.role || 'user';
      if (role === 'assistant_interim') {
        // Update interim text in-place
        let el = document.getElementById('interim');
        if (!el) {
          el = document.createElement('span');
          el.id = 'interim';
          el.className = 'assistant';
          document.getElementById('log').appendChild(el);
        }
        el.textContent = 'Assistant: ' + data.data.text + '\\n';
        document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;
      } else {
        // Final transcription — remove interim and log permanently
        const interim = document.getElementById('interim');
        if (interim) interim.remove();
        const label = role === 'assistant' ? 'Assistant' : 'You';
        log(label + ': ' + data.data.text, role === 'assistant' ? 'assistant' : 'user');
      }
    }
  };

  ws.onclose = () => {
    setStatus('Disconnected');
    log('Disconnected', 'system');
    cleanup();
  };

  ws.onerror = (err) => {
    log('WebSocket error: ' + err, 'system');
    cleanup();
  };
}

function stopVoice() {
  if (ws) ws.close();
  cleanup();
}

function cleanup() {
  document.getElementById('start').disabled = false;
  document.getElementById('stop').disabled = true;
  if (processor) { processor.disconnect(); processor = null; }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (audioCtx && audioCtx.state !== 'closed') audioCtx.close();
}
</script>
</body>
</html>
"""
