"""RoomKit — WebRTC audio+video with vision analysis.

Browser-to-AI assistant with combined audio and video over WebRTC.
Audio goes through the full voice pipeline (STT/TTS), while video
frames are analysed by a vision provider (OpenAI Vision).

Audio flow:
    Browser mic → FastRTC WebRTC → Pipeline → STT → Claude AI → TTS → Browser

Video flow:
    Browser camera → FastRTC WebRTC → on_video_received → Vision AI → context

Requirements:
    pip install roomkit[fastrtc,anthropic] fastapi uvicorn

Run with:
    ANTHROPIC_API_KEY=... \\
    DEEPGRAM_API_KEY=... \\
    ELEVENLABS_API_KEY=... \\
    OPENAI_API_KEY=... \\
    uv run uvicorn examples.webrtc_video:app

Environment variables:
    ANTHROPIC_API_KEY   (required) Anthropic API key
    DEEPGRAM_API_KEY    (required) Deepgram API key
    ELEVENLABS_API_KEY  (required) ElevenLabs API key
    OPENAI_API_KEY      (required) OpenAI API key for vision
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
    AudioVideoChannel,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
)
from roomkit.video.backends.fastrtc import FastRTCVideoBackend, mount_fastrtc_av
from roomkit.video.vision.openai import OpenAIVisionConfig, OpenAIVisionProvider
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc_video")
# Suppress noisy transport logs
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

kit = RoomKit()

# --- Audio settings ---
INPUT_SAMPLE_RATE = 48000
OUTPUT_SAMPLE_RATE = 24000

# --- Backend: FastRTC A/V transport ---
backend = FastRTCVideoBackend(
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

# --- STT ---
stt = DeepgramSTTProvider(
    config=DeepgramConfig(
        api_key=os.environ.get("DEEPGRAM_API_KEY", ""),
        model="nova-3",
        language="en",
        punctuate=True,
        smart_format=True,
        endpointing=300,
    )
)

# --- TTS ---
tts = ElevenLabsTTSProvider(
    config=ElevenLabsConfig(
        api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
        voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        model_id="eleven_multilingual_v2",
        output_format=f"pcm_{OUTPUT_SAMPLE_RATE}",
        optimize_streaming_latency=3,
    )
)

# --- Vision ---
vision = OpenAIVisionProvider(
    config=OpenAIVisionConfig(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        max_tokens=150,
    )
)

# --- AI ---
ai_provider = AnthropicAIProvider(
    AnthropicConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.7,
    )
)

system_prompt = (
    "You are a friendly voice assistant with vision. You can see what "
    "the user's camera shows. Keep responses short and conversational."
)

# --- A/V Channel ---
av = AudioVideoChannel(
    "av",
    stt=stt,
    tts=tts,
    backend=backend,
    pipeline=pipeline,
    vision=vision,
    vision_interval_ms=3000,
)
kit.register_channel(av)

ai = AIChannel("ai", provider=ai_provider, system_prompt=system_prompt)
kit.register_channel(ai)


# --- Session factory ---
async def session_factory(websocket_id: str):
    room = await kit.create_room()
    binding = await kit.attach_channel(room.id, "av")
    await kit.attach_channel(room.id, "ai", category=ChannelCategory.INTELLIGENCE)
    session = await backend.connect(room.id, "browser-user", "av")
    session.metadata["websocket_id"] = websocket_id
    av.bind_session(session, room.id, binding)
    logger.info("A/V session created: session=%s, room=%s", session.id, room.id)
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


# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    mount_fastrtc_av(
        app,
        backend,
        path="/av",
        session_factory=session_factory,
    )
    logger.info("FastRTC A/V backend ready at /av")
    logger.info("Open http://localhost:8000 for the browser client")
    yield
    await kit.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return HTMLResponse(BROWSER_CLIENT_HTML)


@app.get("/health")
async def health():
    return {"status": "ok", "transport": "fastrtc-av"}


BROWSER_CLIENT_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RoomKit WebRTC A/V</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 800px;
         margin: 2rem auto; padding: 0 1rem; }
  video { width: 320px; height: 240px; background: #000;
          border-radius: 8px; }
  button { padding: 0.5rem 1.5rem; font-size: 1rem; cursor: pointer;
           border-radius: 6px; border: 1px solid #ccc; margin: 0.25rem; }
  #start { background: #4CAF50; color: white; }
  #stop { background: #f44336; color: white; }
  #log { margin-top: 1rem; padding: 1rem; background: #f5f5f5;
         border-radius: 6px; max-height: 300px; overflow-y: auto;
         font-size: 0.9rem; white-space: pre-wrap; }
</style>
</head>
<body>
<h2>RoomKit WebRTC A/V</h2>
<p>Combined audio+video over WebRTC with vision analysis.</p>
<div>
  <video id="localVideo" autoplay muted playsinline></video>
</div>
<div>
  <button id="start" onclick="startAV()">Start</button>
  <button id="stop" onclick="stopAV()" disabled>Stop</button>
  <span id="status" style="margin-left:1rem;color:#777">Disconnected</span>
</div>
<div id="log"></div>

<script>
let pc, localStream;

function log(text) {
  const el = document.getElementById('log');
  el.textContent += text + '\\n';
  el.scrollTop = el.scrollHeight;
}

async function startAV() {
  document.getElementById('start').disabled = true;
  document.getElementById('stop').disabled = false;
  document.getElementById('status').textContent = 'Connecting...';

  localStream = await navigator.mediaDevices.getUserMedia({
    audio: true,
    video: { width: 640, height: 480 }
  });
  document.getElementById('localVideo').srcObject = localStream;

  pc = new RTCPeerConnection({
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
  });

  localStream.getTracks().forEach(t => pc.addTrack(t, localStream));

  // FastRTC requires a data channel to unblock handler processing
  const dc = pc.createDataChannel('chat');
  dc.onmessage = e => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'log') log('Server: ' + msg.data);
    } catch(err) {}
  };

  // Play incoming audio/video from server
  pc.ontrack = e => {
    log('Received track: ' + e.track.kind);
    if (e.track.kind === 'audio') {
      const audio = new Audio();
      audio.srcObject = e.streams[0];
      audio.play();
    }
  };

  pc.onicecandidate = e => {
    if (e.candidate) log('ICE candidate: ' + e.candidate.type);
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // Wait for ICE gathering (2s timeout — host candidates arrive instantly)
  await new Promise(resolve => {
    if (pc.iceGatheringState === 'complete') { resolve(); return; }
    const timeout = setTimeout(resolve, 800);
    pc.onicegatheringstatechange = () => {
      if (pc.iceGatheringState === 'complete') {
        clearTimeout(timeout);
        resolve();
      }
    };
  });

  const webrtcId = Math.random().toString(36).substring(2, 10);
  const resp = await fetch('/av/webrtc/offer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
      webrtc_id: webrtcId,
    })
  });
  const answer = await resp.json();
  await pc.setRemoteDescription(new RTCSessionDescription(answer));

  document.getElementById('status').textContent = 'Connected';
  log('WebRTC connected');
}

function stopAV() {
  if (pc) { pc.close(); pc = null; }
  if (localStream) {
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
  }
  document.getElementById('localVideo').srcObject = null;
  document.getElementById('start').disabled = false;
  document.getElementById('stop').disabled = true;
  document.getElementById('status').textContent = 'Disconnected';
  log('Disconnected');
}
</script>
</body>
</html>
"""
