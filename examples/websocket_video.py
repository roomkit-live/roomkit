"""RoomKit — WebSocket video streaming with vision analysis.

Receive video frames from a browser over WebSocket, run vision
analysis, and display results. Useful for browser-based video
applications that send frames separately from audio.

Video flow:
    Browser canvas → WebSocket binary frames → VideoBackend
      → on_video_received → Vision AI → log results

Wire protocol (binary messages):
    [1 byte flags][4 bytes sequence_be][payload]
    flags: bit 0 = keyframe, bits 1-3 = codec (0=h264, 1=vp8, 2=mjpeg, 3=raw)

Requirements:
    pip install roomkit fastapi uvicorn

Run with:
    OPENAI_API_KEY=... uv run uvicorn examples.websocket_video:app

Environment variables:
    OPENAI_API_KEY  (optional) OpenAI API key for vision analysis
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from roomkit import RoomKit
from roomkit.video.backends.websocket import WebSocketVideoBackend, mount_websocket_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websocket_video")

kit = RoomKit()
backend = WebSocketVideoBackend(default_codec="raw_rgb24")

# Track frames
frame_count = 0


def on_video(session, frame):
    global frame_count  # noqa: PLW0603
    frame_count += 1
    if frame_count % 30 == 1:
        logger.info(
            "Frame #%d: codec=%s %dx%d seq=%d %s",
            frame_count,
            frame.codec,
            frame.width,
            frame.height,
            frame.sequence,
            "KEY" if frame.keyframe else "",
        )


backend.on_video_received(on_video)


# --- Session factory: auto-create on WebSocket connect ---
async def session_factory(connection_id: str):
    room = await kit.create_room()
    session = await backend.connect(room.id, connection_id, "video")
    logger.info("Video session: session=%s, room=%s", session.id[:8], room.id)
    return session


backend.set_session_factory(session_factory)


# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    mount_websocket_video(app, backend, path="/video/ws")
    logger.info("WebSocket video endpoint ready at /video/ws")
    logger.info("Open http://localhost:8000 for the browser client")
    yield
    await kit.close()
    await backend.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return HTMLResponse(BROWSER_CLIENT_HTML)


@app.get("/health")
async def health():
    return {"status": "ok", "frames_received": frame_count}


BROWSER_CLIENT_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RoomKit WebSocket Video</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 800px;
         margin: 2rem auto; padding: 0 1rem; }
  video, canvas { width: 320px; height: 240px; background: #000;
                  border-radius: 8px; }
  canvas { display: none; }
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
<h2>RoomKit WebSocket Video</h2>
<p>Streams video frames over WebSocket using the binary frame protocol.</p>
<div>
  <video id="localVideo" autoplay muted playsinline></video>
  <canvas id="canvas" width="640" height="480"></canvas>
</div>
<div>
  <button id="start" onclick="startVideo()">Start</button>
  <button id="stop" onclick="stopVideo()" disabled>Stop</button>
  <span id="status" style="margin-left:1rem;color:#777">Disconnected</span>
  <span id="fps" style="margin-left:1rem;color:#999"></span>
</div>
<div id="log"></div>

<script>
let ws, localStream, captureInterval;
let seq = 0, sentCount = 0, lastCountTime = 0;

function log(text) {
  const el = document.getElementById('log');
  el.textContent += text + '\\n';
  el.scrollTop = el.scrollHeight;
}

async function startVideo() {
  document.getElementById('start').disabled = true;
  document.getElementById('stop').disabled = false;
  document.getElementById('status').textContent = 'Connecting...';

  localStream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 }
  });
  document.getElementById('localVideo').srcObject = localStream;

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/video/ws`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    document.getElementById('status').textContent = 'Connected';
    log('WebSocket connected');

    // Send config
    ws.send(JSON.stringify({
      type: 'config',
      codec: 'raw_rgb24',
      width: 640,
      height: 480
    }));

    // Capture and send frames at ~10 fps
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const video = document.getElementById('localVideo');

    lastCountTime = performance.now();
    captureInterval = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) return;
      ctx.drawImage(video, 0, 0, 640, 480);
      const imageData = ctx.getImageData(0, 0, 640, 480);

      // Convert RGBA to RGB
      const rgba = imageData.data;
      const rgb = new Uint8Array(640 * 480 * 3);
      for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
        rgb[j] = rgba[i];
        rgb[j+1] = rgba[i+1];
        rgb[j+2] = rgba[i+2];
      }

      // Build binary frame: [flags:1][seq:4][payload]
      // flags: keyframe=1 (bit 0), codec=raw_rgb24 (bits 1-3 = 010) => 0x05
      const isKey = seq === 0;
      const flags = (isKey ? 0x01 : 0x00) | (2 << 1);  // codec_id=2 for raw_rgb24
      const header = new ArrayBuffer(5);
      const view = new DataView(header);
      view.setUint8(0, flags);
      view.setUint32(1, seq, false);  // big-endian

      const frame = new Uint8Array(5 + rgb.length);
      frame.set(new Uint8Array(header), 0);
      frame.set(rgb, 5);
      ws.send(frame.buffer);

      seq++;
      sentCount++;

      // Update FPS display
      const now = performance.now();
      if (now - lastCountTime >= 1000) {
        document.getElementById('fps').textContent =
          sentCount + ' fps';
        sentCount = 0;
        lastCountTime = now;
      }
    }, 100);  // ~10 fps
  };

  ws.onclose = () => {
    document.getElementById('status').textContent = 'Disconnected';
    log('Disconnected');
    cleanup();
  };

  ws.onerror = () => {
    log('WebSocket error');
    cleanup();
  };
}

function stopVideo() {
  if (ws) ws.close();
  cleanup();
}

function cleanup() {
  document.getElementById('start').disabled = false;
  document.getElementById('stop').disabled = true;
  if (captureInterval) { clearInterval(captureInterval); captureInterval = null; }
  if (localStream) {
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
  }
  document.getElementById('localVideo').srcObject = null;
  seq = 0;
}
</script>
</body>
</html>
"""
