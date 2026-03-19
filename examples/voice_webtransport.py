"""WebTransport voice echo — low-latency audio via QUIC datagrams.

Demonstrates the WebTransport voice backend using aioquic.  A browser
connects via the WebTransport API and sends audio datagrams.  The server
echoes them back — validating the full round-trip pipeline.

Architecture::

    Browser (WebTransport API)
      ↕  QUIC datagrams (unreliable, UDP-like)
    aioquic server (port 4433)
      ↕
    WebTransportBackend
      → on_audio_received callback
      → send_audio_sync (echo back)

Prerequisites:

    1.  Generate a self-signed TLS certificate (WebTransport requires TLS)::

            openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
                -days 365 -nodes -keyout key.pem -out cert.pem \
                -subj "/CN=localhost"

    2.  Install aioquic::

            pip install 'roomkit[webtransport]'

Run::

    uv run python examples/voice_webtransport.py

Then open a browser (Chrome/Edge) and connect via::

    const transport = new WebTransport("https://localhost:4433/audio");

Note: You need to allow the self-signed certificate.  In Chrome, navigate
to ``chrome://flags/#allow-insecure-localhost`` or launch with::

    google-chrome --origin-to-force-quic-on=localhost:4433

Requirements:
    pip install 'roomkit[webtransport]'

Environment variables:
    WT_HOST          -- Bind address (default: 0.0.0.0)
    WT_PORT          -- QUIC port (default: 4433)
    WT_CERT          -- TLS certificate path (default: cert.pem)
    WT_KEY           -- TLS private key path (default: key.pem)
    WT_SAMPLE_RATE   -- Audio sample rate (default: 16000)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("voice_webtransport")

from roomkit import RoomKit, VoiceChannel
from roomkit.voice.backends.webtransport import WebTransportBackend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = os.environ.get("WT_HOST", "0.0.0.0")
PORT = int(os.environ.get("WT_PORT", "4433"))
CERT = os.environ.get("WT_CERT", "cert.pem")
KEY = os.environ.get("WT_KEY", "key.pem")
SAMPLE_RATE = int(os.environ.get("WT_SAMPLE_RATE", "16000"))

ROOM_ID = "echo-room"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

kit = RoomKit()

backend = WebTransportBackend(
    host=HOST,
    port=PORT,
    certificate=CERT,
    private_key=KEY,
    input_sample_rate=SAMPLE_RATE,
    output_sample_rate=SAMPLE_RATE,
)

voice = VoiceChannel("voice", backend=backend)
kit.register_channel(voice)

# ---------------------------------------------------------------------------
# Echo: send received audio back to the sender
# ---------------------------------------------------------------------------

_session_counter = 0


async def session_factory(connection_id: str):
    global _session_counter  # noqa: PLW0603
    _session_counter += 1
    participant_id = f"wt-user-{_session_counter}"

    session = await kit.join(ROOM_ID, "voice", participant_id=participant_id)
    logger.info("New WebTransport session: %s (%s)", participant_id, connection_id)
    return session


backend.set_session_factory(session_factory)


def on_audio(session, frame):
    """Echo received audio back to the sender."""
    from roomkit.voice.base import AudioChunk

    backend.send_audio_sync(
        session,
        AudioChunk(data=frame.data, sample_rate=frame.sample_rate),
    )


backend.on_audio_received(on_audio)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    if not os.path.exists(CERT) or not os.path.exists(KEY):
        logger.error("TLS certificate not found: %s / %s", CERT, KEY)
        print(
            "\nGenerate a self-signed certificate first:\n\n"
            "  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \\\n"
            "      -days 365 -nodes -keyout key.pem -out cert.pem \\\n"
            '      -subj "/CN=localhost"\n'
        )
        sys.exit(1)

    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "voice")

    await backend.start()

    logger.info("=== WebTransport Voice Echo Server ===")
    logger.info("QUIC server: https://%s:%d/audio", HOST, PORT)
    logger.info("Sample rate: %d Hz", SAMPLE_RATE)
    logger.info("Waiting for WebTransport connections...")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    logger.info("Shutting down...")
    await backend.close()
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
