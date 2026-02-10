"""RoomKit -- RTP receiver with Gradium STT transcription.

Receive voice audio over RTP and transcribe it to text using Gradium
speech-to-text. Transcriptions are printed to the console in real time.

No AI provider or TTS is used — this is a pure listen-and-transcribe example.

Audio flow:
    RTP (8kHz PCMU) → Pipeline (resample to 16kHz) → Gradium STT → print

Gradium handles speech detection and turn endpointing server-side.  An
inactivity watchdog monitors the RTP stream — when no frames arrive for
500 ms, the last partial transcription is promoted to a final result.

Prerequisites:
    pip install roomkit[rtp,gradium]

Run with:
    GRADIUM_API_KEY=... uv run python examples/rtp_gradium_stt.py

Then send RTP audio to the local port (default 10000).  With aiortp:

    python examples/send_wav.py audio.wav 127.0.0.1 10000

Or with ffmpeg:

    ffmpeg -re -i audio.wav -ar 8000 -ac 1 -acodec pcm_mulaw \\
        -f rtp rtp://127.0.0.1:10000

Environment variables:
    GRADIUM_API_KEY     (required) Gradium API key
    RTP_LOCAL_PORT      Local port to bind RTP (default: 10000)
    GRADIUM_REGION      API region (default: us)
    GRADIUM_STT_MODEL   STT model name (default: default)
    LANGUAGE            Language code for STT (default: en)

    --- Debug ---
    DEBUG               Set to 1 for verbose pipeline/STT logging
    RECORD_DIR          Directory to save audio WAVs:
                        - transport_8khz.wav  (raw 8kHz audio from RTP, pre-pipeline)
                        - {session}_{ts}_inbound.wav (16kHz post-pipeline via recorder)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
import wave

from roomkit import (
    ChannelBinding,
    ChannelType,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.voice.backends.rtp import RTPVoiceBackend
from roomkit.voice.pipeline import AudioFormat, AudioPipelineConfig, AudioPipelineContract
from roomkit.voice.pipeline.recorder import (
    RecordingChannelMode,
    RecordingConfig,
    RecordingMode,
    WavFileRecorder,
)
from roomkit.voice.pipeline.resampler import SincResamplerProvider
from roomkit.voice.stt.gradium import GradiumSTTConfig, GradiumSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("rtp_gradium_stt")

if os.environ.get("DEBUG") == "1":
    logging.getLogger("roomkit.voice").setLevel(logging.DEBUG)
    logging.getLogger("roomkit.voice.stt.gradium").setLevel(logging.DEBUG)
    logging.getLogger("roomkit.voice.pipeline").setLevel(logging.DEBUG)
    with contextlib.suppress(ImportError):
        import examples.trace_audio  # noqa: F401


def check_env() -> str:
    """Return the Gradium API key or exit with a helpful message."""
    api_key = os.environ.get("GRADIUM_API_KEY", "")
    if not api_key:
        print("Missing required environment variable:\n")
        print("  GRADIUM_API_KEY  — Gradium API key for STT\n")
        print("Example:\n")
        print("  GRADIUM_API_KEY=... uv run python examples/rtp_gradium_stt.py")
        sys.exit(1)
    return api_key


async def main() -> None:
    api_key = check_env()
    kit = RoomKit()

    # --- Configuration --------------------------------------------------------
    local_port = int(os.environ.get("RTP_LOCAL_PORT", "10000"))

    # --- RTP backend ----------------------------------------------------------
    backend = RTPVoiceBackend(
        local_addr=("0.0.0.0", local_port),
        remote_addr=("127.0.0.1", 9),  # discard port — we only receive, never send
        payload_type=0,  # PCMU (G.711 mu-law)
        clock_rate=8000,
    )

    # --- Pipeline (no VAD — Gradium handles endpointing server-side) ----------
    # Contract tells the pipeline to resample 8kHz RTP audio to 16kHz internally.
    # Without this, raw 8kHz frames reach STT and cause resampling artifacts.
    contract = AudioPipelineContract(
        transport_inbound_format=AudioFormat(sample_rate=8000, channels=1),
        transport_outbound_format=AudioFormat(sample_rate=8000, channels=1),
    )
    # --- Optional recording (RECORD_DIR=/tmp/recordings) -----------------------
    record_dir = os.environ.get("RECORD_DIR")
    recorder = None
    recording_config = None
    if record_dir:
        recorder = WavFileRecorder()
        recording_config = RecordingConfig(
            mode=RecordingMode.INBOUND_ONLY,
            channels=RecordingChannelMode.SEPARATE,
            storage=record_dir,
        )
        logger.info("Recording inbound audio to %s", record_dir)

    pipeline = AudioPipelineConfig(
        contract=contract,
        resampler=SincResamplerProvider(),
        recorder=recorder,
        recording_config=recording_config,
    )

    # --- Gradium STT ----------------------------------------------------------
    region = os.environ.get("GRADIUM_REGION", "us")
    language = os.environ.get("LANGUAGE", "en")
    stt = GradiumSTTProvider(
        config=GradiumSTTConfig(
            api_key=api_key,
            region=region,
            model_name=os.environ.get("GRADIUM_STT_MODEL", "default"),
            input_format="pcm",
            language=language,
        )
    )
    logger.info("STT: Gradium (region=%s, lang=%s)", region, language)

    # --- Voice channel (no TTS — transcribe only) -----------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=MockTTSProvider(),  # placeholder — not used
        backend=backend,
        pipeline=pipeline,
    )
    kit.register_channel(voice)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="rtp-stt")
    await kit.attach_channel("rtp-stt", "voice")

    # --- Hooks: print transcriptions ------------------------------------------
    @kit.hook(HookTrigger.ON_SPEECH_START, execution=HookExecution.ASYNC)
    async def on_speech_start(session_arg, ctx):
        logger.info("Speech started")

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(session_arg, ctx):
        logger.info("Speech ended")

    # Track latest partial for the inactivity watchdog
    last_partial: dict[str, str] = {}

    @kit.hook(HookTrigger.ON_PARTIAL_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def on_partial(result, ctx):
        logger.info("Partial: %r", result.text)
        if result.text:
            last_partial["text"] = result.text

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(text, ctx):
        last_partial.pop("text", None)
        print(f"\n>>> {text}\n")
        # Block further processing (no AI provider to forward to)
        return HookResult.block()

    # --- Start RTP session ----------------------------------------------------
    session = await backend.connect("rtp-stt", "rtp-caller", "voice")
    binding = ChannelBinding(
        room_id="rtp-stt",
        channel_id="voice",
        channel_type=ChannelType.VOICE,
    )
    voice.bind_session(session, "rtp-stt", binding)

    # --- Audio inactivity watchdog --------------------------------------------
    # When an RTP stream ends abruptly (e.g. finite WAV file, RTCP BYE), no
    # more frames arrive.  Without trailing silence, Gradium's server-side VAD
    # may not detect the turn boundary.  This watchdog promotes the last partial
    # transcription to a final result after 500ms of inactivity.
    inactivity_timeout_s = 0.5
    watchdog_state = {"last_frame": 0.0, "active": True}

    # --- Transport-level WAV recording (raw 8kHz, pre-pipeline) -----------------
    transport_wav: wave.Wave_write | None = None
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        transport_wav_path = os.path.join(record_dir, "transport_8khz.wav")
        transport_wav = wave.open(transport_wav_path, "wb")  # noqa: SIM115
        transport_wav.setnchannels(1)
        transport_wav.setsampwidth(2)
        transport_wav.setframerate(8000)
        logger.info("Transport recording: %s", transport_wav_path)

    orig_on_audio = voice._on_audio_received

    def _track_audio(sess, frame):
        watchdog_state["last_frame"] = time.monotonic()
        # Write raw transport audio before any pipeline processing
        if transport_wav is not None:
            transport_wav.writeframes(frame.data)
        return orig_on_audio(sess, frame)

    voice._on_audio_received = _track_audio

    async def _inactivity_watchdog():
        while watchdog_state["active"]:
            await asyncio.sleep(0.1)
            last = watchdog_state["last_frame"]
            if last == 0.0:
                continue
            gap = time.monotonic() - last
            if gap < inactivity_timeout_s:
                continue

            # RTP stream ended — promote last partial to final result.
            text = last_partial.pop("text", None)
            if text:
                print(f"\n>>> {text}\n")
                logger.info("Promoted partial to final after RTP inactivity")

            # Close the STT stream to stop Gradium hallucinating on silence.
            # The continuous-STT loop will reconnect for the next audio.
            stream_state = voice._stt_streams.get(session.id)
            if stream_state is not None:
                voice._flush_stt_buffer(stream_state, session.id)
                with contextlib.suppress(asyncio.QueueFull):
                    stream_state.queue.put_nowait(None)
                logger.info("Closed STT stream after RTP inactivity")

            watchdog_state["last_frame"] = 0.0

    watchdog_task = asyncio.get_running_loop().create_task(
        _inactivity_watchdog(), name="audio_watchdog"
    )

    logger.info("RTP listening on 0.0.0.0:%d", local_port)
    logger.info("Send audio with e.g.:")
    logger.info(
        "  ffmpeg -re -i audio.wav -ar 8000 -ac 1 -acodec pcm_mulaw -f rtp rtp://127.0.0.1:%d",
        local_port,
    )
    logger.info("Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C --------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    watchdog_state["active"] = False
    watchdog_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await watchdog_task

    # --- Cleanup --------------------------------------------------------------
    logger.info("\nStopping...")
    if transport_wav is not None:
        transport_wav.close()
        logger.info("Transport recording saved.")
    voice.unbind_session(session)
    await backend.disconnect(session)
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
