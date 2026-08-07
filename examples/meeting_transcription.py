"""RoomKit — turn a meeting recording into minutes posted in a room.

A recorded meeting is a file. This example takes it the rest of the way::

    recording.wav → GeminiSTTProvider → speaker turns → room → AIChannel → minutes

The transcription is batch by nature: the model receives the whole recording and
answers in one pass, which is exactly what makes the speaker turns and the
timestamps come back with the words — no diarization stage, no merge. It is the
wrong tool for live turn-taking and the right one for a meeting that is over.

Where the recording comes from in production
--------------------------------------------

A room that records fires ``ON_RECORDING_STOPPED`` with the path it wrote, and
that hook is the natural trigger — it is registered below, so a real recorder
drives the same code this example drives by hand. Note that a conference
records one track per participant: transcribe each track separately and you get
the speakers for free, with no diarization at all. Pass ``diarize=False`` there.
This example uses a single mixed file, which is the case where the model's
speaker labels earn their keep.

With no argument, it synthesizes a short two-voice meeting with Gemini TTS so
the example runs on one API key. Pass a path to transcribe your own recording.

Requires:
    pip install roomkit[gemini]

Environment variables:
    GEMINI_API_KEY — Google Gemini API key

Run with:
    uv run python examples/meeting_transcription.py [recording.wav]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import base64
import tempfile

from shared import require_env

from roomkit import ChannelCategory, InboundMessage, RoomEvent, RoomKit, TextContent
from roomkit.channels.ai import AIChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.providers.gemini import GeminiAIProvider, GeminiConfig
from roomkit.voice.stt.gemini import GeminiSTTConfig, GeminiSTTProvider, Transcript
from roomkit.voice.tts.audio_utils import wrap_wav
from roomkit.voice.tts.gemini import GeminiTTSConfig, GeminiTTSProvider

ROOM_ID = "meeting-room"

SCRIPT = [
    ("Kore", "Let's start. The release is blocked on the audio latency issue."),
    ("Puck", "I measured it yesterday. Eight seconds to first audio, all of it upstream."),
    ("Kore", "Then we ship the batch path first and revisit streaming next sprint."),
    ("Puck", "Agreed. I'll open the ticket and take the measurements to the vendor."),
]

MINUTES_PROMPT = """\
You are a meeting scribe. From the transcript you are given, write minutes with \
three sections: Decisions, Action items (with the owner), and Open questions. \
Attribute nothing that is not in the transcript. Be brief."""


async def synthesize_meeting(api_key: str) -> Path:
    """Fabricate a two-voice recording so the example runs without one."""
    tts = GeminiTTSProvider(GeminiTTSConfig(api_key=api_key, language="en-US"))
    pcm = b""
    print("Synthesizing a sample meeting (one request per turn, seconds each)…", flush=True)
    for voice, line in SCRIPT:
        audio = await tts.synthesize(line, voice=voice)
        pcm += base64.b64decode(audio.url.split(",", 1)[1])[44:]
        pcm += b"\x00" * (2 * 24000 // 2)  # half a second between turns
    await tts.close()

    path = Path(tempfile.mkdtemp(prefix="roomkit_meeting_")) / "meeting.wav"
    path.write_bytes(wrap_wav(pcm, 24000))
    print(f"  wrote {path} ({(len(pcm) / 2 / 24000):.1f}s)\n")
    return path


def print_transcript(transcript: Transcript) -> None:
    print(f"=== transcript ({transcript.language}, {len(transcript.segments)} turns) ===")
    for segment in transcript.segments:
        print(f"  [{segment.start}–{segment.end}] {segment.speaker}: {segment.text}")


async def main() -> None:
    env = require_env("GEMINI_API_KEY")
    api_key = env["GEMINI_API_KEY"]

    recording = Path(sys.argv[1]) if len(sys.argv) > 1 else await synthesize_meeting(api_key)

    # --- The room that will hold the minutes ---------------------------------
    kit = RoomKit()
    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "scribe",
        provider=GeminiAIProvider(GeminiConfig(api_key=api_key, model="gemini-3.6-flash")),
        system_prompt=MINUTES_PROMPT,
    )
    kit.register_channel(ws)
    kit.register_channel(ai)

    minutes: list[RoomEvent] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        minutes.append(event)

    ws.register_connection("user-conn", on_receive, room_id=ROOM_ID)

    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "ws-user")
    await kit.attach_channel(ROOM_ID, "scribe", category=ChannelCategory.INTELLIGENCE)

    stt = GeminiSTTProvider(GeminiSTTConfig(api_key=api_key))

    async def transcribe_into_room(path: Path) -> None:
        """Transcribe *path* and let the room's AI channel write the minutes."""
        print("Transcribing (one pass over the whole recording)…", flush=True)
        transcript = await stt.transcribe_recording(path)
        print_transcript(transcript)

        # The transcript enters the room as an ordinary message; the AI channel
        # is attached as INTELLIGENCE, so it answers it.
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="recorder",
                content=TextContent(body=f"Meeting transcript:\n\n{transcript.text}"),
            )
        )

    # --- In production this is what drives it --------------------------------
    # A recording that closes reports where it went; that is the trigger.
    #
    #   @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
    #   async def on_recording_stopped(event, ctx) -> None:
    #       # The trigger carries two shapes: a voice channel reports
    #       # RecordingStoppedEvent.urls (one per recorder), a conference
    #       # reports ConferenceRecordingStopped.url, once per participant track.
    #       for url in getattr(event, "urls", None) or (event.url,):
    #           await transcribe_into_room(Path(url))
    #
    # The demo calls the same function directly, with no recorder in the room.
    await transcribe_into_room(recording)

    print("\n=== minutes ===")
    for event in minutes:
        print(event.content.body)  # type: ignore[union-attr]

    await stt.close()
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
