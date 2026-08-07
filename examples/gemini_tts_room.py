"""RoomKit — a room that speaks whatever is written in it.

Two channels share one room. You type into the CLI channel; the line enters
the room as an ordinary message, the room broadcasts it, and the voice channel
speaks it through the system speakers with Gemini TTS::

    keyboard → CLIChannel → room → broadcast → VoiceChannel → Gemini TTS → speakers

Nothing wires the two channels together — they only share a room. Attach a
third channel (SMS, Telegram, an AI channel) and its messages get spoken too,
because speaking is what this room does with text.

The voice channel runs with ``stt=None``: no microphone transcription, only
delivery. The local backend still opens the mic when the session starts —
audio simply goes nowhere without an STT provider.

Expressions
-----------

Gemini TTS is a generative model, so the delivery is steerable from the text
itself. Bracketed cues are *performed*, not read out::

    [laughs] Okay, that one was actually funny.
    [whispers] Can you keep a secret?
    [excitedly] The release is out!

Two kinds, both inline in the line you type: non-verbal sounds (``[laughs]``,
``[sighs]``, ``[gasp]``, ``[cough]``) and delivery modifiers (``[whispers]``,
``[shouting]``, ``[excitedly]``, ``[bored]``, ``[very slowly]``, ``[singing]``).
Google documents no closed list — any descriptive cue is interpreted — so
uncommon ones deserve a listen before you ship them: a tag the model does not
recognise can be spoken aloud instead of performed. With a non-English
transcript, keep the tags in English. Type ``/tags`` in the prompt for the
short list.

A whole-utterance direction belongs in ``style_prompt`` instead (see below);
tags steer a word or a phrase, ``style_prompt`` steers the performance.

Gemini TTS answers in seconds, not milliseconds, so expect a pause between
pressing Enter and hearing the line. The ``BEFORE_TTS`` hook below prints what
is about to be spoken so the wait is visible.

Requires:
    pip install roomkit[gemini,local-audio]

Environment variables:
    GEMINI_API_KEY — Google Gemini API key

Run with:
    uv run python examples/gemini_tts_room.py

Type a line and press Enter. Ctrl+D (or Ctrl+C) to quit.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio

from shared import require_env

from roomkit import CLIChannel, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.tts.gemini import GeminiTTSConfig, GeminiTTSProvider

ROOM_ID = "gemini-tts-room"

TAGS_HELP = """\
  Performed, not read — put them inline in the line you type:
    sounds     [laughs] [sighs] [gasp] [cough] [clears throat]
    delivery   [whispers] [shouting] [excitedly] [bored] [reluctantly] [sarcastically]
    pacing     [very slowly] [very fast] [short pause] [long pause]
    character  [singing] [asmr] [like a cartoon dog]
  Google documents no closed list — any descriptive cue is interpreted, so listen
  to an unusual one before shipping it: unrecognised cues can be read aloud.
  Keep the tags in English even when the transcript is not.
  Durations are interpreted, never executed: a pause asked for in seconds comes
  back seconds off, and differently on every run. SSML is not an input mode here.
  Example: [whispers] I have a secret. [laughs] Just kidding!\
"""


async def main() -> None:
    env = require_env("GEMINI_API_KEY")

    kit = RoomKit()

    cli = CLIChannel("cli")
    voice = VoiceChannel(
        "voice",
        tts=GeminiTTSProvider(
            GeminiTTSConfig(
                api_key=env["GEMINI_API_KEY"],
                voice="Kore",
                language="en-US",
                # style_prompt="Read this in a calm, reassuring voice",
            )
        ),
        # Default output rate is 24 kHz — Gemini's fixed rate, so no resampling.
        backend=LocalAudioBackend(),
    )
    kit.register_channel(cli)
    kit.register_channel(voice)

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def announce(text: str, ctx: object) -> HookResult:
        print(f"  ♪ speaking: {text}", flush=True)
        return HookResult.allow()

    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "cli")
    # Attaching the voice channel auto-starts a local audio session, because
    # LocalAudioBackend declares auto_connect — that session is what the room
    # speaks into.
    await kit.attach_channel(ROOM_ID, "voice")

    async def show_tags(_argument: str) -> None:
        print(TAGS_HELP, flush=True)

    try:
        await cli.run(
            kit,
            room_id=ROOM_ID,
            welcome=(
                "Type a line — the room will say it. "
                "Try [laughs] or [whispers] inline; /tags lists more. Ctrl+D to quit."
            ),
            commands={"/tags": show_tags},
        )
    finally:
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
