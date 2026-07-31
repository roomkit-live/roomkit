"""A real conference: two humans, one bot, a LiveKit SFU.

The mock quickstart (``conference_quickstart.py``) proves the wiring; this
example is what you run to see the thing work. Your browser connects straight
to the SFU — RoomKit is never in the media path between humans (RFC §12.10.1).
It is the participant named "roomkit" that joins your meeting, subscribes to
your microphone, transcribes what you say attributed to who said it, and
publishes the AI's audio as its own track.

Start a LiveKit dev server (dev credentials: ``devkey`` / ``secret``):

    cat > livekit.yaml <<'EOF'
    port: 7880
    rtc:
      tcp_port: 7881
      udp_port: 7882
      use_external_ip: false
      node_ip: 127.0.0.1
    EOF
    docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \\
        -e LIVEKIT_CONFIG="$(cat livekit.yaml)" \\
        livekit/livekit-server --dev --bind 0.0.0.0

Then run this script (``pip install roomkit[livekit]``):

    uv run python examples/conference_livekit.py

It prints one meeting URL per human — open each in a browser tab. Everything
works without a single API key: the STT is stubbed (your real audio still
crosses the resampler and the VAD, and the ATTRIBUTION is real) and the TTS is
a 440 Hz beep (either you hear the bot or you don't, which is the question).
Export ``DEEPGRAM_API_KEY`` and/or ``ELEVENLABS_API_KEY`` to use the real
providers.

Export ``ANTHROPIC_API_KEY`` as well and the loop closes for real: a live
``AIChannel`` answers what the meeting says — you speak, the model reads the
attributed transcription like any other RoomEvent, and the bot says the answer
on its own track (STT -> LLM -> TTS, no extra wiring; the deterministic
version of this loop is ``conference_ai_meeting.py``). Typed text then becomes
a silent prompt to the model instead of the bot's script. All three keys
together make it a conversation; without the AI key, what you type is what
the bot says. Two knobs: ``ANTHROPIC_MODEL`` picks the model (default
``claude-haiku-4-5`` — a spoken answer is judged by its latency first) and
``DEEPGRAM_LANGUAGE`` the STT language (default ``en`` — set ``fr`` to speak
French; the assistant answers in the language it is addressed in):

    DEEPGRAM_LANGUAGE=fr uv run python examples/conference_livekit.py

The default energy-threshold VAD loses softly-spoken words and never closes an
utterance over background noise. For real utterance boundaries, drop a neural
VAD model next to this script (sherpa-onnx extra required; pin 1.13.3 — the
1.13.4 macOS arm64 wheel is broken):

    curl -sL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx \\
        -o examples/models/ten-vad.onnx

Resume mode (a restart over a live meeting): run once, keep a browser tab
open, Ctrl-C here — the bot leaves, YOU stay — then:

    ROOMKIT_RESUME=1 uv run python examples/conference_livekit.py

In resume mode the script mints nothing and creates no participant: the only
possible join trigger is the attach's occupancy probe (RFC §12.10.4 step 1),
and the roster can only refill through the join's catch-up. The bot must come
back, and the participants still in the meeting must reappear on the roster —
rebuilt from scratch, in a fresh process, with no persistent store.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import sys
from array import array
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote

from roomkit import RoomKit
from roomkit.channels.base import Channel
from roomkit.channels.conference import ConferenceChannel
from roomkit.conference.livekit import LiveKitConferenceBackend, LiveKitConfig
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.voice.base import AudioChunk
from roomkit.voice.pipeline.config import AudioPipelineConfig, AudioPipelineContract
from roomkit.voice.tts.base import TTSProvider

URL = os.getenv("ROOMKIT_LIVEKIT_URL", "ws://127.0.0.1:7880")
API_KEY = os.getenv("ROOMKIT_LIVEKIT_API_KEY", "devkey")
API_SECRET = os.getenv("ROOMKIT_LIVEKIT_API_SECRET", "secret")
ROOM_ID = os.getenv("ROOMKIT_ROOM", "demo-meeting")
RESUME = os.getenv("ROOMKIT_RESUME") == "1"

SAMPLE_RATE = 48_000
HUMANS = ("alice", "bob")


class TextSource(Channel):
    """A channel that turns stdin lines into room events.

    Its type decides what the conference does with them, because the
    conference speaks only AI-typed events aloud — that is the channel's rule,
    not this script's: a meeting is not a place to read other channels'
    traffic aloud. As ``ChannelType.AI`` this stands in for an AIChannel and
    what you type is what the bot says; as any other type (the real-AI mode
    below) it is a silent inlet — what you type prompts the model instead of
    being read out.
    """

    def __init__(self, channel_id: str, *, channel_type: ChannelType = ChannelType.AI) -> None:
        super().__init__(channel_id)
        self._channel_type = channel_type

    @property
    def channel_type(self) -> ChannelType:
        return self._channel_type

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self._channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class BeepTTS(TTSProvider):
    """A "synthesizer" that emits a 440 Hz beep. No key, no model.

    It says nothing intelligible on purpose: the question it answers is "does
    the audio the bot publishes reach an ear", not "is the voice pretty". The
    duration follows the text's length so a long sentence is audible long
    enough to talk over — that is what makes interruption testable by ear.
    """

    @property
    def name(self) -> str:
        return "beep"

    async def synthesize(self, text: str, *, voice: str | None = None) -> Any:
        raise NotImplementedError("BeepTTS only streams")

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        frames = max(20, min(len(text) * 12, 1500))  # 200 ms to 15 s, in 10 ms steps
        per_frame = SAMPLE_RATE // 100
        for index in range(frames):
            samples = array("h")
            for n in range(per_frame):
                position = index * per_frame + n
                fade = min(1.0, (frames - index) / 10)  # avoid the closing click
                samples.append(
                    int(0.25 * 32767 * fade * math.sin(2 * math.pi * 440 * position / SAMPLE_RATE))
                )
            yield AudioChunk(
                data=samples.tobytes(),
                sample_rate=SAMPLE_RATE,
                channels=1,
                is_final=(index == frames - 1),
            )
            # No sleep here, deliberately. Pacing a 10 ms frame with
            # asyncio.sleep(0.01) actually costs ~11 ms (scheduler
            # granularity): production runs at 0.89x real time, the source's
            # queue starves, and LiveKit plays silence between frames — it
            # sounds choppy. The backend's backpressure (capture_frame
            # blocking when its queue is full) is what sets the pace.


def build_providers() -> tuple[Any, TTSProvider, list[str]]:
    """Real providers when their keys are present, stubs otherwise.

    The STT is not decorative: without one the channel consumes no audio
    tracks, so it does not bring the bot in when you publish your microphone —
    subscription is selective by design. A stubbed STT is enough to open the
    door: the bot joins, subscribes, and your real audio crosses the resampler
    and the VAD. Only the produced text is scripted.
    """
    notes: list[str] = []
    if os.getenv("DEEPGRAM_API_KEY"):
        from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider

        # language= is not optional in practice: the provider's default is
        # "en", and speaking anything else at it transcribes gibberish.
        language = os.getenv("DEEPGRAM_LANGUAGE", "en")
        stt: Any = DeepgramSTTProvider(
            DeepgramConfig(api_key=os.environ["DEEPGRAM_API_KEY"], language=language)
        )
        notes.append(f"STT: Deepgram ({language}) — transcriptions will be real words")
    else:
        from roomkit.voice.stt.mock import MockSTTProvider

        stt = MockSTTProvider(transcripts=["(stubbed text — set DEEPGRAM_API_KEY)"])
        notes.append(
            "STT: stub — the text is fake, but the bot joins, subscribes, and your\n"
            "        real audio crosses the resampler and the VAD. The ATTRIBUTION is real."
        )

    if os.getenv("ELEVENLABS_API_KEY"):
        from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

        tts: TTSProvider = ElevenLabsTTSProvider(
            ElevenLabsConfig(
                api_key=os.environ["ELEVENLABS_API_KEY"],
                # Not the provider's mp3 default: the conference backend
                # publishes decoded PCM and refuses encoded audio — encoding
                # belongs to the backend (RFC §12.10.1 principle 6). The
                # LiveKit publisher takes the chunk's declared rate as-is,
                # and pcm_24000 is available on every ElevenLabs tier.
                output_format="pcm_24000",
                voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            )
        )
        notes.append(
            "TTS: ElevenLabs pcm_24000 (the bot really speaks; "
            "ELEVENLABS_VOICE_ID to pick the voice)"
        )
    else:
        tts = BeepTTS()
        notes.append("TTS: 440 Hz beep — no key, but audibility tests the same")
    return stt, tts, notes


def build_realtime(notes: list[str]) -> Any | None:
    """The speech-to-speech composition, when asked for and a key is present.

    ``ROOMKIT_REALTIME=1`` swaps the STT->LLM->TTS loop for one realtime
    provider (RFC §12.10.12): the meeting is mixed N->1 into a single
    speech-to-speech session and the provider's voice publishes on the bot
    track. The per-track STT stays beside it — attribution ends at the
    provider's boundary, so the lanes are what keep the transcript naming
    who spoke. ``tts=`` is mutually exclusive with this: one bot track, one
    voice. The deterministic version is examples/conference_realtime_ai.py.
    """
    if os.getenv("ROOMKIT_REALTIME") != "1":
        return None
    from roomkit import ConferenceRealtimeConfig

    system_prompt = (
        "You are the meeting's voice assistant. Answer briefly — one or two "
        "spoken sentences — in the language you are addressed in."
    )
    if os.getenv("GEMINI_API_KEY"):
        from roomkit.providers.gemini.realtime import GeminiLiveProvider

        model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
        notes.append(f"REALTIME: Gemini Live {model} — one session hears the mixed meeting")
        return ConferenceRealtimeConfig(
            provider=GeminiLiveProvider(api_key=os.environ["GEMINI_API_KEY"], model=model),
            system_prompt=system_prompt,
        )
    if os.getenv("OPENAI_API_KEY"):
        from roomkit.providers.openai.realtime import OpenAIRealtimeProvider

        model = os.getenv("OPENAI_MODEL", "gpt-realtime-2")
        notes.append(f"REALTIME: OpenAI {model} — one session hears the mixed meeting")
        return ConferenceRealtimeConfig(
            provider=OpenAIRealtimeProvider(api_key=os.environ["OPENAI_API_KEY"], model=model),
            system_prompt=system_prompt,
        )
    notes.append(
        "REALTIME: requested but no key — set GEMINI_API_KEY or OPENAI_API_KEY; "
        "falling back to the STT->LLM->TTS loop"
    )
    return None


def build_ai(notes: list[str]) -> Any | None:
    """The real intelligence when its key is present, nothing otherwise.

    This is the whole STT -> LLM -> TTS wiring: registering the AIChannel *is*
    the loop. Transcriptions are RoomEvents (RFC §12.10.1 principle 2), the
    model answers them like any other room traffic, and the conference speaks
    AI events on the bot's track. See examples/conference_ai_meeting.py for
    the same loop deterministic on the mock backend.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        notes.append(
            "AI: none — the bot does not answer; what you type is what it says\n"
            "        (set ANTHROPIC_API_KEY to talk WITH the bot instead)"
        )
        return None
    from roomkit.channels.ai import AIChannel
    from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

    # Named explicitly rather than left to the provider default, and defaulted
    # to Haiku: a spoken answer is judged by its latency before its depth, and
    # the model is the caller's call anyway — ANTHROPIC_MODEL overrides.
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    notes.append(f"AI: Anthropic {model} — speak to the meeting and the bot answers out loud")
    return AIChannel(
        "ai",
        provider=AnthropicAIProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"], model=model)
        ),
        system_prompt=(
            "You are the meeting's voice assistant. Answer briefly — one or two "
            "spoken sentences — in the language you are addressed in."
        ),
    )


def build_vad(notes: list[str]) -> Any:
    """The neural VAD when its model is around, calibrated energy otherwise.

    The VAD is what answers "when does the audio go to the STT": nothing is
    sent while you speak, the lane accumulates, and the whole utterance leaves
    in one block once the VAD has seen silence_threshold_ms of consecutive
    silence. The utterance boundary is the VAD's call.
    """
    model = os.getenv("VAD_MODEL") or os.path.join(
        os.path.dirname(__file__), "models", "ten-vad.onnx"
    )
    if os.path.exists(model):
        from roomkit.voice.pipeline.vad.sherpa_onnx import (
            SherpaOnnxVADConfig,
            SherpaOnnxVADProvider,
        )

        notes.append(f"VAD: TEN-VAD (sherpa-onnx), 700 ms silence — {os.path.basename(model)}")
        # threshold 0.5: the provider's recommendation without a denoiser.
        # silence 700 ms: conversational speech pauses longer than the 500 ms
        # default — the knob to turn (500-800) if your sentences get cut short
        # or drag on.
        return SherpaOnnxVADProvider(
            SherpaOnnxVADConfig(model=model, threshold=0.5, silence_threshold_ms=700)
        )

    from roomkit.voice.pipeline.vad.energy import EnergyVADProvider

    notes.append(
        "VAD: energy (RMS) — fallback. For real utterance boundaries:\n"
        "        curl -sL https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/ten-vad.onnx -o examples/models/ten-vad.onnx"
    )
    return EnergyVADProvider(silence_threshold_ms=700)


def meet_url(access: Any) -> str:
    """The LiveKit web client's URL, pre-filled with the minted credential."""
    return (
        "https://meet.livekit.io/custom"
        f"?liveKitUrl={quote(access.url, safe='')}&token={quote(access.token, safe='')}"
    )


async def show_conference_state(kit: RoomKit, conference: ConferenceChannel, backend: Any) -> None:
    """The full conference state, through the three APIs a UI would read.

    Three different authorities, deliberately not one aggregate call:

    1. ``backend.list_participants()`` — MEDIA presence, the SFU's truth: who
       is connected, since when, with which tracks (kind, mute) and which
       metadata the SFU asserts (a dial-in carries its number here).
    2. ``kit.store.list_participants()`` — the room ROSTER RoomKit keeps:
       role, status, identification, resolved identity. In resume mode it was
       rebuilt by the join's catch-up (RFC §12.10.3) into a store born empty.
    3. ``conference.info()`` — the RFC §17.7 DISCLOSURE: the bot (present,
       session, hidden) and what is being done with the media — collection,
       STT active, recording active — at the moment you ask.
    """
    deadline = asyncio.get_running_loop().time() + 15.0
    roster = await kit.store.list_participants(ROOM_ID)
    while not roster and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.5)
        roster = await kit.store.list_participants(ROOM_ID)

    info = conference.info()
    bot_identity = info["bot_identity"]

    print("\n" + "-" * 74)
    print("  1) MEDIA PRESENCE — backend.list_participants() (the SFU's truth)")
    for p in await backend.list_participants(ROOM_ID):
        tag = " (the channel's bot)" if p.participant_id == bot_identity else ""
        print(
            f"    * {p.participant_id}{tag} — shown as {p.display_name or '(no name)'!r}"
            f" — connected since {p.connected_at:%H:%M:%S}"
        )
        for track in p.tracks:
            state = "muted" if track.muted else "open"
            print(f"        {track.kind.value} track [{state}] id={track.id}")

    print("\n  2) ROOM ROSTER — kit.store.list_participants() (RoomKit's records)")
    for r in roster:
        print(
            f"    * {r.id} — role {r.role.value}, status {r.status.value},"
            f" identification {r.identification.value}"
        )
        print(f"        display: {r.display_name or '-'}, joined at {r.joined_at:%H:%M:%S}")

    print("\n  3) RFC §17.7 DISCLOSURE — conference.info()")
    print(
        f"    backend {info['backend']!r}, bot {bot_identity!r}"
        f" (hidden: {info['bot_hidden']}), stt {info['stt_provider'] or 'none'},"
        f" recording configured: {info['recording_configured']}"
    )
    for key, value in (info["rooms"].get(ROOM_ID) or {}).items():
        print(f"    {key}: {value}")
    print("-" * 74)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    stt, tts, notes = build_providers()
    vad = build_vad(notes)
    realtime = build_realtime(notes)
    ai = None if realtime is not None else build_ai(notes)
    backend = LiveKitConferenceBackend(
        LiveKitConfig(url=URL, api_key=API_KEY, api_secret=API_SECRET)
    )
    kit = RoomKit()
    conference = ConferenceChannel(
        "conf",
        backend=backend,
        stt=stt,
        # One bot track, one voice: the realtime provider replaces the TTS —
        # and the AIChannel too, or the meeting would have two brains.
        tts=None if realtime is not None else tts,
        realtime=realtime,
        # Without pipeline= the channel builds an energy VAD by default; the
        # contract stays either way — it is the format normalization
        # (48 kHz browser stereo -> 16 kHz mono STT).
        pipeline=AudioPipelineConfig(vad=vad, contract=AudioPipelineContract()),
    )
    kit.register_channel(conference)
    if ai is not None:
        # The real loop: the model answers what the meeting says, and stdin
        # becomes a silent side-inlet to it — typed text prompts the model
        # without being read aloud (a non-AI source is not spoken).
        kit.register_channel(ai)
        typed = TextSource("host", channel_type=ChannelType.WEBSOCKET)
    else:
        # No key: stdin stands in for the AI, and what you type is what the
        # bot says.
        typed = TextSource("ai")
    kit.register_channel(typed)

    # What the bot heard, attributed. source.participant_id is the identity
    # the lane attributed the voice to — it must say alice or bob, never
    # "unknown", never the other one.
    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def heard(event: Any, ctx: Any) -> None:
        body = getattr(event.content, "body", None)
        if not body:
            return
        who = event.source.participant_id or "-"
        if event.source.channel_id == typed.channel_id:
            origin = "typed"
        elif event.source.channel_id == "ai" or not event.source.participant_id:
            # An AIChannel's answer, or the realtime provider's transcript —
            # the one voice in the room no lane attributes to a human.
            origin = "AI"
        else:
            origin = "HEARD"
        print(f"\n  [{origin} * {who}] {body}\n> ", end="", flush=True)

    # The VAD's speech edges, per participant and track: the real-time
    # "X started / X finished", without waiting for the STT round-trip.
    @kit.hook(HookTrigger.ON_SPEECH_START)
    async def speech_start(event: Any, ctx: Any) -> None:
        who = event.content.data.get("participant_id", "?")
        print(f"\n  [speech] {who} starts speaking\n> ", end="", flush=True)

    @kit.hook(HookTrigger.ON_SPEECH_END)
    async def speech_end(event: Any, ctx: Any) -> None:
        who = event.content.data.get("participant_id", "?")
        print(f"\n  [speech] {who} finished\n> ", end="", flush=True)

    # The SFU's dominant speaker — server-side energy detection, no track
    # subscription needed. It cannot say that nobody is speaking; the end of a
    # turn is the VAD's ON_SPEECH_END above.
    @kit.hook(HookTrigger.ON_ACTIVE_SPEAKER_CHANGED)
    async def speaking(event: Any, ctx: Any) -> None:
        who = event.content.data.get("participant_id", "?")
        print(f"\n  [speech] {who} is speaking\n> ", end="", flush=True)

    # Connection quality as the SFU sees it — a management UI's quality bars.
    # The bot itself is included: this is the notetaker's own health signal.
    @kit.hook(HookTrigger.ON_CONNECTION_QUALITY_CHANGED)
    async def network(event: Any, ctx: Any) -> None:
        data = event.content.data
        print(
            f"\n  [network] {data.get('participant_id', '?')}: {data.get('quality', '?')}\n> ",
            end="",
            flush=True,
        )

    # Mute/unmute per track, kind included: a muted VIDEO track is how most
    # clients say "camera off" — microphone and camera indicators both read
    # from this pair. No subscription required.
    kinds = {"audio": "microphone", "video": "camera", "screen_share": "screen share"}

    @kit.hook(HookTrigger.ON_CONFERENCE_TRACK_MUTED)
    async def track_muted(event: Any, ctx: Any) -> None:
        data = event.content.data
        what = kinds.get(data.get("kind", ""), "track")
        print(
            f"\n  [media] {data.get('participant_id', '?')} mutes {what}\n> ", end="", flush=True
        )

    @kit.hook(HookTrigger.ON_CONFERENCE_TRACK_UNMUTED)
    async def track_unmuted(event: Any, ctx: Any) -> None:
        data = event.content.data
        what = kinds.get(data.get("kind", ""), "track")
        print(
            f"\n  [media] {data.get('participant_id', '?')} unmutes {what}\n> ", end="", flush=True
        )

    @kit.hook(HookTrigger.ON_CONFERENCE_TRACK_PUBLISHED)
    async def track_on(event: Any, ctx: Any) -> None:
        data = event.content.data
        what = kinds.get(data.get("kind", ""), "track")
        print(
            f"\n  [media] {data.get('participant_id', '?')} publishes {what}\n> ",
            end="",
            flush=True,
        )

    @kit.hook(HookTrigger.ON_SCREEN_SHARE_STARTED)
    async def screen_on(event: Any, ctx: Any) -> None:
        who = event.content.data.get("participant_id", "?")
        print(f"\n  [media] {who} shares their screen\n> ", end="", flush=True)

    @kit.hook(HookTrigger.ON_SCREEN_SHARE_STOPPED)
    async def screen_off(event: Any, ctx: Any) -> None:
        who = event.content.data.get("participant_id", "?")
        print(f"\n  [media] {who} stops sharing\n> ", end="", flush=True)

    # Roster movements, live. Registered BEFORE the attach: in resume mode the
    # join started by the occupancy probe redelivers the humans still
    # connected within its first seconds (the catch-up, RFC §12.10.3), and a
    # listener registered after the attach would miss them.
    @kit.on("conference_started")
    async def bot_in(event: Any) -> None:
        print(f"\n  [roster] bot joined (session {event.data.get('bot_session_id', '?')})")

    @kit.on("conference_ended")
    async def bot_out(event: Any) -> None:
        print(f"\n  [roster] bot left (after {event.data.get('duration_ms', '?')} ms)")

    @kit.on("conference_participant_joined")
    async def human_in(event: Any) -> None:
        print(f"\n  [roster] + {event.data.get('participant_id', '?')}")

    @kit.on("conference_participant_left")
    async def human_out(event: Any) -> None:
        print(f"\n  [roster] - {event.data.get('participant_id', '?')}")

    room = await kit.create_room(ROOM_ID)
    await kit.attach_channel(room.id, "conf")
    if ai is not None:
        await kit.attach_channel(room.id, "ai")
    await kit.attach_channel(room.id, typed.channel_id)

    print("\n" + "=" * 74)
    for note in notes:
        print(f"  {note}")
    print("=" * 74)

    if RESUME:
        print("""
RESUME MODE (no mint, no participant created by this process). The only
possible join trigger is the attach's occupancy probe — the INFO log
"found N participant(s) already in ... at attach" above names it. In the tab
you kept open, "roomkit" must be back without anyone having spoken, minted or
clicked; speak, and [HEARD * ...] must still attribute you correctly.
""")
        await show_conference_state(kit, conference, backend)
    else:
        print("\nOpen BOTH links, each in its own tab (or one on your phone):\n")
        for human in HUMANS:
            # Access is minted for a ROOM participant (RFC §12.10.2): that is
            # what gives transcriptions and hooks someone to attribute speech
            # to. The mint is also the lazy join's trigger — by the time you
            # open the tab, "roomkit" is already on the participant list.
            await kit.ensure_participant(room.id, "conf", human, display_name=human.capitalize())
            access = await conference.mint_access(room.id, human)
            print(f"  - {human}:\n    {meet_url(access)}\n")
        print("=" * 74)
        if realtime is not None:
            print("""
What to check, in order (SPEECH-TO-SPEECH, RFC §12.10.12):

  1. THE BOT IS IN   - open a tab: "roomkit" must ALREADY be on the
                       participant list.
  2. IT HEARS YOU    - speak. [HEARD * alice] still appears HERE with the
                       right identity — the per-track lanes attribute, the
                       provider only gets the anonymous mix.
  3. IT ANSWERS, FAST- keep talking: the provider answers with its OWN voice
                       on the bot track, sub-second — no STT->LLM->TTS relay.
                       Its words appear as [AI * -] lines.
  4. TEXT PROMPTS    - type + Enter: injected into the provider's context,
                       not read aloud.
  5. INTERRUPTION    - talk over the bot: the lane VAD latches, the SFU's
                       queue is dropped, and the provider's response is
                       cancelled — policy stays the conference's, not the
                       provider's.
  6. TEARDOWN        - Ctrl-C: session closed, "roomkit" gone from the tabs.

Type to prompt the bot silently, speak to converse, Ctrl-C to end.
""")
        elif ai is not None:
            print("""
What to check, in order:

  1. THE BOT IS IN   - open a tab: "roomkit" must ALREADY be on the
                       participant list. The mint brought it in, not a word.
  2. IT HEARS YOU    - speak. A [HEARD * alice] line must appear HERE, with
                       the RIGHT identity — never "unknown", never the other.
  3. IT ANSWERS YOU  - keep talking: an [AI * -] line appears and the bot
                       SAYS it in both tabs. This is the whole loop — lane,
                       transcription, AIChannel, TTS, bot track.
  4. TEXT PROMPTS    - type a sentence + Enter: it is NOT read aloud (silent
                       inlet), the model answers it out loud instead.
  5. INTERRUPTION    - talk over the bot mid-answer. It stops.
  6. TEARDOWN        - Ctrl-C here: "roomkit" must vanish from both tabs.

Type to prompt the bot silently, speak to converse, Ctrl-C to end.
""")
        else:
            print("""
What to check, in order:

  1. THE BOT IS IN   - open a tab: "roomkit" must ALREADY be on the
                       participant list. The mint brought it in, not a word.
  2. IT HEARS YOU    - speak. A [HEARD * alice] line must appear HERE, with
                       the RIGHT identity — never "unknown", never the other.
  3. YOU HEAR IT     - type a sentence + Enter. It must sound in both tabs.
  4. INTERRUPTION    - type a long sentence, then talk over it. It stops.
  5. TEARDOWN        - Ctrl-C here: "roomkit" must vanish from both tabs.
  6. RESUME          - keep a tab open and rerun with ROOMKIT_RESUME=1.

Type a sentence to make the bot speak, or Ctrl-C to end.
""")

    loop = asyncio.get_running_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            if text := line.strip():
                await kit.send_event(room.id, typed.channel_id, TextContent(body=text))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\nBot leaving...")
        await kit.close()
        print("Done. Check the tabs: 'roomkit' must be gone.")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
