"""The speech-to-speech provider as a conference's intelligence.

One RealtimeVoiceProvider session per conference: the mixer feeds it every
subscribed track as one stream, and its voice publishes on the bot track
through the same turn-taking, latch and terminal-chunk machinery as TTS — a
provider response and a synthesized answer are indistinguishable to the
backend and to a barge-in (RFC 12.10.12).

Attribution ends at the provider boundary. The provider's transcription of
what it heard names nobody — the mix has no speaker identity — so user-role
transcriptions are discarded; the attributed transcript is the per-track STT
lanes', running in parallel when configured. Assistant-role finals are the
one record of what the AI said — no AIChannel generation stands behind this
voice — and are emitted as room events attributed to the channel.

The session follows the bot. Connecting is lazy — the first mixed window or
the first text to inject establishes it — and a connect failure fails
neither the join nor the plug: the configuration stands, and the next
trigger retries after a cooldown rather than on every 20 ms window.

Split from ConferenceChannel for room, not for isolation: everything here is
steered by the channel that owns it, through the seams it was built with.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from roomkit.channels._conference_mixer import ConferenceMixer
from roomkit.channels._conference_operations import ConferenceResource
from roomkit.channels._conference_tools import (
    ConferenceToolGate,
    bound_result,
    catalogue,
    declared_tools,
    warn_unused_role_overrides,
)
from roomkit.channels._realtime_endings import SparedCalls, abandon_calls, interrupt_for_ending
from roomkit.channels._realtime_host_hooks import fire_session_error, fire_text_injected
from roomkit.channels._realtime_tool_calls import RealtimeToolCall, ToolCallBook
from roomkit.channels._realtime_tool_executor import (
    ABANDONED_BY_PROVIDER,
    SESSION_ENDED,
    ToolCallDoor,
    report_cancelled_call,
    report_interrupted_calls,
    run_tool_call,
    serve_unbooked,
    serving_tool_call,
    submit_tool_outcome,
    tool_loop_context,
)
from roomkit.channels._served_tools import CollisionLog, dict_tool_name, warn_tools_uncallable
from roomkit.channels._tool_registry import schema_tool
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.core.task_utils import log_task_exception
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.tool_call import ToolCallEvent
from roomkit.tools._human_input_channel import ChannelHumanInput
from roomkit.tools._outcome import ToolOutcome
from roomkit.tools.result import GateRefusal, declined_answer, result_text
from roomkit.tools.timeout import answer_within
from roomkit.voice.base import AudioChunk, VoiceSession, VoiceSessionState
from roomkit.voice.realtime._answer_depth import AnswerDepth
from roomkit.voice.realtime.injection import VoiceInjectionResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from roomkit.channels._conference_operations import ConferenceOperations
    from roomkit.channels._conference_voice import ConferencePlayback, ConferenceVoice
    from roomkit.conference.models import BotSession, ConferenceRealtimeConfig
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.tools.context import _ToolLoopContext
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.conference")

_LEFT_THE_ROOM = "The conference left the room"
"""Why a call a detach cut was cancelled before its result."""

# Opens the bot's media connection for a room — the channel's _ensure_bot.
EnsureBot = Callable[[str], "Awaitable[BotSession]"]

# The room's binding as the channel last saw it — the channel's _binding_of.
BindingOf = Callable[[str], "ChannelBinding | None"]

CONNECT_COOLDOWN_S = 5.0
"""How long a failed provider connect holds further attempts off.

The mixer asks again every window; without this a provider outage would be
retried fifty times a second from every conference the channel serves.
"""

_RUNNING_REPORT: ContextVar[asyncio.Task[Any] | None] = ContextVar(
    "conference_running_report", default=None
)
"""The tracked report (or announcement) the current code runs under, so a
teardown its hook starts never waits for the report that is running it."""


@dataclass
class _Utterance:
    """One provider response on its way to the bot track.

    A queue-backed bridge between two pacings: the provider pushes audio
    deltas as it generates them, and the voice's pump pulls them through its
    turn on the track at the backend's pace. ``None`` on the queue is the end of the
    response — the pump publishes the terminal chunk behind it.
    """

    queue: asyncio.Queue[AudioChunk | None] = field(default_factory=asyncio.Queue)
    discarded: bool = False
    """A barge-in landed: everything still arriving for this response is
    dropped rather than queued behind a latch that will never publish it."""

    playback: ConferencePlayback | None = None
    transcript: str = ""

    def finish(self) -> None:
        self.queue.put_nowait(None)


def _idle_event() -> asyncio.Event:
    event = asyncio.Event()
    event.set()
    return event


@dataclass
class _RoomRealtime:
    """One room's share of the provider: its session, and the response in flight."""

    session: VoiceSession | None = None
    connecting: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_connect_at: float = 0.0
    starting: VoiceSession | None = None
    """The session ``connect()`` is establishing for the room."""
    start_calls: list[RealtimeToolCall] = field(default_factory=list)
    """The calls the provider issued while that session started, served once
    it is the room's (RFC §12.4)."""
    utterance: _Utterance | None = None
    tasks: set[asyncio.Task[None]] = field(default_factory=set)
    answer_depth: AnswerDepth = field(default_factory=AnswerDepth)
    """What the room's model heard last, which its answer's depth follows."""
    speaking: int = 0
    """Responses still waiting for or taking their turn on the bot track."""
    hearing: bool = False
    """The provider's own VAD hears the room's people speak."""
    idle: asyncio.Event = field(default_factory=_idle_event)
    """Set while nothing is in flight in the room (RFC §22.2)."""

    def hold_off(self) -> None:
        """Hold further connect attempts off for the cooldown."""
        self.next_connect_at = asyncio.get_running_loop().time() + CONNECT_COOLDOWN_S

    def settle(self) -> None:
        """Idle when no response is open or still published and nobody is heard."""
        open_response = self.utterance is not None and not self.utterance.discarded
        if open_response or self.speaking or self.hearing:
            self.idle.clear()
        else:
            self.idle.set()

    def spawn(self, coro: Awaitable[None]) -> asyncio.Task[None]:
        task = asyncio.ensure_future(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(log_task_exception)
        return task


class ConferenceRealtime:
    """Session lifecycle, provider callbacks and the utterance bridge.

    Inert until :meth:`configure` installs a ``ConferenceRealtimeConfig``:
    every provider callback and every entry point reads the configuration
    first and stands down without one, which is what makes the hot-plug's
    deactivate-first ordering safe against a provider still emitting.
    """

    def __init__(
        self,
        *,
        channel_id: str,
        bot_identity: str,
        voice: ConferenceVoice,
        operations: ConferenceOperations,
        ensure_bot: EnsureBot,
        binding_of: BindingOf,
    ) -> None:
        self._channel_id = channel_id
        # Tools the config gives twice (RFC §21.1), each reported once.
        self._collisions = CollisionLog(channel_id)
        self._bot_identity = bot_identity
        self._voice = voice
        self._operations = operations
        self._ensure_bot = ensure_bot
        self._binding_of = binding_of
        self._config: ConferenceRealtimeConfig | None = None
        # The person's tools of the configuration in force (RFC §9.3).
        self._human_input = ChannelHumanInput(None, ChannelType.CONFERENCE)
        self._framework: RoomKit | None = None
        self._rooms: dict[str, _RoomRealtime] = {}
        self._tools = ConferenceToolGate(channel_id)
        # The tool calls in flight, per session, each delivered and reported once.
        self._tool_calls = ToolCallBook()
        self._reports: set[asyncio.Task[None]] = set()
        # The calls an ending spared (their handler caused it) that may still
        # run: off the books with their room, waited for or cut at close.
        self._spared_calls = SparedCalls()
        # Providers register callbacks append-only, so each instance is wired
        # exactly once, ever — a re-plug of the same provider reuses the
        # registration, and the per-session identity guards make callbacks
        # for a session this channel no longer holds a no-op.
        self._wired: dict[int, RealtimeVoiceProvider] = {}
        self.mixer = ConferenceMixer(send=self.send_mixed)

    @property
    def channel_id(self) -> str:
        return self._channel_id

    @property
    def config(self) -> ConferenceRealtimeConfig | None:
        """The configuration in force, or ``None`` when nothing is plugged."""
        return self._config

    def set_framework(self, framework: RoomKit) -> None:
        self._framework = framework
        self._tools.set_framework(framework)
        self._register_human_input()

    def _register_human_input(self) -> None:
        """Announce the person's requests through the kit's
        ``ON_USER_INPUT_REQUIRED`` hooks, once both are known."""
        human, framework = self._human_input, self._framework
        if human.given and framework is not None:
            hook = framework._build_on_user_input_required_hook(self._channel_id)
            human.register(self._channel_id, hook)

    async def close_human_input(self) -> None:
        """Settle the person's requests the unplug or the close left open:
        the configuration that asked them is gone."""
        human = self._human_input
        self._human_input = ChannelHumanInput(None, ChannelType.CONFERENCE)
        await human.close(self._channel_id)

    def session_for(self, room_id: str) -> VoiceSession | None:
        """The provider session serving a room, if one is connected."""
        room = self._rooms.get(room_id)
        return None if room is None else room.session

    # -------------------------------------------------------------------------
    # Configuration — the plug's two halves
    # -------------------------------------------------------------------------

    def configure(self, config: ConferenceRealtimeConfig) -> None:
        """Install a configuration: the mixer runs and barge-ins reach the provider."""
        provider = config.provider
        if id(provider) not in self._wired:
            self._wired[id(provider)] = provider
            provider.on_audio(self._on_audio)
            provider.on_transcription(self._on_transcription)
            provider.on_response_start(self._on_response_start)
            provider.on_response_end(self._on_response_end)
            provider.on_speech_start(partial(self._on_hearing, hearing=True))
            provider.on_speech_end(partial(self._on_hearing, hearing=False))
            provider.on_tool_call(self._on_tool_call)
            provider.on_tool_call_cancelled(self._on_tool_call_cancelled)
            provider.on_error(self._on_provider_error)
        warn_unused_role_overrides(config, self._channel_id)
        warn_tools_uncallable(config.tools, "tool(s)", config.provider, self._channel_id)
        self._config = config
        self._human_input = ChannelHumanInput(config.human_input_handler, ChannelType.CONFERENCE)
        definitions = self._human_input.definitions
        warn_tools_uncallable(
            definitions, "human-input tool(s)", config.provider, self._channel_id
        )
        self._register_human_input()
        self.mixer.configure(input_sample_rate=config.input_sample_rate)
        self._voice.set_on_interrupted(self.interrupt)

    def deactivate(self) -> list[VoiceSession]:
        """Take the configuration out, returning the sessions left to disconnect.

        Everything downstream goes quiet at once: the mixer stops feeding,
        provider callbacks find no configuration and stand down, responses in
        flight are given their end so no pump waits on audio that will never
        come. The sessions come back by value — the unplug disconnects them
        with the provider it still holds, exactly as a detach does.
        """
        self._config = None
        self._voice.set_on_interrupted(None)
        self.mixer.deactivate()
        sessions: list[VoiceSession] = []
        for room_id in list(self._rooms):
            session = self.detach_room(room_id)
            if session is not None:
                sessions.append(session)
        return sessions

    # -------------------------------------------------------------------------
    # The session — lazily connected, scoped to the bot
    # -------------------------------------------------------------------------

    async def ensure_session(self, room_id: str) -> VoiceSession | None:
        """The room's provider session, connecting it if this is the first need.

        ``None`` where there is nothing to connect or connecting failed. A
        failure never propagates — the lazy-join discipline of RFC 12.10.4,
        held here for the session behind the join — and the cooldown is what
        keeps a down provider from being redialed every mixing window.
        """
        config = self._config
        if config is None:
            return None
        room = self._room(room_id)
        if room.session is not None:
            return room.session
        loop = asyncio.get_running_loop()
        if loop.time() < room.next_connect_at:
            return None
        async with room.connecting:
            if self._config is not config:
                return None
            if room.session is not None:
                return room.session
            if loop.time() < room.next_connect_at:
                return None
            try:
                bot = await self._ensure_bot(room_id)
            except Exception:
                room.hold_off()
                logger.debug(
                    "Conference channel %r has no bot to hang a realtime session on in "
                    "room %s; the provider stays unconnected",
                    self._channel_id,
                    room_id,
                    exc_info=True,
                )
                return None
            session = await self._connect_provider(config, room, room_id, bot)
            if session is not None:
                room.session = session
                self._serve_start_calls(room)
            return session

    async def _connect_provider(
        self,
        config: ConferenceRealtimeConfig,
        room: _RoomRealtime,
        room_id: str,
        bot: BotSession,
    ) -> VoiceSession | None:
        """Connect the provider for a new session of *room_id*: the session,
        or ``None`` once a start that failed is let go. The calls it issues
        meanwhile wait in ``room.start_calls``; a start cancelled is let go
        the same way before the cancellation goes on."""
        session = VoiceSession(
            id=f"conf-rt-{uuid.uuid4().hex}",
            room_id=room_id,
            participant_id=self._bot_identity,
            channel_id=self._channel_id,
            metadata={"bot_session_id": bot.id},
        )
        room.starting = session
        try:
            await self._connect(config, session)
        except asyncio.CancelledError:
            self._fail_start(config, room, session)
            raise
        except Exception:
            logger.warning(
                "Conference channel %r could not connect its realtime provider for "
                "room %s; retrying on the next need after %.0fs",
                self._channel_id,
                room_id,
                CONNECT_COOLDOWN_S,
                exc_info=True,
            )
            self._fail_start(config, room, session)
            return None
        finally:
            room.starting = None
        if session.state == VoiceSessionState.ENDED:
            # The provider ended it while connecting, and said so through its
            # error callback: there is no session to hold.
            self._fail_start(config, room, session)
            return None
        return session

    async def _connect(self, config: ConferenceRealtimeConfig, session: VoiceSession) -> None:
        """Open *session* on the provider with the configuration in force."""
        with self._operations.use(
            ConferenceResource.REALTIME,
            what=f"connecting the realtime provider for room {session.room_id}",
        ):
            self._human_input.warn_unoffered(catalogue(config) or [], self._channel_id)
            await config.provider.connect(
                session,
                system_prompt=config.system_prompt,
                voice=config.voice,
                tools=declared_tools(config, self._collisions),
                temperature=config.temperature,
                input_sample_rate=config.input_sample_rate,
                output_sample_rate=config.output_sample_rate,
                server_vad=config.server_vad,
                provider_config=config.provider_config,
            )

    def _fail_start(
        self, config: ConferenceRealtimeConfig, room: _RoomRealtime, session: VoiceSession
    ) -> None:
        """Let a start that failed go, as a realtime voice channel rolls one
        back: the cooldown armed, each call the provider issued meanwhile
        reported once, cancelled (RFC §12.4), the provider told to disconnect."""
        room.hold_off()
        calls, room.start_calls = room.start_calls, []
        if calls:
            self._track_report(report_interrupted_calls(self, calls, SESSION_ENDED))
        self._track_report(self._disconnect(config.provider, session))

    def _serve_start_calls(self, room: _RoomRealtime) -> None:
        """Serve the calls the provider issued while the room's session
        started, now that it is the room's (RFC §12.4)."""
        calls, room.start_calls = room.start_calls, []
        for call in calls:
            self._take_call(call)

    async def send_mixed(self, room_id: str, data: bytes) -> None:
        """The mixer's sender: one mixed window to the provider.

        A send failure propagates — the mixer logs it and keeps its clock —
        and a session that could not be established is silence the provider
        never notices it missed.
        """
        config = self._config
        if config is None:
            return
        session = await self.ensure_session(room_id)
        if session is None:
            return
        with self._operations.use(
            ConferenceResource.REALTIME, what=f"mixed audio for room {room_id}"
        ):
            await config.provider.send_audio(session, data)

    async def deliver_text(
        self,
        room_id: str,
        text: str,
        *,
        role: str,
        silent: bool,
        chain_depth: int = 0,
        injected_from: RoomEvent | None = None,
    ) -> None:
        """Inject a broadcast text event into the provider's context.

        The realtime counterpart of speaking it: a 1:1 realtime channel
        injects rather than synthesizes, and the conference follows suit.
        Contained, because a provider that cannot take the text right now
        must not fail the broadcast that carried it. The model's answer to
        it is one deeper than the event (RFC §12.10.12); *silent* asks none,
        as a muted binding does on a realtime voice channel (RFC §7.5).
        """
        if self._config is None:
            return
        session = await self.ensure_session(room_id)
        if session is None:
            return
        try:
            await self.inject_text(
                session,
                text,
                role=role,
                silent=silent,
                chain_depth=chain_depth,
                injected_from=injected_from,
            )
        except Exception:
            logger.warning(
                "Conference channel %r could not inject a text event into the realtime "
                "session of room %s",
                self._channel_id,
                room_id,
                exc_info=True,
            )

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str,
        silent: bool = False,
        chain_depth: int = 0,
        injected_from: RoomEvent | None = None,
    ) -> VoiceInjectionResult | None:
        """Inject *text* into the room session *session*, as a realtime voice
        channel injects into its own (RFC §12.4): the model's answer is one
        deeper than *chain_depth* unless *silent*; a broadcast names the event
        it came from to the hook (*injected_from*). Not sent when the session
        is no longer the room's (an unplug, a detach, a reconnect)."""
        config = self._config
        if config is None or self._guarded(session) is None:
            return VoiceInjectionResult(status="not_sent", reason="realtime_session_gone")
        with self._operations.use(
            ConferenceResource.REALTIME, what=f"text injection for room {session.room_id}"
        ):
            result = await config.provider.inject_text(session, text, role=role, silent=silent)
        if result is None or result.status != "sent":
            return result
        room = self._rooms.get(session.room_id)
        if room is not None and not silent:
            room.answer_depth.injected(chain_depth)
        source = self._source(config, session)
        await fire_text_injected(
            self._framework, source, session, text, role=role, injected_from=injected_from
        )
        return result

    # -------------------------------------------------------------------------
    # Barge-in — the latch's upstream half
    # -------------------------------------------------------------------------

    async def interrupt(self, room_id: str) -> None:
        """Carry a landed barge-in to the provider (ConferenceVoice's tap).

        The latch has stopped the pump and ``stop_playback`` has silenced the
        backend by the time this runs; what is left is the generation. The
        response in flight is discarded first, so deltas the provider emits
        before the cancellation lands are dropped rather than queued, then the
        provider is told — best-effort by the ABC: a provider that cannot
        cancel simply finishes into the discard.
        """
        config = self._config
        room = self._rooms.get(room_id)
        if config is None or room is None:
            return
        utterance = room.utterance
        if utterance is not None and not utterance.discarded:
            utterance.discarded = True
            utterance.finish()
            room.settle()
        session = room.session
        if session is None:
            return
        with self._operations.use(
            ConferenceResource.REALTIME, what=f"interrupting the response in room {room_id}"
        ):
            await config.provider.interrupt(session)

    # -------------------------------------------------------------------------
    # Provider callbacks — each one guards its session first
    # -------------------------------------------------------------------------

    def _guarded(self, session: VoiceSession) -> _RoomRealtime | None:
        """The room a callback belongs to, or ``None`` when it is stale.

        A callback carries the session the provider fired it for; a room that
        holds a different one — after an unplug, a detach, a reconnect — makes
        the callback a leftover of a session this channel no longer speaks
        for, and it stands down.
        """
        if self._config is None:
            return None
        room = self._rooms.get(session.room_id)
        if room is None or room.session is not session:
            return None
        return room

    async def _on_response_start(self, session: VoiceSession) -> None:
        room = self._guarded(session)
        if room is None:
            return
        self._open_utterance(room, session.room_id)

    def _open_utterance(self, room: _RoomRealtime, room_id: str) -> _Utterance:
        # A response the provider never closed is closed here: its pump would
        # otherwise wait forever on a queue nothing feeds, holding the turn
        # against the response that just started.
        previous = room.utterance
        if previous is not None and not previous.discarded:
            previous.finish()
        utterance = _Utterance()
        room.utterance = utterance
        room.speaking += 1
        room.settle()
        room.spawn(self._speak(room, room_id, utterance))
        return utterance

    async def _speak(self, room: _RoomRealtime, room_id: str, utterance: _Utterance) -> None:
        """Run one response through its turn on the bot track, start to terminal chunk."""

        def attach(playback: ConferencePlayback) -> None:
            utterance.playback = playback

        try:
            await self._voice.speak_stream(room_id, self._chunks(utterance), on_playback=attach)
        except Exception:
            logger.exception(
                "Conference channel %r could not publish a realtime response in room %s",
                self._channel_id,
                room_id,
            )
        finally:
            room.speaking -= 1
            room.settle()

    async def _chunks(self, utterance: _Utterance) -> AsyncIterator[AudioChunk]:
        while True:
            chunk = await utterance.queue.get()
            if chunk is None:
                return
            yield chunk

    def _on_audio(self, session: VoiceSession, audio: bytes) -> None:
        config = self._config
        room = self._guarded(session)
        if config is None or room is None:
            return
        binding = self._binding_of(session.room_id)
        if binding is not None and binding.output_muted:
            # The provider's audio is not forwarded under an output-muted
            # binding, as on a realtime voice channel (RFC §5).
            return
        utterance = room.utterance
        if utterance is None:
            # Nothing promised on_response_start in the ABC: audio with no
            # open response opens one, so a provider that skips the callback
            # still speaks.
            utterance = self._open_utterance(room, session.room_id)
        if utterance.discarded:
            return
        utterance.queue.put_nowait(AudioChunk(data=audio, sample_rate=config.output_sample_rate))

    async def _on_response_end(self, session: VoiceSession) -> None:
        room = self._guarded(session)
        if room is None:
            return
        utterance = room.utterance
        if utterance is None:
            return
        room.utterance = None
        if not utterance.discarded:
            utterance.finish()
        room.settle()

    async def _on_hearing(self, session: VoiceSession, *, hearing: bool) -> None:
        """The provider's VAD heard the room's people start or stop speaking."""
        room = self._guarded(session)
        if room is None:
            return
        room.hearing = hearing
        room.settle()

    async def wait_idle(self, room_id: str, timeout: float = 15.0) -> None:
        """Wait until nothing is in flight in the room (RFC §22.2): the model's
        last answer has ended and reached the bot track, or a barge-in cut it,
        and its VAD hears nobody speak. A room with no session is idle."""
        room = self._rooms.get(room_id)
        if room is not None and not room.idle.is_set():
            await asyncio.wait_for(room.idle.wait(), timeout=timeout)

    async def _on_transcription(
        self, session: VoiceSession, text: str, role: str, is_final: bool
    ) -> None:
        """Keep the assistant's words; drop the provider's guess at the room's.

        User-role transcriptions are unattributed by construction — the
        provider heard a mix — and are discarded (RFC 12.10.12); the lanes'
        STT is the attributed transcript. Assistant partials keep the active
        playback's text abreast for ON_BARGE_IN; assistant finals become the
        room's record of what the AI said.
        """
        room = self._guarded(session)
        if room is None:
            return
        if role != "assistant":
            # Discarded, but it says the model heard the room's people,
            # whose words open a chain.
            room.answer_depth.user_spoke()
            return
        utterance = room.utterance
        if not is_final:
            if utterance is not None and text:
                utterance.transcript += text
                if utterance.playback is not None:
                    utterance.playback.text = utterance.transcript
            return
        final = text.strip()
        if not final:
            return
        if utterance is not None and utterance.playback is not None:
            utterance.playback.text = final
        await self._emit_assistant_text(session.room_id, final, room.answer_depth.answer)

    async def _emit_assistant_text(self, room_id: str, text: str, chain_depth: int) -> None:
        config = self._config
        if config is None or self._framework is None:
            return
        try:
            await self._framework.send_event(
                room_id,
                self._channel_id,
                TextContent(body=text),
                chain_depth=chain_depth,
                metadata={"source": "conference_realtime", "role": "assistant"},
                provider=config.provider.name,
            )
        except Exception:
            logger.warning(
                "Conference channel %r could not record what its realtime provider said "
                "in room %s; the words were heard on the bot track and are absent from "
                "the room's events",
                self._channel_id,
                room_id,
                exc_info=True,
            )

    async def _on_tool_call(
        self, session: VoiceSession, call_id: str, name: str, arguments: dict[str, Any] | str
    ) -> None:
        call = RealtimeToolCall.from_provider(
            session, call_id, name, arguments, room_id=session.room_id
        )
        if not self._held_for_start(call):
            self._take_call(call)

    def _held_for_start(self, call: RealtimeToolCall) -> bool:
        """Whether *call* waits for its session's start to end: issued while
        ``connect()`` runs, it is served once the session is the room's."""
        room = self._rooms.get(call.room_id or "")
        if self._config is None or room is None or room.starting is not call.session:
            return False
        room.start_calls.append(call)
        return True

    def _take_call(self, call: RealtimeToolCall) -> None:
        """Serve *call* on its room, or report it when its session is no
        longer the room's."""
        room = self._guarded(call.session)
        if room is None:
            self._report_stale_call(call)
            return
        if not self._tool_calls.open(call):
            # No result can name it: refused on the path of any call, nothing
            # sent. Tracked beside the teardown, not on the room: a detach or
            # an unplug does not cut it, and on a room gone it is reported
            # cancelled (RFC §12.4).
            self._track_report(
                serve_unbooked(self, call, lambda: self._answer_tool(call), _LEFT_THE_ROOM)
            )
            return
        call.task = room.spawn(self._answer_tool(call))
        call.task.add_done_callback(lambda _: self._tool_calls.close(call))

    def _on_provider_error(self, session: VoiceSession, code: str, message: str) -> None:
        """The provider failed *session*: ON_ERROR (``realtime_provider``) for
        every failure, and a session it ended is dropped from its room, as on
        a realtime voice channel (RFC §12.5). Scheduled, never awaited: this
        runs inside the provider's receive loop."""
        config = self._config
        room = self._rooms.get(session.room_id)
        if config is None or room is None:
            return
        if room.session is not session and room.starting is not session:
            return
        logger.error(
            "Conference channel %r: realtime provider error for room %s: [%s] %s",
            self._channel_id,
            session.room_id,
            code,
            message,
        )
        self._announce_error(config, session, code, message)
        if session.state == VoiceSessionState.ENDED and room.session is session:
            self._drop_ended_session(config, room, session)

    def _announce_error(
        self, config: ConferenceRealtimeConfig, session: VoiceSession, code: str, message: str
    ) -> None:
        """Fire ON_ERROR (``realtime_provider``) for the provider's failure of
        *session*, beside the teardown."""
        self._track_report(
            fire_session_error(
                self._framework,
                self._source(config, session),
                session,
                error=message,
                error_type=code,
                category="realtime_provider",
            )
        )

    def _source(self, config: ConferenceRealtimeConfig, session: VoiceSession) -> EventSource:
        """The conference as the source of what its realtime session does."""
        return EventSource(
            channel_id=self._channel_id,
            channel_type=ChannelType.CONFERENCE,
            participant_id=session.participant_id,
            provider=config.provider.name,
        )

    def _drop_ended_session(
        self, config: ConferenceRealtimeConfig, room: _RoomRealtime, session: VoiceSession
    ) -> None:
        """Take a session its provider ended off its room, as a detach takes
        it off; the room stays, and its next need reconnects after the
        cooldown."""
        self._end_room_session(room, SESSION_ENDED, [])
        room.hearing = False
        room.hold_off()
        room.settle()
        self._track_report(self._disconnect(config.provider, session))

    async def _on_tool_call_cancelled(self, session: VoiceSession, call_ids: list[str]) -> None:
        """The model will not read these calls' results: interrupt them as a
        realtime voice channel does (:func:`abandon_calls`), send nothing, and
        report each to ON_TOOL_CALL's observers as cancelled (RFC §12.4)."""
        if self._abandon_start_calls(session, call_ids):
            return
        if self._guarded(session) is None:
            return
        for call in abandon_calls(self._tool_calls, session.id, call_ids):
            # Off the provider's callback: an audit hook must not hold up the
            # interruption it reports. Tracked beside the teardown, as a
            # detach's reports are: a detach does not cut it.
            self._track_report(report_cancelled_call(self, call, ABANDONED_BY_PROVIDER))

    def _abandon_start_calls(self, session: VoiceSession, call_ids: list[str]) -> bool:
        """Whether *session* is still starting; its held calls the provider
        abandoned then leave the hold unserved, each reported once, cancelled
        (RFC §12.4), as a realtime voice channel's start journal plays them."""
        room = self._rooms.get(session.room_id)
        if self._config is None or room is None or room.starting is not session:
            return False
        ids = set(call_ids)
        abandoned = [call for call in room.start_calls if call.call_id in ids]
        room.start_calls = [call for call in room.start_calls if call.call_id not in ids]
        for call in abandoned:
            self._track_report(report_cancelled_call(self, call, ABANDONED_BY_PROVIDER))
        return True

    async def _answer_tool(self, call: RealtimeToolCall) -> None:
        """Answer one tool call through the realtime tool executor.

        A refused or failing call still submits a result: the provider's turn
        is waiting on it, and a turn nothing answers wedges the conversation.
        A refusal is reported once it is on the wire, never before (RFC 12.4).
        A call the realtime's unplug left with nothing to serve it is
        reported, cancelled.
        """
        config = self._config
        if config is None:
            await report_cancelled_call(self, call, _LEFT_THE_ROOM)
            return
        outcome = await run_tool_call(self, call, _ConferenceDoor(self, config))
        logger.info(
            "Tool call %s(%s) %s for room %s", call.name, call.call_id, outcome.kind, call.room_id
        )

    # -- ToolCallHost: the steps the executor serves a call with -------------

    def _tool_framework(self, call: RealtimeToolCall) -> RoomKit | None:
        return self._framework if call.room_id else None

    def _tool_event(self, call: RealtimeToolCall, result: str | None) -> ToolCallEvent:
        return self._tools.event(call, result)

    async def _authorize_call(
        self, call: RealtimeToolCall, door: ToolCallDoor
    ) -> tuple[GateRefusal | None, RoomContext | None]:
        config = self._config
        if config is None:
            return GateRefusal(
                json.dumps({"error": "The conference has no realtime model."})
            ), None
        return await self._tools.authorize(config, call), None

    def _call_ended(self, call: RealtimeToolCall) -> bool:
        room = self._rooms.get(call.room_id or "")
        return room is None or room.session is not call.session

    async def _serve_channel_tool(
        self, call: RealtimeToolCall, door: ToolCallDoor, carrying: RoomContext | None
    ) -> ToolOutcome | None:
        return None  # a conference serves no tool of its own (RFC §21.1)

    async def _answer_call(self, call: RealtimeToolCall, carrying: RoomContext | None) -> str:
        """The answer of the person's tools, else of the configured handler,
        inside the call's tool call context, at the depth of the answer that
        issued it (RFC §21.4, §8.3)."""
        config = self._config
        server = self._server(config, call) if config is not None else None
        if config is None or server is None:
            # Nothing serves it: unserved, which the hooks may still serve,
            # as on every channel (RFC §9.3, §21.4).
            raise UnservedToolCallError(call.name)
        serve, asks = server
        loop_ctx = await self._call_context(config, str(call.room_id))
        with serving_tool_call(call, self._channel_id, loop_ctx):
            bound = config.tool_bound(call.name, waits=asks)
            answered = await answer_within(bound, call.name, serve())
        # A person's answer is what they said: only the host's may be the
        # "not mine" envelope (RFC §21.4).
        return result_text(answered if asks else declined_answer(answered, call.name))

    def _server(
        self, config: ConferenceRealtimeConfig, call: RealtimeToolCall
    ) -> tuple[Callable[[], Awaitable[Any]], bool] | None:
        """What serves *call*, and whether it asks a person: the person's
        tools before the configured handler; ``None`` when nothing does."""
        human = self._human_input
        if human.serves(call.name):
            return partial(human.serve, call.name, call.arguments), True
        handler = config.tool_handler
        if handler is None:
            return None
        return partial(handler, str(call.room_id), call.name, call.arguments), False

    async def _call_context(
        self, config: ConferenceRealtimeConfig, room_id: str
    ) -> _ToolLoopContext:
        """The tool call context a call of *room_id* is served in: the room,
        the depth of its answer, and the session's declared toolset."""
        room = self._rooms.get(room_id)
        loop_ctx = await tool_loop_context(
            self._framework,
            room_id,
            actor_id=None,  # the mix names no participant
            chain_depth=room.answer_depth.answer if room is not None else 0,
        )
        # What the session declares is the call's resolved toolset (RFC §21.4).
        # A session that declares none names no list, its gate still judging.
        declared = declared_tools(config, self._collisions)
        if declared:
            loop_ctx.all_context_tools = [
                schema_tool(tool)
                for tool in declared
                if dict_tool_name(tool)  # a provider's native tool has no name
            ]
        return loop_ctx

    def _bound_call_result(self, call: RealtimeToolCall, text: str, *, served: bool = True) -> str:
        return bound_result(text, call.name)

    async def _submit_tool_result(
        self, config: ConferenceRealtimeConfig, call: RealtimeToolCall, outcome: ToolOutcome
    ) -> bool:
        if self._call_ended(call):
            # A call its own ending spared: its room's session is gone, and
            # nothing is left to answer.
            return False
        try:
            with self._operations.use(
                ConferenceResource.REALTIME, what=f"tool result for room {call.room_id}"
            ):
                await submit_tool_outcome(
                    config.provider,
                    call.session,
                    call.call_id,
                    result_text(outcome.result),
                    failed=outcome.failed,
                )
        except Exception:
            logger.warning(
                "Conference channel %r could not return the result of tool %r to the "
                "realtime provider in room %s",
                self._channel_id,
                call.name,
                call.room_id,
                exc_info=True,
            )
            return False
        return True

    # -------------------------------------------------------------------------
    # Lifecycle — the session ends where the bot does
    # -------------------------------------------------------------------------

    def _room(self, room_id: str) -> _RoomRealtime:
        room = self._rooms.get(room_id)
        if room is None:
            room = self._rooms[room_id] = _RoomRealtime()
        return room

    def detach_room(self, room_id: str) -> VoiceSession | None:
        """Take a room off the books, returning the session to disconnect.

        Synchronous bookkeeping only, by value — the awaited disconnect
        belongs to the caller's teardown, so a teardown deferred past a
        re-attach can never disconnect the session a new attachment minted.
        """
        self.mixer.forget_room(room_id)
        room = self._rooms.pop(room_id, None)
        if room is None:
            return None
        session = self._end_room_session(room, _LEFT_THE_ROOM, list(room.tasks))
        # A room off the books has nothing in flight: a delivery waiting on it
        # goes on to find its session gone.
        room.idle.set()
        return session

    def _end_room_session(
        self, room: _RoomRealtime, why: str, tasks: list[asyncio.Task[None]]
    ) -> VoiceSession | None:
        """End the room's session for *why*: the response in flight discarded,
        its calls and *tasks* cut but the call that caused the ending, each
        cut call reported once, cancelled (RFC §12.4). The session ended.

        The reports run beside the teardown; a disconnect of the channel's
        (a detach's, an unplug's, the close's) waits for them, so none races
        the store's release at close.
        """
        utterance, room.utterance = room.utterance, None
        if utterance is not None and not utterance.discarded:
            utterance.discarded = True
            utterance.finish()
        session, room.session = room.session, None
        calls = self._tool_calls.take(session.id) if session is not None else []
        interrupted, _ = interrupt_for_ending(calls, tasks)
        self._spared_calls.keep(call for call in calls if call not in interrupted)
        if interrupted:
            self._track_report(report_interrupted_calls(self, interrupted, why))
        return session

    def _report_stale_call(self, call: RealtimeToolCall) -> None:
        """Report a call a session this channel no longer speaks for issued
        (after a detach or a reconnect): nothing serves it and nothing is
        sent, and it still gets its one report, cancelled (RFC §9.3), after
        a detach as after an unplug: the report needs no realtime config."""
        self._track_report(report_cancelled_call(self, call, _LEFT_THE_ROOM))

    def _track_report(self, coro: Awaitable[None]) -> None:
        """Run a report, an announcement or a disconnect beside the teardown;
        a disconnect of the channel's waits for it."""
        report = asyncio.ensure_future(_run_tracked(coro))
        self._reports.add(report)
        report.add_done_callback(self._reports.discard)
        report.add_done_callback(log_task_exception)

    async def settle_spared(self) -> None:
        """At the channel's close, wait for the calls an ending spared that
        still run, then cut and report the rest (RFC §12.4)."""
        await self._spared_calls.settle(self, _LEFT_THE_ROOM)

    async def _settle_reports(self) -> None:
        """Wait for the tracked reports, but the one this code runs under: a
        hook that unplugs or closes from an ON_ERROR announcement would wait
        for itself. A wait cancelled leaves them running: they are not this
        waiter's to cut."""
        running = _RUNNING_REPORT.get()
        pending = [report for report in self._reports if report is not running]
        if pending:
            await asyncio.wait(pending)

    def abandon_all(self) -> list[VoiceSession]:
        """Every room off the books at once — the channel is closing."""
        sessions: list[VoiceSession] = []
        for room_id in list(self._rooms):
            session = self.detach_room(room_id)
            if session is not None:
                sessions.append(session)
        return sessions

    async def disconnect_detached(self, session: VoiceSession | None) -> None:
        """Disconnect one session ``detach_room`` returned. Best-effort.

        Quiet on a configuration already unplugged: the unplug disconnected
        everything it held, and a detach racing it has nothing left to do.
        """
        await self._settle_reports()
        config = self._config
        if session is None or config is None:
            return
        await self._disconnect(config.provider, session)

    async def disconnect_sessions(
        self, provider: RealtimeVoiceProvider, sessions: list[VoiceSession]
    ) -> None:
        """Disconnect what an unplug or a close took off the books, together.

        The provider arrives as an argument because the configuration is
        already gone by the time this runs — deactivate-first is what made
        the callbacks inert — and each failure is contained: one session the
        provider will not release is not a reason to leave the rest held.
        """
        await self._settle_reports()
        for session in sessions:
            await self._disconnect(provider, session)

    async def _disconnect(self, provider: RealtimeVoiceProvider, session: VoiceSession) -> None:
        """Disconnect one session, contained: a session the provider will not
        release is logged, never raised into the teardown."""
        try:
            with self._operations.use(
                ConferenceResource.REALTIME,
                what=f"disconnecting the realtime session of room {session.room_id}",
            ):
                await provider.disconnect(session)
        except Exception:
            logger.warning(
                "Conference channel %r could not disconnect the realtime session of "
                "room %s; the provider may still hold it",
                self._channel_id,
                session.room_id,
                exc_info=True,
            )

    async def close_provider(self) -> None:
        """Close the provider. The shutdown coordinator's closer for REALTIME."""
        config = self._config
        if config is not None:
            await config.provider.close()


async def _run_tracked(coro: Awaitable[None]) -> None:
    """Run a tracked report, naming its task to what runs under it."""
    _RUNNING_REPORT.set(asyncio.current_task())
    await coro


class _ConferenceDoor:
    """A conference provider's function call: its outcome goes back as the
    call's result, through the realtime resource."""

    channel_serves = False
    can_activate = False

    def __init__(self, realtime: ConferenceRealtime, config: ConferenceRealtimeConfig) -> None:
        self._realtime = realtime
        self._config = config

    async def deliver(self, call: RealtimeToolCall, outcome: ToolOutcome) -> bool:
        return await self._realtime._submit_tool_result(self._config, call, outcome)
