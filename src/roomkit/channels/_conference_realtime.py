"""The speech-to-speech provider as a conference's intelligence.

One RealtimeVoiceProvider session per conference: the mixer feeds it every
subscribed track as one stream, and its voice publishes on the bot track
through the same floor, latch and terminal-chunk machinery as TTS — a
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.channels._conference_mixer import ConferenceMixer
from roomkit.channels._conference_operations import ConferenceResource
from roomkit.core.task_utils import log_task_exception
from roomkit.models.event import TextContent
from roomkit.voice.base import AudioChunk, VoiceSession

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from roomkit.channels._conference_operations import ConferenceOperations
    from roomkit.channels._conference_voice import ConferencePlayback, ConferenceVoice
    from roomkit.conference.models import BotSession, ConferenceRealtimeConfig
    from roomkit.core.framework import RoomKit
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.conference")

# Opens the bot's media connection for a room — the channel's _ensure_bot.
EnsureBot = Callable[[str], "Awaitable[BotSession]"]

CONNECT_COOLDOWN_S = 5.0
"""How long a failed provider connect holds further attempts off.

The mixer asks again every window; without this a provider outage would be
retried fifty times a second from every conference the channel serves.
"""


@dataclass
class _Utterance:
    """One provider response on its way to the bot track.

    A queue-backed bridge between two pacings: the provider pushes audio
    deltas as it generates them, and the voice's pump pulls them through the
    floor at the backend's pace. ``None`` on the queue is the end of the
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


@dataclass
class _RoomRealtime:
    """One room's share of the provider: its session, and the response in flight."""

    session: VoiceSession | None = None
    connecting: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_connect_at: float = 0.0
    utterance: _Utterance | None = None
    tasks: set[asyncio.Task[None]] = field(default_factory=set)

    def spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(log_task_exception)


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
    ) -> None:
        self._channel_id = channel_id
        self._bot_identity = bot_identity
        self._voice = voice
        self._operations = operations
        self._ensure_bot = ensure_bot
        self._config: ConferenceRealtimeConfig | None = None
        self._framework: RoomKit | None = None
        self._rooms: dict[str, _RoomRealtime] = {}
        # Providers register callbacks append-only, so each instance is wired
        # exactly once, ever — a re-plug of the same provider reuses the
        # registration, and the per-session identity guards make callbacks
        # for a session this channel no longer holds a no-op.
        self._wired: dict[int, RealtimeVoiceProvider] = {}
        self.mixer = ConferenceMixer(send=self.send_mixed)

    @property
    def config(self) -> ConferenceRealtimeConfig | None:
        """The configuration in force, or ``None`` when nothing is plugged."""
        return self._config

    def set_framework(self, framework: RoomKit) -> None:
        self._framework = framework

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
            provider.on_tool_call(self._on_tool_call)
        self._config = config
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
                room.next_connect_at = loop.time() + CONNECT_COOLDOWN_S
                logger.debug(
                    "Conference channel %r has no bot to hang a realtime session on in "
                    "room %s; the provider stays unconnected",
                    self._channel_id,
                    room_id,
                    exc_info=True,
                )
                return None
            session = VoiceSession(
                id=f"conf-rt-{uuid.uuid4().hex}",
                room_id=room_id,
                participant_id=self._bot_identity,
                channel_id=self._channel_id,
                metadata={"bot_session_id": bot.id},
            )
            try:
                with self._operations.use(
                    ConferenceResource.REALTIME,
                    what=f"connecting the realtime provider for room {room_id}",
                ):
                    await config.provider.connect(
                        session,
                        system_prompt=config.system_prompt,
                        voice=config.voice,
                        tools=config.tools,
                        temperature=config.temperature,
                        input_sample_rate=config.input_sample_rate,
                        output_sample_rate=config.output_sample_rate,
                        server_vad=config.server_vad,
                        provider_config=config.provider_config,
                    )
            except Exception:
                room.next_connect_at = loop.time() + CONNECT_COOLDOWN_S
                logger.warning(
                    "Conference channel %r could not connect its realtime provider for "
                    "room %s; retrying on the next need after %.0fs",
                    self._channel_id,
                    room_id,
                    CONNECT_COOLDOWN_S,
                    exc_info=True,
                )
                return None
            room.session = session
            return session

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

    async def deliver_text(self, room_id: str, text: str, *, role: str) -> None:
        """Inject a broadcast text event into the provider's context.

        The realtime counterpart of speaking it: a 1:1 realtime channel
        injects rather than synthesizes, and the conference follows suit.
        Contained, because a provider that cannot take the text right now
        must not fail the broadcast that carried it.
        """
        config = self._config
        if config is None:
            return
        session = await self.ensure_session(room_id)
        if session is None:
            return
        try:
            with self._operations.use(
                ConferenceResource.REALTIME, what=f"text injection for room {room_id}"
            ):
                await config.provider.inject_text(session, text, role=role)
        except Exception:
            logger.warning(
                "Conference channel %r could not inject a text event into the realtime "
                "session of room %s",
                self._channel_id,
                room_id,
                exc_info=True,
            )

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
        # otherwise wait forever on a queue nothing feeds, holding the floor
        # against the response that just started.
        previous = room.utterance
        if previous is not None and not previous.discarded:
            previous.finish()
        utterance = _Utterance()
        room.utterance = utterance
        room.spawn(self._speak(room_id, utterance))
        return utterance

    async def _speak(self, room_id: str, utterance: _Utterance) -> None:
        """Run one response through the voice's floor, start to terminal chunk."""

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
        if room is None or role != "assistant":
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
        await self._emit_assistant_text(session.room_id, final)

    async def _emit_assistant_text(self, room_id: str, text: str) -> None:
        config = self._config
        if config is None or self._framework is None:
            return
        try:
            await self._framework.send_event(
                room_id,
                self._channel_id,
                TextContent(body=text),
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
        self, session: VoiceSession, call_id: str, name: str, arguments: dict[str, Any]
    ) -> None:
        room = self._guarded(session)
        if room is None:
            return
        room.spawn(self._answer_tool(session, call_id, name, arguments))

    async def _answer_tool(
        self, session: VoiceSession, call_id: str, name: str, arguments: dict[str, Any]
    ) -> None:
        """Answer one tool call, with an error result rather than silence.

        A handler that raises — or a call for which none was configured —
        submits what went wrong: the provider's turn is waiting on this
        result, and a turn nothing answers wedges the conversation.
        """
        config = self._config
        if config is None:
            return
        if config.tool_handler is None:
            result = json.dumps({"error": f"no handler is configured for tool {name!r}"})
        else:
            try:
                result = await config.tool_handler(session.room_id, name, arguments)
            except Exception as error:
                logger.exception(
                    "Conference channel %r: the tool handler failed on %r in room %s",
                    self._channel_id,
                    name,
                    session.room_id,
                )
                result = json.dumps({"error": f"{type(error).__name__}: {error}"})
        try:
            with self._operations.use(
                ConferenceResource.REALTIME, what=f"tool result for room {session.room_id}"
            ):
                await config.provider.submit_tool_result(session, call_id, result)
        except Exception:
            logger.warning(
                "Conference channel %r could not return the result of tool %r to the "
                "realtime provider in room %s",
                self._channel_id,
                name,
                session.room_id,
                exc_info=True,
            )

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
        utterance = room.utterance
        if utterance is not None and not utterance.discarded:
            utterance.discarded = True
            utterance.finish()
        for task in list(room.tasks):
            task.cancel()
        session, room.session = room.session, None
        return session

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
        config = self._config
        if session is None or config is None:
            return
        try:
            with self._operations.use(
                ConferenceResource.REALTIME,
                what=f"disconnecting the realtime session of room {session.room_id}",
            ):
                await config.provider.disconnect(session)
        except Exception:
            logger.warning(
                "Conference channel %r could not disconnect the realtime session of "
                "room %s; the provider may still hold it",
                self._channel_id,
                session.room_id,
                exc_info=True,
            )

    async def disconnect_sessions(
        self, provider: RealtimeVoiceProvider, sessions: list[VoiceSession]
    ) -> None:
        """Disconnect what an unplug or a close took off the books, together.

        The provider arrives as an argument because the configuration is
        already gone by the time this runs — deactivate-first is what made
        the callbacks inert — and each failure is contained: one session the
        provider will not release is not a reason to leave the rest held.
        """
        for session in sessions:
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
