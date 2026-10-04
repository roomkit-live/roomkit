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
from roomkit.channels._realtime_text_injected import fire_text_injected
from roomkit.channels._realtime_tool_calls import RealtimeToolCall, ToolCallBook
from roomkit.channels._realtime_tool_executor import (
    ABANDONED_BY_PROVIDER,
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
from roomkit.models.event import EventSource, TextContent
from roomkit.models.tool_call import ToolCallEvent
from roomkit.tools._human_input_channel import ChannelHumanInput
from roomkit.tools._outcome import ToolOutcome
from roomkit.tools.result import GateRefusal, declined_answer, result_text
from roomkit.tools.timeout import answer_within
from roomkit.voice.base import AudioChunk, VoiceSession
from roomkit.voice.realtime._answer_depth import AnswerDepth
from roomkit.voice.realtime.injection import VoiceInjectionResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from roomkit.channels._conference_operations import ConferenceOperations
    from roomkit.channels._conference_voice import ConferencePlayback, ConferenceVoice
    from roomkit.conference.models import BotSession, ConferenceRealtimeConfig
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.tools.context import _ToolLoopContext
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.conference")

_LEFT_THE_ROOM = "The conference left the room"
"""Why a call a detach cut was cancelled before its result."""

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
    utterance: _Utterance | None = None
    tasks: set[asyncio.Task[None]] = field(default_factory=set)
    answer_depth: AnswerDepth = field(default_factory=AnswerDepth)
    """What the room's model heard last, which its answer's depth follows."""
    speaking: int = 0
    """Responses still on their way through the floor to the bot track."""
    hearing: bool = False
    """The provider's own VAD hears the room's people speak."""
    idle: asyncio.Event = field(default_factory=_idle_event)
    """Set while nothing is in flight in the room (RFC §22.2)."""

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
    ) -> None:
        self._channel_id = channel_id
        # Tools the config gives twice (RFC §21.1), each reported once.
        self._collisions = CollisionLog(channel_id)
        self._bot_identity = bot_identity
        self._voice = voice
        self._operations = operations
        self._ensure_bot = ensure_bot
        self._config: ConferenceRealtimeConfig | None = None
        # The person's tools of the configuration in force (RFC §9.3).
        self._human_input = ChannelHumanInput(None, ChannelType.CONFERENCE)
        self._framework: RoomKit | None = None
        self._rooms: dict[str, _RoomRealtime] = {}
        self._tools = ConferenceToolGate(channel_id)
        # The tool calls in flight, per session, each delivered and reported once.
        self._tool_calls = ToolCallBook()
        self._reports: set[asyncio.Task[None]] = set()
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
                    offered = {dict_tool_name(tool) for tool in catalogue(config) or []}
                    self._human_input.warn_unoffered(offered, self._channel_id)
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

    async def deliver_text(
        self, room_id: str, text: str, *, role: str, chain_depth: int = 0
    ) -> None:
        """Inject a broadcast text event into the provider's context.

        The realtime counterpart of speaking it: a 1:1 realtime channel
        injects rather than synthesizes, and the conference follows suit.
        Contained, because a provider that cannot take the text right now
        must not fail the broadcast that carried it. The model's answer to
        it is one deeper than the event (RFC §12.10.12).
        """
        if self._config is None:
            return
        session = await self.ensure_session(room_id)
        if session is None:
            return
        try:
            await self.inject_text(session, text, role=role, chain_depth=chain_depth)
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
    ) -> VoiceInjectionResult | None:
        """Inject *text* into the room session *session*, as a realtime voice
        channel injects into its own (RFC §12.4): the model's answer is one
        deeper than *chain_depth* unless *silent*. Not sent when the session
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
        source = EventSource(
            channel_id=self._channel_id,
            channel_type=ChannelType.CONFERENCE,
            participant_id=session.participant_id,
            provider=config.provider.name,
        )
        await fire_text_injected(self._framework, source, session, text, role=role)
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
        # otherwise wait forever on a queue nothing feeds, holding the floor
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
        room = self._guarded(session)
        call = RealtimeToolCall.from_provider(
            session, call_id, name, arguments, room_id=session.room_id
        )
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

    async def _on_tool_call_cancelled(self, session: VoiceSession, call_ids: list[str]) -> None:
        """The model will not read these calls' results: interrupt their
        handlers, send nothing, and report them to ON_TOOL_CALL's observers as
        cancelled.

        Each call's id is freed, as the provider freed it, so nothing is sent
        for the call and a call issued under the id is a new one (RFC §12.4).
        A call whose outcome ON_TOOL_CALL already has is left to finish,
        sending nothing: a second report would put two outcomes on one call.
        """
        room = self._guarded(session)
        if room is None:
            return
        for call_id in call_ids:
            call = self._tool_calls.release(session.id, call_id)
            if call is None or not call.interruptible:
                continue
            assert call.task is not None  # interruptible  # noqa: S101
            call.task.cancel()
            # Off the provider's callback: an audit hook must not hold up the
            # interruption it reports. Tracked beside the teardown, as a
            # detach's reports are: a detach does not cut it.
            self._track_report(report_cancelled_call(self, call, ABANDONED_BY_PROVIDER))

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
        await run_tool_call(self, call, _ConferenceDoor(self, config))

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
        utterance = room.utterance
        if utterance is not None and not utterance.discarded:
            utterance.discarded = True
            utterance.finish()
        for task in list(room.tasks):
            task.cancel()
        # A room off the books has nothing in flight: a delivery waiting on it
        # goes on to find its session gone.
        room.idle.set()
        session, room.session = room.session, None
        if session is not None:
            self._report_detached_calls(session)
        return session

    def _report_stale_call(self, call: RealtimeToolCall) -> None:
        """Report a call a session this channel no longer speaks for issued
        (after a detach or a reconnect): nothing serves it and nothing is
        sent, and it still gets its one report, cancelled (RFC §9.3), after
        a detach as after an unplug: the report needs no realtime config."""
        self._track_report(report_cancelled_call(self, call, _LEFT_THE_ROOM))

    def _track_report(self, coro: Awaitable[None]) -> None:
        """Run a report beside the teardown; the disconnect waits for it."""
        report = asyncio.ensure_future(coro)
        self._reports.add(report)
        report.add_done_callback(self._reports.discard)
        report.add_done_callback(log_task_exception)

    def _report_detached_calls(self, session: VoiceSession) -> None:
        """Report each call the detach interrupted, once, as cancelled (RFC §12.4).

        The reports run beside the teardown; the disconnect that follows waits
        for them, so none races the store's release at close.
        """
        calls = self._tool_calls.take(session.id)
        if not calls:
            return
        self._track_report(report_interrupted_calls(self, calls, _LEFT_THE_ROOM))

    async def _settle_reports(self) -> None:
        """Wait for the reports of the calls detaches interrupted. A wait
        cancelled leaves them running: they are not this waiter's to cut."""
        if self._reports:
            await asyncio.wait(list(self._reports))

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
        await self._settle_reports()
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
