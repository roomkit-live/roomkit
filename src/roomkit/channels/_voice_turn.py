"""VoiceChannel mixin — turn detection and text routing."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit import _clock
from roomkit.channels._voice_speakers import routed_sender
from roomkit.channels._voice_unheard import UnheardTurns
from roomkit.core._authors import People
from roomkit.models.delivery import SUPERSEDED, DeliveryHandle, InboundResult
from roomkit.models.enums import EventType, HookTrigger

if TYPE_CHECKING:
    from roomkit.channels._room_hooks import RoomHooks
    from roomkit.channels._voice_speakers import SpeakerAttribution
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.turn.base import TurnEntry

logger = logging.getLogger("roomkit.voice")

# Cap pending audio buffer at ~1MB (≈32s at 16kHz mono 16-bit).
# SmartTurnDetector uses the last 8s (≈256KB) so this is generous.
_MAX_PENDING_AUDIO_BYTES = 1_048_576

# How long, past its silence, a turn's wait holds for a transcript still on its
# way (RFC §12): beyond the STT stream's own 5 s and a batch fallback, it is lost.
_TRANSCRIPT_WAIT_MAX_S = 10.0
_TRANSCRIPT_POLL_S = 0.05


class DueTranscript:
    """The transcript of a speech segment that ended, on its way to the session's turn.

    While one is due, an incomplete turn's wait does not end (RFC §12): a
    streaming STT finalizes after the speech, and the words join the turn
    they continue instead of becoming a turn of their own. ``settle`` is
    idempotent: called once the words have joined and been judged, and again
    on every way out of the segment's processing.
    """

    def __init__(self, due: dict[str, int], lock: threading.Lock, session_id: str) -> None:
        self._due = due
        self._lock = lock
        self._session_id = session_id
        self._settled = False
        with lock:
            due[session_id] = due.get(session_id, 0) + 1

    def settle(self) -> None:
        if self._settled:
            return
        self._settled = True
        with self._lock:
            left = self._due.get(self._session_id, 0) - 1
            if left > 0:
                self._due[self._session_id] = left
            else:
                self._due.pop(self._session_id, None)


@runtime_checkable
class VoiceTurnHost(Protocol):
    """Contract: capabilities a host class must provide for VoiceTurnMixin.

    Every attribute listed here is initialized by ``VoiceChannel.__init__``.

    Attributes:
        channel_id: Unique identifier for this channel instance.
        _framework: Reference to the owning RoomKit framework (None before registration).
        _backend: The voice backend transport (None before configuration).
        _pipeline_config: Audio pipeline configuration (None when no pipeline).
        _session_bindings: Map of session ID to (room_id, binding) pairs.
        _pending_turns: Accumulated turn entries per session, awaiting turn completion.
        _pending_audio: Accumulated raw audio per session for audio-native turn detectors.
        _turn_speech_state: Per session, whether the user is speaking and the
            ``_clock.monotonic()`` of the last speech onset or end, written from VAD
            events, so a wait for an incomplete turn measures silence, not time.
        _turn_transcripts_due: Per session, how many ended segments' transcripts
            have not reached the turn yet (see :class:`DueTranscript`).
        _turn_wait_tasks: The pending wait per session, cancelled when replaced.
        _scheduled_tasks: Background tasks that ``close()`` cancels.
        _unheard_turns: The routed turn per session whose response is not heard yet.
        _barge_in_threshold_ms: Speech that lasts this long supersedes an unheard turn.
        _state_lock: Guards state touched from the audio thread.
    """

    channel_id: str
    _framework: RoomKit | None
    _backend: VoiceBackend | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _pending_turns: dict[str, list[TurnEntry]]
    _pending_turn_speakers: dict[str, SpeakerAttribution]
    _transcript_locks: dict[str, asyncio.Lock]
    _pending_audio: dict[str, bytearray]
    _turn_speech_state: dict[str, tuple[bool, float]]
    _turn_transcripts_due: dict[str, int]
    _turn_wait_tasks: dict[str, asyncio.Task[None]]
    _scheduled_tasks: set[asyncio.Task[Any]]
    _unheard_turns: UnheardTurns
    _barge_in_threshold_ms: int
    _state_lock: threading.Lock


class VoiceTurnMixin:
    """Turn detection and text routing for VoiceChannel.

    Host contract: :class:`VoiceTurnHost`.
    """

    # -- attributes provided by VoiceChannel.__init__ (see VoiceTurnHost) --
    channel_id: str
    _framework: RoomKit | None
    _backend: VoiceBackend | None
    _pipeline_config: AudioPipelineConfig | None
    _session_bindings: dict[str, tuple[str, ChannelBinding]]
    _pending_turns: dict[str, list[TurnEntry]]
    _pending_turn_speakers: dict[str, SpeakerAttribution]
    _transcript_locks: dict[str, asyncio.Lock]
    _pending_audio: dict[str, bytearray]
    _turn_speech_state: dict[str, tuple[bool, float]]
    _turn_transcripts_due: dict[str, int]
    _turn_wait_tasks: dict[str, asyncio.Task[None]]
    _scheduled_tasks: set[asyncio.Task[Any]]
    _unheard_turns: UnheardTurns
    _barge_in_threshold_ms: int
    _state_lock: threading.Lock

    # -- cross-mixin methods (annotated as Any to avoid MRO shadowing) --
    _task_done: Any  # VoiceChannel._task_done
    _pipeline_audio_rate: Any  # VoicePipelineMixin

    async def _evaluate_turn(
        self,
        session: VoiceSession,
        text: str,
        room_id: str,
        hooks: RoomHooks,
        *,
        audio_bytes: bytes | None = None,
        speaker: SpeakerAttribution | None = None,
        await_delivery: bool = True,
        transcript: DueTranscript | None = None,
    ) -> None:
        """Evaluate turn completion using the configured TurnDetector.

        ``speaker`` is who said ``text`` when a diarizing STT labelled it. A
        turn has one speaker (RFC §12.2.3): a pending turn with another is
        routed first, as its own message. ``await_delivery=False`` returns once
        a routed turn is committed, without waiting for the reply to it.
        ``transcript`` is settled once ``text`` has joined the turn and been
        judged, so a wait the turn is under may end from there.
        """
        if not self._framework or not self._pipeline_config:
            return
        turn_detector = self._pipeline_config.turn_detector
        if turn_detector is None:
            await self._route_text(
                session, text, room_id, speaker=speaker, await_delivery=await_delivery
            )
            return

        from roomkit.voice.pipeline.turn.base import TurnContext, TurnEntry

        await self._close_turn_of_another_speaker(session, room_id, hooks, speaker)
        # Accumulate entry
        entries = self._pending_turns.setdefault(session.id, [])
        entries.append(TurnEntry(text=text, role="user"))
        if speaker is not None:
            self._pending_turn_speakers[session.id] = speaker

        # Accumulate audio for audio-native turn detectors
        if audio_bytes:
            buf = self._pending_audio.setdefault(session.id, bytearray())
            buf.extend(audio_bytes)
            if len(buf) > _MAX_PENDING_AUDIO_BYTES:
                trim = len(buf) - _MAX_PENDING_AUDIO_BYTES
                trim = trim + (trim % 2)  # Align to 2-byte (16-bit PCM) boundary
                del buf[:trim]

        accumulated_audio = bytes(self._pending_audio.get(session.id, b"")) or None
        sample_rate = self._pipeline_audio_rate(session)

        turn_ctx = TurnContext(
            conversation_history=list(entries),
            silence_duration_ms=0.0,
            transcript=text,
            is_final=True,
            session_id=session.id,
            audio_bytes=accumulated_audio,
            audio_sample_rate=sample_rate,
        )
        decision = await asyncio.to_thread(turn_detector.evaluate, turn_ctx)
        if transcript is not None:
            transcript.settle()
        logger.debug(
            "Turn %s by %s (confidence %.2f, %s)",
            "complete" if decision.is_complete else "incomplete",
            turn_detector.name,
            decision.confidence,
            decision.reason,
        )

        if decision.is_complete and self._user_speaking_again(session.id):
            # The user resumed while the turn was judged: it goes on, and the new
            # speech joins it (RFC §12). The wait still routes it after silence.
            logger.debug("Turn judged complete, but the user is speaking again: holding it")
            self._arm_turn_wait(session, room_id, hooks, None)
        elif decision.is_complete:
            await self._complete_turn(
                session, room_id, hooks, decision.confidence, await_delivery=await_delivery
            )
        else:
            # Fire ON_TURN_INCOMPLETE hook
            combined_so_far = " ".join(e.text for e in entries)
            try:
                from roomkit.voice.events import TurnIncompleteEvent

                incomplete_event = TurnIncompleteEvent(
                    session=session,
                    text=combined_so_far,
                    confidence=decision.confidence,
                )
                await hooks.run_async(HookTrigger.ON_TURN_INCOMPLETE, incomplete_event)
            except Exception:
                logger.exception("Error firing ON_TURN_INCOMPLETE hook")
            self._arm_turn_wait(session, room_id, hooks, decision.suggested_wait_ms)

    async def _close_turn_of_another_speaker(
        self,
        session: VoiceSession,
        room_id: str,
        hooks: RoomHooks,
        speaker: SpeakerAttribution | None,
    ) -> None:
        """Route the pending turn now if someone else said it (RFC §12.2.3).

        Its wait is dropped: the turn ended when the other voice started.
        """
        pending = self._pending_turn_speakers.get(session.id)
        if pending is None or speaker is None or pending.same_voice(speaker):
            return
        if not self._pending_turns.get(session.id):
            return
        task = self._turn_wait_tasks.pop(session.id, None)
        if task is not None:
            task.cancel()
        await self._complete_turn(session, room_id, hooks, confidence=0.0, await_delivery=False)

    async def _complete_turn(
        self,
        session: VoiceSession,
        room_id: str,
        hooks: RoomHooks,
        confidence: float,
        *,
        await_delivery: bool = True,
    ) -> None:
        """Route the session's accumulated turn, after its ON_TURN_COMPLETE hook."""
        if not self._framework:
            return
        entries = self._pending_turns.pop(session.id, [])
        speaker = self._pending_turn_speakers.pop(session.id, None)
        self._pending_audio.pop(session.id, None)
        if not entries:
            return
        combined = " ".join(e.text for e in entries)
        try:
            from roomkit.voice.events import TurnCompleteEvent

            event = TurnCompleteEvent(session=session, text=combined, confidence=confidence)
            await hooks.run_async(HookTrigger.ON_TURN_COMPLETE, event)
        except Exception:
            logger.exception("Error firing ON_TURN_COMPLETE hook")

        await self._route_text(
            session, combined, room_id, speaker=speaker, await_delivery=await_delivery
        )

    def _arm_turn_wait(
        self,
        session: VoiceSession,
        room_id: str,
        hooks: RoomHooks,
        suggested_wait_ms: float | None,
    ) -> None:
        """Wait for more speech on an incomplete turn, then route it (RFC §12).

        A later evaluation of the same turn replaces the wait: it either completes
        the turn or arms a new one.
        """
        default_ms = self._pipeline_config.turn_incomplete_wait_ms if self._pipeline_config else 0
        wait_ms = suggested_wait_ms if suggested_wait_ms is not None else default_ms
        previous = self._turn_wait_tasks.pop(session.id, None)
        if previous is not None:
            previous.cancel()
        task = asyncio.get_running_loop().create_task(
            self._route_turn_after_wait(session, room_id, hooks, wait_ms),
            name=f"turn_wait:{session.id}",
        )
        task.add_done_callback(self._task_done)
        self._scheduled_tasks.add(task)
        self._turn_wait_tasks[session.id] = task

    async def _route_turn_after_wait(
        self, session: VoiceSession, room_id: str, hooks: RoomHooks, wait_ms: float
    ) -> None:
        await self._wait_out_turn_silence(session.id, max(wait_ms, 0) / 1000)
        if self._turn_wait_tasks.get(session.id) is asyncio.current_task():
            self._turn_wait_tasks.pop(session.id, None)
        if session.id not in self._session_bindings or session.id not in self._pending_turns:
            return
        logger.info(
            "Turn judged incomplete, then %.0f ms of silence: routing it (long_pause)", wait_ms
        )
        try:
            lock = self._transcript_locks.get(session.id)
            if lock is None:
                await self._complete_turn(session, room_id, hooks, confidence=0.0)
            else:
                await self._complete_waited_turn_in_order(session, room_id, hooks, lock)
        except Exception:
            logger.exception("Error routing the turn after its wait")

    async def _wait_out_turn_silence(self, session_id: str, wait_s: float) -> None:
        """Return once the user has been silent *wait_s* and no transcript is due (RFC §12).

        A transcript still on its way when the silence is over holds the turn:
        it continues it. Its evaluation then completes the turn or arms a new
        wait; one that is lost holds it no longer than ``_TRANSCRIPT_WAIT_MAX_S``.
        """
        silence_since = _clock.monotonic()
        while True:
            with self._state_lock:
                speaking, changed_at = self._turn_speech_state.get(session_id, (False, 0.0))
                transcript_due = session_id in self._turn_transcripts_due
            if speaking:
                # The user is talking: the turn goes on, and silence restarts when they stop.
                await _clock.sleep(wait_s)
                continue
            silence_since = max(silence_since, changed_at)
            overdue = _clock.monotonic() - (silence_since + wait_s)
            if overdue < 0:
                await _clock.sleep(-overdue)
            elif not transcript_due:
                return
            elif overdue < _TRANSCRIPT_WAIT_MAX_S:
                await _clock.sleep(_TRANSCRIPT_POLL_S)
            else:
                logger.warning(
                    "Transcript of %s still due %.0f s after the turn's wait: routing without it",
                    session_id,
                    overdue,
                )
                return

    def _expect_transcript(self, session_id: str) -> DueTranscript:
        """A segment ended: its transcript is due to the session's turn until settled."""
        return DueTranscript(self._turn_transcripts_due, self._state_lock, session_id)

    async def _complete_waited_turn_in_order(
        self, session: VoiceSession, room_id: str, hooks: RoomHooks, lock: asyncio.Lock
    ) -> None:
        """Route a waited turn in its place among a diarized session's finals.

        The session's finals go through ``lock`` in order (RFC §12.2.3). A final
        that took the lock first may already have routed this turn (a change
        of voice closes it) and started another: only the turn waited for is
        routed here, and the reply is not awaited under the lock.
        """
        waited = self._pending_turns.get(session.id)
        async with lock:
            if waited is None or self._pending_turns.get(session.id) is not waited:
                return
            await self._complete_turn(
                session, room_id, hooks, confidence=0.0, await_delivery=False
            )

    def _user_speaking_again(self, session_id: str) -> bool:
        """Whether the VAD hears speech that started after the segment being judged."""
        with self._state_lock:
            speaking, _ = self._turn_speech_state.get(session_id, (False, 0.0))
        return speaking

    def _note_turn_speech(self, session_id: str, speaking: bool) -> None:
        """Record a VAD speech onset or end for the turn wait. Safe from the audio thread."""
        with self._state_lock:
            self._turn_speech_state[session_id] = (speaking, _clock.monotonic())

    def _cancel_turn_wait(self, session_id: str) -> None:
        """Drop the session's wait and speech state (session unbound)."""
        task = self._turn_wait_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
        with self._state_lock:
            self._turn_speech_state.pop(session_id, None)
            self._turn_transcripts_due.pop(session_id, None)

    async def _route_text(
        self,
        session: VoiceSession,
        text: str,
        room_id: str,
        *,
        speaker: SpeakerAttribution | None = None,
        await_delivery: bool = True,
    ) -> None:
        """Route transcribed text through the inbound pipeline.

        The sender stays the session's participant, the owner of the audio
        stream, unless an ``ON_TRANSCRIPTION`` hook matched the speaker to a
        room participant; ``speaker`` adds who said it when a diarizing STT
        labelled it (``speaker_label``, ``speaker_epoch``, ``sender_name``,
        RFC §12.2.3).

        ``await_delivery=False`` returns once the message is committed and
        waits for its delivery (the reply) in a task of its own: a diarized
        session's finals are routed in order under a lock, and holding it
        through the whole reply would hold back the next speaker's words.
        """
        if not self._framework:
            return
        from roomkit.models.delivery import InboundMessage
        from roomkit.models.event import TextContent
        from roomkit.telemetry.context import reset_span, set_current_span

        speaker = await self._speaker_reached(room_id, speaker)
        metadata: dict[str, Any] = {"voice_session_id": session.id, "source": "voice"}
        if speaker is not None:
            metadata.update(speaker.metadata())
        inbound = InboundMessage(
            channel_id=self.channel_id,
            sender_id=routed_sender(session.participant_id, speaker),
            content=TextContent(body=text),
            metadata=metadata,
        )
        # Set voice session span as parent so INBOUND_PIPELINE is a child
        session_span = getattr(self, "_voice_session_spans", {}).get(session.id)
        token = set_current_span(session_span) if session_span else None
        try:
            # Deferred, so the user's message is committed before the turn can be
            # superseded: resuming speech cancels the delivery, never the commit.
            result = await self._framework.process_inbound(
                inbound, room_id=room_id, defer_delivery=True
            )
        finally:
            if token is not None:
                reset_span(token)
        handle = result.delivery
        if handle is None:
            return
        with self._state_lock:
            speaking, since = self._turn_speech_state.get(session.id, (False, 0.0))
        self._unheard_turns.register(
            session.id, handle, speaking_since=since if speaking else None
        )
        if await_delivery:
            await self._await_delivery(session, handle)
            return
        task = asyncio.get_running_loop().create_task(
            self._await_delivery(session, handle), name=f"voice_delivery:{session.id}"
        )
        task.add_done_callback(self._task_done)
        self._scheduled_tasks.add(task)

    async def _speaker_reached(
        self, room_id: str, speaker: SpeakerAttribution | None
    ) -> SpeakerAttribution | None:
        """*speaker*, with the participant an ``ON_TRANSCRIPTION`` hook named
        only when the room reaches them through this channel (RFC §12.2.3,
        §5.5): an id names its participant on a channel they are reached
        through, so any other leaves the words the stream owner's, and the
        log says why."""
        if speaker is None or not speaker.participant_id or self._framework is None:
            return speaker
        people = People(await self._framework.store.list_participants(room_id))
        if people.named_on(speaker.participant_id, self.channel_id) is not None:
            return speaker
        logger.warning(
            "ON_TRANSCRIPTION named participant %r, whom room %s does not reach through "
            "voice channel %s: the words stay the session participant's (RFC §12.2.3)",
            speaker.participant_id,
            room_id,
            self.channel_id,
        )
        return dataclasses.replace(speaker, participant_id=None)

    async def _await_delivery(self, session: VoiceSession, handle: Any) -> None:
        """Wait for a committed turn's delivery; giving up on it cancels it."""
        try:
            await handle.wait()
        except asyncio.CancelledError:
            # A caller that gives up on the turn takes its delivery with it.
            with contextlib.suppress(TimeoutError):
                await handle.cancel(reason="caller_cancelled")
            raise
        finally:
            self._unheard_turns.discard(session.id, handle)

    def _release_unheard_turn(self, session_id: str) -> None:
        """Let the held response play, unless the user is already speaking again.

        Speech that started since holds it in turn, and its own end decides.
        """
        if not self._user_speaking_again(session_id):
            self._unheard_turns.release(session_id)

    async def _supersede_unheard_turn(self, session: VoiceSession) -> bool:
        """Cancel the turn the user's ended speech continues, before routing it (RFC §12.3.12).

        True when a turn was superseded. Its response was never heard: it is
        marked so no intelligence channel replays it, and the new transcript
        is routed on its own.
        """
        handle = self._unheard_turns.take_superseded(session.id, self._barge_in_threshold_ms)
        if handle is None:
            return False
        logger.info("User resumed before the response was heard: superseding the routed turn")
        try:
            result = await handle.cancel(reason=SUPERSEDED)
        except TimeoutError:
            # Still draining: its events are marked once it has.
            logger.warning("Superseded turn still draining for session %s", session.id)
            task = asyncio.get_running_loop().create_task(
                self._mark_when_drained(handle), name=f"superseded:{session.id}"
            )
            task.add_done_callback(self._task_done)
            self._scheduled_tasks.add(task)
            return True
        await self._mark_superseded(result)
        return True

    async def _mark_when_drained(self, handle: DeliveryHandle) -> None:
        """Mark a superseded turn still draining once it has drained. Its wait
        is begun here, by the task marking it: begun before, a task cancelled
        before its first step would leave the wait never awaited."""
        await self._mark_superseded(await handle.wait())

    async def _mark_superseded(self, result: InboundResult) -> None:
        """Mark a superseded turn's responses so no intelligence channel replays them.

        A response already fully generated was stored as finished, though nobody
        heard it: the mark is the voice channel's to add. Best effort, per event:
        a failed mark must not keep the user's new words from being routed.
        """
        for event in result.response_events:
            if event.type != EventType.MESSAGE or self._framework is None:
                continue
            metadata = {**event.metadata, "cancelled": True, "cancellation_reason": SUPERSEDED}
            try:
                await self._framework.update_event(event.room_id, event.id, metadata=metadata)
            except Exception:
                logger.exception("Could not mark superseded response %s", event.id)
