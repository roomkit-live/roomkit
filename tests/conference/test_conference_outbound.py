"""ConferenceChannel outbound path (RFC §12.10.4, §12.10.11).

One bot track, heard by everyone: the AI is synthesized once and published
once — and, because there is only the one track, one utterance at a time. The
other half of these tests guards the restrictive default on
``speak_text_events``, which is normative rather than conventional — a
conference is a meeting, and reading every inbound SMS aloud into one is
disruptive to everyone at once.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from roomkit import (
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels import _conference_activity as activity_module
from roomkit.channels.base import Channel
from roomkit.channels.conference import ConferenceChannel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import AudioContent, EventSource, RoomEvent, TextContent
from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.mock import MockTTSProvider

if TYPE_CHECKING:
    import pytest

ROOM = "room-1"
OTHER = "room-2"


class _Source(Channel):
    """A channel that only exists to originate events of a given type."""

    def __init__(self, channel_id: str, channel_type: ChannelType) -> None:
        super().__init__(channel_id)
        self._type = channel_type

    @property
    def channel_type(self) -> ChannelType:
        return self._type

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self._type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _WordTTS(TTSProvider):
    """Synthesizes one chunk per word, awaiting between them.

    Both halves are load-bearing. Awaiting is what opens the window two
    utterances could interleave in — a synthesizer that never suspends
    serialises itself and would prove nothing. Carrying the word in the chunk
    is what makes an interleaving *visible*: chunks of zeros are
    indistinguishable, so a record of them cannot say which answer they came
    from.

    ``final`` is what a synthesizer that never closes its own stream looks
    like. Nothing obliges a TTS provider to set ``is_final``, and the bot track
    still owes the backend an utterance boundary.
    """

    def __init__(self, *, final: bool = True) -> None:
        self._final = final
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.gated = False

    @property
    def default_voice(self) -> str:
        return "word"

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        gate = self.gated and not self.started.is_set()
        self.calls.append(text)
        words = text.split()
        for index, word in enumerate(words):
            # Suspends before every chunk, not only between them: the first
            # publication of a second utterance has to be able to land ahead of
            # the first utterance's second chunk.
            await asyncio.sleep(0)
            yield AudioChunk(
                data=word.encode(),
                sample_rate=16000,
                is_final=self._final and index == len(words) - 1,
            )
            if gate and index == 0:
                self.started.set()
                await self.release.wait()

    async def close(self) -> None:
        return None


async def _until(predicate, *, timeout: float = 5.0) -> None:  # type: ignore[no-untyped-def]
    """Wait until a predicate holds, rather than towards when it might."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0)


async def _conference(
    *,
    source_type: ChannelType = ChannelType.AI,
    tts: TTSProvider | None = None,
    rooms: tuple[str, ...] = (ROOM,),
    **kwargs: object,
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend, TTSProvider]:
    backend = MockConferenceBackend()
    tts = tts or MockTTSProvider()
    channel = ConferenceChannel("conf", backend=backend, tts=tts, **kwargs)  # type: ignore[arg-type]
    kit = RoomKit()
    kit.register_channel(channel)
    kit.register_channel(_Source("src", source_type))
    for room_id in rooms:
        await kit.create_room(room_id)
        await kit.attach_channel(room_id, "conf")
        await kit.attach_channel(room_id, "src")
    return kit, channel, backend, tts


class TestSingleBotTrack:
    async def test_an_ai_response_is_synthesized_once_and_published_once(self) -> None:
        """The SFU distributes the bot's track to everyone, so there is no
        per-participant audio to produce and no reason to synthesize twice.
        """
        kit, _, backend, tts = await _conference()

        await kit.send_event(ROOM, "src", TextContent(body="bonjour tout le monde"))

        assert len(tts.calls) == 1
        assert tts.calls[0]["text"] == "bonjour tout le monde"
        assert backend.published_audio

    async def test_everything_is_published_on_one_bot_session(self) -> None:
        kit, _, backend, _ = await _conference()

        await kit.send_event(ROOM, "src", TextContent(body="un"))
        await kit.send_event(ROOM, "src", TextContent(body="deux"))

        assert len(backend.bots) == 1
        published = [c for c in backend.calls if c.method == "publish_audio"]
        assert {c.args["bot"] for c in published} == {backend.bots[0].id}

    async def test_delivery_alone_brings_the_bot_in(self) -> None:
        """The bot joins on first need, and a delivery is a need — nobody has
        to speak first for the AI to be heard.
        """
        kit, _, backend, _ = await _conference()
        assert backend.bots == []

        await kit.send_event(ROOM, "src", TextContent(body="bonjour"))

        assert len(backend.bots) == 1

    async def test_two_answers_at_once_do_not_interleave(self) -> None:
        """One track means one utterance at a time.

        Each delivery used to publish on a loop of its own, so two answers went
        out chunk by alternating chunk on the single track every participant
        hears. Mixed together they are audible as neither, and the two
        ``is_final`` that arrived closed utterances the backend was never given
        apart.

        The room lock serialises the *locked* pipeline, which is why this
        drives the channel's own entry point: a streaming AI response is
        consumed outside that lock (``RoomKit._process_streaming_responses``),
        so its re-entry broadcast and the next message's broadcast reach the
        conference binding together. The single track is the channel's to
        protect either way.
        """
        _, channel, backend, _ = await _conference(tts=_WordTTS())
        # Long enough that the second answer's own setup — hooks, a context
        # build, the bot lock — finishes well inside the first one's stream.
        # Two short utterances can pass by accident.
        alpha, bravo = "alpha " * 12, "bravo " * 12

        first = asyncio.create_task(channel._voice.speak(ROOM, alpha))
        await _until(lambda: len(backend.published_audio) >= 1)
        second = asyncio.create_task(channel._voice.speak(ROOM, bravo))
        await asyncio.gather(first, second)

        assert [utterance.data for utterance in backend.utterances] == [
            b"alpha" * 12,
            b"bravo" * 12,
        ]
        assert all(utterance.complete for utterance in backend.utterances)

    async def test_one_room_speaking_does_not_hold_up_another(self) -> None:
        """The turn is a property of a track, and every room has its own bot."""
        tts = _WordTTS()
        tts.gated = True
        _, channel, backend, _ = await _conference(tts=tts, rooms=(ROOM, OTHER))

        held = asyncio.create_task(channel._voice.speak(ROOM, "alpha alpha alpha"))
        await asyncio.wait_for(tts.started.wait(), timeout=5.0)

        await asyncio.wait_for(channel._voice.speak(OTHER, "bravo bravo bravo"), timeout=5.0)
        tts.release.set()
        await held

        other = next(bot for bot in backend.bots if bot.room_id == OTHER)
        assert [u.data for u in backend.utterances_for(other)] == [b"bravo" * 3]
        assert all(u.complete for u in backend.utterances_for(other))

    async def test_an_answer_waiting_its_turn_is_dropped_by_a_detach(self) -> None:
        """Queueing an utterance must not outlive the conference it queued for.

        The wait is the new window: an answer that has not started speaking is
        not in the publishing loop the detach knows how to stop, so it has to be
        reachable while it waits — otherwise it takes the floor the first
        answer releases and publishes into a room the channel has left.
        """
        tts = _WordTTS()
        tts.gated = True
        kit, channel, backend, _ = await _conference(tts=tts)

        held = asyncio.create_task(channel._voice.speak(ROOM, "alpha alpha alpha"))
        await asyncio.wait_for(tts.started.wait(), timeout=5.0)
        queued = asyncio.create_task(channel._voice.speak(ROOM, "bravo bravo bravo"))
        await _until(lambda: len(channel._voice._speaking[ROOM].playbacks) == 2)

        await kit.detach_channel(ROOM, "conf")
        tts.release.set()
        await asyncio.gather(held, queued)

        assert tts.calls == ["alpha alpha alpha"], "the queued answer was synthesized anyway"
        assert not any(b"bravo" in chunk.data for chunk in backend.published_audio)

    async def test_without_tts_nothing_is_published(self) -> None:
        backend = MockConferenceBackend()
        channel = ConferenceChannel("conf", backend=backend)
        kit = RoomKit()
        kit.register_channel(channel)
        kit.register_channel(_Source("src", ChannelType.AI))
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        await kit.attach_channel(ROOM, "src")

        await kit.send_event(ROOM, "src", TextContent(body="bonjour"))

        assert backend.published_audio == []


class TestUtteranceBoundary:
    """What the backend is told about where an utterance ends.

    ``is_final`` is the only thing that says so (RFC §12.10.3), and it is what
    a backend reconstructs the AI's turns from — a recording, a transcript
    aligned to the bot track, a client that shows who is speaking. An utterance
    the channel never closes is one the backend believes is still going.
    """

    async def test_an_utterance_ends_on_one_final_chunk_and_no_more(self) -> None:
        kit, _, backend, _ = await _conference(tts=_WordTTS())

        await kit.send_event(ROOM, "src", TextContent(body="alpha bravo charlie"))

        assert [chunk.is_final for chunk in backend.published_audio] == [False, False, True]
        assert [utterance.data for utterance in backend.utterances] == [b"alphabravocharlie"]

    async def test_a_synthesizer_that_never_closes_is_closed_for_it(self) -> None:
        """Nothing obliges a TTS provider to mark its last chunk final, and the
        bot track owes the backend a boundary regardless of which one it was
        handed.
        """
        kit, _, backend, _ = await _conference(tts=_WordTTS(final=False))

        await kit.send_event(ROOM, "src", TextContent(body="alpha bravo"))

        assert backend.published_audio[-1].is_final is True
        assert backend.published_audio[-1].data == b""
        assert [utterance.data for utterance in backend.utterances] == [b"alphabravo"]
        assert backend.utterances[-1].complete is True

    async def test_a_detach_closes_nothing(self) -> None:
        """The one exception RFC §12.10.4 makes to the boundary guarantee, and
        the reason it makes it: a detach is not an utterance ending, it is a
        conference going away. The terminal chunk would be published into a
        session the channel has left — the one thing the abandoned flag exists
        to prevent — and it would not arrive ahead of the ``leave()`` behind it
        in any case. A backend ends the utterance on the session going away,
        which is what ``complete is False`` here records: the utterance was
        never closed in band, and nothing pretends it was.
        """
        tts = _WordTTS()
        tts.gated = True
        kit, channel, backend, _ = await _conference(tts=tts)

        held = asyncio.create_task(channel._voice.speak(ROOM, "alpha bravo charlie"))
        await asyncio.wait_for(tts.started.wait(), timeout=5.0)

        await kit.detach_channel(ROOM, "conf")
        tts.release.set()
        await held

        methods = [call.method for call in backend.calls]
        assert "publish_audio" not in methods[methods.index("leave") :]
        assert backend.utterances[-1].complete is False


class TestSpeakTextEvents:
    async def test_text_from_another_channel_is_silent_by_default(self) -> None:
        """Normative, not conventional: VoiceChannel speaks every TextContent
        it receives, and a conference deliberately does not. Do not "align"
        this with the voice channel.
        """
        kit, _, backend, tts = await _conference(source_type=ChannelType.SMS)

        await kit.send_event(ROOM, "src", TextContent(body="rappel de rendez-vous"))

        assert tts.calls == []
        assert backend.published_audio == []

    async def test_text_from_another_channel_is_spoken_when_enabled(self) -> None:
        kit, _, backend, tts = await _conference(
            source_type=ChannelType.SMS, speak_text_events=True
        )

        await kit.send_event(ROOM, "src", TextContent(body="rappel de rendez-vous"))

        assert len(tts.calls) == 1
        assert backend.published_audio

    async def test_ai_responses_are_spoken_regardless_of_the_flag(self) -> None:
        kit, _, backend, tts = await _conference(source_type=ChannelType.AI)

        await kit.send_event(ROOM, "src", TextContent(body="voici la réponse"))

        assert len(tts.calls) == 1


class TestNonSpeakableEvents:
    async def test_system_events_are_never_spoken(self) -> None:
        """System events carry orchestration metadata — handoff notices and the
        like — which has no business being read aloud in a meeting.
        """
        kit, _, backend, tts = await _conference(speak_text_events=True)

        await kit.send_event(ROOM, "src", TextContent(body="handoff"), event_type=EventType.SYSTEM)

        assert tts.calls == []
        assert backend.published_audio == []


class TestClose:
    async def test_close_releases_bot_backend_and_providers(self) -> None:
        """Closing a channel that owns its providers must release them, or a
        long-lived process leaks a client per conference.
        """
        closed: list[str] = []

        class _TrackingTTS(MockTTSProvider):
            async def close(self) -> None:
                closed.append("tts")

        tts = _TrackingTTS()
        kit, channel, backend, _ = await _conference(tts=tts)
        await kit.send_event(ROOM, "src", TextContent(body="bonjour"))
        assert len(backend.bots) == 1

        await channel.close()

        assert closed == ["tts"]
        assert backend.bots == []
        assert any(c.method == "close" for c in backend.calls)

    async def test_injected_providers_are_left_open_when_asked(self) -> None:
        """An integrator sharing one provider across channels closes it itself."""
        closed: list[str] = []

        class _TrackingTTS(MockTTSProvider):
            async def close(self) -> None:
                closed.append("tts")

        backend = MockConferenceBackend()
        channel = ConferenceChannel(
            "conf", backend=backend, tts=_TrackingTTS(), close_providers=False
        )

        await channel.close()

        assert closed == []


class TestACancelledUtterance:
    """An answer whose publication is cancelled while the conference runs on.

    The exception RFC §12.10.4 makes is for a session going away, and this is
    not that: the channel is attached, the bot is in the meeting, and the only
    thing that ended is one caller's interest in this answer — an orchestration
    that dropped it, a `deliver()` someone gave up on. Left unclosed, the next
    thing the bot says is heard as the continuation of it.
    """

    @staticmethod
    async def _cancel_mid_utterance(channel: ConferenceChannel, tts: _WordTTS) -> None:
        """Start an answer, let one chunk out, then cancel the caller."""
        speaking = asyncio.create_task(channel._voice.speak(ROOM, "alpha bravo charlie"))
        await asyncio.wait_for(tts.started.wait(), timeout=5.0)
        speaking.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await speaking

    async def test_the_backend_is_told_where_the_utterance_ended(self) -> None:
        tts = _WordTTS()
        tts.gated = True
        _, channel, backend, _ = await _conference(tts=tts)

        await self._cancel_mid_utterance(channel, tts)
        tts.release.set()
        await _until(lambda: backend.utterances[-1].complete)

        assert backend.published_audio[-1].is_final is True
        assert backend.published_audio[-1].data == b""
        assert backend.utterances[-1].complete is True

    async def test_the_next_answer_is_not_heard_as_a_continuation(self) -> None:
        """What an unclosed utterance actually costs: the boundary the backend
        reconstructs turns from lands in the wrong place, so two answers arrive
        as one.
        """
        tts = _WordTTS()
        tts.gated = True
        _, channel, backend, _ = await _conference(tts=tts)

        await self._cancel_mid_utterance(channel, tts)
        tts.release.set()
        await _until(lambda: backend.utterances[-1].complete)

        await channel._voice.speak(ROOM, "delta")

        assert len(backend.utterances) == 2, backend.utterances
        assert backend.utterances[-1].data == b"delta"

    async def test_a_cancellation_during_a_detach_still_closes_nothing(self) -> None:
        """The exception stands where it applies. A cancellation that *is* the
        conference going away must not publish into the session on its way out.
        """
        tts = _WordTTS()
        tts.gated = True
        kit, channel, backend, _ = await _conference(tts=tts)

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "alpha bravo charlie"))
        await asyncio.wait_for(tts.started.wait(), timeout=5.0)
        await kit.detach_channel(ROOM, "conf")
        speaking.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await speaking
        tts.release.set()

        methods = [call.method for call in backend.calls]
        assert "publish_audio" not in methods[methods.index("leave") :]
        assert backend.utterances[-1].complete is False


class _ShieldedPublishBackend(MockConferenceBackend):
    """Publishes under ``asyncio.shield``, so cancelling does not stop it.

    An SDK that runs its send under a shield — or hands it to a connection
    pool task of its own — behaves exactly like this: the caller is released
    with a CancelledError while the chunk goes out regardless. Which makes the
    framework's record of "what was published" a claim about its own await
    rather than about the conference.
    """

    def __init__(self) -> None:
        super().__init__()
        self.publishing = asyncio.Event()
        self.gate = asyncio.Event()

    async def publish_audio(self, bot, chunk):  # type: ignore[no-untyped-def]
        return await asyncio.shield(self._publish(bot, chunk))

    async def _publish(self, bot, chunk):  # type: ignore[no-untyped-def]
        if chunk.data:
            self.publishing.set()
            await self.gate.wait()
        return await super().publish_audio(bot, chunk)


class _HoldsTheBoundary(MockConferenceBackend):
    """Holds the chunk that ends an utterance, so a test can sit inside it.

    The one chunk a cancellation must not be able to take away, held where the
    framework can neither see it nor hurry it — which is what an SFU that has
    stopped acknowledging looks like from this side.
    """

    def __init__(self) -> None:
        super().__init__()
        self.closing = asyncio.Event()
        self.gate = asyncio.Event()

    async def publish_audio(self, bot, chunk):  # type: ignore[no-untyped-def]
        if chunk.is_final:
            self.closing.set()
            await self.gate.wait()
        return await super().publish_audio(bot, chunk)


class TestACancelledPublicationStillEnds:
    """The windows a cancellation leaves around one chunk.

    A backend may shield its own send, so the audio goes out while this side is
    cancelled; and the cancellation can land on the terminal chunk itself. In
    both, the framework had published audio and told the backend nothing about
    where it ended — which merges the next answer into this one.
    """

    async def test_a_shielded_publication_is_still_closed(self) -> None:
        backend = _ShieldedPublishBackend()
        channel = ConferenceChannel("conf", backend=backend, tts=_WordTTS())
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "alpha bravo"))
        await asyncio.wait_for(backend.publishing.wait(), timeout=5.0)
        speaking.cancel()
        backend.gate.set()
        with contextlib.suppress(asyncio.CancelledError):
            await speaking
        await _until(lambda: bool(backend.utterances) and backend.utterances[-1].complete)

        assert backend.published_audio[-1].is_final is True
        assert backend.utterances[-1].complete is True

    async def test_a_cancellation_on_the_terminal_chunk_still_publishes_it(self) -> None:
        """The boundary is published from a task this channel owns, so a
        cancellation arriving during it reaches the caller and not the chunk.
        """
        backend = _HoldsTheBoundary()
        channel = ConferenceChannel("conf", backend=backend, tts=_WordTTS(final=False))
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "alpha"))
        await asyncio.wait_for(backend.closing.wait(), timeout=5.0)
        speaking.cancel()
        backend.gate.set()
        with contextlib.suppress(asyncio.CancelledError):
            await speaking
        await _until(lambda: bool(backend.utterances) and backend.utterances[-1].complete)

        assert backend.published_audio[-1].is_final is True
        assert backend.utterances[-1].complete is True


class TestAnUnclosedUtteranceStopsTheNextOne:
    """A boundary that never went out is not something the next answer can
    publish past.

    Waiting for it is the ordinary case; the deadline passing is not, and going
    ahead anyway is what RFC §12.10.4 forbids outright — the previous utterance
    has no end, so what follows is heard as its continuation and the boundary
    still to come lands in the middle of it. The answer goes unheard instead,
    which the RFC leaves to the implementation.
    """

    async def test_the_next_answer_is_dropped_rather_than_run_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.05)
        backend = _HoldsTheBoundary()
        channel = ConferenceChannel("conf", backend=backend, tts=_WordTTS(final=False))
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "alpha"))
        # Held inside the boundary from here on: the utterance has no end, and
        # cannot be given one within the budget the next answer waits for.
        await asyncio.wait_for(backend.closing.wait(), timeout=5.0)
        speaking.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await speaking

        try:
            await asyncio.wait_for(channel._voice.speak(ROOM, "delta"), timeout=5.0)

            assert [chunk.data for chunk in backend.published_audio] == [b"alpha"], (
                "the second answer was published onto an utterance with no end"
            )
        finally:
            backend.gate.set()


class _RefusesTheBoundary(MockConferenceBackend):
    """Publishes audio and refuses the chunk that would end the utterance."""

    def __init__(self) -> None:
        super().__init__()
        self.refusals = 0

    async def publish_audio(self, bot, chunk):  # type: ignore[no-untyped-def]
        if chunk.is_final:
            self.refusals += 1
            raise OSError("the SFU refused the boundary")
        return await super().publish_audio(bot, chunk)


class TestABoundaryNeverOvertakesItsAudio:
    """The terminal chunk closes the audio before it, so it cannot arrive first.

    Published while an earlier chunk was still in the backend — which is what a
    shielded send leaves behind when this side is cancelled — it ends the
    utterance before its own last words, and the audio lands after the boundary
    that was supposed to close it.
    """

    async def test_the_boundary_waits_for_the_chunk_it_closes(self) -> None:
        backend = _ShieldedPublishBackend()
        channel = ConferenceChannel("conf", backend=backend, tts=_WordTTS(final=False))
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "alpha"))
        await asyncio.wait_for(backend.publishing.wait(), timeout=5.0)
        speaking.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await speaking
        backend.gate.set()
        await _until(lambda: bool(backend.utterances) and backend.utterances[-1].complete)

        assert [(chunk.data, chunk.is_final) for chunk in backend.published_audio] == [
            (b"alpha", False),
            (b"", True),
        ]

    async def test_a_final_chunk_that_never_arrived_is_replaced(self) -> None:
        """A synthesizer's own last chunk carries ``is_final``, so a publication
        of it that failed leaves the utterance open while looking closed. The
        boundary is owed on what the backend *accepted*, not on what was handed
        to it.
        """
        backend = _FailsThenAccepts()
        channel = ConferenceChannel("conf", backend=backend, tts=_WordTTS())
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")

        # The refusal is the caller's to hear; what it must not do is leave
        # the utterance open behind it.
        with contextlib.suppress(OSError):
            await channel._voice.speak(ROOM, "alpha bravo")

        assert backend.published_audio[-1].is_final is True
        assert backend.published_audio[-1].data == b""
        assert backend.utterances[-1].complete is True


class _FailsThenAccepts(MockConferenceBackend):
    """Refuses the synthesizer's own final chunk, and accepts what follows."""

    def __init__(self) -> None:
        super().__init__()
        self._refused = False

    async def publish_audio(self, bot, chunk):  # type: ignore[no-untyped-def]
        if chunk.is_final and chunk.data and not self._refused:
            self._refused = True
            raise OSError("the SFU refused the last chunk")
        return await super().publish_audio(bot, chunk)


class TestATrackNothingCouldCloseStaysOutOfUse:
    """An utterance whose boundary the backend refused leaves the bot track in a
    state nothing may publish on.

    Read as a successful close — which dropping the closing task without looking
    at its exception did — the next answer goes out onto an utterance that never
    ended, and the backend reconstructs the two as one. RFC §12.10.4 states both
    obligations together, and this breaks both at once.
    """

    async def _stranded(self) -> tuple[RoomKit, ConferenceChannel, _RefusesTheBoundary]:
        backend = _RefusesTheBoundary()
        channel = ConferenceChannel("conf", backend=backend, tts=_WordTTS(final=False))
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        # The refusal reaches the caller — an answer whose end could not be
        # published is not one that was delivered — and what it leaves behind
        # is the subject of the tests below.
        with contextlib.suppress(OSError):
            await channel._voice.speak(ROOM, "alpha")
        await _until(lambda: backend.refusals > 0)
        return kit, channel, backend

    async def test_the_next_answer_is_not_published(self) -> None:
        _, channel, backend = await self._stranded()

        await channel._voice.speak(ROOM, "delta")

        assert [chunk.data for chunk in backend.published_audio] == [b"alpha"]
        assert len(backend.utterances) == 1

    async def test_the_room_says_why_the_bot_went_quiet(self) -> None:
        """An integrator whose AI stopped speaking reads the reason rather than
        interpreting a silence (RFC §17.7).
        """
        _, channel, _ = await self._stranded()

        assert "OSError" in channel.info()["rooms"][ROOM]["bot_track_unterminated"]

    async def test_a_new_bot_session_clears_it(self) -> None:
        """What made the track unusable was an utterance open on that session.
        Re-attaching joins as a new one, whose track has nothing open on it.
        """
        kit, channel, backend = await self._stranded()

        await kit.detach_channel(ROOM, "conf")
        await kit.attach_channel(ROOM, "conf")
        # This backend refuses every boundary, so the answer still ends in a
        # refusal — what matters is that it was published at all.
        with contextlib.suppress(OSError):
            await channel._voice.speak(ROOM, "delta")

        assert b"delta" in [chunk.data for chunk in backend.published_audio]
