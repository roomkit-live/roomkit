"""`flush_partial_tts` and `keep_partial_transcript` decide something (RFC §12.3.13).

Both fields were defined, documented and defaulted — and read nowhere in `src/`.
Setting `flush_partial_tts=False` still cut the audio off; setting
`keep_partial_transcript=True` still recorded nothing, so the timeline said the
bot delivered its whole line when the room had heard a third of it.
"""

from __future__ import annotations

from roomkit import RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceCapability
from roomkit.voice.interruption import InterruptionConfig


async def _speaking(config: InterruptionConfig) -> tuple[RoomKit, VoiceChannel, object]:
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel("voice-1", backend=backend, interruption=config)
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    channel._playing_sessions[session.id] = TTSPlaybackState(  # noqa: SLF001
        session_id=session.id, text="Your appointment is confirmed for Tuesday at ten."
    )
    return kit, channel, session


async def _recorded(kit: RoomKit) -> list[object]:
    """The interrupted-utterance events, not the room's own system chatter."""
    return [e for e in await kit.get_timeline("r1") if e.metadata.get("interrupted")]


def _cancels(channel: VoiceChannel) -> list[str]:
    return [
        c.args["session_id"]
        for c in channel._backend.calls  # noqa: SLF001
        if c.method == "cancel_audio"
    ]


class TestFlushPartialTTS:
    async def test_true_cancels_the_buffered_audio(self) -> None:
        _, channel, session = await _speaking(InterruptionConfig(flush_partial_tts=True))

        await channel.interrupt(session, reason="barge_in")

        assert _cancels(channel) == [session.id]

    async def test_false_lets_the_utterance_finish(self) -> None:
        """Some deployments would rather the sentence completes than be cut."""
        _, channel, session = await _speaking(InterruptionConfig(flush_partial_tts=False))

        await channel.interrupt(session, reason="barge_in")

        assert _cancels(channel) == []


class TestKeepPartialTranscript:
    async def test_true_records_what_was_cut_off(self) -> None:
        kit, channel, session = await _speaking(InterruptionConfig(keep_partial_transcript=True))

        await channel.interrupt(session, reason="barge_in")
        recorded = await _recorded(kit)

        assert len(recorded) == 1
        assert recorded[0].metadata["interrupted"] is True
        assert recorded[0].metadata["played_ms"] >= 0
        assert "appointment" in recorded[0].content.body

    async def test_false_records_nothing(self) -> None:
        kit, channel, session = await _speaking(InterruptionConfig(keep_partial_transcript=False))

        await channel.interrupt(session, reason="barge_in")

        assert await _recorded(kit) == []

    async def test_the_percentage_follows_a_known_duration(self) -> None:
        kit, channel, session = await _speaking(InterruptionConfig(keep_partial_transcript=True))
        channel._playing_sessions[session.id].total_duration_ms = 4000  # noqa: SLF001

        await channel.interrupt(session, reason="barge_in")
        recorded = await _recorded(kit)

        assert "played_percentage" in recorded[0].metadata
        assert 0 <= recorded[0].metadata["played_percentage"] <= 100

    async def test_an_unknown_duration_reports_no_percentage(self) -> None:
        """Better absent than invented — nothing populates the total today."""
        kit, channel, session = await _speaking(InterruptionConfig(keep_partial_transcript=True))

        await channel.interrupt(session, reason="barge_in")
        recorded = await _recorded(kit)

        assert "played_percentage" not in recorded[0].metadata

    async def test_nothing_is_recorded_when_no_playback_was_running(self) -> None:
        kit, channel, session = await _speaking(InterruptionConfig(keep_partial_transcript=True))
        channel._playing_sessions.clear()  # noqa: SLF001

        assert await channel.interrupt(session, reason="barge_in") is False
        assert await _recorded(kit) == []
