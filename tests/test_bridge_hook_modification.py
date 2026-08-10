"""A bridge hook that modifies a frame forwards the modification (RFC §12.7, §12.8).

Both triggers are declared "can block/modify" (RFC §9.2). Only the block half
was wired: `HookResult.modify(event=...)` returned a reshaped frame and the
bridge forwarded the original, so a hook that redacted, muted or watermarked a
frame changed nothing and said so nowhere.
"""

from __future__ import annotations

from roomkit import HookTrigger, RoomKit, VideoChannel, VoiceChannel
from roomkit.models.hook import HookResult
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.video.events import BridgeVideoEvent
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.events import BridgeAudioEvent

SILENCE = b"\x00\x00" * 160
TONE = b"\x11\x22" * 160


async def _voice_pair() -> tuple[RoomKit, VoiceChannel, MockVoiceBackend]:
    backend = MockVoiceBackend()
    channel = VoiceChannel("voice-1", backend=backend, bridge=True)
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    room = await kit.create_room(room_id="r1")
    await kit.attach_channel(room.id, "voice-1")
    return kit, channel, backend


class TestAudioBridge:
    async def test_a_modified_frame_is_the_one_forwarded(self) -> None:
        kit, channel, backend = await _voice_pair()
        a = await kit.connect_voice("r1", "user-a", "voice-1")
        await kit.connect_voice("r1", "user-b", "voice-1")

        @kit.hook(HookTrigger.BEFORE_BRIDGE_AUDIO)
        async def redact(event: BridgeAudioEvent, ctx: object) -> HookResult:
            quiet = AudioFrame(data=SILENCE, sample_rate=event.frame.sample_rate)
            return HookResult.modify(
                event=BridgeAudioEvent(session=event.session, frame=quiet, room_id=event.room_id)
            )

        forwarded: list[AudioFrame] = []
        channel._bridge.forward = lambda session, frame: forwarded.append(frame)  # noqa: SLF001

        await channel._fire_bridge_audio_and_forward(  # noqa: SLF001
            a, AudioFrame(data=TONE, sample_rate=16000)
        )

        assert len(forwarded) == 1
        assert forwarded[0].data == SILENCE

    async def test_an_untouched_frame_passes_through(self) -> None:
        kit, channel, _ = await _voice_pair()
        a = await kit.connect_voice("r1", "user-a", "voice-1")

        @kit.hook(HookTrigger.BEFORE_BRIDGE_AUDIO)
        async def observe(event: BridgeAudioEvent, ctx: object) -> HookResult:
            return HookResult.allow()

        forwarded: list[AudioFrame] = []
        channel._bridge.forward = lambda session, frame: forwarded.append(frame)  # noqa: SLF001

        await channel._fire_bridge_audio_and_forward(  # noqa: SLF001
            a, AudioFrame(data=TONE, sample_rate=16000)
        )

        assert forwarded[0].data == TONE

    async def test_block_still_drops_the_frame(self) -> None:
        kit, channel, _ = await _voice_pair()
        a = await kit.connect_voice("r1", "user-a", "voice-1")

        @kit.hook(HookTrigger.BEFORE_BRIDGE_AUDIO)
        async def drop(event: BridgeAudioEvent, ctx: object) -> HookResult:
            return HookResult.block("no")

        forwarded: list[AudioFrame] = []
        channel._bridge.forward = lambda session, frame: forwarded.append(frame)  # noqa: SLF001

        await channel._fire_bridge_audio_and_forward(  # noqa: SLF001
            a, AudioFrame(data=TONE, sample_rate=16000)
        )

        assert forwarded == []


class TestVideoBridge:
    async def test_a_modified_frame_is_the_one_forwarded(self) -> None:
        backend = MockVideoBackend()
        channel = VideoChannel("video-1", backend=backend, bridge=True)
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")
        session = await kit.connect_video("r1", "user-a", "video-1")

        @kit.hook(HookTrigger.BEFORE_BRIDGE_VIDEO)
        async def censor(event: BridgeVideoEvent, ctx: object) -> HookResult:
            blurred = VideoFrame(data=b"\x00" * 8, keyframe=event.frame.keyframe)
            return HookResult.modify(
                event=BridgeVideoEvent(session=event.session, frame=blurred, room_id=event.room_id)
            )

        forwarded: list[VideoFrame] = []
        channel._bridge.forward = lambda s, f: forwarded.append(f)  # noqa: SLF001

        await channel._fire_bridge_video_and_forward(  # noqa: SLF001
            session, VideoFrame(data=b"\xff" * 64, keyframe=True)
        )

        assert len(forwarded) == 1
        assert forwarded[0].data == b"\x00" * 8
