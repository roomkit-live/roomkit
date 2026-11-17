"""Framework injection is opt-in by inheritance, not by method name.

``register_channel`` hands session-based channels the framework they were
registered with.  Selecting them by the name of the method would call anything
that happens to own it — with a ``RoomKit`` instance it never asked for.
Inheriting :class:`FrameworkAwareChannel` is the declaration that makes the call
wanted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roomkit import (
    AudioVideoChannel,
    ConferenceChannel,
    FrameworkAwareChannel,
    RealtimeAudioVideoChannel,
    RealtimeVoiceChannel,
    RoomKit,
    VideoChannel,
    VoiceChannel,
)
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import RoomEvent
from roomkit.store import SQLiteStore


class _Inert(Channel):
    """The least a channel can implement and still be registered."""

    channel_type = ChannelType.WEBSOCKET

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _Declared(FrameworkAwareChannel, _Inert):
    """A channel that asks for the framework the way the contract says to."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.framework: RoomKit | None = None

    def set_framework(self, framework: RoomKit) -> None:
        self.framework = framework


class _Homonym(_Inert):
    """A third-party channel whose own ``set_framework`` means something else.

    Not a contrived name collision: a channel wrapping another system may well
    own the word.  Its argument is not a :class:`RoomKit`, and calling it with
    one is the defect this test exists for.
    """

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.called_with: list[str] = []

    def set_framework(self, framework: str) -> None:
        self.called_with.append(framework)


class TestFrameworkInjection:
    def test_a_declared_channel_receives_the_framework(self) -> None:
        kit = RoomKit()
        channel = _Declared("declared")

        kit.register_channel(channel)

        assert channel.framework is kit

    async def test_an_application_channel_uses_the_configured_store(self, tmp_path: Path) -> None:
        store = SQLiteStore(tmp_path / "rooms.db")
        async with RoomKit(store=store) as kit:
            channel = _Declared("application")
            kit.register_channel(channel)
            assert channel.framework is kit
            assert channel.framework.store is store
            await channel.framework.create_room(room_id="example")
            assert (await kit.get_room("example")).id == "example"

    def test_injection_failure_leaves_the_inserted_channel_registered(self) -> None:
        class Refuses(_Declared):
            def set_framework(self, framework: RoomKit) -> None:
                assert framework.get_channel(self.channel_id) is self
                raise RuntimeError("setup failed")

        kit = RoomKit()
        channel = Refuses("application")
        with pytest.raises(RuntimeError, match="^setup failed$"):
            kit.register_channel(channel)
        assert kit.get_channel("application") is channel

    def test_a_homonym_is_left_alone(self) -> None:
        """Selecting by name would call a method that means something else."""
        kit = RoomKit()
        channel = _Homonym("homonym")

        kit.register_channel(channel)

        assert channel.called_with == []

    def test_the_contract_must_be_implemented(self) -> None:
        """A channel that declares the contract and leaves set_framework out is
        refused, rather than built and never handed the framework."""

        class _Undeclared(FrameworkAwareChannel, _Inert):
            pass

        with pytest.raises(TypeError):
            _Undeclared("undeclared")  # type: ignore[abstract]


class TestSessionChannelsDeclareIt:
    """Every channel that routes its own inbound media declares the contract.

    ``AudioVideoChannel`` and ``RealtimeAudioVideoChannel`` inherit the
    declaration from their parent; they are here so a split of either hierarchy
    cannot quietly drop it.
    """

    @pytest.mark.parametrize(
        "channel_class",
        [
            VoiceChannel,
            RealtimeVoiceChannel,
            VideoChannel,
            ConferenceChannel,
            AudioVideoChannel,
            RealtimeAudioVideoChannel,
        ],
    )
    def test_session_channel_is_framework_aware(self, channel_class: type[Channel]) -> None:
        assert issubclass(channel_class, FrameworkAwareChannel)
