"""A transport that records what it delivers, and an agent that answers each
event it is asked with a fixed reply: channels for tests and examples."""

from __future__ import annotations

from dataclasses import dataclass

from roomkit._media import content_text
from roomkit.channels.addressing import addressing_of
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventDirection
from roomkit.models.event import EventSource, RoomEvent, TextContent


@dataclass(frozen=True)
class Delivery:
    """A delivery a :class:`RecordingTransport` made."""

    event: RoomEvent
    """The event delivered, as transcoded for the transport."""
    text: str
    """Its text, as a person on the transport would read it."""


class RecordingTransport(Channel):
    """A text transport of *channel_type* that records every event delivered
    to it, in order.

    Its destination is its record, so it needs no recipient: it names no
    recipient key, whatever its type (RFC §10.2 step 3d). It addresses its
    senders as its type does: an SMS recording transport writes them in
    E.164 (RFC §5.7). *capabilities* are those it declares: one that edits
    and deletes natively receives an edit or a delete as it is (RFC §6.6).
    """

    category = ChannelCategory.TRANSPORT

    def __init__(
        self,
        channel_id: str,
        channel_type: ChannelType,
        *,
        capabilities: ChannelCapabilities | None = None,
    ) -> None:
        super().__init__(channel_id)
        self.channel_type = channel_type
        self._capabilities = capabilities
        self._deliveries: list[Delivery] = []

    @property
    def provider_name(self) -> str | None:
        return "RecordingTransport"

    @property
    def deliveries(self) -> list[Delivery]:
        """What the transport delivered so far, in order."""
        return list(self._deliveries)

    @property
    def texts(self) -> list[str]:
        """The texts the transport delivered so far, in order."""
        return [delivery.text for delivery in self._deliveries]

    def capabilities(self) -> ChannelCapabilities:
        if self._capabilities is not None:
            return self._capabilities
        return super().capabilities()

    def normalize_address(self, address: str) -> str:
        return addressing_of(self.channel_type).normalize(address) if address else address

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            type=message.event_type,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self._deliveries.append(Delivery(event, content_text(event.content) or ""))
        return ChannelOutput.empty()


class EchoAgent(Channel):
    """An agent that answers every event it is asked with *reply*, followed
    by the event's text when *quote* is set."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str, reply: str, *, quote: bool = False) -> None:
        super().__init__(channel_id)
        self.reply = reply
        self.quote = quote

    @property
    def provider_name(self) -> str | None:
        return "EchoAgent"

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError("An EchoAgent answers in its room; nothing sends it a message")

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        text = self.reply + (content_text(event.content) or "") if self.quote else self.reply
        answer = RoomEvent(
            room_id=event.room_id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=ChannelType.AI,
                direction=EventDirection.OUTBOUND,
            ),
            content=TextContent(body=text),
            chain_depth=event.chain_depth + 1,
        )
        return ChannelOutput(responded=True, response_events=[answer])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()
