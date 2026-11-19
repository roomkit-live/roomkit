"""RoomKit.post() — the host's own voice in a room (RFC §10.5, RMK-947)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit import (
    Access,
    AgentResponsePolicy,
    ChannelCategory,
    ChannelNotFoundError,
    ChannelNotRegisteredError,
    ChannelType,
    Discussion,
    EventDirection,
    EventFilter,
    EventType,
    FrameworkEvent,
    HookExecution,
    HookResult,
    HookTrigger,
    HostChannel,
    InboundMessage,
    Participant,
    ParticipantNotFoundError,
    Pipeline,
    ProcessTimeoutError,
    RoomClosedError,
    RoomEvent,
    RoomKit,
    RoomNotFoundError,
    SMSChannel,
    SQLiteStore,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.agent import Agent
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelDirection, ChannelMediaType
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.store.memory import InMemoryStore

CUSTOMER = "+15550100001"

# max_chain_depth=1: a reentry budget of 10 passes per cascade.
BUDGET = 10


class _CountingHost(HostChannel):
    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered = 0

    async def deliver(self, event, binding, context):
        self.delivered += 1
        return await super().deliver(event, binding, context)


class _IntelligentHost(HostChannel):
    """A host channel that declares itself an intelligence channel."""

    category = ChannelCategory.INTELLIGENCE


class _SlowAttachHost(HostChannel):
    """A host channel whose attach waits until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.attaching = asyncio.Event()
        self.release = asyncio.Event()

    async def on_room_attached(self, room_id, binding):
        self.attaching.set()
        await self.release.wait()


class _OutlastingHost(HostChannel):
    """A host channel whose attach takes *seconds*, a post's whole
    ``process_timeout`` in the test that uses it."""

    def __init__(self, *, seconds: float) -> None:
        super().__init__()
        self._seconds = seconds

    async def on_room_attached(self, room_id, binding):
        await asyncio.sleep(self._seconds)


@pytest.fixture
async def kit():
    kit = RoomKit()
    yield kit
    await kit.close()


def _store(name: str) -> InMemoryStore | SQLiteStore:
    return InMemoryStore() if name == "memory" else SQLiteStore(":memory:")


async def _sms_room(
    kit: RoomKit, *, agent: MockAIProvider | None, **room: object
) -> MockSMSProvider:
    """Room "r": the customer on SMS, and an agent "assistant" when given a provider."""
    sms = MockSMSProvider()
    kit.register_channel(SMSChannel("sms", provider=sms))
    if agent is not None:
        kit.register_channel(Agent("assistant", provider=agent))
    await kit.create_room(room_id="r", **room)
    await kit.attach_channel("r", "sms", metadata={"phone_number": CUSTOMER})
    if agent is not None and "orchestration" not in room:
        await kit.attach_channel("r", "assistant", category=ChannelCategory.INTELLIGENCE)
    return sms


def _texted(sms: MockSMSProvider) -> list[tuple[str, str]]:
    return [(item["to"], item["event"].content.body) for item in sms.sent]


def _customer(body: str) -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id=CUSTOMER, content=TextContent(body=body))


async def _events(kit: RoomKit) -> list[RoomEvent]:
    return await kit.store.list_events(
        "r", limit=1000, event_filter=EventFilter(include_blocked=True)
    )


async def _host_attachments(kit: RoomKit) -> int:
    return sum(
        1
        for e in await _events(kit)
        if e.type == EventType.CHANNEL_ATTACHED and e.content.data.get("channel_id") == "host"
    )


async def _written(kit: RoomKit) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """What the room holds: its events, its bindings and the room itself."""
    bindings = await kit.store.list_bindings("r")
    room = await kit.store.get_room("r")
    assert room is not None
    return (
        [e.id for e in await _events(kit)],
        [b.model_dump() for b in bindings],
        room.model_dump(),
    )


@pytest.mark.parametrize("organization_id", [None, "acme"], ids=["unscoped", "scoped"])
async def test_post_texts_the_correspondent_and_solicits_no_agent(kit, organization_id):
    ai = MockAIProvider(["should not answer"])
    sms = await _sms_room(kit, agent=ai, organization_id=organization_id)

    event = await kit.post(
        "r",
        "Your order shipped.",
        metadata={"job": "shipping-notice"},
        organization_id=organization_id,
    )

    assert _texted(sms) == [(CUSTOMER, "Your order shipped.")]
    assert ai.calls == []
    assert (event.source.channel_type, event.source.direction) == (
        ChannelType.HOST,
        EventDirection.INBOUND,
    )
    stored = next(e for e in await _events(kit) if e.id == event.id)
    assert stored.metadata["job"] == "shipping-notice"
    # The host may post any content: its binding declares every media type.
    binding = await kit.store.get_binding("r", "host")
    assert binding is not None
    assert set(binding.capabilities.media_types) == set(ChannelMediaType)


async def test_post_addressed_to_an_agent_is_answered_and_names_its_speaker(kit):
    """The speaker, a participant connected via the host channel the post
    comes from, is shown by name (RFC §10.5, §10.1 step 12)."""
    ai = MockAIProvider(["It left the warehouse today."])
    sms = await _sms_room(kit, agent=ai)
    host = _CountingHost("desk-host")
    kit.register_channel(host)
    # Turns are labelled with their speaker once two people take part (RFC §6.4).
    await kit.store.add_participant(
        Participant(id=CUSTOMER, room_id="r", channel_id="sms", display_name="Marie")
    )
    await kit.store.add_participant(
        Participant(id="desk", room_id="r", channel_id="desk-host", display_name="Acme Support")
    )

    await kit.post(
        "r",
        "Tell the customer where the order is.",
        participant_id="desk",
        addressed_to=["assistant"],
        channel_id="desk-host",
    )

    asked = [m.content for m in ai.calls[0].messages if m.role == "user"]
    assert asked[-1].startswith('Acme Support: "Tell')
    assert _texted(sms)[-1] == (CUSTOMER, "It left the warehouse today.")
    assert host.delivered == 0


@pytest.mark.parametrize("store", ["memory", "sqlite"])
async def test_concurrent_first_posts_attach_the_host_channel_once(store):
    """The first post's attach is one section under the room lock: on a
    store that yields, concurrent first posts attach the channel once."""
    kit = RoomKit(store=_store(store))
    try:
        await _sms_room(kit, agent=None)

        await asyncio.gather(*(kit.post("r", f"Notice {n}.") for n in range(5)))

        assert await _host_attachments(kit) == 1
        assert await kit.store.get_binding("r", "host") is not None
    finally:
        await kit.close()


@pytest.mark.parametrize("store", ["memory", "sqlite"])
@pytest.mark.parametrize("when", ["before", "concurrently"])
async def test_a_host_binding_the_integrator_sets_keeps_its_settings(store, when):
    kit = RoomKit(store=_store(store))
    try:
        await _sms_room(kit, agent=None)
        kit.register_channel(HostChannel())
        attach = kit.attach_channel("r", "host", access=Access.WRITE_ONLY, metadata={"desk": "1"})

        if when == "before":
            await attach
            await kit.post("r", "Notice.")
        else:
            await asyncio.gather(attach, kit.post("r", "Notice."))

        binding = await kit.store.get_binding("r", "host")
        assert binding is not None
        assert (binding.access, binding.metadata) == (Access.WRITE_ONLY, {"desk": "1"})
    finally:
        await kit.close()


class _UnansweredBindings(InMemoryStore):
    """A store whose binding reads, once held, wait until released."""

    def __init__(self) -> None:
        super().__init__()
        self.held = False
        self.released = asyncio.Event()

    async def get_binding(self, room_id, channel_id):
        if self.held:
            await self.released.wait()
        return await super().get_binding(room_id, channel_id)


type _Release = Callable[[], Awaitable[None]]


async def _hold_the_room_lock(kit: RoomKit, store: _UnansweredBindings) -> _Release:
    held, released = asyncio.Event(), asyncio.Event()

    async def hold() -> None:
        async with kit._lock_manager.locked("r"):
            held.set()
            await released.wait()

    holder = asyncio.create_task(hold())
    await held.wait()

    async def release() -> None:
        released.set()
        await holder

    return release


async def _hold_the_binding_read(kit: RoomKit, store: _UnansweredBindings) -> _Release:
    store.held = True

    async def release() -> None:
        store.released.set()

    return release


@pytest.mark.parametrize(
    "hold",
    [_hold_the_room_lock, _hold_the_binding_read],
    ids=["the room lock", "the off-lock binding read"],
)
async def test_a_first_post_waits_within_process_timeout(hold):
    """The post's off-lock read of its binding and the first post's wait for
    the room lock spend the call's ``process_timeout``, as step 2's attach
    does (RFC §13.6): past it, ``ProcessTimeoutError`` with its framework
    event, nothing written."""
    store = _UnansweredBindings()
    kit = RoomKit(store=store, process_timeout=0.05)
    timeouts: list[FrameworkEvent] = []

    @kit.on("process_timeout")
    async def timed_out(event: FrameworkEvent) -> None:
        timeouts.append(event)

    try:
        sms = await _sms_room(kit, agent=None)
        before = await _written(kit)
        release = await hold(kit, store)
        try:
            with pytest.raises(ProcessTimeoutError):
                await asyncio.wait_for(kit.post("r", "Notice."), 1)
        finally:
            await release()
        assert [event.room_id for event in timeouts] == ["r"]
        assert await _written(kit) == before
        assert sms.sent == []
    finally:
        await kit.close()


async def test_a_first_post_s_attach_spends_the_deadline_its_message_shares():
    """One ``process_timeout`` from the call's start to its commit (RFC
    §13.6), not one for the attach and another for the message: an attach
    that outlasts it runs to its end and stays, with its record, and the
    post is refused, its message not written."""
    kit = RoomKit(process_timeout=0.02)
    try:
        sms = await _sms_room(kit, agent=None)
        kit.register_channel(_OutlastingHost(seconds=0.02))

        with pytest.raises(ProcessTimeoutError):
            await asyncio.wait_for(kit.post("r", "Notice."), 1)

        assert await kit.store.get_binding("r", "host") is not None
        assert await _host_attachments(kit) == 1
        assert [e for e in await _events(kit) if e.type == EventType.MESSAGE] == []
        assert sms.sent == []
    finally:
        await kit.close()


async def test_a_cancelled_first_post_attaches_the_host_channel_to_its_end(kit):
    """Once the lock is held, the attach runs to its end whatever becomes of
    the caller (RFC §13.6): its binding comes with its CHANNEL_ATTACHED
    record, and the cancelled post sends nothing."""
    sms = await _sms_room(kit, agent=None)
    host = _SlowAttachHost()
    kit.register_channel(host)
    posting = asyncio.create_task(kit.post("r", "Notice."))
    await host.attaching.wait()

    posting.cancel()
    await asyncio.sleep(0)
    host.release.set()
    with pytest.raises(asyncio.CancelledError):
        await posting

    assert await kit.store.get_binding("r", "host") is not None
    assert await _host_attachments(kit) == 1
    assert sms.sent == []


async def test_post_reaches_the_correspondent_of_a_discussion_room(kit):
    sms = await _sms_room(
        kit, agent=MockAIProvider(["unused"]), orchestration=Discussion(agents=["assistant"])
    )

    await kit.post("r", "The meeting moved to 3 pm.")

    assert _texted(sms) == [(CUSTOMER, "The meeting moved to 3 pm.")]


_ROOMS: dict[str, Callable[[], dict[str, object]]] = {
    "ADDRESSED_ONLY": lambda: {"agent_response_policy": AgentResponsePolicy.ADDRESSED_ONLY},
    "Pipeline": lambda: {"orchestration": Pipeline(agents=["a", "b"])},
}


@pytest.mark.parametrize(
    ("room", "speak", "turns"),
    [
        ("ADDRESSED_ONLY", "post", (0, 0)),
        ("ADDRESSED_ONLY", "post to a", (1, 0)),
        ("ADDRESSED_ONLY", "unaddressed", (1, 1)),
        ("Pipeline", "post", (0, 0)),
        ("Pipeline", "post to a", (1, 0)),
        ("Pipeline", "unaddressed", (1, 0)),
    ],
)
async def test_the_host_s_words_are_a_person_s_message_to_the_room_s_routing(
    kit, room, speak, turns
):
    """A post solicits no agent unless it names one; unaddressed, from the
    host channel, the host's words are routed as a person's message: every
    agent of an ``ADDRESSED_ONLY`` room answers, where an agent's output
    would solicit none, and a pipeline's first stage does (RFC §19.3)."""
    a, b = MockAIProvider(["A says."]), MockAIProvider(["B says."])
    kit.register_channel(Agent("a", provider=a))
    kit.register_channel(Agent("b", provider=b))
    settings = _ROOMS[room]()
    sms = await _sms_room(kit, agent=None, **settings)
    if "orchestration" not in settings:
        for agent in ("a", "b"):
            await kit.attach_channel("r", agent, category=ChannelCategory.INTELLIGENCE)

    if speak == "unaddressed":
        await kit.post("r", "Your order shipped.")
        await kit.send_event("r", "host", TextContent(body="Any question?"))
    else:
        await kit.post("r", "Any question?", addressed_to=["a"] if speak == "post to a" else ())

    assert (len(a.calls), len(b.calls)) == turns
    assert (CUSTOMER, "Any question?") in _texted(sms)


async def _detach_after_a_post(kit: RoomKit) -> None:
    await kit.post("r", "An earlier notice.", idempotency_key="notice")
    await kit.detach_channel("r", "host")


async def _close(kit: RoomKit) -> None:
    await kit.close_room("r")


async def _close_after_a_post(kit: RoomKit) -> None:
    await kit.post("r", "An earlier notice.", idempotency_key="notice")
    await kit.close_room("r")


async def _detach_then_close(kit: RoomKit) -> None:
    await _detach_after_a_post(kit)
    await kit.close_room("r")


async def _take_host_id(kit: RoomKit) -> None:
    kit.register_channel(SMSChannel("host", provider=MockSMSProvider()))


async def _nothing(kit: RoomKit) -> None:
    return None


async def _customer_on_sms(kit: RoomKit) -> None:
    await kit.add_member("r", "sms", CUSTOMER, display_name="Marie")


async def _an_agent(kit: RoomKit) -> None:
    kit.register_channel(Agent("assistant", provider=MockAIProvider(["unused"])))
    await kit.attach_channel("r", "assistant", category=ChannelCategory.INTELLIGENCE)


_REFUSALS: dict[str, tuple[Callable[[RoomKit], Awaitable[None]], dict[str, Any], type]] = {
    "a channel of another type": (_nothing, {"channel_id": "sms"}, ValueError),
    "an unregistered channel": (_nothing, {"channel_id": "ghost"}, ChannelNotRegisteredError),
    "host taken by another type": (_take_host_id, {}, ValueError),
    "a closed room": (_close, {}, RoomClosedError),
    "a detached host channel": (_detach_after_a_post, {}, ChannelNotFoundError),
    # The revocation is read before the room's status (RFC §7.5 rule 7).
    "a detached host channel in a closed room": (_detach_then_close, {}, ChannelNotFoundError),
    # The room's status and binding are read before the key (RFC §10.1).
    "a replay into a room closed since": (
        _close_after_a_post,
        {"idempotency_key": "notice"},
        RoomClosedError,
    ),
    "a replay after a detach": (
        _detach_after_a_post,
        {"idempotency_key": "notice"},
        ChannelNotFoundError,
    ),
    "another organization's room": (_nothing, {"organization_id": "other"}, RoomNotFoundError),
    "a key in the framework's namespace": (
        _nothing,
        {"idempotency_key": "roomkit:publication:k"},
        ValueError,
    ),
    "a negative chain depth": (_nothing, {"chain_depth": -1}, ValueError),
    # The speaker is a participant the host channel reaches (RFC §10.5, §5.5).
    "an unknown speaker": (_nothing, {"participant_id": "ghost"}, ParticipantNotFoundError),
    "a speaker reached through SMS only": (
        _customer_on_sms,
        {"participant_id": CUSTOMER},
        ParticipantNotFoundError,
    ),
    "an agent's channel id as the speaker": (
        _an_agent,
        {"participant_id": "assistant"},
        ParticipantNotFoundError,
    ),
    # Scoped first: another organization's room is not found, whoever the id names.
    "an unknown speaker in another organization's room": (
        _nothing,
        {"participant_id": "ghost", "organization_id": "other"},
        RoomNotFoundError,
    ),
}


@pytest.mark.parametrize(
    ("setup", "arguments", "error"), list(_REFUSALS.values()), ids=list(_REFUSALS)
)
async def test_a_refused_post_writes_nothing(kit, setup, arguments, error):
    """Refused before anything is written: no binding, no attachment record,
    no event, the room as it was (a detached channel's revocation kept)."""
    sms = await _sms_room(kit, agent=None, organization_id="acme")
    await setup(kit)
    before, texted = await _written(kit), list(sms.sent)

    with pytest.raises(error):
        await kit.post("r", "Never sent.", **arguments)

    assert await _written(kit) == before
    assert sms.sent == texted


@pytest.mark.parametrize(
    ("setup", "speaker", "says"),
    [
        (_nothing, "ghost", "names nobody"),
        (_customer_on_sms, CUSTOMER, "reached through other channels only"),
    ],
    ids=["an unknown speaker", "a speaker reached through SMS only"],
)
async def test_a_refused_speaker_is_told_why(kit, setup, speaker, says):
    """The refusal says why, and never suggests adding the refused id on
    the host channel, which would have the host speak in that person's
    name."""
    await _sms_room(kit, agent=None)
    await setup(kit)

    with pytest.raises(ParticipantNotFoundError, match=says) as refused:
        await kit.post("r", "Never sent.", participant_id=speaker)

    assert repr(speaker) not in str(refused.value).split("add_member", 1)[-1]


async def _desk_on_host(kit: RoomKit) -> None:
    await kit.add_member("r", "host", "desk", identity_id="id-desk", display_name="Acme Support")


async def _customer_on_host_too(kit: RoomKit) -> None:
    await _customer_on_sms(kit)
    await kit.add_member("r", "host", CUSTOMER)


_SPEAKERS: dict[str, tuple[Callable[[RoomKit], Awaitable[None]], str | None]] = {
    "a member added on the host channel": (_desk_on_host, "desk"),
    "their identity": (_desk_on_host, "id-desk"),
    "an SMS member added on the host channel too": (_customer_on_host_too, CUSTOMER),
    "nobody": (_nothing, None),
}


@pytest.mark.parametrize(("setup", "speaker"), list(_SPEAKERS.values()), ids=list(_SPEAKERS))
async def test_a_post_speaks_as_a_participant_its_host_channel_reaches(kit, setup, speaker):
    """``participant_id`` names a participant the host channel reaches, by
    their id or their identity's, as a turn's sender is read (RFC §10.5,
    §5.5), or names nobody: the post is sent, its source the id it was
    given. ``add_member()`` on the host channel's id makes a member one."""
    sms = await _sms_room(kit, agent=None)
    await setup(kit)

    event = await kit.post("r", "Your order shipped.", participant_id=speaker)

    assert event.source.participant_id == speaker
    assert _texted(sms) == [(CUSTOMER, "Your order shipped.")]


@pytest.mark.parametrize("room_id", ["r", None], ids=["named room", "routed"])
async def test_a_message_on_the_host_channel_is_refused_before_anything_is_written(kit, room_id):
    """The host channel receives no external payload (RFC §10.5): a message
    on it is refused at the door, no room opened, nothing attached."""
    await _sms_room(kit, agent=None)
    await kit.post("r", "Notice.")
    before = await _written(kit)

    with pytest.raises(ValueError, match="receives no external payload"):
        await kit.process_inbound(
            InboundMessage(channel_id="host", sender_id="x", content=TextContent(body="hi")),
            room_id=room_id,
        )

    assert [room.id for room in await kit.store.list_rooms()] == ["r"]
    assert await _written(kit) == before


@pytest.mark.parametrize(
    ("channel", "category"),
    [(HostChannel, ChannelCategory.INTELLIGENCE), (_IntelligentHost, None)],
    ids=["attached as intelligence", "declared intelligence"],
)
async def test_a_host_channel_is_never_bound_as_an_intelligence_channel(kit, channel, category):
    """The host's words MUST NOT come from an intelligence channel, which
    would make them an agent's output (RFC §10.5)."""
    await _sms_room(kit, agent=None)
    kit.register_channel(channel())

    with pytest.raises(ValueError, match="host channel"):
        await kit.attach_channel("r", "host", category=category)

    assert await kit.store.get_binding("r", "host") is None


@pytest.mark.parametrize(
    ("key", "texts"),
    [("notice-1042", 1), ("", 2), ("   ", 2)],
    ids=["a key", "empty", "spaces"],
)
async def test_a_post_replayed_with_its_key_is_not_texted_again(kit, key, texts):
    """A key its channel holds is a replay that returns the original; a
    blank key is no key (RFC §13.4)."""
    sms = await _sms_room(kit, agent=None)

    first = await kit.post("r", "Your order shipped.", idempotency_key=key)
    again = await kit.post("r", "Your order shipped.", idempotency_key=key)

    assert len(sms.sent) == texts
    assert (again.id == first.id) is (texts == 1)


@pytest.mark.parametrize("needs_lock", [True, False], ids=["locked hook", "off-lock hook"])
async def test_a_post_replayed_from_the_room_s_work_draws_no_pass(needs_lock):
    """A hook posts with one key more times than its cascade's budget holds
    passes, then posts anew: the first post attaches the host channel from
    within the room's work, each replay returns the original, read before
    the budget, and the new post still has its pass (RFC §13.4, §9.5.1)."""
    kit = RoomKit(max_chain_depth=1)
    posted: list[RoomEvent] = []
    try:
        await _sms_room(kit, agent=None)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="ack", needs_lock=needs_lock)
        async def ack(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.source.channel_id == "sms":
                for _ in range(BUDGET + 2):
                    posted.append(await kit.post("r", "Received.", idempotency_key="ack"))
                posted.append(await kit.post("r", "We answer within the hour."))
            return HookResult.allow()

        await asyncio.wait_for(kit.process_inbound(_customer("Hello?")), 5)

        assert {e.id for e in posted[:-1]} == {posted[0].id}
        host = [
            (e.content.body, e.blocked_by)
            for e in await _events(kit)
            if e.type == EventType.MESSAGE and e.source.channel_id == "host"
        ]
        assert host == [("Received.", None), ("We answer within the hour.", None)]
        assert await _host_attachments(kit) == 1
    finally:
        await kit.close()


async def test_a_first_post_past_the_cap_binds_nothing():
    """A hook's body spends its cascade's hook calls, then posts for the
    first time: the post draws its call before the attach, so past the cap it
    is stored ``BLOCKED`` with ``reentry_loop_cap`` and binds nothing, no
    attachment recorded (RFC §9.5.1, §10.5). The next post, outside that
    work, attaches the host channel."""
    kit = RoomKit(max_chain_depth=1)
    posted: list[RoomEvent] = []
    done = asyncio.Event()
    try:
        await _sms_room(kit, agent=None)

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="notice")
        async def notice(event: RoomEvent, ctx: RoomContext) -> None:
            if event.source.channel_id != "sms" or done.is_set():
                return
            try:
                for _ in range(BUDGET):
                    await kit.send_event("r", "sms", TextContent(body="Logged."), addressed_to=[])
                posted.append(await kit.post("r", "We answer within the hour."))
            finally:
                done.set()

        await kit.process_inbound(_customer("Hello?"))
        await asyncio.wait_for(done.wait(), 5)

        assert [e.blocked_by for e in posted] == ["reentry_loop_cap"]
        assert await kit.store.get_binding("r", "host") is None
        assert await _host_attachments(kit) == 0
        await kit.post("r", "Your order shipped.")
        assert await _host_attachments(kit) == 1
    finally:
        await kit.close()


async def _post_once(kit: RoomKit) -> None:
    await kit.post("r", "Your order shipped.")


async def _sms_received_only(kit: RoomKit) -> None:
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    await kit.attach_channel(
        "r", "sms", metadata={"phone_number": CUSTOMER}, direction=ChannelDirection.INBOUND
    )


@pytest.mark.parametrize(
    "bind_another",
    [_post_once, _sms_received_only],
    ids=["the host channel", "an SMS bound INBOUND"],
)
async def test_a_transport_that_does_not_send_narrows_nothing_an_agent_may_answer_with(
    kit, bind_another
):
    """Never delivered to, a transport that does not send is no transport an
    answer reaches: the agent may use what the transports that send carry
    (RFC §4.2, §6.7, §10.5), a WebSocket's six media types and rich text."""
    ai = MockAIProvider(["Sure."])
    ws = WebSocketChannel("ws")
    kit.register_channel(ws)
    kit.register_channel(Agent("assistant", provider=ai))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", "assistant", category=ChannelCategory.INTELLIGENCE)
    await bind_another(kit)

    await kit.process_inbound(
        InboundMessage(channel_id="ws", sender_id="u", content=TextContent(body="Thanks!")),
        room_id="r",
    )

    context, carried = ai.calls[-1], ws.capabilities()
    assert context.target_capabilities is not None
    assert set(context.target_media_types) == set(carried.media_types)
    assert context.target_capabilities.model_dump(exclude={"media_types"}) == carried.model_dump(
        exclude={"media_types"}
    )
