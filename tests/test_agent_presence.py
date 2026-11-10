"""Which executors are online, and which of a room's agents they cover (RFC
§10.6.7, §10.6.8): ``kit.executors()`` and ``kit.agent_presence(room_id)``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import AIChannel, ChannelDeclaration, InboundMessage, RoomKit, TextContent
from roomkit.models.enums import ChannelCategory, HookTrigger
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.realtime import EphemeralEvent, EphemeralEventType
from roomkit.realtime.memory import InMemoryRealtime
from roomkit.store.memory import InMemoryStore
from roomkit.turns import (
    AgentPresence,
    InMemoryTurnBackend,
    TurnBackend,
    TurnExecutor,
    TurnTimeouts,
)
from tests.test_framework import SimpleChannel


def _executor(executor_id: str, *hosts: str) -> TurnExecutor:
    return TurnExecutor(lambda _: None, hosts=set(hosts), executor_id=executor_id)


async def _kit(backend: InMemoryTurnBackend, *agents: str, live: str | None = None) -> RoomKit:
    """A kit holding *agents* by their declaration (and *live* live), each
    attached to rooms ``r1`` and ``r2``."""
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(SimpleChannel("ws"))
    for agent in agents:
        kit.register_channel(ChannelDeclaration(channel_id=agent))
    if live is not None:
        kit.register_channel(AIChannel(live, provider=MockAIProvider(responses=["hi"])))
    for room_id in ("r1", "r2"):
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "ws")
        for agent in [*agents, *([live] if live else [])]:
            await kit.attach_channel(room_id, agent, category=ChannelCategory.INTELLIGENCE)
    return kit


def _presence(presence: list[AgentPresence]) -> dict[str, tuple[bool, list[str]]]:
    return {p.channel_id: (p.online, p.executors) for p in presence}


async def test_the_kit_lists_the_executors_served_and_what_each_declares() -> None:
    backend = InMemoryTurnBackend()
    pool, host = _executor("pool", "acme-*"), _executor("host", "coder")
    await backend.serve(pool)
    await backend.serve(host)
    kit = await _kit(backend, "coder")
    try:
        before = {(i.executor_id, i.hosts, i.online) for i in await kit.executors()}
        await host.close()
        after = [i.executor_id for i in await kit.executors()]
    finally:
        await kit.close()
        await pool.close()

    assert before == {("pool", frozenset({"acme-*"}), True), ("host", frozenset({"coder"}), True)}
    assert after == ["pool"]


async def test_a_room_s_agent_presence_names_the_executors_covering_each_agent() -> None:
    backend = InMemoryTurnBackend()
    pool, host = _executor("pool", "acme-*"), _executor("host", "coder")
    await backend.serve(pool)
    await backend.serve(host)
    kit = await _kit(backend, "acme-sales", "coder", "stranger", live="helper")
    try:
        before = _presence(await kit.agent_presence("r1"))
        await host.close()
        after = _presence(await kit.agent_presence("r1"))
        own = kit.turn_executor.executor_id
    finally:
        await kit.close()
        await pool.close()

    assert before == {
        "acme-sales": (True, ["pool"]),
        "coder": (True, ["host"]),
        "helper": (True, [own]),
        "stranger": (False, []),
    }
    assert after["coder"] == (False, []) and after["acme-sales"] == (True, ["pool"])


async def test_an_executor_lacking_a_room_s_hook_does_not_cover_its_agent_there() -> None:
    backend = InMemoryTurnBackend()
    plain, audited = _executor("plain", "coder"), _executor("audited", "coder")

    @audited.room_hook("audit", HookTrigger.BEFORE_TOOL_USE)
    async def audit(event: Any, context: Any) -> None: ...

    await backend.serve(plain)
    kit = await _kit(backend, "coder")
    try:
        await kit.enable_room_hook("r1", "audit")
        hooked_room = _presence(await kit.agent_presence("r1"))
        other_room = _presence(await kit.agent_presence("r2"))
        await backend.serve(audited)
        served = _presence(await kit.agent_presence("r1"))
    finally:
        await kit.close()
        await plain.close()
        await audited.close()

    assert hooked_room == {"coder": (False, [])}
    assert other_room == {"coder": (True, ["plain"])}
    assert served == {"coder": (True, ["audited"])}


# -- Published in the rooms (RFC §10.6.7, §8.4) -------------------------------------


_PRESENCE = {EphemeralEventType.PRESENCE_ONLINE, EphemeralEventType.PRESENCE_OFFLINE}


async def _published(kit: RoomKit, *room_ids: str) -> dict[str, list[tuple[str, str]]]:
    """What each room's subscribers hear of the agents' presence, from now."""
    heard: dict[str, list[tuple[str, str]]] = {room_id: [] for room_id in room_ids}
    for room_id in room_ids:

        async def record(event: EphemeralEvent, room_id: str = room_id) -> None:
            if event.type in _PRESENCE:
                heard[room_id].append((event.type.value.removeprefix("presence_"), event.user_id))

        await kit.realtime.subscribe_to_room(room_id, record)
    return heard


async def _rooms(kit: RoomKit, *room_ids: str) -> None:
    kit.register_channel(SimpleChannel("ws"))
    for room_id in room_ids:
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "ws")


async def _heard_all(kit: RoomKit) -> None:
    """Lets the realtime backend hand what it was published."""
    for _ in range(20):
        await asyncio.sleep(0)


async def test_an_agent_s_presence_is_published_on_attach_and_as_its_coverage_changes() -> None:
    backend = InMemoryTurnBackend()
    host, late = _executor("host", "coder"), _executor("late", "coder")
    await backend.serve(host)
    kit = RoomKit(turn_backend=backend)
    for agent in ("coder", "stranger"):
        kit.register_channel(ChannelDeclaration(channel_id=agent))
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    await _rooms(kit, "r1", "r2")
    heard = await _published(kit, "r1", "r2")
    try:
        for room_id, agent in (
            ("r1", "coder"),
            ("r1", "stranger"),
            ("r1", "helper"),
            ("r2", "coder"),
        ):
            await kit.attach_channel(room_id, agent, category=ChannelCategory.INTELLIGENCE)
        await host.close()
        await backend.serve(late)
        await _heard_all(kit)
    finally:
        await kit.close()
        await late.close()

    assert heard == {
        "r1": [
            ("online", "coder"),
            ("offline", "stranger"),
            ("online", "helper"),
            ("offline", "coder"),
            ("online", "coder"),
        ],
        "r2": [("online", "coder"), ("offline", "coder"), ("online", "coder")],
    }


async def test_a_room_hook_no_executor_defines_takes_its_agent_offline_in_that_room() -> None:
    backend = InMemoryTurnBackend()
    plain, audited = _executor("plain", "coder"), _executor("audited", "coder")

    @audited.room_hook("audit", HookTrigger.BEFORE_TOOL_USE)
    async def audit(event: Any, context: Any) -> None: ...

    await backend.serve(plain)
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(ChannelDeclaration(channel_id="coder"))
    await _rooms(kit, "r1", "r2")
    heard = await _published(kit, "r1", "r2")
    try:
        for room_id in ("r1", "r2"):
            await kit.attach_channel(room_id, "coder", category=ChannelCategory.INTELLIGENCE)
        await kit.enable_room_hook("r1", "audit")
        await backend.serve(audited)
        await _heard_all(kit)
    finally:
        await kit.close()
        await plain.close()
        await audited.close()

    assert heard == {
        "r1": [("online", "coder"), ("offline", "coder"), ("online", "coder")],
        "r2": [("online", "coder")],
    }


async def test_a_detach_forgets_what_was_published_so_a_new_attach_publishes() -> None:
    backend = InMemoryTurnBackend()
    host = _executor("host", "coder")
    await backend.serve(host)
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(ChannelDeclaration(channel_id="coder"))
    await _rooms(kit, "r1")
    heard = await _published(kit, "r1")
    try:
        await kit.attach_channel("r1", "coder", category=ChannelCategory.INTELLIGENCE)
        await kit.detach_channel("r1", "coder")
        await kit.attach_channel("r1", "coder", category=ChannelCategory.INTELLIGENCE)
        await _heard_all(kit)
    finally:
        await kit.close()
        await host.close()

    assert heard == {"r1": [("online", "coder"), ("online", "coder")]}


class _Failing(InMemoryRealtime):
    """A realtime backend whose next presence publications fail."""

    def __init__(self) -> None:
        super().__init__()
        self.failures = 0

    async def publish_to_room(self, room_id: str, event: EphemeralEvent) -> None:
        if event.type in _PRESENCE and self.failures:
            self.failures -= 1
            raise ConnectionError("realtime down")
        await super().publish_to_room(room_id, event)


async def test_a_presence_that_failed_to_publish_is_published_at_the_next_look() -> None:
    """The realtime backend fails as the agent is attached to r2, then as it
    goes offline in r1: the attach still succeeds, and each lost publication
    is published at the next look at the executors, by the owner holding its
    claim."""
    backend = InMemoryTurnBackend()
    host = _executor("host", "coder")
    looks = [_executor(f"look-{n}", f"unrelated-{n}") for n in range(2)]
    await backend.serve(host)
    realtime = _Failing()
    kit = RoomKit(turn_backend=backend, realtime=realtime)
    kit.register_channel(ChannelDeclaration(channel_id="coder"))
    await _rooms(kit, "r1", "r2")
    heard = await _published(kit, "r1", "r2")
    await _heard_all(kit)  # the watch's first look runs before the attach
    try:
        realtime.failures = 1
        binding = await kit.attach_channel("r2", "coder", category=ChannelCategory.INTELLIGENCE)
        await kit.attach_channel("r1", "coder", category=ChannelCategory.INTELLIGENCE)
        await backend.serve(looks[0])  # a look: r2's online, lost at the attach
        realtime.failures = 1
        await host.close()  # offline in r1 (lost) and r2
        await backend.serve(looks[1])  # a look: r1's offline
        await _heard_all(kit)
    finally:
        await kit.close()
        for look in looks:
            await look.close()

    assert binding.channel_id == "coder"
    assert heard == {
        "r1": [("online", "coder"), ("offline", "coder")],
        "r2": [("online", "coder"), ("offline", "coder")],
    }


class _ClaimsDown(InMemoryTurnBackend):
    """A backend whose presence store is down."""

    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        raise ConnectionError("presence store down")


async def test_a_presence_failure_never_fails_an_attach_nor_a_detach() -> None:
    backend = _ClaimsDown()
    host = _executor("host", "coder")
    await backend.serve(host)
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(ChannelDeclaration(channel_id="coder"))
    await _rooms(kit, "r1")
    detached: list[str] = []
    kit.on("room_channel_detached")(lambda event: _noted(detached, event.channel_id))
    try:
        binding = await kit.attach_channel("r1", "coder", category=ChannelCategory.INTELLIGENCE)
        removed = await kit.detach_channel("r1", "coder")
    finally:
        await kit.close()
        await host.close()

    assert binding.channel_id == "coder" and removed
    assert detached == ["coder"]


async def _noted(seen: list[str], value: str | None) -> None:
    seen.append(str(value))


def _built(store: InMemoryStore, backend: InMemoryTurnBackend) -> RoomKit:
    """A kit made with no running loop, as at a module's import."""
    kit = RoomKit(store=store, turn_backend=backend)
    kit.register_channel(SimpleChannel("ws"))
    kit.register_channel(ChannelDeclaration(channel_id="coder"))
    return kit


async def test_a_kit_made_outside_a_loop_publishes_from_its_first_turn() -> None:
    """A kit restarted on a store whose room already holds the agent, made
    with no running loop: its first turn starts the watch, whose first look
    publishes the agent's presence in the room."""
    store = InMemoryStore()
    earlier = RoomKit(store=store)
    earlier.register_channel(SimpleChannel("ws"))
    earlier.register_channel(ChannelDeclaration(channel_id="coder"))
    await earlier.create_room(room_id="r1")
    await earlier.attach_channel("r1", "ws")
    await earlier.attach_channel("r1", "coder", category=ChannelCategory.INTELLIGENCE)
    await earlier.close()
    backend = InMemoryTurnBackend()
    host = _executor("host", "coder")
    await backend.serve(host)
    kit = await asyncio.to_thread(_built, store, backend)
    heard = await _published(kit, "r1")
    online = asyncio.Event()
    await kit.realtime.subscribe_to_room("r1", lambda event: _set_when_presence(online, event))
    try:
        hi = InboundMessage(channel_id="ws", sender_id="u", content=TextContent(body="Hi."))
        await kit.process_inbound(hi, room_id="r1")
        await asyncio.wait_for(online.wait(), 1)
        await _heard_all(kit)
    finally:
        await kit.close()
        await host.close()

    assert heard == {"r1": [("online", "coder")]}


async def _set_when_presence(flag: asyncio.Event, event: EphemeralEvent) -> None:
    if event.type in _PRESENCE:
        flag.set()


class _Unlisting(InMemoryTurnBackend):
    """A backend that does not list its executors, as a custom one may not."""

    executors = TurnBackend.executors  # type: ignore[assignment]


async def test_a_backend_that_does_not_list_its_executors_publishes_no_declared_agent() -> None:
    """The kit cannot tell whether an agent it holds by its declaration is
    served: it publishes nothing for it; a live agent's presence still is."""
    backend = _Unlisting()
    host = _executor("host", "coder")
    await backend.serve(host)
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(ChannelDeclaration(channel_id="coder"))
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    await _rooms(kit, "r1")
    heard = await _published(kit, "r1")
    try:
        for agent in ("coder", "helper"):
            await kit.attach_channel("r1", agent, category=ChannelCategory.INTELLIGENCE)
        await host.close()
        await _heard_all(kit)
    finally:
        await kit.close()

    assert heard == {"r1": [("online", "helper")]}


async def test_a_room_hook_change_publishes_each_agent_though_one_fails() -> None:
    backend = InMemoryTurnBackend()
    host = _executor("host", "alpha", "beta")
    look = _executor("look", "unrelated")
    await backend.serve(host)
    realtime = _Failing()
    kit = RoomKit(turn_backend=backend, realtime=realtime)
    for agent in ("alpha", "beta"):
        kit.register_channel(ChannelDeclaration(channel_id=agent))
    await _rooms(kit, "r1")
    for agent in ("alpha", "beta"):
        await kit.attach_channel("r1", agent, category=ChannelCategory.INTELLIGENCE)
    heard = await _published(kit, "r1")
    await _heard_all(kit)  # the watch's first look runs before the change
    try:
        realtime.failures = 1
        await kit.enable_room_hook("r1", "audit")  # neither covered there now
        await backend.serve(look)  # a look: the one that failed
        await _heard_all(kit)
    finally:
        await kit.close()
        await host.close()
        await look.close()

    assert sorted(heard["r1"]) == [("offline", "alpha"), ("offline", "beta")]


async def test_a_live_agent_s_presence_that_failed_is_published_again() -> None:
    """A kit holding only live agents still looks again for a publication
    that failed."""
    realtime = _Failing()
    kit = RoomKit(realtime=realtime)
    kit.register_channel(AIChannel("helper", provider=MockAIProvider(responses=["hi"])))
    await _rooms(kit, "r1")
    heard = await _published(kit, "r1")
    online = asyncio.Event()
    await kit.realtime.subscribe_to_room("r1", lambda event: _set_when_presence(online, event))
    try:
        realtime.failures = 1
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        await asyncio.wait_for(online.wait(), 3 * TurnTimeouts().heartbeat_interval)
    finally:
        await kit.close()

    assert heard == {"r1": [("online", "helper")]}


class _ClaimFailsOnce(InMemoryTurnBackend):
    """A backend whose presence store fails once, when told to."""

    failing = False

    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        if self.failing and online is not None:
            self.failing = False
            raise ConnectionError("presence store down")
        return await super().claim_presence(channel_id, room_id, online)


async def test_a_change_whose_claim_failed_is_published_at_a_later_look() -> None:
    backend = _ClaimFailsOnce()
    host, look = _executor("host", "coder"), _executor("look", "unrelated")
    await backend.serve(host)
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(ChannelDeclaration(channel_id="coder"))
    await _rooms(kit, "r1")
    await kit.attach_channel("r1", "coder", category=ChannelCategory.INTELLIGENCE)
    heard = await _published(kit, "r1")
    try:
        backend.failing = True
        await host.close()  # its claim fails
        await backend.serve(look)  # a look
        await _heard_all(kit)
    finally:
        await kit.close()
        await look.close()

    assert heard == {"r1": [("offline", "coder")]}


async def test_a_room_s_close_forgets_its_agents_presence_before_a_new_attach() -> None:
    """The room closes, reopens, and the agent is attached again at once: the
    close's forget runs first, so the attach publishes."""
    backend = InMemoryTurnBackend()
    host = _executor("host", "coder")
    await backend.serve(host)
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(ChannelDeclaration(channel_id="coder"))
    await _rooms(kit, "r1")
    heard = await _published(kit, "r1")
    try:
        await kit.attach_channel("r1", "coder", category=ChannelCategory.INTELLIGENCE)
        await kit.close_room("r1")
        await kit.reopen_room("r1")
        await kit.attach_channel("r1", "coder", category=ChannelCategory.INTELLIGENCE)
        await _heard_all(kit)
    finally:
        await kit.close()
        await host.close()

    assert heard == {"r1": [("online", "coder"), ("online", "coder")]}
