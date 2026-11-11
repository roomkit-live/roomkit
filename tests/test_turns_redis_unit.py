"""The Redis turn backend's parts that need no Redis (RFC §10.6.7, §10.6.8):
its inbox entries, what an executor declares, how an owner counts a beat an
executor makes outside the turn's messages, and the optional dependency."""

from __future__ import annotations

import asyncio
import contextvars
import json
import sys
import time
from typing import Any
from unittest.mock import patch

import pytest

from roomkit import HookTrigger
from roomkit.core.exceptions import TurnLinkError
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.turns import InMemoryTurnBackend, TurnExecutor, TurnTimeouts, _redis_routes
from roomkit.turns._error_codec import ROOMKIT_ERRORS
from roomkit.turns._hosting import Hosted
from roomkit.turns._organizations import RoomNeeds
from roomkit.turns._owner_turn import OwnerTurn
from roomkit.turns._redis_beat import LiveExecutors
from roomkit.turns._redis_clients import redis_clients
from roomkit.turns._redis_inbox import Envelope
from roomkit.turns._redis_links import RedisExecutorLink
from roomkit.turns._redis_outbox import Outbox
from roomkit.turns._redis_placement import Placement
from roomkit.turns._redis_presence import Declaration, View, member_of
from roomkit.turns._redis_routes import REFUSED, TAKEN, Routes
from roomkit.turns.owner import registry_turn_client, turn_request
from roomkit.turns.redis import RedisTurnBackend
from roomkit.turns.wire import (
    Accepted,
    ChannelControlEnvelope,
    ChannelResult,
    End,
    Ephemeral,
    HostResult,
    OwnerMessageEnvelope,
    RoomCall,
    RoomDetached,
    RoomPublication,
)
from tests.private_module import load_private_copy

QUICK = TurnTimeouts(heartbeat_interval=0.05, heartbeat_timeout=0.2, accept=0.2, cancel=0.2)
# A turn's bounds ten times shorter, for the tests that wait one out.
TICK = TurnTimeouts(heartbeat_interval=0.005, heartbeat_timeout=0.02, accept=0.02, cancel=0.02)


def test_an_inbox_entry_reads_back_from_bytes_or_text() -> None:
    sent = Envelope("frame", "node-a", '{"x": 1}', target="turn-1", ref="")
    as_bytes = {key.encode(): value.encode() for key, value in sent.fields().items()}

    assert Envelope.parse(as_bytes) == sent
    assert Envelope.parse(sent.fields()) == sent


def test_a_declaration_takes_its_channels_with_every_room_hook_it_defines() -> None:
    declared = Declaration("node-a", "host-a", Hosted(frozenset({"meteo"})), frozenset({"audit"}))
    pool = Declaration("node-b", "pool", Hosted(frozenset({"org-*", "meteo"})), frozenset())

    assert declared.takes("meteo", RoomNeeds())
    assert declared.takes("meteo", RoomNeeds(("audit",)))
    assert not declared.takes("meteo", RoomNeeds(("audit", "billing")))
    assert not declared.takes("other", RoomNeeds())
    assert Declaration.load(declared.dump()) == declared
    assert pool.takes("org-sales", RoomNeeds()) and pool.takes("meteo", RoomNeeds())
    assert not pool.takes("org", RoomNeeds()) and not pool.takes("other", RoomNeeds())
    assert Declaration.load(pool.dump()) == pool


def test_a_declaration_takes_the_rooms_of_the_organizations_it_serves() -> None:
    """RFC §10.6.7: what a beat tells of an executor's organizations is what
    the room owners of other processes place on; a beat of an executor of an
    earlier release names none, and serves the rooms without an organization."""
    hosted, hooks = Hosted(frozenset({"support"})), frozenset[str]()
    acme = Declaration("node-a", "acme", hosted, hooks, frozenset({"acme"}))
    every = Declaration("node-a", "every", hosted, hooks, frozenset({"*"}))
    plain = Declaration("node-a", "plain", hosted, hooks)

    assert acme.takes("support", RoomNeeds(organization_id="acme"))
    assert not acme.takes("support", RoomNeeds(organization_id="globex"))
    assert not acme.takes("support", RoomNeeds())
    assert every.takes("support", RoomNeeds(organization_id="globex"))
    assert every.takes("support", RoomNeeds())
    assert plain.takes("support", RoomNeeds())
    assert not plain.takes("support", RoomNeeds(organization_id="acme"))
    for declaration in (acme, every, plain):
        assert Declaration.load(declaration.dump()) == declaration
    earlier = json.loads(plain.dump())
    del earlier["organizations"]
    assert Declaration.load(json.dumps(earlier)) == plain


def test_an_executor_given_its_channels_tells_them() -> None:
    listed = TurnExecutor(lambda _: None, hosts={"meteo", "billing"})
    prefixed = TurnExecutor(lambda _: None, hosts={"org-*"})
    starred = TurnExecutor(lambda _: None, hosts={"*"})
    predicate = TurnExecutor(lambda _: None, hosts=lambda cid: cid == "meteo")
    every = TurnExecutor(lambda _: None)

    assert listed.hosted == {"meteo", "billing"} and listed.declares("billing")
    assert prefixed.hosted == {"org-*"} and prefixed.declares("org-sales")
    assert not prefixed.declares("meteo")
    assert starred.hosted == {"*"} and starred.declares("anything")
    assert predicate.hosted is None and predicate.declares("meteo")
    assert every.hosted is None and every.declares("anything")
    with pytest.raises(TypeError, match="not one id"):
        TurnExecutor(lambda _: None, hosts="meteo")
    with pytest.raises(ValueError, match="prefix"):
        TurnExecutor(lambda _: None, hosts={"org-*-sales"})


def test_an_executor_tells_the_names_of_its_room_hooks() -> None:
    executor = TurnExecutor(lambda _: None, hosts={"meteo"})

    @executor.room_hook("audit", HookTrigger.ON_AI_RESPONSE)
    async def audit(event: Any, context: Any) -> None: ...

    assert executor.hooks.defined_room_hooks() == {"audit"}


def test_a_backend_s_timeouts_are_the_kit_s_and_its_own_executor_s() -> None:
    backend = InMemoryTurnBackend()
    backend.timeouts = QUICK

    client = registry_turn_client({}, backend)

    assert client._timeouts is QUICK
    assert client.executor is not None and client.executor._timeouts is QUICK


# -- A beat outside the turn's messages ---------------------------------------------


class _SilentLink:
    """An executor took the turn, then sends nothing: it beats elsewhere."""

    def __init__(self, request: Any, *, beats: bool) -> None:
        self.request = request
        self.taken = True
        self._beats = beats
        self._accepted = False
        self.asked = 0

    async def receive(self) -> Any:
        if not self._accepted:
            self._accepted = True
            return 1, Accepted(executor_id="far")
        await asyncio.Event().wait()

    async def send(self, message: Any) -> None:
        return None

    def release(self) -> None:
        return None

    def beat_heard(self) -> bool:
        self.asked += 1
        return self._beats


def _request() -> Any:
    return turn_request(
        RoomEvent(
            room_id="room",
            source=EventSource(channel_id="sms", channel_type=ChannelType.SMS),
            content=TextContent(body="hello"),
        ),
        ChannelBinding(
            channel_id="agent",
            room_id="room",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        ),
        RoomContext(room=Room(id="room")),
    )


async def test_a_turn_whose_executor_beats_outside_its_messages_is_not_lost() -> None:
    link = _SilentLink(_request(), beats=True)
    turn = OwnerTurn(link, TICK, ROOMKIT_ERRORS)  # type: ignore[arg-type]
    output = asyncio.ensure_future(turn.output())
    # Twice the silent intervals that lose a turn, each asking for the beat.
    async with asyncio.timeout(5):
        while link.asked < 2 * TICK.silent_intervals:
            await asyncio.sleep(TICK.heartbeat_interval)

    assert not output.done()
    output.cancel()
    await asyncio.gather(output, return_exceptions=True)
    turn.abandon()


async def test_a_turn_whose_executor_stopped_beating_is_lost() -> None:
    turn = OwnerTurn(_SilentLink(_request(), beats=False), TICK, ROOMKIT_ERRORS)  # type: ignore[arg-type]

    with pytest.raises(TurnLinkError) as raised:
        await asyncio.wait_for(turn.output(), 5 * TICK.heartbeat_timeout)

    assert raised.value.reason == "executor_lost"


class _NeverAccepted(_SilentLink):
    """No executor took the turn, though something beats."""

    async def receive(self) -> Any:
        await asyncio.Event().wait()


async def test_a_beat_before_an_executor_took_the_turn_does_not_hold_it() -> None:
    turn = OwnerTurn(_NeverAccepted(_request(), beats=True), TICK, ROOMKIT_ERRORS)  # type: ignore[arg-type]

    with pytest.raises(TurnLinkError):
        await asyncio.wait_for(turn.output(), 5 * TICK.accept)


# -- The optional dependency --------------------------------------------------------


def test_the_backend_without_redis_names_the_extra() -> None:
    fakes: dict[str, Any] = {"redis": None, "redis.asyncio": None}
    with patch.dict(sys.modules, fakes):
        module = load_private_copy("roomkit.turns.redis", fakes)
        with pytest.raises(ImportError, match=r"pip install roomkit\[redis\]"):
            module.RedisTurnBackend()


def test_two_rooms_whose_ids_read_alike_hold_apart() -> None:
    """``support:fr`` in ``acme`` and ``support`` in ``fr:acme`` are two holders."""
    placement = Placement(None, "p", "node", None, loop_back=False)  # type: ignore[arg-type]

    assert placement._holder_key("support:fr", "acme") != placement._holder_key(
        "support", "fr:acme"
    )


def test_an_executor_restarted_elsewhere_is_another_member() -> None:
    assert member_of("node-a", "host-1") != member_of("node-b", "host-1")


def test_a_backend_given_a_client_needs_the_url_its_beat_connects_to() -> None:
    pytest.importorskip("redis.asyncio")

    with pytest.raises(ValueError, match="needs url"):
        RedisTurnBackend(client=object())


# -- An entry written twice ---------------------------------------------------------


class _Posted:
    """What a process posted: (node, kind, body, ref), and how many times
    each may be written."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str, str]] = []
        self.writes: list[tuple[int | None, float | None]] = []

    def __call__(
        self,
        node: str,
        kind: str,
        body: str,
        target: str,
        ref: str = "",
        *,
        failed: Any = None,
        writes: int | None = 1,
        within: float | None = None,
    ) -> None:
        self.messages.append((node, kind, body, ref))
        self.writes.append((writes, within))

    def of(self, kind: str) -> list[tuple[str, str, str, str]]:
        return [m for m in self.messages if m[1] == kind]


class _Taker:
    """An executor of this process counting the turns and controls it got."""

    executor_id = "host-a"
    closed = False

    def __init__(self) -> None:
        self.taken: list[str] = []
        self.applied: list[str] = []

    def take(self, link: Any) -> None:
        self.taken.append(link.request.turn_id)

    async def control(self, message: Any) -> None:
        await asyncio.sleep(0)
        self.applied.append(message.type)


class _Owner:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.publications: list[str] = []

    async def room_call(self, message: RoomCall) -> HostResult:
        self.calls.append(message.verb)
        return HostResult(call_id=message.call_id)

    async def room_publication(self, message: RoomPublication) -> None:
        self.publications.append(message.turn_id)


def _routes() -> tuple[Routes, _Posted, list[asyncio.Task[None]]]:
    posted = _Posted()
    tasks: list[asyncio.Task[None]] = []
    routes = Routes(
        "node-b",
        posted,
        lambda work, name: tasks.append(asyncio.ensure_future(work)),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        end_bound=1.0,
    )
    return routes, posted, tasks


def test_a_request_written_twice_is_taken_once_and_answered_twice() -> None:
    routes, posted, _ = _routes()
    taker = _Taker()
    routes.shared["host-a"] = taker  # type: ignore[assignment]
    request = _request()
    asked = Envelope("request", "node-a", request.model_dump_json(), "host-a", "ref-1")

    routes.route(asked)
    routes.route(asked)

    assert taker.taken == [request.turn_id]
    assert posted.of("answer") == [("node-a", "answer", TAKEN, "ref-1")] * 2


async def test_a_channel_control_written_twice_is_applied_once_and_answered_twice() -> None:
    routes, posted, tasks = _routes()
    taker = _Taker()
    routes.shared["host-a"] = taker  # type: ignore[assignment]
    raw = ChannelControlEnvelope(message=RoomDetached(channel_id="agent", room_id="room"))
    asked = Envelope("channel", "node-a", raw.model_dump_json(), "host-a", "ref-2")

    routes.route(asked)
    routes.route(asked)  # while the first is applied: its answer is coming
    await asyncio.gather(*tasks)
    routes.route(asked)  # once answered: answered again

    assert taker.applied == ["room_detached"]
    assert [m[3] for m in posted.of("answer")] == ["ref-2", "ref-2"]


async def test_work_after_a_turn_written_twice_reaches_its_owner_once() -> None:
    routes, posted, tasks = _routes()
    owner = _Owner()
    routes.owner = owner  # type: ignore[assignment]
    call = RoomCall(
        channel_id="agent", room_id="room", executor_id="host-a", turn_id="t-1", verb="send"
    )
    published = RoomPublication(
        channel_id="agent",
        room_id="room",
        executor_id="host-a",
        turn_id="t-1",
        frame=Ephemeral(channel="room:room", event={}),
    )
    for kind, message, ref in (("owner", call, "ref-3"), ("work", published, "ref-4")):
        body = OwnerMessageEnvelope(message=message).model_dump_json()
        routes.route(Envelope(kind, "node-a", body, "", ref))
        routes.route(Envelope(kind, "node-a", body, "", ref))
    await asyncio.gather(*tasks)

    assert owner.calls == ["send"]
    assert owner.publications == ["t-1"]


async def test_answers_and_work_after_a_turn_are_written_until_they_are() -> None:
    routes, posted, _ = _routes()
    routes.shared["host-a"] = _Taker()  # type: ignore[assignment]
    request = _request()
    routes.route(Envelope("request", "node-a", request.model_dump_json(), "host-a", "ref-5"))

    await routes.port.room_publication(
        RoomPublication(
            channel_id="agent",
            room_id="room",
            executor_id="host-a",
            turn_id=request.turn_id,
            frame=Ephemeral(channel="room:room", event={}),
        )
    )

    assert [m[1] for m in posted.messages] == ["answer", "work"]
    # An answer until it is written; work after a turn for less long than
    # its receiver remembers it (written twice, published once).
    assert posted.writes == [(None, None), (None, _redis_routes._WORK_FOR)]
    assert _redis_routes._WORK_FOR < _redis_routes._PUBLISHED_FOR
    assert posted.of("work")[0][3] != ""  # a reference: handled once on arrival


def test_a_request_reaching_a_closing_process_is_refused_to_be_placed_again() -> None:
    routes, posted, _ = _routes()
    taker = _Taker()
    routes.shared["host-a"] = taker  # type: ignore[assignment]
    routes.close()

    routes.route(Envelope("request", "node-a", _request().model_dump_json(), "host-a", "ref-6"))

    assert taker.taken == []
    assert posted.of("answer") == [("node-a", "answer", REFUSED, "ref-6")]


async def test_a_closing_process_asks_nothing() -> None:
    routes, posted, _ = _routes()
    routes.close()

    with pytest.raises(TurnLinkError) as raised:
        await routes.ask("node-a", "owner", "{}", "")

    assert raised.value.reason == "backend_closed"
    assert posted.messages == []


class _Raises(_Taker):
    async def control(self, message: Any) -> None:
        raise RuntimeError("the channel's factory is broken")


class _OwnerRaises(_Owner):
    async def room_call(self, message: RoomCall) -> HostResult:
        raise RuntimeError("the store is down")


async def test_a_channel_control_that_raised_is_answered_with_its_error() -> None:
    """Its owner waits for the answer while this process beats."""
    routes, posted, tasks = _routes()
    routes.shared["host-a"] = _Raises()  # type: ignore[assignment]
    raw = ChannelControlEnvelope(message=RoomDetached(channel_id="agent", room_id="room"))
    routes.route(Envelope("channel", "node-a", raw.model_dump_json(), "host-a", "ref-7"))
    await asyncio.gather(*tasks)

    (answer,) = posted.of("answer")
    assert "the channel's factory is broken" in answer[2]


async def test_a_room_call_that_raised_is_answered_with_its_error() -> None:
    routes, posted, tasks = _routes()
    routes.owner = _OwnerRaises()  # type: ignore[assignment]
    call = RoomCall(
        channel_id="agent", room_id="room", executor_id="host-a", turn_id="t-1", verb="send"
    )
    body = OwnerMessageEnvelope(message=call).model_dump_json()
    routes.route(Envelope("owner", "node-a", body, "", "ref-8"))
    await asyncio.gather(*tasks)

    (answer,) = posted.of("answer")
    assert HostResult.model_validate_json(answer[2]).error is not None


def test_a_question_handled_long_ago_is_forgotten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_redis_routes, "_HANDLED_FOR", 0.0)
    routes, posted, _ = _routes()
    taker = _Taker()
    routes.shared["host-a"] = taker  # type: ignore[assignment]
    first, second = _request(), _request()

    routes.route(Envelope("request", "node-a", first.model_dump_json(), "host-a", "ref-9"))
    routes.route(Envelope("request", "node-a", second.model_dump_json(), "host-a", "ref-10"))

    assert list(routes._handled) == ["ref-10"]


# -- The view of the others ---------------------------------------------------------


class _Registry:
    def __init__(self) -> None:
        self.down = False
        self.silent = False
        self.fail_once = False
        self.failed = asyncio.Event()
        self.hanging = asyncio.Event()

    async def view(self, forget_after_ms: int) -> View:
        if self.silent:  # the host stopped answering without a reset
            self.hanging.set()
            await asyncio.Event().wait()
        if self.down or self.fail_once:
            self.fail_once = False
            self.failed.set()
            raise ConnectionError("Connection refused")
        return View(now=1_000, beats={"node-a:w": 1_000}, owners={"node-a": 1_000})

    async def running(self, turns: dict[str, list[str]]) -> dict[str, frozenset[str]]:
        return {}


async def _read_once(live: LiveExecutors) -> None:
    """Until *live*'s first read is in: node-a's executor known alive."""
    async with asyncio.timeout(5):
        while not live.alive("node-a:w"):
            await asyncio.sleep(0.005)


async def _unread_within(live: LiveExecutors, window: float) -> None:
    """Until *live* no longer takes node-a's executor for alive, failing past
    *window*: the time the test grants the view to give it up."""
    async with asyncio.timeout(window):
        while live.alive("node-a:w"):
            await asyncio.sleep(0.005)


@pytest.mark.wallclock
async def test_the_others_unread_for_the_heartbeat_timeout_are_not_alive() -> None:
    """Redis unreachable: an owner stops waiting for an executor it cannot
    hear of, rather than trust the last view for good."""
    registry = _Registry()
    live = LiveExecutors(registry, 0.01, 0.1, dict)  # type: ignore[arg-type]
    live.start()
    try:
        await _read_once(live)
        assert live.owner_alive("node-a")

        registry.down = True
        await _unread_within(live, 0.3)

        assert not live.owner_alive("node-a")
    finally:
        await live.close()


# -- The writers --------------------------------------------------------------------

_TURN: contextvars.ContextVar[str] = contextvars.ContextVar("turn")


class _Client:
    """A Redis client whose pipelines take *delay* to write."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.written: list[str] = []
        self.contexts: list[str | None] = []

    def pipeline(self, transaction: bool) -> _Pipe:
        return _Pipe(self)


class _Pipe:
    def __init__(self, client: _Client) -> None:
        self._client = client
        self._bodies: list[str] = []

    def xadd(self, key: str, fields: dict[str, str]) -> None:
        self._bodies.append(fields["body"])

    def pexpire(self, key: str, ttl: int) -> None:
        return None

    async def execute(self) -> None:
        await asyncio.sleep(self._client.delay)
        self._client.contexts.append(_TURN.get(None))
        self._client.written.extend(self._bodies)


def _writers() -> list[asyncio.Task[Any]]:
    return [t for t in asyncio.all_tasks() if t.get_name().startswith("roomkit-turns-outbox-")]


def _message(body: str) -> Envelope:
    return Envelope("frame", "node-a", body, "turn-1")


async def test_a_writer_left_idle_goes_and_the_next_message_starts_another() -> None:
    client = _Client()
    outbox = Outbox(client, lambda node: f"inbox:{node}", ttl_ms=50)
    for node in ("a", "b", "c"):
        outbox.post(node, _message(node))
    async with asyncio.timeout(0.2):  # idle past their ttl, the writers go
        while _writers():
            await asyncio.sleep(0.005)

    outbox.post("a", _message("again"))
    async with asyncio.timeout(1):
        while len(client.written) < 4:
            await asyncio.sleep(0.005)

    assert client.written == ["a", "b", "c", "again"]
    await outbox.close(drain=1.0)


async def test_a_writer_carries_no_context_of_the_turn_that_posted_first() -> None:
    client = _Client()
    outbox = Outbox(client, lambda node: f"inbox:{node}", ttl_ms=60_000)
    _TURN.set("turn-1 span")
    outbox.post("a", _message("m"))
    async with asyncio.timeout(1):
        while not client.contexts:
            await asyncio.sleep(0.005)

    assert client.contexts == [None]
    await outbox.close(drain=1.0)


async def test_closing_writes_what_is_queued() -> None:
    client = _Client(delay=0.02)
    outbox = Outbox(client, lambda node: f"inbox:{node}", ttl_ms=60_000)
    for body in ("m1", "m2", "m3"):
        outbox.post("a", _message(body))
        await asyncio.sleep(0)

    await outbox.close(drain=1.0)

    assert client.written == ["m1", "m2", "m3"]


async def test_closing_tells_the_senders_of_what_it_could_not_write() -> None:
    client = _Client(delay=1.0)
    outbox = Outbox(client, lambda node: f"inbox:{node}", ttl_ms=60_000)
    told: list[str] = []
    outbox.post("a", _message("m1"), lambda error: told.append("m1"))
    await asyncio.sleep(0.01)  # m1 is being written
    outbox.post("a", _message("m2"), lambda error: told.append("m2"))

    await outbox.close(drain=0.05)
    outbox.post("a", _message("m3"), lambda error: told.append("m3"))

    assert client.written == []
    assert told == ["m1", "m2", "m3"]


async def test_a_sender_is_told_its_message_was_written_once_redis_took_it() -> None:
    client = _Client(delay=0.05)
    outbox = Outbox(client, lambda node: f"inbox:{node}", ttl_ms=60_000)
    told: list[list[str]] = []
    outbox.post("a", _message("end"), written=lambda: told.append(list(client.written)))
    await asyncio.sleep(0.01)  # being written

    assert told == []
    await outbox.close(drain=1.0)
    assert told == [["end"]]


async def test_a_message_redis_refused_is_never_told_written() -> None:
    outbox = Outbox(_RefusesAll(), lambda node: f"inbox:{node}", ttl_ms=60_000)
    told: list[str] = []
    outbox.post(
        "a",
        _message("end"),
        lambda error: told.append("refused"),
        written=lambda: told.append("written"),
    )
    async with asyncio.timeout(1):
        while not told:
            await asyncio.sleep(0.005)

    assert told == ["refused"]
    await outbox.close(drain=0.05)


@pytest.mark.wallclock
async def test_the_others_read_by_a_read_that_hangs_are_not_alive() -> None:
    registry = _Registry()
    live = LiveExecutors(registry, 0.01, 0.1, dict)  # type: ignore[arg-type]
    live.start()
    try:
        await _read_once(live)
        registry.silent = True
        await _unread_within(live, 0.5)  # two reads hang a heartbeat timeout each
    finally:
        await live.close()


@pytest.mark.wallclock
async def test_a_stall_of_this_process_after_one_failed_read_is_not_the_others_silence() -> None:
    """RFC §10.6.8: silence is counted only while the side counting it runs."""
    registry = _Registry()
    live = LiveExecutors(registry, 0.01, 0.1, dict)  # type: ignore[arg-type]
    live.start()
    try:
        await _read_once(live)
        registry.fail_once = True
        await registry.failed.wait()  # a read failed, and the next is not made yet
        time.sleep(0.2)  # noqa: ASYNC251 - this process's event loop stalls, twice the timeout

        assert live.alive("node-a:w")
    finally:
        await live.close()


class _RefusesFirst(_Client):
    """Refuses its first write."""

    def __init__(self) -> None:
        super().__init__()
        self.refused = False

    def pipeline(self, transaction: bool) -> _Pipe:
        if not self.refused:
            self.refused = True
            return _RefusedPipe(self)
        return _Pipe(self)


class _RefusedPipe(_Pipe):
    async def execute(self) -> None:
        raise ConnectionError("OOM command not allowed")


async def test_a_message_written_again_keeps_its_place_before_later_ones() -> None:
    """A typing indicator's start refused once still reaches the room
    before its stop."""
    client = _RefusesFirst()
    outbox = Outbox(client, lambda node: f"inbox:{node}", ttl_ms=60_000, retry_delay=0.02)
    told: list[str] = []
    outbox.post("a", _message("start"), writes=None)
    outbox.post("a", _message("frame"), lambda error: told.append("frame"))
    await asyncio.sleep(0.005)
    outbox.post("a", _message("stop"), writes=None)
    async with asyncio.timeout(0.1):  # the refused start, written again before the stop
        while len(client.written) < 2:
            await asyncio.sleep(0.005)

    assert client.written == ["start", "stop"]
    assert told == ["frame"]  # a turn's message is not written again
    await outbox.close(drain=1.0)


class _SlowRegistry(_Registry):
    """Every read answers, a little later than a heartbeat interval."""

    async def view(self, forget_after_ms: int) -> View:
        await asyncio.sleep(0.05)
        return await super().view(forget_after_ms)


@pytest.mark.wallclock
async def test_the_others_read_slower_than_an_interval_are_still_alive() -> None:
    live = LiveExecutors(_SlowRegistry(), 0.04, 0.2, dict)  # type: ignore[arg-type]
    await live.refresh()
    live.start()
    try:
        await asyncio.sleep(0.4)  # twice the heartbeat timeout, every read slow

        assert live.alive("node-a:w")
    finally:
        await live.close()


class _RefusesAll(_Client):
    def pipeline(self, transaction: bool) -> _Pipe:
        return _RefusedPipe(self)


async def test_a_turn_s_message_gets_its_write_behind_a_batch_written_again() -> None:
    """Redis refusing: answers kept to be written again never keep a turn's
    message from its one write, nor its sender from learning it failed."""
    outbox = Outbox(_RefusesAll(), lambda node: f"inbox:{node}", ttl_ms=60_000, retry_delay=0.01)
    told: list[str] = []
    for i in range(300):
        outbox.post("a", _message(f"answer-{i}"), writes=None)
    await asyncio.sleep(0.02)
    outbox.post("a", _message("frame"), lambda error: told.append("frame"))
    async with asyncio.timeout(0.1):  # its one write behind the batch, refused
        while not told:
            await asyncio.sleep(0.005)

    assert told == ["frame"]
    await outbox.close(drain=0.05)


async def test_a_channel_control_for_an_executor_this_process_never_served_fails() -> None:
    routes, posted, tasks = _routes()
    raw = ChannelControlEnvelope(message=RoomDetached(channel_id="agent", room_id="room"))
    routes.route(Envelope("channel", "node-a", raw.model_dump_json(), "host-x", "ref-11"))
    await asyncio.gather(*tasks)

    (answer,) = posted.of("answer")
    assert ChannelResult.model_validate_json(answer[2]).error is not None


async def test_what_reaches_a_closing_process_is_refused_without_being_applied() -> None:
    routes, posted, tasks = _routes()
    taker = _Taker()
    owner = _Owner()
    routes.shared["host-a"] = taker  # type: ignore[assignment]
    routes.owner = owner  # type: ignore[assignment]
    routes.close()
    control = ChannelControlEnvelope(message=RoomDetached(channel_id="agent", room_id="room"))
    call = RoomCall(
        channel_id="agent", room_id="room", executor_id="host-a", turn_id="t-1", verb="send"
    )
    published = RoomPublication(
        channel_id="agent",
        room_id="room",
        executor_id="host-a",
        turn_id="t-1",
        frame=Ephemeral(channel="room:room", event={}),
    )

    routes.route(Envelope("channel", "node-a", control.model_dump_json(), "host-a", "ref-12"))
    routes.route(
        Envelope(
            "owner", "node-a", OwnerMessageEnvelope(message=call).model_dump_json(), "", "r13"
        )
    )
    body = OwnerMessageEnvelope(message=published).model_dump_json()
    routes.route(Envelope("work", "node-a", body, "", "ref-14"))
    await asyncio.gather(*tasks)

    assert taker.applied == [] and owner.calls == [] and owner.publications == []
    channel_answer, call_answer = (m[2] for m in posted.of("answer"))
    assert channel_answer == REFUSED  # placed again by its owner
    assert HostResult.model_validate_json(call_answer).error is not None


@pytest.mark.wallclock
async def test_a_stall_of_this_process_with_a_read_in_flight_then_one_failure_is_not_silence() -> (
    None
):
    """The read in flight across the stall times out, the next one fails
    once: two failures, but not the others' silence for a heartbeat timeout."""
    registry = _Registry()
    live = LiveExecutors(registry, 0.02, 0.2, dict)  # type: ignore[arg-type]
    live.start()
    try:
        await _read_once(live)
        registry.silent = True  # the read in flight hangs
        await registry.hanging.wait()
        time.sleep(0.3)  # noqa: ASYNC251 - this process's event loop stalls past the timeout
        registry.silent = False
        registry.fail_once = True
        registry.failed.clear()
        await registry.failed.wait()

        assert live.alive("node-a:w")
    finally:
        await live.close()


async def test_a_message_written_again_for_a_while_is_given_up_after_it() -> None:
    outbox = Outbox(_RefusesAll(), lambda node: f"inbox:{node}", ttl_ms=60_000, retry_delay=0.01)
    told: list[str] = []
    outbox.post("a", _message("work"), lambda error: told.append("work"), writes=None, within=0.05)
    async with asyncio.timeout(0.2):  # given up past its 0.05 s
        while not told:
            await asyncio.sleep(0.005)

    assert told == ["work"]
    await outbox.close(drain=0.05)


class _HangsThenRefuses(_Registry):
    """Redis freezes (a read hangs), then refuses every connection."""

    def __init__(self) -> None:
        super().__init__()
        self.frozen = False
        self.hung = 0

    async def view(self, forget_after_ms: int) -> View:
        if self.frozen and not self.hung:
            self.hung += 1
            await asyncio.Event().wait()
        if self.frozen:
            raise ConnectionError("Connection refused")
        return await super().view(forget_after_ms)


@pytest.mark.wallclock
async def test_a_read_that_hung_counts_from_its_start() -> None:
    """A read that hung then one refused: the others are unread within
    about a heartbeat timeout of Redis freezing, not two."""
    registry = _HangsThenRefuses()
    live = LiveExecutors(registry, 0.1, 0.5, dict)  # type: ignore[arg-type]
    live.start()
    try:
        await _read_once(live)
        registry.frozen = True
        await _unread_within(live, 0.9)
    finally:
        await live.close()


class _ClientTimeoutError(Exception):
    """What a Redis client raises when its connection's bound cuts a call:
    not the builtin TimeoutError (redis-py's is not one)."""


class _CutThenRefuses(_Registry):
    """Redis freezes: the read in flight is cut by its connection's bound,
    half the heartbeat timeout; then every connection is refused."""

    def __init__(self, bound: float) -> None:
        super().__init__()
        self.bound = bound
        self.frozen = False
        self.cut = 0

    async def view(self, forget_after_ms: int) -> View:
        if self.frozen and not self.cut:
            self.cut += 1
            await asyncio.sleep(self.bound)
            raise _ClientTimeoutError("Timeout reading from redis")
        if self.frozen:
            raise ConnectionError("Connection refused")
        return await super().view(forget_after_ms)


@pytest.mark.wallclock
async def test_a_read_its_connection_cut_counts_from_its_start() -> None:
    """RMK-859: a read the client's bound cut counts as one that hung, from
    its start: the others are unread about a heartbeat timeout after Redis
    froze, not that plus the bound."""
    registry = _CutThenRefuses(bound=0.5)
    live = LiveExecutors(
        registry,  # type: ignore[arg-type]
        0.1,
        1.0,
        dict,
        timeout_errors=(TimeoutError, _ClientTimeoutError),
    )
    live.start()
    try:
        await _read_once(live)
        registry.frozen = True
        await _unread_within(live, 1.35)
    finally:
        await live.close()


@pytest.mark.wallclock
async def test_a_stall_just_over_the_timeout_with_a_read_in_flight_is_not_silence() -> None:
    """The read in flight across a stall a little longer than the timeout
    times out late, the next fails once: not the others' silence."""
    registry = _Registry()
    live = LiveExecutors(registry, 0.08, 0.2, dict)  # type: ignore[arg-type]
    live.start()
    try:
        await _read_once(live)
        registry.silent = True
        await registry.hanging.wait()  # a read is in flight
        time.sleep(0.25)  # noqa: ASYNC251 - this process's event loop stalls, 1.25 timeouts
        registry.silent = False
        registry.fail_once = True
        registry.failed.clear()
        await registry.failed.wait()

        assert live.alive("node-a:w")
    finally:
        await live.close()


async def test_a_channel_control_for_a_closed_executor_is_refused_to_be_placed_again() -> None:
    routes, posted, tasks = _routes()
    closed = _Taker()
    closed.closed = True  # type: ignore[misc]
    routes.shared["host-a"] = closed  # type: ignore[assignment]
    raw = ChannelControlEnvelope(message=RoomDetached(channel_id="agent", room_id="room"))

    routes.route(Envelope("channel", "node-a", raw.model_dump_json(), "host-a", "ref-15"))
    await asyncio.gather(*tasks)

    assert closed.applied == []
    assert posted.of("answer") == [("node-a", "answer", REFUSED, "ref-15")]


class _Holders:
    """No room has a holder yet."""

    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, ex: int) -> None:
        return None


class _Alive:
    def __init__(self, members: list[str]) -> None:
        agent = Hosted(frozenset({"agent"}))
        self._declared = {
            member: Declaration(member.split(":")[0], "w", agent, frozenset())
            for member in members
        }

    async def current(self) -> dict[str, Declaration]:
        return self._declared


async def test_an_event_held_for_a_room_no_one_holds_goes_where_its_next_turn_goes() -> None:
    """RFC §10.6.3: a committed event goes where the room's turns would."""
    placement = Placement(
        _Holders(), "p", "node-a", _Alive(["node-b:w", "node-c:w"]), loop_back=False
    )  # type: ignore[arg-type]

    for _ in range(3):
        held = await placement.hold("agent", "room", RoomNeeds())
        placed = await placement.place("agent", "room", RoomNeeds())

        assert held == placed


class _HangingPipeline:
    """A Redis pipeline on a connection dropped without a word: its write
    never returns."""

    def xadd(self, key: str, fields: dict[str, str]) -> None:
        pass

    def pexpire(self, key: str, ttl: int) -> None:
        pass

    async def execute(self) -> None:
        await asyncio.Event().wait()


class _HangingClient:
    def pipeline(self, transaction: bool) -> _HangingPipeline:
        return _HangingPipeline()


async def test_a_turn_whose_end_write_hangs_stops_running_after_the_end_bound() -> None:
    outbox = Outbox(_HangingClient(), lambda node: f"inbox:{node}", ttl_ms=60_000)

    def post(
        node: str,
        kind: str,
        body: str,
        target: str,
        ref: str = "",
        failed: Any = None,
        *,
        writes: int | None = 1,
        within: float | None = None,
        written: Any = None,
    ) -> None:
        envelope = Envelope(kind, "node-b", body, target=target, ref=ref)
        outbox.post(node, envelope, failed, writes=writes, within=within, written=written)

    let_go: list[RedisExecutorLink] = []
    request = _request()
    link = RedisExecutorLink(request, "node-a", post, let_go.append, end_bound=0.05)
    try:
        await link.send(End())
        link.turn_ended()
        await asyncio.sleep(0.01)
        # Its end is on its way: the turn still runs here.
        assert let_go == []
        async with asyncio.timeout(0.1):  # past the end bound, it is let go
            while not let_go:
                await asyncio.sleep(0.005)
        assert let_go == [link]
    finally:
        await outbox.close(drain=0.01)


@pytest.mark.parametrize(
    ("timeouts", "bound", "inbox"),
    [
        (TurnTimeouts(), 5.0, 6.0),
        (TurnTimeouts(heartbeat_interval=0.5, heartbeat_timeout=2.0), 1.0, 1.5),
    ],
)
def test_every_connection_gives_up_within_half_the_silence_bound(
    timeouts: TurnTimeouts, bound: float, inbox: float
) -> None:
    """RMK-859: a call on a half-open connection gives up within half of
    heartbeat_timeout, the inbox's blocking read the time the server holds
    it more (a second, never over a beat's interval), whatever redis-py's
    own default, so the call tried again at once lands inside the silence
    bound."""
    exceptions = pytest.importorskip("redis.exceptions")
    clients = redis_clients("redis://localhost:6379", timeouts)
    beat = clients.beat()

    for client, waits in ((clients.pooled, bound), (clients.inbox, inbox), (beat, bound)):
        options = client.connection_pool.connection_kwargs
        assert (options["socket_timeout"], options["socket_connect_timeout"]) == (waits, bound)
        assert "socket_keepalive" not in options
    assert exceptions.TimeoutError in clients.timeout_errors
    beat.close()


def test_the_clients_build_on_a_redis_package_imported_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RMK-859: a redis package imported again while its submodules stay
    loaded lacks them as attributes: the clients still name the error a cut
    call raises, read from its own module."""
    exceptions = pytest.importorskip("redis.exceptions")
    monkeypatch.delitem(sys.modules, "redis")

    clients = redis_clients("redis://localhost:6379", TurnTimeouts())

    assert exceptions.TimeoutError in clients.timeout_errors


def test_a_client_given_is_the_pool_and_none_is_built() -> None:
    """RMK-859: with client=, the backend writes through it: no pool of its
    own is built."""
    pytest.importorskip("redis")
    given = object()

    with patch("redis.asyncio.BlockingConnectionPool.from_url") as pool:
        clients = redis_clients("redis://localhost:6379", TurnTimeouts(), given)

    assert clients.pooled is given
    assert not pool.called
