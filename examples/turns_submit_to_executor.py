"""A turn submitted to the worker a host reserved (RFC §10.6.7).

RoomKit places a room's turns itself: on the executor that ran the room's
last one while it lives, else on one that declares the agent. A host that
places them by rules of its own (free slots, memory, an operating system)
reserves a worker first, then hands the turn to that worker and no other:
``TurnBackend.submit_to(request, executor)``.

``SlotBackend`` is such a host's backend. It wraps the turn backend the kit
is given, holds a count of free slots per worker, and for each turn reserves
the worker with the most free ones, submits the turn to it, and gives the
slot back when the turn is let go. What ``submit_to`` raises tells it what it
may do next:

- ``no_executor``: the worker did not take the turn (not served, closed,
  not serving the room's organization or hooks) and nothing was handed, to it
  or to any other. The host gives the slot back and reserves another worker;
- any other reason comes after the hand-over (``executor_lost`` when the
  worker never answered): the worker may have begun the turn, so the host
  never submits it again elsewhere.

1. ``small`` (one slot) runs the room's first turn.
2. ``big`` (four slots) starts: the host reserves it for the next turn, which
   runs there though ``small`` held the room, and the room follows.
3. ``big`` closes while the host still counts its slots: its reservation is
   refused before anything is handed, and the turn runs on ``small``.

Everything runs in this process (``InMemoryTurnBackend``); ``RedisTurnBackend``
takes the same call across processes, where an executor is named by its
``executor_id`` and, when several processes serve that id, by its ``node``
too, as ``kit.executors()`` lists both.

Run with:
    uv run python examples/turns_submit_to_executor.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial

from roomkit import (
    Agent,
    AgentIdentity,
    ChannelCategory,
    ChannelDeclaration,
    InboundMessage,
    RoomKit,
    TextContent,
    TurnLinkError,
    WebSocketChannel,
)
from roomkit.providers.ai import MockAIProvider
from roomkit.turns import (
    ChannelControl,
    ChannelResult,
    ExecutorInfo,
    InMemoryTurnBackend,
    RoomOwner,
    TurnBackend,
    TurnExecutor,
    TurnLink,
    TurnRequest,
)
from roomkit.turns.wire import Control, Frame


class SlotLink(TurnLink):
    """A turn's link that gives its worker's slot back once the turn is let go."""

    def __init__(self, link: TurnLink, give_back: Callable[[], None]) -> None:
        self.request = link.request
        self._link = link
        self._give_back: Callable[[], None] | None = give_back

    @property
    def taken(self) -> bool:
        return self._link.taken

    async def receive(self) -> tuple[int, Frame]:
        return await self._link.receive()

    async def send(self, message: Control) -> None:
        await self._link.send(message)

    def release(self) -> None:
        self._link.release()
        give_back, self._give_back = self._give_back, None
        if give_back is not None:
            give_back()

    def beat_heard(self) -> bool:
        return self._link.beat_heard()


class SlotBackend(TurnBackend):
    """The host's turn backend: each turn goes to the worker whose slot the
    host reserved, by a rule RoomKit knows nothing of."""

    def __init__(self, inner: TurnBackend, slots: dict[str, int]) -> None:
        self._inner = inner
        self.timeouts = inner.timeouts
        self._free = dict(slots)

    async def submit(self, request: TurnRequest) -> TurnLink:
        while True:
            worker = self._reserve()
            try:
                link = await self._inner.submit_to(request, ExecutorInfo(executor_id=worker))
            except BaseException as error:
                if not _refused(error, request):
                    # Handed, or cut: the worker may have begun it. Never elsewhere.
                    self._give_back(worker)
                    raise
                print(f"    {worker} refused it before it was handed: reserving another")
                del self._free[worker]
                continue
            print(f"    reserved {worker}, which took the turn")
            return SlotLink(link, partial(self._give_back, worker))

    def _reserve(self) -> str:
        """Take a slot of the worker with the most free ones."""
        worker = max(self._free, key=lambda name: self._free[name], default=None)
        if worker is None or self._free[worker] == 0:
            raise TurnLinkError("no_executor", "No worker has a free slot")
        self._free[worker] -= 1
        return worker

    def _give_back(self, worker: str) -> None:
        if worker in self._free:
            self._free[worker] += 1

    def add(self, worker: str, slots: int) -> None:
        self._free[worker] = slots

    # The rest is the wrapped backend's.

    async def serve(self, executor: TurnExecutor) -> None:
        await self._inner.serve(executor)

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        return await self._inner.control(message)

    async def attend(self, owner: RoomOwner) -> None:
        await self._inner.attend(owner)

    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        return await self._inner.claim_presence(channel_id, room_id, online)

    async def executors(self) -> list[ExecutorInfo]:
        return await self._inner.executors()

    async def executors_now(self) -> list[ExecutorInfo]:
        return await self._inner.executors_now()

    def watch_executors(self, listener: Callable[[], Awaitable[None]]) -> None:
        self._inner.watch_executors(listener)

    async def close(self) -> None:
        await self._inner.close()


def _refused(error: BaseException, request: TurnRequest) -> bool:
    """Whether *error* says the worker did not take *request*, nothing handed."""
    refusal = isinstance(error, TurnLinkError) and error.reason == "no_executor"
    return refusal and not request.handed


def worker(executor_id: str) -> TurnExecutor:
    """An executor building ``coder`` with a model that names it."""

    def build(channel_id: str) -> Agent:
        reply = f"{channel_id} on {executor_id}."
        return Agent(channel_id, provider=MockAIProvider(responses=[reply] * 4))

    return TurnExecutor(build, hosts={"coder"}, executor_id=executor_id)


async def ask(kit: RoomKit, text: str) -> str:
    """Send *text* to the room; the agent's answer, or why there is none."""
    message = InboundMessage(channel_id="chat", sender_id="ana", content=TextContent(body=text))
    result = await kit.process_inbound(message, room_id="room")
    answers = [
        event.content.body
        for event in await kit.store.list_events("room")
        if event.source.channel_id == "coder" and isinstance(event.content, TextContent)
    ]
    return f"no answer: {result.error}" if result.error else answers[-1]


async def main() -> None:
    backend = SlotBackend(InMemoryTurnBackend(), {"small": 1})
    small, big = worker("small"), worker("big")
    await backend.serve(small)
    kit = RoomKit(turn_backend=backend)
    kit.register_channel(WebSocketChannel("chat"))
    kit.register_channel(ChannelDeclaration(channel_id="coder", identity=AgentIdentity()))
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "chat")
    await kit.attach_channel("room", "coder", category=ChannelCategory.INTELLIGENCE)

    print("1. small alone:")
    print(f"    {await ask(kit, 'First.')}")

    print("2. big starts with four free slots; small ran the room's last turn:")
    await backend.serve(big)
    backend.add("big", 4)
    print(f"    {await ask(kit, 'Second.')}")

    print("3. big closes, and the host still counts its slots:")
    await big.close()
    print(f"    {await ask(kit, 'Third.')}")

    await kit.close()
    await small.close()


if __name__ == "__main__":
    asyncio.run(main())
