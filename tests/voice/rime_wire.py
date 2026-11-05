"""Local WebSocket fixtures for Rime's three wire protocols."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from websockets.asyncio.server import ServerConnection, serve


@contextlib.asynccontextmanager
async def server(
    handler: Callable[[ServerConnection], Awaitable[None]],
) -> AsyncIterator[str]:
    errors: list[Exception] = []

    async def checked(ws: ServerConnection) -> None:
        try:
            assert ws.request is not None
            assert ws.request.headers["Authorization"] == "Bearer test-key"
            await handler(ws)
        except Exception as exc:
            errors.append(exc)

    async with serve(checked, "127.0.0.1", 0, close_timeout=0.1) as listener:
        port = listener.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}"
    if errors:
        raise errors[0]


async def send(ws: ServerConnection, **event: Any) -> None:
    await ws.send(json.dumps(event))


async def receive(ws: ServerConnection) -> dict[str, Any]:
    return json.loads(await ws.recv())


async def control(ws: ServerConnection) -> dict[str, Any]:
    """Prism keeps sending silence while its commands are tested."""
    while True:
        event = await receive(ws)
        if event.get("type") != "input_audio_buffer.append":
            return event


async def prism_handshake(ws: ServerConnection) -> dict[str, Any]:
    await send(ws, type="session.created", session={"id": "prism-test"})
    event = await control(ws)
    assert event["type"] == "session.update"
    settings = {"id": "prism-test", **event["session"]}
    await send(ws, type="session.updated", session=settings)
    await send(ws, type="prism.typed_input.ready")
    return settings
