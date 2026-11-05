"""One Prism connection: acknowledged commands, continuous input and cleanup."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.core.task_utils import cancel_and_wait, log_task_exception
from roomkit.voice.base import VoiceSessionState
from roomkit.voice.realtime._rime_events import PrismEvents
from roomkit.voice.realtime._rime_tools import PrismTools
from roomkit.voice.realtime.injection import VoiceInjectionResult

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.rime import RimeRealtimeProvider


class PrismRequestError(RuntimeError):
    """The service explicitly refused a command, rather than losing its acknowledgement."""

    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(error.get("message", "Prism request failed"))
        self.code = error.get("code", "unknown")


class PrismSession:
    def __init__(
        self,
        owner: RimeRealtimeProvider,
        session: VoiceSession,
        ws: Any,
        timeout: float,
    ) -> None:
        self.owner, self.session, self.ws, self.timeout = owner, session, ws, timeout
        self.closed = False
        self.stopped = asyncio.Event()
        self.closing_task: asyncio.Task[Any] | None = None
        self.terminal_error = False
        self.settings: dict[str, Any] = {}
        self.ready = asyncio.Event()
        self.send_lock = asyncio.Lock()
        self.update_lock = asyncio.Lock()
        self.inject_lock = asyncio.Lock()
        self.tasks: set[asyncio.Task[Any]] = set()
        self.pending: dict[str, tuple[str, asyncio.Future[dict[str, Any]]]] = {}
        self.audio = bytearray()
        self.response_id: str | None = None
        self.response_requested: str | None = None
        self.speaking = False
        self.output_item: str | None = None
        self.output_bytes = 0
        self.tools = PrismTools(self)
        self.events = PrismEvents(self)

    async def initialize(self, settings: dict[str, Any]) -> None:
        async with asyncio.timeout(self.timeout):
            first = json.loads(await self.ws.recv())
            if first.get("type") != "session.created":
                raise RuntimeError("Prism did not create a session")
            await self.send({"type": "session.update", "session": settings})
            while True:
                event = json.loads(await self.ws.recv())
                if event.get("type") == "error":
                    raise PrismRequestError(event.get("error", {}))
                if event.get("type") == "session.updated":
                    self.settings = event["session"]
                    self.session.provider_session_id = self.settings.get("id")
                    break
        self.session.state = VoiceSessionState.ACTIVE
        self.spawn(self._receive(), "receive")
        self.spawn(self._input(), "input")

    def spawn(self, coroutine: Coroutine[Any, Any, Any], label: str) -> asyncio.Task[Any]:
        task = self.owner._session_task(coroutine, name=f"rime:{label}:{self.session.id}")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(log_task_exception)
        return task

    async def fire(self, kind: str, *args: Any) -> None:
        await self.owner._fire(
            getattr(self.owner, f"_{kind}_callbacks"), self.session, *args, label=f"rime_{kind}"
        )

    async def send(self, event: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("Prism session is closed")
        raw = json.dumps(event)
        if len(raw.encode()) > 1024 * 1024:
            raise ValueError("Prism events must fit in 1 MiB")
        async with self.send_lock:
            await self.ws.send(raw)

    async def request(self, event: dict[str, Any], expected: str) -> dict[str, Any]:
        event_id = uuid4().hex
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.pending[event_id] = (expected, future)
        if expected == "response.created":
            self.response_requested = event_id
        try:
            await self.send({**event, "event_id": event_id})
            return await asyncio.wait_for(future, self.timeout)
        finally:
            self.pending.pop(event_id, None)
            if not future.done():
                future.cancel()

    def acknowledge(self, event: dict[str, Any]) -> None:
        error = event.get("error") if event.get("type") == "error" else None
        event_id = error.get("event_id") if error else event.get("prism_request_event_id")
        pending = self.pending.get(event_id)
        if pending is None or pending[1].done():
            return
        expected, future = pending
        if error:
            future.set_exception(PrismRequestError(error))
        elif event.get("type") == expected:
            future.set_result(event)

    async def inject(
        self,
        event: dict[str, Any],
        expected: str,
        *,
        wait_ready: bool,
    ) -> VoiceInjectionResult:
        async with self.inject_lock:
            if wait_ready:
                try:
                    await asyncio.wait_for(self.ready.wait(), self.timeout)
                except TimeoutError:
                    return VoiceInjectionResult(status="not_sent", reason="busy", retryable=True)
                self.ready.clear()
            if self.closed:
                return VoiceInjectionResult(status="not_sent", reason="disconnected")
            try:
                await self.request(event, expected)
            except PrismRequestError as exc:
                retryable = exc.code in {
                    "typed_turn_busy",
                    "instructions_busy",
                    "proactive_unavailable",
                }
                return VoiceInjectionResult(
                    status="not_sent", reason=exc.code, retryable=retryable
                )
            except Exception:
                # The send may have reached the service: do not invite a replay.
                return VoiceInjectionResult(status="unknown", reason="acknowledgement_lost")
        return VoiceInjectionResult(status="sent")

    def append_audio(self, audio: bytes) -> None:
        if len(audio) % 2:
            raise ValueError("Prism input must contain whole PCM16 samples")
        if len(self.audio) + len(audio) > 5 * 32000:
            raise ValueError("Prism input is more than five seconds ahead of real time")
        self.audio.extend(audio)

    async def _input(self) -> None:
        """Pace 20 ms input frames and fill gaps without replaying capture."""
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        try:
            while not self.closed:
                chunk = bytes(self.audio[:640])
                del self.audio[:640]
                await self.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk.ljust(640, b"\0")).decode(),
                    }
                )
                deadline = max(deadline + 0.02, loop.time())
                await asyncio.sleep(max(0, deadline - loop.time()))
        except Exception:
            if not self.closed:
                await self.fail("input_failed", "Prism audio input failed")

    async def _receive(self) -> None:
        try:
            async for raw in self.ws:
                event = json.loads(raw)
                self.acknowledge(event)
                await self.events.handle(event)
            if not self.closed and not self.terminal_error:
                await self.fail("connection_closed", "Prism closed the connection")
        except Exception:
            if not self.closed and not self.terminal_error:
                await self.fail("connection_failed", "Prism connection failed")
        finally:
            await self.close()

    async def fail(self, code: str, message: str) -> None:
        # The channel recognizes fatal errors by the shared session's state.
        # Mark it before the callback so it also releases its audio transport.
        self.terminal_error = True
        self.session.state = VoiceSessionState.ENDED
        try:
            await self.fire("error", code, message)
        finally:
            await self.close()

    async def close(self) -> None:
        current = asyncio.current_task()
        if self.closed:
            # A cancelled receiver enters close from its finally block. Waiting
            # there would deadlock the task which is already joining it.
            if current not in self.tasks and current is not self.closing_task:
                await self.stopped.wait()
            return
        self.closed = True
        self.closing_task = current
        try:
            await self._stop()
        finally:
            with contextlib.suppress(Exception):
                await self.ws.close()
            self.session.state = VoiceSessionState.ENDED
            if self.owner._sessions.get(self.session.id) is self:
                self.owner._sessions.pop(self.session.id, None)
            self.stopped.set()

    async def _stop(self) -> None:
        self.ready.set()
        for _, future in self.pending.values():
            if not future.done():
                future.set_exception(
                    RuntimeError("Prism connection closed before acknowledgement")
                )
        for task in tuple(self.tasks):
            if task is not self.closing_task:
                await cancel_and_wait(task)
        await self.owner._abandon_open_tool_calls(self.session)
        if self.response_id:
            self.response_id = None
            await self.fire("response_end")
        self.response_requested = None
        self.audio.clear()
        self.tools.clear()
