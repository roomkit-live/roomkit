"""Per-stream serialized offload of inbound DSP onto a thread pool.

The inbound stage chain (resampler → AEC → denoiser → VAD → …) is CPU
work that otherwise runs on the event-loop thread, once per 20 ms frame
per session: the process's call ceiling is one core, and one slow stage
delays every other session's audio and all message traffic with it.

:class:`InboundFrameOffload` moves that chain onto a small thread pool
while keeping the one ordering that matters: frames of a given stream
are processed FIFO, one at a time — a per-stream queue is drained by at
most one worker at any moment, and only *which* streams run in parallel
is decided by the pool. The native stages release the GIL (ctypes,
onnxruntime, numpy), so pool threads yield real multi-core parallelism,
and the per-stream stage locks (e.g. Speex's) already make the stages
safe against the outbound path's ``feed_reference``.

Backpressure is per stream and bounded: when a stream's queue is full
the OLDEST frame is dropped and counted — late audio is worthless audio,
and a stalled consumer must never grow memory or hold the pool hostage.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger("roomkit.voice.pipeline")

# Frames buffered per stream before drop-oldest kicks in (~1.3 s @ 20 ms).
_DEFAULT_MAX_QUEUED_FRAMES = 64

# How many drops between WARNING lines, per stream.
_DROP_LOG_INTERVAL = 50


class InboundFrameOffload:
    """Runs per-stream frame work on a thread pool, FIFO within a stream."""

    def __init__(
        self,
        threads: int,
        *,
        max_queued_frames: int = _DEFAULT_MAX_QUEUED_FRAMES,
    ) -> None:
        if threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")
        self._pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="roomkit-dsp")
        self._lock = threading.Lock()
        self._pending: dict[str, deque[tuple[Callable[..., Any], tuple[Any, ...]]]] = {}
        self._active: set[str] = set()
        self._dropped: dict[str, int] = {}
        self._max_queued = max_queued_frames
        self._closed = False

    def submit(self, stream: str, fn: Callable[..., Any], *args: Any) -> None:
        """Queue ``fn(*args)`` for *stream*; safe from any thread."""
        with self._lock:
            if self._closed:
                return
            dq = self._pending.setdefault(stream, deque())
            if len(dq) >= self._max_queued:
                dq.popleft()
                drops = self._dropped.get(stream, 0) + 1
                self._dropped[stream] = drops
                if drops % _DROP_LOG_INTERVAL == 1:
                    logger.warning(
                        "Inbound DSP queue full for stream %s: dropped %d frame(s) "
                        "— the pool cannot keep up with this stream's frame rate",
                        stream,
                        drops,
                    )
            dq.append((fn, args))
            if stream in self._active:
                return
            self._active.add(stream)
        self._pool.submit(self._drain, stream)

    def _drain(self, stream: str) -> None:
        """Process *stream*'s queue to exhaustion. One drainer per stream."""
        while True:
            with self._lock:
                dq = self._pending.get(stream)
                if not dq:
                    self._active.discard(stream)
                    self._pending.pop(stream, None)
                    return
                fn, args = dq.popleft()
            try:
                fn(*args)
            except Exception:
                logger.exception("Inbound DSP failed for stream %s", stream)

    def release(self, stream: str) -> None:
        """Drop everything still queued for a stream that went away."""
        with self._lock:
            self._pending.pop(stream, None)
            self._dropped.pop(stream, None)

    def dropped(self, stream: str) -> int:
        """Frames dropped for *stream* since it appeared."""
        with self._lock:
            return self._dropped.get(stream, 0)

    def wait_idle(self, *, timeout: float = 5.0) -> bool:
        """Block until every queue is empty and no drainer runs.

        A test/shutdown helper — the frame path never calls it.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._active and not any(self._pending.values()):
                    return True
            time.sleep(0.002)
        return False

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Finish what is queued (bounded by *timeout*), then stop the pool."""
        self.wait_idle(timeout=timeout)
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
