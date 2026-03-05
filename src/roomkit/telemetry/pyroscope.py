"""Pyroscope continuous profiling integration for RoomKit.

Provides CPU profiling with automatic tagging by room, session, and
channel.  Requires the ``pyroscope-io`` package::

    pip install pyroscope-io

Usage::

    from roomkit.telemetry.pyroscope import PyroscopeProfiler

    profiler = PyroscopeProfiler(
        application_name="my-voice-app",
        server_address="http://localhost:4040",
    )
    profiler.start()

    # Profile a specific code path with tags:
    with profiler.tag(room_id="room-1", session_id="sess-abc"):
        process_audio()

    profiler.stop()

For Grafana Cloud::

    profiler = PyroscopeProfiler(
        application_name="my-voice-app",
        server_address="https://profiles-prod-001.grafana.net",
        basic_auth_username="<user>",
        basic_auth_password="<token>",
    )
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("roomkit.telemetry.pyroscope")


class PyroscopeProfiler:
    """Continuous CPU profiler backed by Pyroscope.

    Wraps ``pyroscope-io`` with RoomKit-specific defaults and provides
    a ``tag()`` context manager for attributing CPU samples to rooms,
    sessions, or arbitrary labels.

    Args:
        application_name: Name shown in Pyroscope UI.
        server_address: Pyroscope server or Grafana Cloud endpoint.
        sample_rate: Samples per second (default: 100).
        detect_subprocesses: Profile child processes (default: False).
        oncpu: Only measure on-CPU time (default: True).
        gil_only: Only profile GIL-holding threads (default: True).
        tags: Default tags applied to all samples.
        basic_auth_username: Grafana Cloud username (optional).
        basic_auth_password: Grafana Cloud API token (optional).
        tenant_id: Multi-tenant Pyroscope tenant (optional).
    """

    def __init__(
        self,
        *,
        application_name: str = "roomkit",
        server_address: str = "http://localhost:4040",
        sample_rate: int = 100,
        detect_subprocesses: bool = False,
        oncpu: bool = True,
        gil_only: bool = True,
        tags: dict[str, str] | None = None,
        basic_auth_username: str | None = None,
        basic_auth_password: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self._application_name = application_name
        self._server_address = server_address
        self._sample_rate = sample_rate
        self._detect_subprocesses = detect_subprocesses
        self._oncpu = oncpu
        self._gil_only = gil_only
        self._tags = tags or {}
        self._basic_auth_username = basic_auth_username
        self._basic_auth_password = basic_auth_password
        self._tenant_id = tenant_id
        self._started = False

    def start(self) -> None:
        """Start the Pyroscope profiler."""
        if self._started:
            return
        try:
            import pyroscope
        except ImportError as exc:
            raise ImportError(
                "PyroscopeProfiler requires pyroscope-io. Install with: pip install pyroscope-io"
            ) from exc

        kwargs: dict[str, Any] = {
            "application_name": self._application_name,
            "server_address": self._server_address,
            "sample_rate": self._sample_rate,
            "detect_subprocesses": self._detect_subprocesses,
            "oncpu": self._oncpu,
            "gil_only": self._gil_only,
            "tags": self._tags,
        }
        if self._basic_auth_username:
            kwargs["basic_auth_username"] = self._basic_auth_username
        if self._basic_auth_password:
            kwargs["basic_auth_password"] = self._basic_auth_password
        if self._tenant_id:
            kwargs["tenant_id"] = self._tenant_id

        pyroscope.configure(**kwargs)
        self._started = True
        logger.info(
            "Pyroscope profiler started: app=%s server=%s rate=%d",
            self._application_name,
            self._server_address,
            self._sample_rate,
        )

    def stop(self) -> None:
        """Stop the Pyroscope profiler."""
        if not self._started:
            return
        try:
            import pyroscope

            pyroscope.shutdown()
        except Exception:
            logger.debug("Pyroscope shutdown error", exc_info=True)
        self._started = False
        logger.info("Pyroscope profiler stopped")

    @contextmanager
    def tag(self, **tags: str) -> Generator[None, None, None]:
        """Tag CPU samples within this context with the given labels.

        Usage::

            with profiler.tag(room_id="room-1", backend="sip"):
                process_audio_frame()

        In Pyroscope UI, you can filter/group flamegraphs by these tags
        to isolate CPU usage per room, session, backend, etc.
        """
        if not self._started:
            yield
            return
        import pyroscope

        with pyroscope.tag_wrapper(tags):
            yield

    @contextmanager
    def tag_session(
        self,
        *,
        room_id: str | None = None,
        session_id: str | None = None,
        backend: str | None = None,
        channel_id: str | None = None,
    ) -> Generator[None, None, None]:
        """Tag CPU samples with RoomKit session context.

        Convenience wrapper around :meth:`tag` with well-known keys.
        """
        tags: dict[str, str] = {}
        if room_id:
            tags["room_id"] = room_id
        if session_id:
            tags["session_id"] = session_id
        if backend:
            tags["backend"] = backend
        if channel_id:
            tags["channel_id"] = channel_id
        with self.tag(**tags):
            yield
