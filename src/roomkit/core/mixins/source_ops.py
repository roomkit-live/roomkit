"""SourceOpsMixin — event-driven source management."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from roomkit.core.exceptions import SourceAlreadyAttachedError, SourceNotFoundError
from roomkit.core.mixins.helpers import HelpersMixin

if TYPE_CHECKING:
    from roomkit.models.delivery import InboundMessage, InboundResult
    from roomkit.sources.base import SourceHealth, SourceProvider, SourceStatus


@runtime_checkable
class SourceOpsHost(Protocol):
    """Contract: capabilities a host class must provide for SourceOpsMixin.

    Attributes provided by the host's ``__init__``:
        _sources: Registry of attached source providers, keyed by channel ID.
        _source_tasks: Background tasks running each source's event loop.

    Methods provided by InboundMixin (or equivalent):
        process_inbound: Feed an inbound message into the framework pipeline.
    """

    _sources: dict[str, SourceProvider]
    _source_tasks: dict[str, asyncio.Task[None]]

    async def process_inbound(
        self, message: InboundMessage, *, room_id: str | None = None
    ) -> InboundResult: ...


class SourceOpsMixin(HelpersMixin):
    """Event-driven source attach / detach / health / list operations.

    Host contract: :class:`SourceOpsHost`.
    """

    _sources: dict[str, SourceProvider]
    _source_tasks: dict[str, asyncio.Task[None]]

    # Stub for cross-mixin call — implemented by InboundMixin in the MRO.
    async def process_inbound(
        self, message: InboundMessage, *, room_id: str | None = None
    ) -> InboundResult: ...

    async def attach_source(
        self,
        channel_id: str,
        source: SourceProvider,
        *,
        auto_restart: bool = True,
        restart_delay: float = 5.0,
        max_restart_delay: float = 300.0,
        max_restart_attempts: int | None = None,
        max_concurrent_emits: int | None = 10,
    ) -> None:
        """Attach an event-driven source to a channel.

        The source will start listening for messages and emit them into
        RoomKit's inbound pipeline via ``process_inbound()``.

        Args:
            channel_id: The channel ID to associate with this source.
                Messages from this source will be tagged with this channel_id.
            source: The source provider instance to attach.
            auto_restart: If True (default), automatically restart the source
                if it exits unexpectedly. Set to False for one-shot sources.
            restart_delay: Initial delay in seconds before restarting after
                failure. Doubles on each consecutive failure (exponential backoff).
            max_restart_delay: Maximum delay between restart attempts in seconds.
                Backoff is capped at this value. Defaults to 300 (5 minutes).
            max_restart_attempts: Maximum number of consecutive restart attempts
                before giving up. If None (default), retries indefinitely.
                When exhausted, emits ``source_exhausted`` framework event.
            max_concurrent_emits: Maximum number of concurrent ``emit()`` calls
                to prevent backpressure buildup. Defaults to 10. Set to None
                for unlimited concurrency (not recommended for high-volume sources).

        Raises:
            SourceAlreadyAttachedError: If a source is already attached to
                this channel_id.

        Example:
            from roomkit.sources.neonize import NeonizeSource

            source = NeonizeSource(session_path="~/.roomkit/wa.db")
            await kit.attach_source(
                "whatsapp-personal",
                source,
                max_restart_attempts=5,      # Give up after 5 failures
                max_concurrent_emits=20,     # Allow 20 concurrent messages
            )
        """
        if channel_id in self._sources:
            raise SourceAlreadyAttachedError(f"Source already attached to channel {channel_id}")

        logger = logging.getLogger("roomkit.sources")

        # Create emit callback with optional backpressure control
        if max_concurrent_emits is not None:
            semaphore = asyncio.Semaphore(max_concurrent_emits)

            async def emit(msg: InboundMessage) -> InboundResult:
                async with semaphore:
                    return await self.process_inbound(msg)
        else:

            async def emit(msg: InboundMessage) -> InboundResult:
                return await self.process_inbound(msg)

        self._sources[channel_id] = source
        self._source_tasks[channel_id] = asyncio.create_task(
            self._run_source(
                channel_id,
                source,
                emit,
                auto_restart,
                restart_delay,
                max_restart_delay,
                max_restart_attempts,
                logger,
            ),
            name=f"source:{channel_id}",
        )

        await self._emit_framework_event(
            "source_attached",
            channel_id=channel_id,
            data={"source_name": source.name},
        )

    async def detach_source(self, channel_id: str) -> None:
        """Detach and stop an event-driven source.

        Args:
            channel_id: The channel ID of the source to detach.

        Raises:
            SourceNotFoundError: If no source is attached to this channel_id.
        """
        source = self._sources.pop(channel_id, None)
        task = self._source_tasks.pop(channel_id, None)
        if source is None:
            raise SourceNotFoundError(f"No source attached to channel {channel_id}")

        # Stop the source
        await source.stop()

        # Cancel the runner task and await its completion
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        await self._emit_framework_event(
            "source_detached",
            channel_id=channel_id,
            data={"source_name": source.name},
        )

    async def _run_source(
        self,
        channel_id: str,
        source: SourceProvider,
        emit: Callable[[InboundMessage], Awaitable[InboundResult]],
        auto_restart: bool,
        restart_delay: float,
        max_restart_delay: float,
        max_restart_attempts: int | None,
        logger: logging.Logger,
    ) -> None:
        """Run a source with optional auto-restart on failure.

        Uses exponential backoff: delay doubles on each failure, capped at
        max_restart_delay. Delay resets after a successful start.
        """

        attempt = 0
        current_delay = restart_delay

        while True:
            try:
                logger.info("Starting source %s for channel %s", source.name, channel_id)
                await source.start(emit)
                # Clean exit - source stopped normally
                logger.info("Source %s stopped cleanly", source.name)
                self._sources.pop(channel_id, None)
                self._source_tasks.pop(channel_id, None)
                break
            except asyncio.CancelledError:
                logger.debug("Source %s cancelled", source.name)
                raise
            except Exception as e:
                attempt += 1
                logger.exception("Source %s failed (attempt %d): %s", source.name, attempt, e)
                await self._emit_framework_event(
                    "source_error",
                    channel_id=channel_id,
                    data={"source_name": source.name, "error": str(e), "attempt": attempt},
                )

                if not auto_restart:
                    raise

                # Check if max attempts exceeded
                if max_restart_attempts is not None and attempt >= max_restart_attempts:
                    logger.error(
                        "Source %s exhausted after %d attempts, giving up",
                        source.name,
                        attempt,
                    )
                    await self._emit_framework_event(
                        "source_exhausted",
                        channel_id=channel_id,
                        data={
                            "source_name": source.name,
                            "attempts": attempt,
                            "last_error": str(e),
                        },
                    )
                    break

                logger.info(
                    "Restarting source %s in %.1f seconds (attempt %d%s)",
                    source.name,
                    current_delay,
                    attempt,
                    f"/{max_restart_attempts}" if max_restart_attempts else "",
                )
                await asyncio.sleep(current_delay)

                # Exponential backoff: double delay, cap at max
                current_delay = min(current_delay * 2, max_restart_delay)

    async def source_health(self, channel_id: str) -> SourceHealth | None:
        """Get health information for an attached source.

        Args:
            channel_id: The channel ID of the source.

        Returns:
            SourceHealth if a source is attached, None otherwise.
        """
        source = self._sources.get(channel_id)
        if source is None:
            return None
        return await source.healthcheck()

    def list_sources(self) -> dict[str, SourceStatus]:
        """List all attached sources and their status.

        Returns:
            Dict mapping channel_id to current SourceStatus.
        """
        return {cid: source.status for cid, source in self._sources.items()}
