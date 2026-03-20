"""Shared signal handling and graceful shutdown for RoomKit examples."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable

from roomkit import RoomKit

logger = logging.getLogger(__name__)


async def run_until_stopped(
    kit: RoomKit,
    *,
    cleanup: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Block until SIGINT/SIGTERM, then run *cleanup* and close *kit*.

    Typical usage at the end of an example's ``main()``::

        await run_until_stopped(kit, cleanup=my_teardown)
    """
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    logger.info("Stopping...")
    if cleanup is not None:
        await cleanup()
    await kit.close()
    logger.info("Done.")
