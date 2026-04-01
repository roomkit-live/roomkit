"""Shared sandbox tool handlers used by AIChannel."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from roomkit.sandbox.tools import SANDBOX_TOOL_PREFIX

if TYPE_CHECKING:
    from roomkit.sandbox.executor import SandboxExecutor

logger = logging.getLogger("roomkit.channels.sandbox")


async def handle_sandbox_command(
    tool_name: str,
    arguments: dict[str, Any],
    executor: SandboxExecutor,
) -> str:
    """Route a sandbox tool call to the executor.

    Strips the ``sandbox_`` prefix and delegates to
    :meth:`SandboxExecutor.execute`.

    Returns:
        JSON-encoded :class:`SandboxResult`.
    """
    command = tool_name.removeprefix(SANDBOX_TOOL_PREFIX)
    try:
        result = await executor.execute(command, arguments)
        return result.model_dump_json()
    except Exception as exc:
        logger.exception("Sandbox command failed: %s", tool_name)
        return json.dumps({"error": f"Sandbox command failed: {exc}"})
