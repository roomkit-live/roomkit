"""External tool handler ABC for provider-executed tools.

When an AI provider executes tools internally (e.g., Claude Code sandbox),
the :class:`ExternalToolHandler` ABC provides control (approve/deny/modify)
and observability (post-execution hooks) without RoomKit trying to execute
the tools locally.

The ABC is transport-agnostic — subclasses decide HOW tool events arrive
(HTTP, WebSocket, in-process queue, etc.). The framework injects hook
callbacks so the handler can fire :attr:`~roomkit.models.enums.HookTrigger.BEFORE_TOOL_USE`
and :attr:`~roomkit.models.enums.HookTrigger.ON_TOOL_CALL` hooks.

Usage::

    from roomkit.tools.external import ExternalToolHandler, ToolDecision

    class MyToolHandler(ExternalToolHandler):
        async def process_tool_call(self, tool_name, tool_input, **kwargs):
            # Apply policy, ask user, etc.
            if tool_name == "Bash":
                return ToolDecision(approved=False, reason="Bash not allowed")
            return ToolDecision(approved=True)

        async def on_tool_result(self, tool_name, tool_input, result, **kwargs):
            print(f"Tool {tool_name} returned: {result[:100]}")

    agent = Agent(
        "my-agent",
        provider=provider,
        external_tool_handler=MyToolHandler(),
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from roomkit.models.enums import ChannelType
from roomkit.models.tool_call import ToolCallEvent


@dataclass
class ToolDecision:
    """Decision for an external tool call.

    Returned by :meth:`ExternalToolHandler.process_tool_call`.
    """

    approved: bool
    """Whether the tool call is allowed to proceed."""

    modified_input: dict[str, Any] | None = None
    """If set, overrides the original tool input."""

    reason: str = ""
    """Human-readable reason for the decision (used in deny messages)."""


# Callback type injected by the framework.
BeforeToolCallback = Callable[[ToolCallEvent], Awaitable[bool]]
OnToolCallback = Callable[[ToolCallEvent], Awaitable[str | None]]


class ExternalToolHandler(ABC):
    """Controls and observes tools executed by an external provider.

    Subclasses decide HOW tool events arrive (HTTP callback, WebSocket,
    in-process, queue, etc.). The ABC defines the contract for processing
    them and bridging to RoomKit's hook system.

    Lifecycle:
        1. ``register_channel()`` injects ``_before_tool_hook`` and ``_on_tool_hook``
        2. ``start()`` is called (subclass sets up transport)
        3. External provider sends tool events via transport
        4. Subclass calls ``process_tool_call()`` / ``on_tool_result()``
        5. These fire RoomKit hooks via injected callbacks
        6. ``stop()`` is called on shutdown

    Attributes set by the framework (do not override):
        _before_tool_hook: Fires BEFORE_TOOL_USE sync hooks. Returns True if allowed.
        _on_tool_hook: Fires ON_TOOL_CALL sync hooks. Returns optional result override.
        _channel_id: Channel ID this handler is attached to.
    """

    _before_tool_hook: BeforeToolCallback | None = None
    _on_tool_hook: OnToolCallback | None = None
    _channel_id: str = ""

    @abstractmethod
    async def process_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_call_id: str = "",
        job_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
        room_id: str | None = None,
    ) -> ToolDecision:
        """Called BEFORE the external provider executes a tool.

        The implementation should apply its own logic (policy checks,
        user approval, etc.) and return a :class:`ToolDecision`.

        The default implementation fires ``BEFORE_TOOL_USE`` hooks via
        the injected callback. Subclasses that override this should call
        ``await self._fire_before_hook(...)`` to preserve hook integration.

        This method MAY block (e.g., waiting for user approval via UI).
        The external provider's hook (e.g., Claude Code's ``PreToolUse``)
        blocks until this returns.

        Args:
            tool_name: Name of the tool (e.g., "Write", "Bash", "Read").
            tool_input: Tool arguments as a dict.
            tool_call_id: Provider-assigned ID for this tool call.
            job_id: Job identifier for tracking.
            session_id: Session identifier.
            tenant_id: Tenant identifier for multi-tenant isolation.
            room_id: RoomKit room ID.

        Returns:
            ToolDecision with approved/denied status and optional modified input.
        """

    @abstractmethod
    async def on_tool_result(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: str,
        *,
        is_error: bool = False,
        tool_call_id: str = "",
        job_id: str | None = None,
        room_id: str | None = None,
    ) -> None:
        """Called AFTER the external provider executed a tool.

        Fires ``ON_TOOL_CALL`` hooks via the injected callback for
        observability. Subclasses that override this should call
        ``await self._fire_on_tool_hook(...)`` to preserve hook integration.

        Args:
            tool_name: Name of the tool.
            tool_input: Tool arguments.
            result: Tool execution result (stdout, content, etc.).
            is_error: Whether the tool execution failed.
            tool_call_id: Provider-assigned ID for this tool call.
            job_id: Job identifier.
            room_id: RoomKit room ID.
        """

    async def start(self) -> None:  # noqa: B027
        """Start receiving tool events.

        Override for transports that need setup (HTTP server, WebSocket
        connection, queue subscription, etc.). Default: no-op.
        """

    async def stop(self) -> None:  # noqa: B027
        """Stop and release resources. Default: no-op."""

    # ── Hook bridge helpers ───────────────────────────────────────────

    async def _fire_before_hook(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_call_id: str = "",
        room_id: str | None = None,
    ) -> bool:
        """Fire BEFORE_TOOL_USE hooks. Returns True if allowed."""
        if self._before_tool_hook is None:
            return True
        event = ToolCallEvent(
            channel_id=self._channel_id,
            channel_type=ChannelType.AI,
            tool_call_id=tool_call_id,
            name=tool_name,
            arguments=tool_input,
            result=None,
            room_id=room_id,
        )
        return await self._before_tool_hook(event)

    async def _fire_on_tool_hook(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: str,
        *,
        tool_call_id: str = "",
        room_id: str | None = None,
    ) -> None:
        """Fire ON_TOOL_CALL hooks for observation."""
        if self._on_tool_hook is None:
            return
        event = ToolCallEvent(
            channel_id=self._channel_id,
            channel_type=ChannelType.AI,
            tool_call_id=tool_call_id,
            name=tool_name,
            arguments=tool_input,
            result=result,
            room_id=room_id,
        )
        await self._on_tool_hook(event)


class PolicyExternalToolHandler(ExternalToolHandler):
    """Auto-approve tools based on a ToolPolicy. No UI, no blocking.

    Useful for standalone/testing scenarios where no human approval is needed.
    Fires all RoomKit hooks for observability.

    Usage::

        handler = PolicyExternalToolHandler(
            policy=ToolPolicy(deny=["Bash", "Write"]),
        )
    """

    def __init__(self, policy: Any = None) -> None:
        from roomkit.tools.policy import ToolPolicy

        self._policy: ToolPolicy | None = policy

    async def process_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_call_id: str = "",
        job_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
        room_id: str | None = None,
    ) -> ToolDecision:
        # Fire BEFORE_TOOL_USE hook first
        hook_allowed = await self._fire_before_hook(
            tool_name, tool_input, tool_call_id=tool_call_id, room_id=room_id
        )
        if not hook_allowed:
            return ToolDecision(approved=False, reason="Denied by BEFORE_TOOL_USE hook")

        # Apply policy
        if self._policy and not self._policy.is_allowed(tool_name):
            return ToolDecision(
                approved=False,
                reason=f"Tool '{tool_name}' denied by policy",
            )

        return ToolDecision(approved=True)

    async def on_tool_result(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: str,
        *,
        is_error: bool = False,
        tool_call_id: str = "",
        job_id: str | None = None,
        room_id: str | None = None,
    ) -> None:
        await self._fire_on_tool_hook(
            tool_name,
            tool_input,
            result,
            tool_call_id=tool_call_id,
            room_id=room_id,
        )
