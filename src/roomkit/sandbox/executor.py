"""Sandbox executor abstract base class.

The :class:`SandboxExecutor` ABC defines the contract for sandboxed
command execution.  When provided to an :class:`~roomkit.channels.ai.AIChannel`,
the channel automatically injects the executor's tools and routes
tool calls through it.

Usage::

    from roomkit import Agent, SandboxExecutor, SandboxResult
    from roomkit.sandbox.tools import SANDBOX_TOOL_SCHEMAS

    class DockerSandboxExecutor(SandboxExecutor):
        async def execute(self, command, arguments=None):
            # Route to container: rtk <command> <args>
            result = await container.exec_command(["rtk", command, ...])
            return SandboxResult(exit_code=result.exit_code, output=result.stdout)

        def tool_definitions(self):
            return SANDBOX_TOOL_SCHEMAS

    agent = Agent(
        "my-agent",
        provider=my_provider,
        sandbox=DockerSandboxExecutor(),
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from roomkit.sandbox.models import SandboxResult


class SandboxExecutor(ABC):
    """Execute commands in a sandboxed environment.

    No default implementation is provided — the execution environment
    (container runtime, image, security policy) is always the
    integrator's responsibility.

    Implementations should:

    1. Override :meth:`execute` to run commands in a sandbox (Docker,
       Kubernetes, local process, etc.).
    2. Override :meth:`tool_definitions` to declare which tools are
       available.  Use the reference schemas from
       :data:`roomkit.sandbox.tools.SANDBOX_TOOL_SCHEMAS` or define
       custom ones.
    3. Optionally override :meth:`close` to release resources.
    """

    @abstractmethod
    async def execute(
        self,
        command: str,
        arguments: dict[str, Any] | None = None,
    ) -> SandboxResult:
        """Run a sandbox command.

        The ``command`` is the tool name with the ``sandbox_`` prefix
        stripped.  For example, a tool call to ``sandbox_git`` arrives
        here as ``command="git"``.

        Args:
            command: Command name (e.g. ``"read"``, ``"grep"``, ``"git"``).
            arguments: Command-specific arguments from the tool call.

        Returns:
            Execution result with output.
        """
        ...

    @abstractmethod
    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return available tool schemas for injection into agent context.

        Each entry is a dict with ``name``, ``description``, and
        ``parameters`` (JSON Schema) keys.  Implementations may return
        a subset of the reference schemas defined in
        :mod:`roomkit.sandbox.tools` or provide entirely custom tools.

        All tool names **must** start with the ``sandbox_`` prefix.
        """
        ...

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""
