"""Script executor abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from roomkit.skills.models import ScriptResult, Skill


class ScriptExecutor(ABC):
    """Execute skill scripts with integrator-defined policy.

    No default implementation is provided — the execution policy
    (sandboxing, timeouts, allowed interpreters) is always the
    integrator's responsibility.
    """

    @abstractmethod
    async def execute(
        self,
        skill: Skill,
        script_name: str,
        arguments: dict[str, str] | None = None,
    ) -> ScriptResult:
        """Run a script from a skill's scripts/ directory.

        Args:
            skill: The skill that owns the script.
            script_name: Filename of the script to run.
            arguments: Optional key-value arguments to pass.

        Returns:
            Result of the script execution.
        """
        ...

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""
