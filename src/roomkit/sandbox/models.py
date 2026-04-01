"""Data models for Sandbox execution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, computed_field


class SandboxResult(BaseModel):
    """Result of executing a sandbox command."""

    exit_code: int
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success(self) -> bool:
        """Derived from ``exit_code``: ``True`` when ``exit_code == 0``."""
        return self.exit_code == 0
