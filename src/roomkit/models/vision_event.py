"""Vision result event for ON_VISION_RESULT hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VisionEvent:
    """Vision analysis result passed to ON_VISION_RESULT hooks.

    Hooks can inspect and modify the description, labels, or text
    before it's injected into AI channels.  Return ``HookResult.block()``
    to suppress the result entirely.
    """

    session: Any
    """The video session that produced this vision result."""

    description: str
    """Natural language description of the frame."""

    labels: list[str] = field(default_factory=list)
    """Detected object/scene labels."""

    confidence: float = 0.0
    """Overall confidence score."""

    text: str | None = None
    """OCR text visible in the frame."""

    faces: list[Any] = field(default_factory=list)
    """Detected faces."""

    elapsed_ms: int = 0
    """Vision analysis duration in milliseconds."""
