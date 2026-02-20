"""Steering directives for mid-run AI channel interaction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class InjectMessage(BaseModel):
    """Inject a message into the AI context between tool rounds."""

    type: Literal["inject_message"] = "inject_message"
    content: str
    role: str = "user"


class Cancel(BaseModel):
    """Cancel an active tool loop."""

    type: Literal["cancel"] = "cancel"
    reason: str = "cancelled"


class UpdateSystemPrompt(BaseModel):
    """Append text to the system prompt between tool rounds."""

    type: Literal["update_system_prompt"] = "update_system_prompt"
    append: str


SteeringDirective = InjectMessage | Cancel | UpdateSystemPrompt
