"""Session configuration for the OpenAI GPT-Live provider.

The delegation modes a session opens with, the voice catalog, the wire
constants, and the per-session connection state the provider and its event
handlers share. Everything here is fixed when ``session.start`` is sent; what
changes afterwards lives on the provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from roomkit.providers.openai.live_events import PendingResponse, TurnGrouper, format_backend_tools
from roomkit.voice._g711 import _G711Codec
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.provider import VoiceInfo

_DEFAULT_BASE_URL = "wss://api.openai.com/v1/live/sessions"

_DEFAULT_MODEL = "gpt-live-1"

_CONNECT_TIMEOUT = 30.0

_CLOSE_TIMEOUT = 2.0

_LOG_TAG = "GPT-Live"

_VOICES: list[VoiceInfo] = [
    VoiceInfo(id="marin", name="Marin", language="en", description="Default GPT-Live voice"),
    VoiceInfo(id="cedar", name="Cedar", language="en"),
    VoiceInfo(id="quartz", name="Quartz", language="en-AU", gender="female"),
    VoiceInfo(id="ripple", name="Ripple", language="en-AU", gender="male"),
    VoiceInfo(id="vesper", name="Vesper", language="en-GB", gender="male"),
    VoiceInfo(id="willow", name="Willow", language="en-IE", gender="female"),
    VoiceInfo(id="stone", name="Stone", language="en-IE", gender="male"),
    VoiceInfo(id="gleam", name="Gleam", language="en-US", gender="female"),
    VoiceInfo(id="meridian", name="Meridian", language="en-US", gender="male"),
    VoiceInfo(
        id="delta", name="Delta", language="en-US", gender="female", description="Southern US"
    ),
    VoiceInfo(
        id="cinder", name="Cinder", language="en-US", gender="male", description="Southern US"
    ),
    VoiceInfo(id="beacon", name="Beacon", language="en-PH", gender="male"),
    VoiceInfo(id="bossa", name="Bossa", language="pt-BR", gender="female"),
    VoiceInfo(id="tempo", name="Tempo", language="pt-BR", gender="male"),
]


@dataclass(frozen=True)
class HostedReasoning:
    """Hosted backend: OpenAI runs the Responses model the live model delegates to.

    The channel's tools become this model's tools; its function calls reach
    the channel through ``on_tool_call`` and are answered through
    ``submit_tool_result``, exactly as with any other provider (RFC §12.4.1,
    hosted backend).

    Attributes:
        model: Responses model id (e.g. ``"gpt-5.6-terra"``). Required.
        instructions: The backend's own instructions.
        reasoning_effort: ``"low"``/``"medium"``/``"high"`` when the backend
            model grades it; omitted otherwise.
        max_output_tokens: Output cap for one delegated response.
        service_tier: Responses service tier (``auto``, ``default``, ``flex``,
            ``priority``).
        parallel_tool_calls: Whether the backend may call several tools at once.
        extra: Other ``delegation.responses`` fields, merged last.
    """

    model: str
    instructions: str | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    service_tier: str | None = None
    parallel_tool_calls: bool | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_config(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Render the ``delegation`` block of ``session.start``."""
        responses: dict[str, Any] = {"model": self.model}
        if self.instructions:
            responses["instructions"] = self.instructions
        if self.reasoning_effort:
            responses["reasoning"] = {"effort": self.reasoning_effort}
        if self.max_output_tokens is not None:
            responses["max_output_tokens"] = int(self.max_output_tokens)
        if self.service_tier:
            responses["service_tier"] = self.service_tier
        if self.parallel_tool_calls is not None:
            responses["parallel_tool_calls"] = bool(self.parallel_tool_calls)
        if tools:
            responses["tools"] = format_backend_tools(tools)
        responses.update(self.extra)
        return {"type": "responses", "responses": responses}


@dataclass(frozen=True)
class IntegratorReasoning:
    """Integrator backend: the channel's ReasoningBackend answers delegations.

    The model signals only that it is handing work over; the provider fires
    ``on_delegation(session, delegation_id, "integrator")`` and the answer
    returns through ``submit_delegation_output`` (RFC §12.4.1, integrator
    backend). Tools given to ``connect()`` stay off the wire: they are the
    backend's, and the channel keeps them as its declared catalogue.
    """

    def to_config(self, tools: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: ARG002
        """Render the ``delegation`` block of ``session.start``."""
        return {"type": "client"}


@dataclass
class _LiveSession:
    """Per-session connection state."""

    ws: Any
    session: VoiceSession
    session_rate: int
    input_rate: int
    output_rate: int
    codec: _G711Codec | None
    system_prompt: str | None
    voice: str | None
    tools: list[dict[str, Any]]
    provider_config: dict[str, Any]
    user_turn: TurnGrouper
    assistant_turn: TurnGrouper
    started: asyncio.Event = field(default_factory=asyncio.Event)
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    start_error: str | None = None
    receive_task: asyncio.Task[None] | None = None
    responding: bool = False
    # Hosted delegation: Responses runs by delegation id, and the open
    # function calls each still owes an output.
    pending: dict[str, PendingResponse] = field(default_factory=dict)
    open_calls: dict[str, str] = field(default_factory=dict)
    live_seconds: float = 0.0
