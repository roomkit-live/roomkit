"""Abstract base class for AI providers."""

from __future__ import annotations

import time as _time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from roomkit.models.channel import ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelMediaType
from roomkit.models.task import Observation, Task


class AITextPart(BaseModel):
    """Text part of a multimodal message."""

    type: Literal["text"] = "text"
    text: str


class AIImagePart(BaseModel):
    """Image part of a multimodal message."""

    type: Literal["image"] = "image"
    url: str
    mime_type: str | None = None


class AITool(BaseModel):
    """Tool definition for function calling."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class AIToolCall(BaseModel):
    """A tool call from the AI response."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AIToolCallPart(BaseModel):
    """Assistant's tool call in conversation history."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AIToolResultPart(BaseModel):
    """Tool execution result in conversation history."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    result: str


class AIThinkingPart(BaseModel):
    """AI reasoning/thinking block in conversation history.

    Used to preserve thinking blocks across tool-loop turns (required by
    providers like Anthropic that mandate round-trip fidelity).

    Attributes:
        thinking: The reasoning text produced by the model.
        signature: Provider-specific opaque token for caching/validation
            (e.g. Anthropic's thinking block signature).
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ProviderError(Exception):
    """Error from an AI provider SDK call.

    Attributes:
        retryable: Whether the caller should retry the request.
        provider: Name of the provider that raised the error.
        status_code: HTTP status code from the provider, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        provider: str = "",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.provider = provider
        self.status_code = status_code


class AIMessage(BaseModel):
    """A message in the AI conversation context."""

    role: str  # "system", "user", "assistant", "tool"
    content: (
        str | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart]
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class AIContext(BaseModel):
    """Context passed to AI provider for generation."""

    model_config = {"arbitrary_types_allowed": True}

    messages: list[AIMessage] = Field(default_factory=list)
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    thinking_budget: int | None = None
    tools: list[AITool] = Field(default_factory=list)
    room: RoomContext | None = None
    target_capabilities: ChannelCapabilities | None = None
    target_media_types: list[ChannelMediaType] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    response_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Merged into the metadata of every MESSAGE response event built "
            "for this turn, on both the streaming and non-streaming paths. "
            "Set by hosts (e.g. a BEFORE_AI_GENERATION hook) to attach "
            "turn-level attribution such as RAG sources to the reply."
        ),
    )


class AIResponse(BaseModel):
    """Response from an AI provider."""

    content: str
    thinking: str | None = None
    thinking_signature: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tasks: list[Task] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    tool_calls: list[AIToolCall] = Field(default_factory=list)


class StreamThinkingDelta(BaseModel):
    """A thinking/reasoning delta from a streaming AI response.

    Emitted before text deltas when the model is reasoning. A delta may carry
    only a ``signature`` (with empty ``thinking``): Anthropic streams the
    thinking block's opaque signature separately, and it must be preserved so
    the block can be echoed back in history without a 400.
    """

    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str
    signature: str | None = None


class StreamTextDelta(BaseModel):
    """A text delta from a streaming AI response."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class StreamToolCall(BaseModel):
    """A complete tool call extracted from a streaming AI response."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamDone(BaseModel):
    """Signals the end of a streaming AI response."""

    type: Literal["done"] = "done"
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


StreamEvent = StreamThinkingDelta | StreamTextDelta | StreamToolCall | StreamDone


class ModelInfo(BaseModel):
    """Metadata describing a single model offered by an AI provider.

    Both the curated catalog (:meth:`AIProvider.available_models`) and the
    live API query (:meth:`AIProvider.list_models`) return these. Only ``id``
    is guaranteed; the remaining fields are best-effort and may be ``None``
    when the source does not report them.

    Attributes:
        id: Exact model identifier accepted by the provider's API
            (e.g. ``"claude-sonnet-4-20250514"``, ``"gpt-4o"``).
        display_name: Human-friendly name (e.g. ``"Claude Sonnet 4"``).
        context_window: Input context window in tokens, if known.
        supports_vision: Whether the model accepts image input, if known.
        deprecated: Whether the provider marks the model deprecated.
        capabilities: Provider-reported capability tags (e.g. Ollama's
            ``"completion"``, ``"embedding"``, ``"vision"``, ``"tools"``).
            Empty when the source does not report them — consumers treat
            empty as "unknown, allow everywhere" rather than "none".
    """

    id: str
    display_name: str | None = None
    context_window: int | None = None
    supports_vision: bool | None = None
    deprecated: bool = False
    capabilities: list[str] = Field(default_factory=list)


class AIProvider(ABC):
    """AI model provider for generating responses."""

    @property
    def name(self) -> str:
        """Provider name (e.g. 'anthropic', 'openai')."""
        return self.__class__.__name__

    @property
    def supports_vision(self) -> bool:
        """Whether this provider can process images."""
        return False

    @property
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming token generation."""
        return False

    @property
    def supports_structured_streaming(self) -> bool:
        """Whether this provider supports structured streaming with tool calls."""
        return False

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of models known to this provider.

        Returns a hand-maintained list — no API key or network required —
        so an integrator can discover configurable models by calling this
        on the class before instantiating. The base returns an empty list;
        each provider overrides it with its catalog.
        """
        return []

    async def list_models(self) -> list[ModelInfo]:
        """Models reported live by the provider's API.

        The base implementation returns the curated :meth:`available_models`.
        Providers whose API exposes a models endpoint override this to query
        it, backfilling missing metadata from the catalog via
        :meth:`_merge_curated`.
        """
        return self.available_models()

    @classmethod
    def _merge_curated(cls, live: list[ModelInfo]) -> list[ModelInfo]:
        """Backfill metadata absent from live results using the curated catalog.

        A live models endpoint typically returns ids with little metadata.
        For each live model that also appears in :meth:`available_models`,
        fill any missing ``display_name``/``context_window``/``supports_vision``
        from the curated entry, keeping whatever the API did report.
        """
        curated = {m.id: m for m in cls.available_models()}
        merged: list[ModelInfo] = []
        for model in live:
            match = curated.get(model.id)
            if match is None:
                merged.append(model)
                continue
            merged.append(
                model.model_copy(
                    update={
                        "display_name": model.display_name or match.display_name,
                        "context_window": model.context_window or match.context_window,
                        "supports_vision": (
                            model.supports_vision
                            if model.supports_vision is not None
                            else match.supports_vision
                        ),
                    }
                )
            )
        return merged

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier (e.g. 'claude-sonnet-4-20250514', 'gpt-4o')."""
        ...

    @abstractmethod
    async def generate(self, context: AIContext) -> AIResponse:
        """Generate an AI response from the given context.

        Args:
            context: Conversation context including messages, system prompt,
                temperature, and target channel capabilities.

        Returns:
            The AI response with content, usage stats, and optional
            tasks/observations.
        """
        ...

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text deltas as they arrive. Override for streaming providers."""
        raise NotImplementedError(f"{self.name} does not support streaming generation")
        yield  # pragma: no cover

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events (thinking deltas, text deltas, tool calls, done).

        Default implementation wraps ``generate()`` so every provider works
        without changes.  Override for true streaming support.
        """
        response = await self.generate(context)
        if response.thinking:
            yield StreamThinkingDelta(thinking=response.thinking)
        if response.content:
            yield StreamTextDelta(text=response.content)
        for tc in response.tool_calls:
            yield StreamToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
        yield StreamDone(
            finish_reason=response.finish_reason,
            usage=response.usage,
            metadata=response.metadata,
        )

    def _record_ttfb(self, t0: float) -> None:
        """Record time-to-first-byte metric via telemetry (if propagated)."""
        from roomkit.telemetry.noop import NoopTelemetryProvider  # avoid circular import

        ttfb_ms = (_time.monotonic() - t0) * 1000
        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        telemetry.record_metric(
            "roomkit.llm.ttfb_ms",
            ttfb_ms,
            unit="ms",
            attributes={"provider": self.name, "model": self.model_name},
        )

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""
