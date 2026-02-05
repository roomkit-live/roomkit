"""Abstract base class for AI providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
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

    role: str  # "system", "user", "assistant"
    content: str | list[AITextPart | AIImagePart]
    metadata: dict[str, Any] = Field(default_factory=dict)


class AIContext(BaseModel):
    """Context passed to AI provider for generation."""

    model_config = {"arbitrary_types_allowed": True}

    messages: list[AIMessage] = Field(default_factory=list)
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    tools: list[AITool] = Field(default_factory=list)
    room: RoomContext | None = None
    target_capabilities: ChannelCapabilities | None = None
    target_media_types: list[ChannelMediaType] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AIResponse(BaseModel):
    """Response from an AI provider."""

    content: str
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tasks: list[Task] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    tool_calls: list[AIToolCall] = Field(default_factory=list)


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

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""
