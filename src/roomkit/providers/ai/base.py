"""Abstract base class for AI providers."""

from __future__ import annotations

import time as _time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from datetime import date
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, SecretStr, field_validator

from roomkit.models.channel import ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelMediaType
from roomkit.models.response_metadata import ResponseMetadata
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
    # English keyword aliases scored by Tool Search alongside name/description.
    # A language-invariant search surface: a French/Spanish query (normalized to
    # English by the model) matches these even when the tool's name/description
    # is in another language. Optional — tools without tags score as before.
    tags: list[str] = Field(default_factory=list)


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
    """Tool execution result in conversation history.

    ``result`` is a plain string for text results, or a list of content parts
    (text and/or image) when a tool returns multimodal output — e.g. an edge
    tool that returns a screenshot. Providers that support image tool results
    (Anthropic) render the parts as content blocks; the rest flatten via
    ``as_text()``.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    result: str | list[AITextPart | AIImagePart]
    # MCP CallToolResult.structuredContent, captured before the LLM-facing
    # string is flattened and possibly evicted. Never rendered to providers —
    # it rides the part so tool-call events can hand it to UI surfaces
    # (MCP Apps widgets) verbatim.
    structured_content: dict[str, Any] | None = None

    def as_text(self) -> str:
        """Flatten the result to plain text for providers without image support.

        A string result is returned unchanged; a list concatenates its text
        parts and substitutes a ``[image]`` placeholder for each image part.
        """
        if isinstance(self.result, str):
            return self.result
        return "\n".join(p.text if isinstance(p, AITextPart) else "[image]" for p in self.result)

    def split_for_message(self) -> tuple[str, list[AIImagePart]]:
        """Split the result into the tool-message text and its image parts.

        Unlike Anthropic — whose Messages API accepts image blocks *inside* a
        ``tool_result`` — most providers reject image content in a tool /
        function-response message; the image has to ride on a following ``user``
        message instead. This returns the text to keep on the tool message
        (text parts joined, or a ``"[see image below]"`` placeholder when the
        result was image-only, so the tool-call/result pairing stays non-empty
        and valid) together with the image parts to render natively elsewhere.

        A string result yields ``(result, [])`` and a text-only list yields
        ``(joined_text, [])`` — the no-op path that keeps every existing text
        tool byte-for-byte unchanged. Only a list carrying an image populates
        the second element and triggers a provider's synthetic-image path.
        """
        if isinstance(self.result, str):
            return self.result, []
        texts = [p.text for p in self.result if isinstance(p, AITextPart)]
        images = [p for p in self.result if isinstance(p, AIImagePart)]
        text = "\n".join(texts) if texts else ("[see image below]" if images else "")
        return text, images


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
        context_overflow: Did the request exceed the model's context window?
            Tri-state. ``True`` and ``False`` are a structural classification
            (measurement, an error code) and are believed as stated, in both
            directions. ``None`` means nobody classified, and the message
            wording decides as a fallback — an envelope may rewrap the
            provider's prose, and prose must never override an explicit
            answer.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        provider: str = "",
        status_code: int | None = None,
        context_overflow: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.provider = provider
        self.status_code = status_code
        self.context_overflow = context_overflow


# HTTP status codes that are transient and worth retrying for any AI provider.
# Providers may extend this set with their own (e.g. Anthropic's 529 "overloaded").
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503})


# The wordings observed across providers for a context-window refusal. One
# list for every layer that must read provider prose — a consumer keeping its
# own copy drifts from this one, and both then miss what the other catches.
# "request too large" is deliberately absent: it is OpenAI's wording for a
# tokens-per-minute rate limit (a 429 worth retrying), not an overflow.
_CONTEXT_OVERFLOW_PHRASES: tuple[str, ...] = (
    "context length",  # OpenAI / vLLM: "maximum context length", "context length exceeded"
    "context_length",  # OpenAI error-code spelling
    "maximum context",
    "context window",
    "token limit",
    "too many tokens",
    "prompt is too long",  # Anthropic
    "exceeds the model",
    "range of input length",  # Qwen / DashScope
)


def is_context_overflow_message(text: str) -> bool:
    """Does this provider error text describe a context-window overflow?

    Phrase matching is the fallback, not the fact: an error classified
    structurally carries ``ProviderError.context_overflow`` instead, which
    survives envelopes that rewrap the provider's prose.
    """
    haystack = text.lower()
    return any(phrase in haystack for phrase in _CONTEXT_OVERFLOW_PHRASES)


class AIMessage(BaseModel):
    """A message in the AI conversation context."""

    role: str  # "system", "user", "assistant", "tool"
    content: (
        str | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart]
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


API_KEY_METADATA_KEY = "api_key"
"""``AIContext.metadata`` key holding the credential to use for *this* request.

A provider is built once, and in a multi-tenant host the object it becomes is
shared by every conversation it serves — so a key fixed at construction is
necessarily everyone's key. When the credential belongs to the individual making
the request (their own subscription or account, which sharing would violate),
the host resolves it per turn and leaves it here instead, typically from a
``BEFORE_AI_GENERATION`` hook: the same way it already attaches turn-level
attribution through :attr:`AIContext.response_metadata`.

Providers that honour it fall back to their configured key when the entry is
absent, so a host that never sets it sees no change at all.
"""


class _AIContextMetadata(dict[str, Any]):
    """Metadata mapping that never renders a per-request credential in clear text.

    Hooks intentionally mutate ``AIContext.metadata`` in place, so protecting
    only model construction would leave the common assignment path exposed.
    This small dict subtype wraps that one reserved value on every mutation;
    ordinary metadata retains normal ``dict`` behaviour and equality.
    """

    def __init__(self, values: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        if values is not None:
            self.update(values)
        if kwargs:
            self.update(kwargs)

    @staticmethod
    def _protected(key: str, value: Any) -> Any:
        if key == API_KEY_METADATA_KEY and isinstance(value, str) and value:
            return SecretStr(value)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._protected(key, value))

    def update(self, *args: Any, **kwargs: Any) -> None:
        if len(args) > 1:
            raise TypeError(f"update expected at most 1 argument, got {len(args)}")
        if args:
            values = args[0]
            items = (
                ((key, values[key]) for key in values.keys())  # noqa: SIM118
                if hasattr(values, "keys")
                else values
            )
            for key, value in items:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def __ior__(self, values: Any) -> Self:
        self.update(values)
        return self


class AIContext(BaseModel):
    """Context passed to AI provider for generation."""

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    messages: list[AIMessage] = Field(default_factory=list)
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    """Output cap for this turn. ``None`` means "not set for this turn", which
    lets each provider fall back to its own configured ``max_tokens``. A
    non-``None`` default here would shadow that config and make it dead."""
    thinking_budget: int | None = None
    enable_thinking: bool | None = None
    """Turn this turn's reasoning block on or off, for providers that expose
    the switch. ``None`` defers to the provider's own configuration, and then
    to the model's default."""
    reasoning_effort: str | None = None
    """Reasoning verbosity for this turn, for providers that grade it.
    Accepted values are the provider's own; ``None`` defers to its config."""
    tools: list[AITool] = Field(default_factory=list)
    room: RoomContext | None = None
    target_capabilities: ChannelCapabilities | None = None
    target_media_types: list[ChannelMediaType] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    response_metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description=(
            "The turn's one response-metadata record, merged into every MESSAGE "
            "event built for the turn, on both the streaming and non-streaming "
            "paths — each event carries the record as it stands when the event "
            "is created. Live for the whole turn: a memory provider writes it "
            "while the context is built, a BEFORE_AI_GENERATION hook through "
            "this attribute, a tool handler through "
            "roomkit.tools.current_response_metadata(). Shared by identity, so "
            "a host that keeps the instance keeps the turn's record."
        ),
    )

    @field_validator("metadata", mode="after")
    @classmethod
    def _protect_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _AIContextMetadata(value)

    def model_post_init(self, __context: Any) -> None:
        """Protect metadata even when Pydantic's validation was bypassed."""
        if not isinstance(self.metadata, _AIContextMetadata):
            self.metadata = _AIContextMetadata(self.metadata)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Preserve secret wrapping when ``model_copy(update=...)`` bypasses validation."""
        protected_update = dict(update) if update is not None else None
        if protected_update is not None and "metadata" in protected_update:
            metadata = protected_update["metadata"]
            if not isinstance(metadata, Mapping):
                raise TypeError("AIContext metadata must be a mapping")
            protected_update["metadata"] = _AIContextMetadata(metadata)
        copied = super().model_copy(update=protected_update, deep=deep)
        if not isinstance(copied.metadata, _AIContextMetadata):
            copied.metadata = _AIContextMetadata(copied.metadata)
        return copied


def request_api_key(context: AIContext) -> str | None:
    """Return the per-request credential carried by ``context``, or ``None``.

    ``None`` means "use the configured key". An absent entry, an empty string
    and a non-string value all collapse to it, because each is a host that did
    not supply a usable credential for this turn — and because metadata is an
    open bag, so a provider must not hand whatever it finds to a client
    constructor.
    """
    value = context.metadata.get(API_KEY_METADATA_KEY)
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if isinstance(value, str) and value:
        return value
    return None


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


class ModelPricing(BaseModel):
    """List price of one model, per million tokens, as its vendor published it.

    Rates mirror the keys roomkit itself reports in ``usage``
    (:attr:`AIResponse.usage`) — input, output, cache reads, cache writes —
    so a consumer can price a response without inventing a mapping. What is
    *not* here is deliberate: per-client negotiated rates, discounts and
    currency conversion belong to whoever bills, not to a shared catalog.

    A rate is volatile in a way a model id is not, hence :attr:`verified`:
    the entry states what the vendor published on that date, and a consumer
    can decide for itself when that is too old to trust.

    Attributes:
        input_per_million: Price of a million uncached input tokens.
        output_per_million: Price of a million output tokens.
        cache_read_per_million: Price of a million tokens re-read from the
            prompt cache. ``None`` means this catalog represents no separate
            per-token charge for that counter. Catalogs must explicitly repeat
            the input rate when cache reads are billed as ordinary input.
        cache_write_per_million: Price of a million tokens written to the
            prompt cache — Anthropic's 5-minute write premium (1.25x input),
            which is the TTL roomkit's ``ephemeral`` markers request. ``None``
            where a write is not billed per token. Google cache storage, for
            example, is billed by time and cannot be represented here.
        image_input_per_million: Price of a million image *input* tokens, where
            the vendor quotes them apart from text — a reference image handed
            to an image model, say. ``None`` where the catalog represents no
            separate charge, which is every conversational model here: they
            bill a vision token as an ordinary input token.
        image_output_per_million: Price of a million *generated*-image tokens.
            An image model is billed per token like any other, only with the
            picture counted on its own meter and at a rate an order of
            magnitude above the text one — which is why it is a field and not
            an approximation folded into ``output_per_million``. ``None`` for
            a model that generates no images.
        long_context_threshold_tokens: Total input-token threshold above which
            the model's published long-context multipliers apply. ``None`` for
            models with flat pricing.
        long_context_input_multiplier: Multiplier applied to all represented
            input and cache charges above the long-context threshold.
        long_context_output_multiplier: Multiplier applied to output charges
            above the long-context threshold.
        currency: ISO 4217 code the rates are quoted in. Every vendor roomkit
            ships a catalog for publishes in USD.
        verified: Date the rates were read from the vendor's own price list.
    """

    input_per_million: float = Field(ge=0, allow_inf_nan=False)
    output_per_million: float = Field(ge=0, allow_inf_nan=False)
    cache_read_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cache_write_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    image_input_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    image_output_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    long_context_threshold_tokens: int | None = Field(default=None, gt=0)
    long_context_input_multiplier: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    long_context_output_multiplier: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    currency: str = Field(default="USD", min_length=1)
    verified: date

    def cost_for(self, usage: Mapping[str, int]) -> float:
        """Price a single response's ``usage`` dict, in :attr:`currency`.

        Reads the keys roomkit's providers report — ``input_tokens``,
        ``output_tokens``, ``cache_read_input_tokens``,
        ``cache_creation_input_tokens``, plus ``input_image_tokens`` and
        ``output_image_tokens`` from an image generation
        (:class:`~roomkit.providers.image.base.ImageProvider`) — and ignores
        anything else, so a provider reporting extra counters neither breaks
        nor inflates the total. Missing keys count as zero.

        Every counter is **disjoint**: a token is charged under exactly one of
        them. Providers that receive image tokens nested inside a total
        subtract them before reporting, so summing here bills each token once.

        A counter with no corresponding rate is omitted: ``None`` means the
        catalog does not represent a separate per-token charge for it. When the
        response crosses a published long-context threshold, the configured
        input and output multipliers are applied automatically.

        Args:
            usage: A response's token counters, as reported by the provider.

        Returns:
            The cost of that response, in :attr:`currency`.
        """
        counters = {
            name: self._usage_counter(usage, name)
            for name in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "input_image_tokens",
                "output_image_tokens",
            )
        }
        input_total = counters["input_tokens"] * self.input_per_million
        for counter, rate in (
            ("cache_read_input_tokens", self.cache_read_per_million),
            ("cache_creation_input_tokens", self.cache_write_per_million),
            ("input_image_tokens", self.image_input_per_million),
        ):
            if rate is not None:
                input_total += counters[counter] * rate

        output_total = counters["output_tokens"] * self.output_per_million
        if self.image_output_per_million is not None:
            output_total += counters["output_image_tokens"] * self.image_output_per_million
        total_input_tokens = sum(
            counters[counter]
            for counter in (
                "input_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "input_image_tokens",
            )
        )
        if (
            self.long_context_threshold_tokens is not None
            and total_input_tokens > self.long_context_threshold_tokens
        ):
            input_total *= self.long_context_input_multiplier
            output_total *= self.long_context_output_multiplier
        return (input_total + output_total) / 1_000_000

    @staticmethod
    def _usage_counter(usage: Mapping[str, int], name: str) -> int:
        """Read one non-negative integer counter from a provider response."""
        value = usage.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"usage counter {name!r} must be a non-negative integer")
        return value


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
        pricing: Vendor list price for this model, if published. It lives
            here, beside the id, because a lineup and its price list turn
            over together: kept apart, adding a model leaves its price
            behind and the consumer bills nothing. ``None`` where no
            per-token list price exists — locally pulled open weights, a
            private edge, a retired id the vendor stopped quoting.
    """

    id: str
    display_name: str | None = None
    context_window: int | None = None
    supports_vision: bool | None = None
    deprecated: bool = False
    capabilities: list[str] = Field(default_factory=list)
    pricing: ModelPricing | None = None


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
        """Offline metadata for the models roomkit can describe without a key.

        This is **not** the discovery surface. A provider's lineup turns over
        faster than a release cycle, so a hand-maintained list can never be
        the authoritative answer to "what does this provider offer" — that is
        :meth:`list_models`, which asks the provider. What this list is for is
        the metadata roomkit needs *before* any network call exists: it backs
        :attr:`context_window` (a sync property, so it cannot await an API)
        and backfills the sparse ids a live endpoint returns, via
        :meth:`_merge_curated`.

        A model absent from it is an ordinary outcome, not an error: the
        caller gets ``context_window is None`` and degrades, which is the
        point — an unknown window is safer than a stale one. The base returns
        an empty list; providers override it.
        """
        return []

    async def list_models(self) -> list[ModelInfo]:
        """Models reported live by the provider's API — the discovery surface.

        Always current, and the only answer that reflects the caller's own
        account (entitlements, regional availability, locally loaded weights).
        The base implementation falls back to :meth:`available_models` for
        providers whose API exposes no models endpoint; the rest override
        this to query it, backfilling missing metadata via
        :meth:`_merge_curated`.
        """
        return self.available_models()

    @classmethod
    def _merge_curated(cls, live: list[ModelInfo]) -> list[ModelInfo]:
        """Backfill metadata absent from live results using the curated catalog.

        A live models endpoint typically returns ids with little metadata.
        For each live model that also appears in :meth:`available_models`,
        fill any missing
        ``display_name``/``context_window``/``supports_vision``/``pricing``
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
                        "pricing": model.pricing or match.pricing,
                    }
                )
            )
        return merged

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier (e.g. 'claude-opus-5', 'gpt-5.6-sol')."""
        ...

    def catalog_entry(self) -> ModelInfo | None:
        """The offline :class:`ModelInfo` for the active model, if known.

        The single place a provider should read its own model's metadata from.
        A second hardcoded table — a tuple of vision-capable prefixes, say —
        duplicates what :meth:`available_models` already states and rots
        independently of it, which is how a provider ends up reporting a
        current model as text-only.

        Returns ``None`` for an id the catalog does not carry (a custom or
        local model behind ``base_url``, a snapshot newer than this release).
        """
        name = self.model_name
        for model in type(self).available_models():
            if model.id == name:
                return model
        return None

    @property
    def context_window(self) -> int | None:
        """Input context window of the active model in tokens, if known.

        Resolved offline from the curated :meth:`available_models` catalog
        keyed by :attr:`model_name` — no API key or network. Returns ``None``
        when the active model is absent from the catalog (custom / local model
        ids, e.g. an arbitrary vLLM model string), so callers must degrade
        gracefully rather than assume a window.
        """
        entry = self.catalog_entry()
        return entry.context_window if entry else None

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
