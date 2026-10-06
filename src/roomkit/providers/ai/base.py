"""Abstract base class for AI providers."""

from __future__ import annotations

import re
import sys
import time as _time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, Field, SecretStr, field_validator

from roomkit.models.channel import ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelMediaType, ToolCallOutcome
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.task import Observation, Task
from roomkit.providers.ai.json_schema import check_portable_schema
from roomkit.providers.ai.model_tags import with_speech_tags

if TYPE_CHECKING:
    from roomkit.telemetry.base import TelemetryProvider


class AITextPart(BaseModel):
    """Text part of a multimodal message."""

    type: Literal["text"] = "text"
    text: str


class AIImagePart(BaseModel):
    """Image part of a multimodal message."""

    type: Literal["image"] = "image"
    url: str
    mime_type: str | None = None


_ANY_VENDOR_TOOL_NAME = re.compile(r"[A-Za-z0-9_.:-]+")
"""The characters some vendor accepts in a tool name (RFC §6.7); each
provider checks its own vendor's narrower rule before the request."""


def some_vendor_accepts_tool_name(name: str) -> bool:
    """Whether at least one vendor accepts *name* for a tool (RFC §6.7)."""
    return _ANY_VENDOR_TOOL_NAME.fullmatch(name) is not None


class AITool(BaseModel):
    """Tool definition for function calling.

    A name no vendor accepts (empty, or a character other than a letter, a
    digit, ``_``, ``.``, ``:`` or ``-``) is refused here; a name some vendors
    refuse is refused by their provider when it declares the tool (RFC §6.7).
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    # English keyword aliases scored by Tool Search alongside name/description.
    # A language-invariant search surface: a French/Spanish query (normalized to
    # English by the model) matches these even when the tool's name/description
    # is in another language. Optional — tools without tags score as before.
    tags: list[str] = Field(default_factory=list)
    # Declared but unseen: the provider holds the definition out of the
    # model's view (and out of its prompt cache) until a tool result
    # references it (``AIToolResultPart.references``). Only a channel whose
    # provider ``supports_deferred_tools`` sets it (RFC §6.4).
    defer_loading: bool = False

    @field_validator("name")
    @classmethod
    def _a_name_some_vendor_accepts(cls, name: str) -> str:
        if not some_vendor_accepts_tool_name(name):
            raise ValueError(
                f"tool name {name!r} is accepted by no provider: use letters, digits, "
                "'_', '.', ':' or '-'"
            )
        return name


class ServedCall(BaseModel):
    """What a provider that runs its own tools (an MCP gateway, a sandbox, a
    proxy) reports of a call it already ran: its result, and whether it
    failed. Only the provider sets it; the model's arguments never do, so a
    model that writes a ``_result`` key gets an argument like any other, and
    its call is judged by the gate (RFC §9.3)."""

    result: str = ""
    is_error: bool = False


class AIToolCall(BaseModel):
    """A tool call from the AI response."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    served: ServedCall | None = None
    """Set by a provider that already ran the call: the channel reports its
    outcome rather than serving it (RFC §9.3)."""
    partial: bool = False
    """The call's arguments do not read as an object: the tool loop does not
    run it and tells the model why (RFC §6.4)."""
    garbled: bool = False
    """The model wrote this ``partial`` call's arguments unreadable: the
    response was not cut short over them. A partial call that is not garbled
    was cut (the output cap or the context window, a content filter or a
    refusal, a stream that ended without a stop reason), and the model reads
    which (RFC §6.4)."""


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
    # Whether this result is a failure or a refusal. Carried, never inferred:
    # the tool loop knows (it caught the exception, or it refused the call
    # itself), while a reader of ``result`` would have to recognise both the
    # error envelopes and the prose sentence — and a tool whose own output
    # happens to look like either would be misread. It rides the part so
    # tool-call events and the ON_TOOL_CALL hook can state the outcome
    # instead of guessing it, and a provider whose format flags an error
    # result renders it (Anthropic's ``is_error``, RFC §6.7).
    is_error: bool = False
    # MCP CallToolResult.structuredContent, captured before the LLM-facing
    # string is flattened and possibly evicted. Never rendered to providers —
    # it rides the part so tool-call events can hand it to UI surfaces
    # (MCP Apps widgets), unevicted; the event bounds its binary payloads.
    structured_content: dict[str, Any] | None = None
    # How the call ended: served, refused, failed, blocked, unserved or
    # cancelled. Never rendered to providers; it rides the part to the tool
    # row the room stores (``ToolCallContent.outcome``).
    outcome: ToolCallOutcome | None = None
    # Tools this result makes callable when the provider holds them unseen
    # (``AITool.defer_loading``): a ``find_tools`` result naming them, an
    # ``activate_skill`` result opening them. Rendered only by a provider
    # that ``supports_deferred_tools`` (RFC §6.4).
    references: list[str] = Field(default_factory=list)

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
        redacted: The opaque data of a block the vendor redacted (Anthropic's
            ``redacted_thinking``), replayed as received; ``thinking`` is
            then empty.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None
    redacted: str | None = None


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

    def __str__(self) -> str:
        """The provider's message, led by the provider and the status it answered:
        ``cerebras (402): Payment required…``. An OpenAI-compatible provider is
        reached through the OpenAI SDK, whose message names no provider, so a log
        line or a traceback otherwise says nothing of which one failed. A message
        that already names its provider is left as it is."""
        message = super().__str__()
        if not self.provider or re.search(
            rf"\b{re.escape(self.provider)}\b", message, re.IGNORECASE
        ):
            return message
        status = f" ({self.status_code})" if self.status_code is not None else ""
        return f"{self.provider}{status}: {message}"


# HTTP status codes that are transient and worth retrying for any AI provider.
# Providers may extend this set with their own (e.g. Anthropic's 529 "overloaded").
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503})

# What an SDK error that lost its status says of a transient one: some SDKs
# (Mistral, google-genai) raise HTTP errors that carry no code, and their
# message is all there is to read.
_RETRYABLE_ERROR_TERMS: tuple[str, ...] = ("rate", "limit", "429", "500", "502", "503")


def is_transport_failure(exc: BaseException) -> bool:
    """Whether *exc*, or an exception it was raised from, is a transport
    failure: the connection refused, reset or timed out, before the status
    or while the body was read, any error of the HTTP client's transport.

    It is the same failure whichever SDK surfaces it (Anthropic's and
    OpenAI's ``APIConnectionError`` are raised from it, Mistral and
    google-genai let it through as it is), and it is worth retrying on every
    provider. ``httpx`` and ``httpx2`` (Anthropic's client) are read only if
    already imported: an exception of theirs cannot exist otherwise.
    """
    transport: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)
    for client in ("httpx", "httpx2"):
        module = sys.modules.get(client)
        if module is not None:
            transport = (*transport, module.TransportError)
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, transport):
            return True
        seen.add(id(current))
        current = current.__cause__
    return False


def failure_retryable(status_code: int | None, exc: BaseException) -> bool:
    """Whether an SDK failure is worth retrying, for an SDK that may lose the
    status (Mistral, google-genai): a transient status when it carries one;
    without one, a transport failure (:func:`is_transport_failure`) or an
    error whose message names a transient status."""
    if status_code:
        return status_code in RETRYABLE_STATUS_CODES
    if is_transport_failure(exc):
        return True
    text = str(exc).lower()
    return any(term in text for term in _RETRYABLE_ERROR_TERMS)


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
    "exceeds the maximum number of tokens",  # Gemini / Vertex: "The input token count ..."
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
    ``none`` turns reasoning off (RFC §6.7). A provider whose vendor takes
    an effort passes the value as the vendor spells it; one that maps it to
    a vendor setting of its own (Gemini's ``thinking_level``, Ollama's
    ``think``) reads ``minimal``, ``low``, ``medium``, ``high`` and
    ``xhigh``, sent as the nearest level the model takes. ``None`` defers
    to its config."""
    tools: list[AITool] = Field(default_factory=list)
    response_schema: dict[str, Any] | None = None
    """JSON Schema the answer must satisfy, within the portable subset of
    :func:`~roomkit.providers.ai.json_schema.check_portable_schema` (checked
    here, on construction and on assignment). ``generate()`` then returns one
    JSON document in ``content`` or raises
    :class:`~roomkit.providers.ai.response_schema.ResponseSchemaError`; a
    provider never ignores it. See :attr:`AIProvider.supports_response_schema`."""
    room: RoomContext | None = None
    target_capabilities: ChannelCapabilities | None = None
    target_media_types: list[ChannelMediaType] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    response_metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description=(
            "The turn's one response-metadata record, merged into every MESSAGE "
            "event built for the turn — each event carries the record as it stands when the event "
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

    @field_validator("response_schema", mode="after")
    @classmethod
    def _portable_response_schema(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None:
            check_portable_schema(value)
        return value

    def model_post_init(self, __context: Any) -> None:
        """Protect metadata even when Pydantic's validation was bypassed."""
        if not isinstance(self.metadata, _AIContextMetadata):
            self.metadata = _AIContextMetadata(self.metadata)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Preserve secret wrapping and the schema check that ``update=`` bypasses."""
        protected_update = dict(update) if update is not None else None
        if protected_update is not None and protected_update.get("response_schema") is not None:
            check_portable_schema(protected_update["response_schema"])
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
    thinking_parts: list[AIThinkingPart] | None = None
    """The reasoning blocks to replay, in the order the vendor sent them, each
    with its signature or redacted data (RFC §6.4); ``None`` for a provider
    whose reasoning has no blocks. A block the response cut before its
    signature is not among them. ``thinking`` stays all the reasoning text,
    ``thinking_signature`` the last signature."""
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
    block: int | None = None
    """The reasoning block the delta belongs to, for a vendor whose reasoning
    comes in blocks each with its own signature (Anthropic's content block
    index); ``None`` for one whose reasoning is a single block per round."""
    redacted: str | None = None
    """A block the vendor redacted: its opaque data, replayed as received."""


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
    served: ServedCall | None = None
    """As :attr:`AIToolCall.served`: the provider already ran the call."""
    partial: bool = False
    """As :attr:`AIToolCall.partial`: the call's arguments do not read."""
    garbled: bool = False
    """As :attr:`AIToolCall.garbled`: written unreadable, not cut."""


class StreamToolCallDelta(BaseModel):
    """A fragment of a tool call's arguments, as the model composes it.

    A model that calls a tool spends the whole composition of its arguments
    producing tokens the provider delivers fragment by fragment — for a large
    argument (a document, an SVG, base64) that is minutes during which the
    stream would otherwise carry nothing at all. This item surfaces that work
    as it happens; the complete :class:`StreamToolCall` still follows and
    remains the unit of execution and persistence.

    ``arguments_delta`` is the raw fragment, not valid JSON on its own.
    ``index`` tells the response's calls apart: the call's position among
    them, in the order they first appeared (OpenAI and the providers built on
    it, Mistral, PolarGrid), or the content block's index (Anthropic).
    Providers that deliver whole tool calls (Gemini, Ollama) never emit this.
    """

    type: Literal["tool_call_delta"] = "tool_call_delta"
    id: str
    name: str
    index: int = 0
    arguments_delta: str = ""


class StreamDone(BaseModel):
    """Signals the end of a streaming AI response."""

    type: Literal["done"] = "done"
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


StreamEvent = (
    StreamThinkingDelta | StreamTextDelta | StreamToolCallDelta | StreamToolCall | StreamDone
)


def answered_by(answered: str | None, asked: str) -> dict[str, str]:
    """A response's ``metadata`` naming its model (RFC §6.7): the one that
    answered, as the response names it, else the one asked for. One rule for
    a response and for the end of a stream, so both modes report the same."""
    return {"model": answered or asked}


def stream_done(
    finish_reason: str | None,
    usage: dict[str, int],
    answered: str | None,
    asked: str,
    refusal: str = "",
) -> StreamDone:
    """A stream's end: its stop reason, its usage, its model as
    :func:`answered_by` names it, and any refusal the stream carried."""
    metadata = answered_by(answered, asked)
    if refusal:
        metadata["refusal"] = refusal
    return StreamDone(finish_reason=finish_reason, usage=usage, metadata=metadata)


def thinking_parts_of(response: AIResponse) -> list[AIThinkingPart]:
    """A response's reasoning blocks: its ``thinking_parts``, or for a
    provider that reports one block its ``thinking`` with its signature."""
    if response.thinking_parts is not None:
        return list(response.thinking_parts)
    if response.thinking or response.thinking_signature:
        return [
            AIThinkingPart(thinking=response.thinking or "", signature=response.thinking_signature)
        ]
    return []


def stream_call_of(call: AIToolCall) -> StreamToolCall:
    """The streamed form of a tool call, every field it carries kept."""
    return StreamToolCall(
        id=call.id,
        name=call.name,
        arguments=call.arguments,
        metadata=call.metadata,
        served=call.served,
        partial=call.partial,
        garbled=call.garbled,
    )


def tool_call_of(event: StreamToolCall) -> AIToolCall:
    """The tool call a streamed one is, every field it carries kept."""
    return AIToolCall(
        id=event.id,
        name=event.name,
        arguments=event.arguments,
        metadata=event.metadata,
        served=event.served,
        partial=event.partial,
        garbled=event.garbled,
    )


def response_stream_events(
    response: AIResponse,
    call_deltas: Callable[[AIToolCall], Iterable[StreamToolCallDelta]] | None = None,
) -> Iterator[StreamEvent]:
    """A whole response as the events a stream of it carries (RFC §6.4).

    Everything ``generate()`` returned reaches the tool loop: each thinking
    block with its signature or redacted data (a signature alone too), the
    text, each call with its
    metadata (a thought signature among them), then the done event's finish
    reason, usage and metadata. *call_deltas* gives the argument fragments a
    stream announces ahead of each call, when there are any.
    """
    for block, part in enumerate(thinking_parts_of(response)):
        yield StreamThinkingDelta(
            thinking=part.thinking,
            signature=part.signature,
            redacted=part.redacted,
            block=block if response.thinking_parts is not None else None,
        )
    if response.content:
        yield StreamTextDelta(text=response.content)
    for call in response.tool_calls:
        if call_deltas is not None:
            yield from call_deltas(call)
        yield stream_call_of(call)
    yield StreamDone(
        finish_reason=response.finish_reason,
        usage=response.usage,
        metadata=response.metadata,
    )


# The usage counters a vendor bills, disjoint: a token is counted under one of
# them only (RFC §6.7). ``reasoning_tokens`` is not one: it is the thinking
# share of ``output_tokens``.
BILLED_USAGE_COUNTERS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "input_image_tokens",
    "output_image_tokens",
)


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
        ``reasoning_tokens`` is not one of them: it is the thinking share of
        ``output_tokens``, a detail this ignores (RFC §6).

        A counter with no corresponding rate is omitted: ``None`` means the
        catalog does not represent a separate per-token charge for it. When the
        response crosses a published long-context threshold, the configured
        input and output multipliers are applied automatically.

        Args:
            usage: A response's token counters, as reported by the provider.

        Returns:
            The cost of that response, in :attr:`currency`.
        """
        counters = {name: self._usage_counter(usage, name) for name in BILLED_USAGE_COUNTERS}
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
        capabilities: Capability tags. In a live listing, the tags the
            source reports (e.g. Ollama's ``"completion"``, ``"embedding"``,
            ``"vision"``, ``"tools"``), or, where it reports no model type
            at all, the public tags of the provider's catalog (PolarGrid);
            and ``"transcription"`` or ``"speech"`` on a speech-to-text or
            text-to-speech model (:mod:`roomkit.providers.ai.model_tags`).
            A curated catalog (``available_models()``) may carry
            provider-specific tags here too, internal routing flags among
            them (``chat_tools_refused``, ``deferred_tools``): only
            ``transcription``, ``speech`` and ``image_gen`` are defined
            once, with one meaning across providers. Empty when the source
            reports none — consumers treat empty as "unknown, allow
            everywhere" rather than "none".
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

    _telemetry: TelemetryProvider | None = None
    """Set by the channel that serves the provider, for its metrics."""

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

    @property
    def supports_response_schema(self) -> bool:
        """Whether :meth:`generate` honours :attr:`AIContext.response_schema`.

        A provider default, not a fact about every model it can reach: a model
        or an OpenAI-compatible server may still refuse the constraint, which
        surfaces as an error rather than as prose.

        The contract is the implementation's to keep (RFC §6.7): refuse a
        schema it cannot carry before sending anything, and never return an
        answer that does not satisfy it. The helpers in
        :mod:`roomkit.providers.ai.response_schema` do both;
        a provider that ignores the field breaks the contract.
        """
        return False

    @property
    def supports_deferred_tools(self) -> bool:
        """Whether the model can hold a tool declared but unseen.

        Such a tool (``AITool.defer_loading``) stays out of the model's view
        and out of the cached prompt prefix until a tool result references it
        (``AIToolResultPart.references``), so a tool that has to appear mid-turn
        does not change the declaration (RFC §6.4). False by default: a
        provider that cannot declares only what the model may see.
        """
        return False

    @property
    def supports_response_schema_with_tools(self) -> bool:
        """Whether a response schema may ride a turn that also carries tools.

        The model then either calls tools or answers in the schema, and only
        the final answer is checked (RFC §6.7). False wherever the constraint
        is a decoding grammar: it stops the model from calling a tool, and the
        model invents a schema-valid answer instead.
        """
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
    def _curated_index(cls) -> dict[str, ModelInfo]:
        """Curated entries :meth:`_merge_curated` backfills from, by model id.

        The advertised catalog, unless a provider recognises more models than
        it advertises — PolarGrid's customer-pilot models, served from no
        public edge but live on the edge a pilot customer is pinned to.
        """
        return {m.id: m for m in cls.available_models()}

    @classmethod
    def _listing(cls, live: list[ModelInfo]) -> list[ModelInfo]:
        """*live* as every ``list_models()`` returns it: backfilled from the
        curated catalog, its speech models tagged."""
        return with_speech_tags(cls._merge_curated(live))

    @classmethod
    def _merge_curated(cls, live: list[ModelInfo]) -> list[ModelInfo]:
        """Backfill metadata absent from live results using the curated catalog.

        A live models endpoint typically returns ids with little metadata.
        For each live model that also appears in :meth:`_curated_index`
        (the advertised catalog, by default), fill any missing
        ``display_name``/``context_window``/``supports_vision``/``pricing``
        from the curated entry, keeping whatever the API did report. A
        catalog's ``capabilities`` stay out: most carry internal routing flags
        (``chat_tools_refused``, ``deferred_tools``), not a public tag set.
        """
        curated = cls._curated_index()
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

        A provider whose wire format fragments a tool call's arguments SHOULD
        also yield :class:`StreamToolCallDelta` per fragment; it is optional,
        and one that delivers whole calls yields none.

        The default reads what the provider has, so every provider serves the
        one tool loop unchanged (RFC §6.4): a turn without tools streams
        through ``generate_stream`` where the provider streams text, and any
        other turn wraps ``generate()``, everything it returned kept. Override
        for true streaming support.
        """
        if self.supports_streaming and not context.tools:
            async for text in self.generate_stream(context):
                yield StreamTextDelta(text=text)
            # A text stream reports no usage and no stop reason.
            yield StreamDone(finish_reason="stop")
            return
        for event in response_stream_events(await self.generate(context)):
            yield event

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


class FirstToken:
    """A stream's time to first token, recorded once: at its first text or
    reasoning, never at a call fragment (the ``roomkit.llm.ttfb_ms`` metric).
    Started when it is made, just before the request."""

    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider
        self._t0: float | None = _time.monotonic()

    def seen(self) -> None:
        """Mark output reaching the stream; the first call records it."""
        if self._t0 is not None:
            self._provider._record_ttfb(self._t0)  # noqa: SLF001
            self._t0 = None
