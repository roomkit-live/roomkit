"""Mistral AI provider — generates responses via the Mistral Chat Completions API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from roomkit.providers.ai.base import (
    RETRYABLE_STATUS_CODES,
    AIContext,
    AIMessage,
    AIProvider,
    AIResponse,
    AIToolCall,
    FirstToken,
    ModelInfo,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    stream_done,
    tool_call_of,
    unstatused_failure_retryable,
)
from roomkit.providers.ai.chat_request import ChatDialect, chat_messages
from roomkit.providers.ai.openai_dialect import (
    ThinkTagParser,
    ToolCallSlots,
    json_schema_format,
)
from roomkit.providers.ai.reasoning import thinking_switch, turn_setting
from roomkit.providers.ai.response_schema import checked_stream, schema_for_generate
from roomkit.providers.ai.tool_declaration import ToolNameRule, chat_tool_declarations
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.mistral.models import MODELS
from roomkit.providers.utils import _aclose_stream
from roomkit.providers.vendor_endpoint import MISTRAL_BASE_URL, is_vendor_endpoint

MISTRAL_CHAT = ChatDialect(names_tool_results=True)
"""Mistral renders a conversation as OpenAI does, each tool message naming its
tool."""

MISTRAL_TOOL_NAMES = ToolNameRule("mistral", r"[A-Za-z0-9_.-]+")
"""The tool names Mistral accepts, a dot included, a colon not (measured
2026-10-02)."""


def _server_call_id(call_id: str | None) -> str | None:
    """The call's id, or None when the server gave none.

    Mistral's SDK fills a missing ``id`` with the string ``"null"``: read as
    an id, every id-less call would share it.
    """
    return None if call_id in (None, "", "null") else call_id


class MistralAIProvider(AIProvider):
    """AI provider using the Mistral AI API.

    Supports streaming, tool calling, vision (multimodal models), and
    ``<think>`` tag parsing for reasoning models.
    """

    def __init__(self, config: MistralConfig) -> None:
        try:
            from mistralai.client import Mistral as _Mistral
        except ImportError as exc:
            raise ImportError(
                "mistralai is required for MistralAIProvider. "
                "Install it with: pip install roomkit[mistral]"
            ) from exc

        self._config = config
        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key.get_secret_value(),
        }
        if config.server_url is not None:
            client_kwargs["server_url"] = config.server_url
        self._client = _Mistral(**client_kwargs)

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_vision(self) -> bool:
        # Vision support is per-model on Mistral and the multimodal lineup
        # keeps shifting (Pixtral is deprecated; Mistral Large 3 is
        # multimodal). Rather than maintain a prefix list that goes stale,
        # pass images through whenever they arrive — the API rejects a
        # non-vision model itself. Returning True keeps the routing layer
        # honest: image content reaches the wire instead of being silently
        # filtered out one layer above.
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    @property
    def supports_response_schema(self) -> bool:
        """Custom structured outputs, through a strict ``json_schema`` format."""
        return True

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Mistral chat/multimodal models."""
        return list(MODELS)

    async def list_models(self) -> list[ModelInfo]:
        """List models the Mistral API currently exposes for this key."""
        resp = await self._client.models.list_async()
        data = getattr(resp, "data", None) or []
        live = [ModelInfo(id=m.id) for m in data if getattr(m, "id", None)]
        return self._listing(live)

    # -- Message formatting ----------------------------------------------------

    def _build_messages(
        self,
        messages: list[AIMessage],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """The conversation as Mistral reads it (``chat_request``)."""
        return chat_messages(messages, system_prompt, MISTRAL_CHAT, provider="mistral")

    def _build_kwargs(self, context: AIContext) -> dict[str, Any]:
        """Build kwargs shared by generate and streaming paths."""
        messages = self._build_messages(context.messages, context.system_prompt)
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": context.max_tokens or self._config.max_tokens,
            "messages": messages,
        }
        if context.temperature is not None:
            kwargs["temperature"] = context.temperature
        effort = self._resolve_reasoning_effort(context)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        if context.tools:
            if is_vendor_endpoint(self._config.server_url, MISTRAL_BASE_URL):
                # Behind another server_url the server decides its names (RFC §6.7).
                MISTRAL_TOOL_NAMES.check(t.name for t in context.tools)
            kwargs["tools"] = chat_tool_declarations(context.tools)
        if context.response_schema is not None:
            kwargs["response_format"] = json_schema_format(context.response_schema)
        return kwargs

    @staticmethod
    def _usage_from(raw: Any) -> dict[str, int]:
        """Separate cached prompt tokens from fresh Mistral input."""
        prompt = raw.prompt_tokens or 0
        # The SDK does not declare the field: it keeps it as a plain dict.
        details = getattr(raw, "prompt_tokens_details", None) or {}
        cached = details.get("cached_tokens") or 0
        usage = {
            "input_tokens": max(prompt - cached, 0),
            "output_tokens": raw.completion_tokens or 0,
        }
        if cached:
            usage["cache_read_input_tokens"] = cached
        return usage

    def _resolve_reasoning_effort(self, context: AIContext) -> str | None:
        """Decide ``reasoning_effort`` for this request.

        The turn's effort outranks the configured one (RFC §6.7).
        ``thinking_budget`` gates per-turn: ``None`` passes that effort through
        verbatim; ``0`` forces ``"none"`` (reasoning off); ``>0`` honors that
        effort or defaults to ``"high"``. Returns ``None`` to omit the
        parameter entirely (model decides — Magistral always reasons).
        """
        effort = turn_setting(context.reasoning_effort, self._config.reasoning_effort)
        switch = thinking_switch(context)
        if switch is None:
            return effort
        if switch is False:
            return "none"
        return effort or "high"

    @staticmethod
    def _chunks_to_segments(chunks: list[Any]) -> list[tuple[str, str]]:
        """Map Mistral structured content chunks to ``(kind, text)`` segments.

        Reasoning models stream ``content`` as a list of typed chunks instead of
        a plain string: a ``ThinkChunk`` (``type == "thinking"``) carries the
        reasoning trace in ``.thinking`` (a list of inner text chunks), while a
        ``TextChunk`` (``type == "text"``) carries answer text in ``.text``.
        Unknown chunk types are skipped.
        """
        segments: list[tuple[str, str]] = []
        for chunk in chunks:
            ctype = getattr(chunk, "type", None)
            if ctype == "thinking":
                for inner in getattr(chunk, "thinking", None) or []:
                    inner_text = getattr(inner, "text", None)
                    if inner_text:
                        segments.append(("thinking", inner_text))
            elif ctype == "text":
                text = getattr(chunk, "text", None)
                if text:
                    segments.append(("text", text))
        return segments

    # -- Structured streaming --------------------------------------------------

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events with ``<think>`` tag parsing.

        Text inside ``<think>...</think>`` is yielded as
        :class:`StreamThinkingDelta`; everything else as
        :class:`StreamTextDelta`.  Tool calls are accumulated from deltas
        and yielded as :class:`StreamToolCall`. A response schema is checked
        before the done event (RFC §6.7).
        """
        schema_for_generate(
            context,
            supported=self.supports_response_schema,
            with_tools=self.supports_response_schema_with_tools,
            provider="mistral",
        )
        stream = checked_stream(
            self._events(context),
            context,
            provider="mistral",
            refusal=lambda _done: None,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await _aclose_stream(stream)

    async def _events(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """The streamed call itself, shared by :meth:`generate`."""
        kwargs = self._build_kwargs(context)
        first_token = FirstToken(self)
        parser = ThinkTagParser()

        # Accumulate tool call deltas across chunks
        tool_call_slots = ToolCallSlots()
        finish_reason: str | None = None
        usage: dict[str, int] = {}

        model: str | None = None
        try:
            response = await self._client.chat.stream_async(**kwargs)
            async for event in response:
                data = event.data
                model = getattr(data, "model", None) or model
                if not data.choices:
                    continue
                delta = data.choices[0].delta
                finish_reason = data.choices[0].finish_reason or finish_reason

                # Extract usage from the stream when available. Normalize to
                # the canonical key names every other provider emits
                # (input_tokens / output_tokens) so downstream usage trackers
                # read one contract — Mistral's SDK calls them prompt/completion.
                if data.usage:
                    usage = self._usage_from(data.usage)

                # Accumulate streamed tool call deltas
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        function = getattr(tc_delta, "function", None)
                        # Surface the call while it is being composed. The
                        # complete StreamToolCall below remains the unit of
                        # execution and persistence.
                        composed = tool_call_slots.fold(
                            getattr(tc_delta, "index", None),
                            _server_call_id(tc_delta.id),
                            function.name if function else None,
                            function.arguments if function else "",
                        )
                        if composed is not None:
                            yield composed

                # Reasoning models stream content as a list of typed chunks
                # (ThinkChunk / TextChunk); older or non-reasoning models stream
                # a plain string with optional inline <think>...</think> tags.
                content = delta.content if hasattr(delta, "content") else None
                if isinstance(content, list):
                    segments = self._chunks_to_segments(content)
                elif content:
                    segments = list(parser.feed(content))
                else:
                    segments = []
                for kind, segment in segments:
                    first_token.seen()
                    if kind == "thinking":
                        yield StreamThinkingDelta(thinking=segment)
                    else:
                        yield StreamTextDelta(text=segment)

            # Flush any remaining buffered text
            for kind, segment in parser.flush():
                first_token.seen()
                if kind == "thinking":
                    yield StreamThinkingDelta(thinking=segment)
                else:
                    yield StreamTextDelta(text=segment)

            # Yield accumulated tool calls
            for call in tool_call_slots.calls(finish_reason):
                yield call

            yield stream_done(finish_reason, usage, model, self._config.model)

        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def generate(self, context: AIContext) -> AIResponse:
        """Generate by consuming the structured stream."""
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        tool_calls: list[AIToolCall] = []
        done_event: StreamDone | None = None

        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamThinkingDelta):
                thinking_parts.append(event.thinking)
            elif isinstance(event, StreamTextDelta):
                text_parts.append(event.text)
            elif isinstance(event, StreamToolCall):
                tool_calls.append(tool_call_of(event))
            elif isinstance(event, StreamDone):
                done_event = event

        finish_reason = done_event.finish_reason if done_event else None
        return AIResponse(
            content="".join(text_parts),
            thinking="".join(thinking_parts) if thinking_parts else None,
            finish_reason=finish_reason,
            usage=done_event.usage if done_event else {},
            metadata=done_event.metadata if done_event else {},
            tool_calls=tool_calls,
        )

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text deltas as they arrive from the Mistral API."""
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text

    # -- Helpers ---------------------------------------------------------------

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """Wrap an SDK exception into a ProviderError."""
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        retryable = (
            status_code in RETRYABLE_STATUS_CODES
            if status_code
            else unstatused_failure_retryable(exc)
        )
        return ProviderError(
            str(exc),
            retryable=retryable,
            provider="mistral",
            status_code=status_code,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if hasattr(self._client, "close"):
            await self._client.close()
