"""OpenAI AI provider — generates responses via the OpenAI Chat Completions API."""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from roomkit.providers.ai.base import (
    RETRYABLE_STATUS_CODES,
    AIContext,
    AIImagePart,
    AIMessage,
    AIProvider,
    AIResponse,
    AITextPart,
    AIThinkingPart,
    AIToolCall,
    AIToolCallPart,
    AIToolResultPart,
    ModelInfo,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.openai.models import MODELS

# Fallback only, for ids the catalog does not carry — a snapshot newer than
# this release, or an OpenAI-compatible server behind ``base_url`` naming its
# own model. A model that is in the catalog is answered from there instead.
_VISION_PREFIXES = (
    "gpt-5",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4-turbo",
    "gpt-4-vision",
    "o1",
    "o3",
    "o4",
)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _field_reasoning(carrier: Any) -> str | None:
    """Read a reasoning trace carried in its own field rather than inline.

    ``<think>`` tags are one of two conventions OpenAI-compatible servers use.
    The other is a dedicated field beside ``content`` — ``reasoning_content``
    for DeepSeek, Qwen and vLLM's reasoning parsers, ``reasoning`` for
    OpenRouter. Both a streaming delta and a complete message carry it under
    the same names, so both paths read it here.
    """
    value = getattr(carrier, "reasoning_content", None) or getattr(carrier, "reasoning", None)
    return value if isinstance(value, str) and value else None


def _merge_thinking(inline: str | None, field: str | None) -> str | None:
    """Combine the two reasoning conventions into one trace.

    A server uses one or the other, so in practice exactly one side is set;
    concatenating rather than picking a winner means neither is dropped if one
    ever emits both.
    """
    if not field:
        return inline
    return f"{inline}{field}" if inline else field


class _ThinkTagParser:
    """Stateful parser for ``<think>...</think>`` tags in a text stream.

    vLLM / Ollama models (DeepSeek-R1, QwQ, etc.) emit reasoning inside
    ``<think>`` tags before the answer text.  This parser classifies
    incoming chunks into ``"thinking"`` and ``"text"`` segments, handling
    tags that are split across chunk boundaries.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._in_thinking = False
        self._buf = ""

    def feed(self, chunk: str) -> list[tuple[Literal["thinking", "text"], str]]:
        """Process *chunk* and return classified ``(kind, content)`` pairs."""
        self._buf += chunk
        results: list[tuple[Literal["thinking", "text"], str]] = []

        while self._buf:
            tag = self._CLOSE if self._in_thinking else self._OPEN
            idx = self._buf.find(tag)

            if idx >= 0:
                before = self._buf[:idx]
                if before:
                    kind: Literal["thinking", "text"] = "thinking" if self._in_thinking else "text"
                    results.append((kind, before))
                self._buf = self._buf[idx + len(tag) :]
                self._in_thinking = not self._in_thinking
            else:
                # Hold back a suffix that could be a partial tag start.
                hold = self._partial_tag_len(self._buf, tag)
                if hold:
                    emit = self._buf[:-hold]
                    if emit:
                        kind = "thinking" if self._in_thinking else "text"
                        results.append((kind, emit))
                    self._buf = self._buf[-hold:]
                    break
                # No partial match — emit everything.
                kind = "thinking" if self._in_thinking else "text"
                results.append((kind, self._buf))
                self._buf = ""

        return results

    def flush(self) -> list[tuple[Literal["thinking", "text"], str]]:
        """Emit any remaining buffered content."""
        if not self._buf:
            return []
        kind: Literal["thinking", "text"] = "thinking" if self._in_thinking else "text"
        result = [(kind, self._buf)]
        self._buf = ""
        return result

    @staticmethod
    def _partial_tag_len(text: str, tag: str) -> int:
        """Length of the longest suffix of *text* that is a prefix of *tag*."""
        max_check = min(len(text), len(tag) - 1)
        for length in range(max_check, 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0


class OpenAIAIProvider(AIProvider):
    """AI provider using the OpenAI Chat Completions API."""

    def __init__(self, config: OpenAIConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAIAIProvider. "
                "Install it with: pip install roomkit[openai]"
            ) from exc
        self._config = config
        self._api_status_error = _openai.APIStatusError
        self._api_connection_error = _openai.APIConnectionError
        self._client = _openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
            default_headers=config.default_headers,
        )

    @property
    def _provider_name(self) -> str:
        """Provider identifier used in error messages and telemetry."""
        return "openai"

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_vision(self) -> bool:
        """Whether the configured model accepts image input.

        Read from the offline catalog, which states it per model, rather than
        from a prefix table that has to be remembered on every release — the
        prefix form predated GPT-5 entirely and silently reported the whole
        current lineup as text-only, dropping images before they reached
        the wire.
        """
        entry = self.catalog_entry()
        if entry is not None and entry.supports_vision is not None:
            return entry.supports_vision
        return self._config.model.startswith(_VISION_PREFIXES)

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of OpenAI chat/multimodal models."""
        return list(MODELS)

    async def list_models(self) -> list[ModelInfo]:
        """List every model id the configured endpoint exposes.

        The OpenAI ``/v1/models`` response carries only ids — metadata for
        known chat models is backfilled from the curated catalog. The raw
        list also includes non-chat models (embeddings, audio); they pass
        through unfiltered since the endpoint reports no capability field.
        """
        page = await self._client.models.list()
        live = [ModelInfo(id=m.id) for m in page.data]
        return self._merge_curated(live)

    def _format_content(
        self,
        content: (
            str
            | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart]
        ),
    ) -> str | list[dict[str, Any]]:
        """Format message content for OpenAI API.

        Converts AITextPart/AIImagePart to OpenAI's content block format.
        AIThinkingPart is re-injected as a ``<think>`` text block so
        vLLM / Ollama models see their own reasoning in history.
        """
        if isinstance(content, str):
            return content

        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, AITextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, AIImagePart):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": part.url},
                    }
                )
            elif isinstance(part, AIThinkingPart):
                # Re-wrap thinking as <think> tags so the model sees its own
                # prior reasoning when the conversation is sent back.
                parts.append({"type": "text", "text": f"<think>{part.thinking}</think>"})
        return parts

    def _build_messages(
        self,
        messages: list[AIMessage],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-formatted messages with tool call/result support."""
        result: list[dict[str, Any]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        for m in messages:
            if isinstance(m.content, list) and any(
                isinstance(p, AIToolCallPart) for p in m.content
            ):
                # Assistant message with tool calls
                tool_calls = []
                content_text = ""
                for p in m.content:
                    if isinstance(p, AITextPart):
                        content_text = p.text
                    elif isinstance(p, AIThinkingPart):
                        # Prepend thinking as <think> tags before text
                        content_text = f"<think>{p.thinking}</think>" + content_text
                    elif isinstance(p, AIToolCallPart):
                        tool_calls.append(
                            {
                                "id": p.id,
                                "type": "function",
                                "function": {
                                    "name": p.name,
                                    "arguments": json.dumps(p.arguments),
                                },
                            }
                        )
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content_text or None,
                    "tool_calls": tool_calls,
                }
                result.append(msg)
            elif isinstance(m.content, list) and any(
                isinstance(p, AIToolResultPart) for p in m.content
            ):
                # Tool results → separate messages with role="tool". Chat
                # Completions accepts only text on a tool message (image_url
                # parts are user-only), so an image result keeps the tool
                # message text-only and the image is split onto a synthetic
                # user message emitted after every tool message — the
                # call/result pairing stays valid. Text results are unchanged.
                pending_images: list[AIImagePart] = []
                for p in m.content:
                    if isinstance(p, AIToolResultPart):
                        text, images = p.split_for_message()
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": p.tool_call_id,
                                "content": text,
                            }
                        )
                        pending_images.extend(images)
                if pending_images:
                    result.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": img.url}}
                                for img in pending_images
                            ],
                        }
                    )
            else:
                result.append(
                    {
                        "role": m.role,
                        "content": self._format_content(m.content),
                    }
                )
        return result

    def _token_limit_kwarg(self, value: int) -> dict[str, int]:
        """Build the output-cap kwarg under the name the endpoint expects.

        OpenAI's newer models reject the deprecated ``max_tokens`` and require
        ``max_completion_tokens``; OpenAI-compatible servers (vLLM, older Azure)
        only understand ``max_tokens``. ``use_max_completion_tokens`` on the
        config selects between them.
        """
        key = "max_completion_tokens" if self._config.use_max_completion_tokens else "max_tokens"
        return {key: value}

    def _apply_sampling_kwargs(self, kwargs: dict[str, Any], context: AIContext) -> None:
        """Add temperature and reasoning_effort to a request when applicable.

        Temperature is dropped for models that only accept the default
        (``supports_custom_temperature=False``). GPT-5.6 function tools on the
        official Chat Completions endpoint require effective reasoning ``none``;
        omission is not equivalent because that family defaults to ``medium``.
        Older models keep the existing conservative behavior of omitting an
        explicitly configured effort on tool turns.
        """
        if context.temperature is not None and self._config.supports_custom_temperature:
            kwargs["temperature"] = context.temperature

        official_gpt_5_6_tool_turn = (
            bool(context.tools)
            and self._provider_name == "openai"
            and getattr(self._config, "base_url", None) is None
            and self._config.model.startswith("gpt-5.6")
        )
        if official_gpt_5_6_tool_turn:
            kwargs["reasoning_effort"] = "none"
        elif self._config.reasoning_effort is not None and not context.tools:
            kwargs["reasoning_effort"] = self._config.reasoning_effort

    def _apply_extra_body(self, kwargs: dict[str, Any]) -> None:
        """Merge configured ``extra_body`` (server-specific request fields).

        Carries params the OpenAI schema omits — vLLM guided decoding and
        extra sampling knobs — through the SDK's ``extra_body`` passthrough.
        Merges into any ``extra_body`` a subclass already populated (e.g.
        OpenRouter's ``reasoning``) instead of replacing it; keys already set
        on the request win, so static config never clobbers a per-turn value.
        """
        if self._config.extra_body:
            kwargs["extra_body"] = {**self._config.extra_body, **kwargs.get("extra_body", {})}

    @staticmethod
    def _usage_from(raw: Any) -> dict[str, int]:
        """Map an OpenAI-shaped usage object to roomkit's canonical counters.

        ``prompt_tokens`` includes cache reads and explicit cache writes, while
        roomkit reports those counters separately. Subtract both from ordinary
        input so a budget or cost dashboard cannot charge them twice.
        """
        prompt = raw.prompt_tokens or 0
        details = getattr(raw, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) if details else 0) or 0
        written = (
            (getattr(details, "cache_write_tokens", 0) if details else 0)
            or getattr(raw, "cache_write_tokens", 0)
            or 0
        )
        usage = {
            "input_tokens": max(prompt - cached - written, 0),
            "output_tokens": raw.completion_tokens or 0,
        }
        if cached:
            usage["cache_read_input_tokens"] = cached
        if written:
            usage["cache_creation_input_tokens"] = written
        return usage

    # -- Non-streaming ---------------------------------------------------------

    async def generate(self, context: AIContext) -> AIResponse:
        messages = self._build_messages(context.messages, context.system_prompt)

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            **self._token_limit_kwarg(context.max_tokens or self._config.max_tokens),
            "messages": messages,
        }
        self._apply_sampling_kwargs(kwargs, context)
        self._apply_extra_body(kwargs)

        # Add tools if provided
        if context.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in context.tools
            ]

        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except ProviderError:
            raise
        except self._api_connection_error as exc:
            raise ProviderError(
                str(exc),
                retryable=True,
                provider=self._provider_name,
            ) from exc
        except self._api_status_error as exc:
            retryable = exc.status_code in RETRYABLE_STATUS_CODES
            raise ProviderError(
                str(exc),
                retryable=retryable,
                provider=self._provider_name,
                status_code=exc.status_code,
            ) from exc
        except Exception as exc:
            raise ProviderError(
                str(exc),
                retryable=False,
                provider=self._provider_name,
                status_code=None,
            ) from exc

        ttfb_ms = (time.monotonic() - t0) * 1000
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        telemetry.record_metric(
            "roomkit.llm.ttfb_ms",
            ttfb_ms,
            unit="ms",
            attributes={"provider": self._provider_name, "model": self._config.model},
        )

        if not response.choices:
            return AIResponse(content="")

        choice = response.choices[0]
        usage: dict[str, int] = {}
        if response.usage:
            usage = self._usage_from(response.usage)

        # Extract tool calls from response
        tool_calls: list[AIToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc.function.arguments}
                tool_calls.append(
                    AIToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        # Extract <think>...</think> tags from response text.
        raw_text = choice.message.content or ""
        thinking, content = _extract_think_tags(raw_text)
        thinking = _merge_thinking(thinking, _field_reasoning(choice.message))

        return AIResponse(
            content=content,
            thinking=thinking,
            finish_reason=choice.finish_reason,
            usage=usage,
            metadata={"model": response.model},
            tool_calls=tool_calls,
        )

    # -- Streaming -------------------------------------------------------------

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events with ``<think>`` tag parsing.

        Text inside ``<think>...</think>`` is yielded as
        :class:`StreamThinkingDelta`; everything else as
        :class:`StreamTextDelta`.  Tool calls are collected from the final
        chunks and yielded as :class:`StreamToolCall`.
        """
        messages = self._build_messages(context.messages, context.system_prompt)
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "stream": True,
        }
        if self._config.include_stream_usage:
            kwargs["stream_options"] = {"include_usage": True}
        self._apply_sampling_kwargs(kwargs, context)
        self._apply_extra_body(kwargs)
        if context.max_tokens is not None:
            kwargs.update(self._token_limit_kwarg(context.max_tokens))
        if context.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in context.tools
            ]

        t0 = time.monotonic()
        first_token = True
        parser = _ThinkTagParser()

        # Accumulate tool call deltas across chunks
        tool_call_accum: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, int] = {}

        try:
            response = await self._client.chat.completions.create(**kwargs)
            async for chunk in response:
                # With include_usage, the final chunk has usage but empty choices
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = self._usage_from(chunk.usage)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason or finish_reason

                # Accumulate streamed tool call deltas
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_accum:
                            tool_call_accum[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        acc = tool_call_accum[idx]
                        if tc_delta.id:
                            acc["id"] = tc_delta.id
                        if hasattr(tc_delta, "function") and tc_delta.function:
                            if tc_delta.function.name:
                                acc["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                acc["arguments"] += tc_delta.function.arguments

                # OpenAI-compatible reasoning models (DeepSeek-R1, vLLM with a
                # reasoning parser) stream reasoning in a dedicated field instead
                # of inline <think> tags. Surface it as thinking when present.
                reasoning = _field_reasoning(delta)
                if reasoning:
                    if first_token:
                        self._record_ttfb(t0)
                        first_token = False
                    yield StreamThinkingDelta(thinking=reasoning)

                # Process text content through the think-tag parser
                text = delta.content if hasattr(delta, "content") else None
                if text:
                    for kind, segment in parser.feed(text):
                        if first_token:
                            self._record_ttfb(t0)
                            first_token = False
                        if kind == "thinking":
                            yield StreamThinkingDelta(thinking=segment)
                        else:
                            yield StreamTextDelta(text=segment)

            # Flush any remaining buffered text from the parser
            for kind, segment in parser.flush():
                if first_token:
                    self._record_ttfb(t0)
                    first_token = False
                if kind == "thinking":
                    yield StreamThinkingDelta(thinking=segment)
                else:
                    yield StreamTextDelta(text=segment)

            # Yield accumulated tool calls
            for _idx in sorted(tool_call_accum):
                acc = tool_call_accum[_idx]
                try:
                    args = json.loads(acc["arguments"]) if acc["arguments"] else {}
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": acc["arguments"]}
                yield StreamToolCall(id=acc["id"], name=acc["name"], arguments=args)

            yield StreamDone(finish_reason=finish_reason, usage=usage)

        except self._api_connection_error as exc:
            raise ProviderError(
                str(exc),
                retryable=True,
                provider=self._provider_name,
            ) from exc
        except self._api_status_error as exc:
            raise ProviderError(
                str(exc),
                retryable=exc.status_code in RETRYABLE_STATUS_CODES,
                provider=self._provider_name,
                status_code=exc.status_code,
            ) from exc
        except Exception as exc:
            raise ProviderError(
                str(exc),
                retryable=False,
                provider=self._provider_name,
            ) from exc

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text deltas (thinking content filtered out)."""
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()


def _extract_think_tags(text: str) -> tuple[str | None, str]:
    """Extract ``<think>...</think>`` content from *text*.

    Returns:
        ``(thinking, clean_text)`` — *thinking* is ``None`` when no tags
        are present.
    """
    matches = _THINK_RE.findall(text)
    if not matches:
        return None, text
    thinking = "\n".join(m.strip() for m in matches if m.strip())
    clean = _THINK_RE.sub("", text).strip()
    return thinking or None, clean
