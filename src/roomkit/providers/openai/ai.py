"""OpenAI AI provider — generates responses via the OpenAI Chat Completions API."""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from roomkit.providers.ai.base import (
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
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)
from roomkit.providers.openai.config import OpenAIConfig

# OpenAI models that support vision
_VISION_MODELS = (
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4-vision",
    "o1",
    "o3",
)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


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
        self._client = _openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout,
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
        """GPT-4o and GPT-4-turbo models support vision."""
        return any(self._config.model.startswith(prefix) for prefix in _VISION_MODELS)

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

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
                # Tool results → separate messages with role="tool"
                for p in m.content:
                    if isinstance(p, AIToolResultPart):
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": p.tool_call_id,
                                "content": p.result,
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

    # -- Non-streaming ---------------------------------------------------------

    async def generate(self, context: AIContext) -> AIResponse:
        messages = self._build_messages(context.messages, context.system_prompt)

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": context.max_tokens or self._config.max_tokens,
            "messages": messages,
        }
        if context.temperature is not None:
            kwargs["temperature"] = context.temperature

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
        except self._api_status_error as exc:
            retryable = exc.status_code in (429, 500, 502, 503)
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
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

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
        if context.temperature is not None:
            kwargs["temperature"] = context.temperature
        if context.max_tokens is not None:
            kwargs["max_tokens"] = context.max_tokens
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
                    usage = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }
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

        except self._api_status_error as exc:
            raise ProviderError(
                str(exc),
                retryable=exc.status_code in {429, 500, 502, 503},
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

    def _record_ttfb(self, t0: float) -> None:
        """Record time-to-first-byte metric."""
        ttfb_ms = (time.monotonic() - t0) * 1000
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        telemetry.record_metric(
            "roomkit.llm.ttfb_ms",
            ttfb_ms,
            unit="ms",
            attributes={"provider": self._provider_name, "model": self._config.model},
        )

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
