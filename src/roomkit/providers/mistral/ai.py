"""Mistral AI provider — generates responses via the Mistral Chat Completions API."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

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
from roomkit.providers.mistral.config import MistralConfig

# Mistral models that support vision (Pixtral family)
_VISION_MODELS = ("pixtral",)


class MistralAIProvider(AIProvider):
    """AI provider using the Mistral AI API.

    Supports streaming, tool calling, vision (Pixtral models), and
    ``<think>`` tag parsing for reasoning models.
    """

    def __init__(self, config: MistralConfig) -> None:
        try:
            from mistralai import Mistral as _Mistral
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
        """Pixtral models support vision."""
        return any(self._config.model.startswith(prefix) for prefix in _VISION_MODELS)

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    # -- Message formatting ----------------------------------------------------

    def _format_content(
        self,
        content: (
            str
            | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart]
        ),
    ) -> str | list[dict[str, Any]]:
        """Format message content for the Mistral API.

        AIThinkingPart is re-injected as ``<think>`` text so the model sees
        its own prior reasoning when the conversation is sent back.
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
                parts.append({"type": "text", "text": f"<think>{part.thinking}</think>"})
        return parts

    def _build_messages(
        self,
        messages: list[AIMessage],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build Mistral-formatted messages with tool call/result support."""
        result: list[dict[str, Any]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        for m in messages:
            if isinstance(m.content, list) and any(
                isinstance(p, AIToolCallPart) for p in m.content
            ):
                tool_calls = []
                content_text = ""
                for p in m.content:
                    if isinstance(p, AITextPart):
                        content_text = p.text
                    elif isinstance(p, AIThinkingPart):
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
                for p in m.content:
                    if isinstance(p, AIToolResultPart):
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": p.tool_call_id,
                                "name": p.name,
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
        return kwargs

    # -- Structured streaming --------------------------------------------------

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events with ``<think>`` tag parsing.

        Text inside ``<think>...</think>`` is yielded as
        :class:`StreamThinkingDelta`; everything else as
        :class:`StreamTextDelta`.  Tool calls are accumulated from deltas
        and yielded as :class:`StreamToolCall`.
        """
        from roomkit.providers.openai.ai import _ThinkTagParser

        kwargs = self._build_kwargs(context)
        t0 = time.monotonic()
        first_token = True
        parser = _ThinkTagParser()

        # Accumulate tool call deltas across chunks
        tool_call_accum: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, int] = {}

        try:
            response = await self._client.chat.stream_async(**kwargs)
            async for event in response:
                data = event.data
                if not data.choices:
                    continue
                delta = data.choices[0].delta
                finish_reason = data.choices[0].finish_reason or finish_reason

                # Extract usage from the stream when available
                if data.usage:
                    usage = {
                        "prompt_tokens": data.usage.prompt_tokens,
                        "completion_tokens": data.usage.completion_tokens,
                    }

                # Accumulate streamed tool call deltas
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index if hasattr(tc_delta, "index") else 0
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

            # Flush any remaining buffered text
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

            yield StreamDone(
                finish_reason=finish_reason,
                usage=usage,
                metadata={"model": self._config.model},
            )

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
                tool_calls.append(
                    AIToolCall(id=event.id, name=event.name, arguments=event.arguments)
                )
            elif isinstance(event, StreamDone):
                done_event = event

        return AIResponse(
            content="".join(text_parts),
            thinking="".join(thinking_parts) if thinking_parts else None,
            finish_reason=done_event.finish_reason if done_event else None,
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
            status_code in (429, 500, 502, 503)
            if status_code
            else any(
                term in str(exc).lower() for term in ["rate", "limit", "429", "500", "502", "503"]
            )
        )
        return ProviderError(
            str(exc),
            retryable=retryable,
            provider="mistral",
            status_code=status_code,
        )

    def _record_ttfb(self, t0: float) -> None:
        """Record time-to-first-byte metric."""
        ttfb_ms = (time.monotonic() - t0) * 1000
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        telemetry.record_metric(
            "roomkit.llm.ttfb_ms",
            ttfb_ms,
            unit="ms",
            attributes={"provider": "mistral", "model": self._config.model},
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if hasattr(self._client, "close"):
            await self._client.close()
