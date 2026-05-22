"""Ollama AI provider — generates responses via the native Ollama API.

This is *not* an OpenAI-compatible shim. It calls Ollama's native
``/api/chat`` endpoint through the official ``ollama-python`` SDK so
the full feature surface is available — most importantly, the
``think`` parameter and the dedicated ``thinking`` field on streamed
chunks. The OpenAI-compatible endpoint Ollama also exposes silently
ignores ``think`` and folds thinking into the response in a way that
loses the streamed-deltas property; use this provider whenever you
want real-time thinking or explicit control over the reasoning phase.
"""

from __future__ import annotations

import time
import uuid
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
    AITool,
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
from roomkit.providers.ollama.config import OllamaConfig


class OllamaAIProvider(AIProvider):
    """AI provider using Ollama's native API via the ollama-python SDK."""

    def __init__(self, config: OllamaConfig) -> None:
        try:
            import ollama as _ollama
        except ImportError as exc:
            raise ImportError(
                "ollama is required for OllamaAIProvider. "
                "Install it with: pip install roomkit[ollama]"
            ) from exc
        self._config = config
        self._response_error = _ollama.ResponseError
        self._client = _ollama.AsyncClient(host=config.host, timeout=config.timeout)

    @property
    def _provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_vision(self) -> bool:
        # Vision support is per-model on Ollama (llava, llama3.2-vision,
        # qwen2.5-vl, etc.). The provider passes images through whenever
        # they arrive; the server rejects unsupported models with a
        # ResponseError. Returning True here keeps the routing layer
        # honest — image content reaches the wire instead of being
        # silently filtered out one layer above.
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    # -- Message + tool conversion ------------------------------------------

    def _build_messages(
        self,
        messages: list[AIMessage],
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        """Convert RoomKit messages to Ollama's native chat format.

        Ollama messages carry ``role``, ``content``, optionally
        ``thinking`` (assistant), ``tool_calls`` (assistant), and
        ``images`` (any role). Tool results go on a separate message
        with ``role="tool"`` and ``tool_name`` so the model can match
        them to the originating call.
        """
        result: list[dict[str, Any]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for m in messages:
            if isinstance(m.content, str):
                result.append({"role": m.role, "content": m.content})
                continue

            # Tool results split into their own message(s) — Ollama uses
            # role="tool" with a tool_name field. One result per message
            # so the model sees them paired with their calls in order.
            tool_results = [p for p in m.content if isinstance(p, AIToolResultPart)]
            if tool_results:
                for r in tool_results:
                    result.append(
                        {
                            "role": "tool",
                            "content": r.result,
                            "tool_name": r.name,
                        }
                    )
                continue

            # Anything else is reassembled into a single message.
            text_parts: list[str] = []
            images: list[str] = []
            thinking_text = ""
            tool_calls: list[dict[str, Any]] = []
            for part in m.content:
                if isinstance(part, AITextPart):
                    text_parts.append(part.text)
                elif isinstance(part, AIImagePart):
                    images.append(part.url)
                elif isinstance(part, AIThinkingPart):
                    thinking_text = part.thinking
                elif isinstance(part, AIToolCallPart):
                    tool_calls.append(
                        {
                            "function": {
                                "name": part.name,
                                "arguments": part.arguments,
                            }
                        }
                    )
            msg: dict[str, Any] = {
                "role": m.role,
                "content": "".join(text_parts),
            }
            if thinking_text:
                msg["thinking"] = thinking_text
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if images:
                msg["images"] = images
            result.append(msg)
        return result

    def _build_tools(self, tools: list[AITool]) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _build_options(self, context: AIContext) -> dict[str, Any]:
        """Translate AIContext + config to Ollama's options dict."""
        options: dict[str, Any] = {}
        if context.temperature is not None:
            options["temperature"] = context.temperature
        num_predict = context.max_tokens or self._config.max_tokens
        if num_predict is not None:
            options["num_predict"] = num_predict
        if self._config.num_ctx is not None:
            options["num_ctx"] = self._config.num_ctx
        return options

    def _resolve_think(self, context: AIContext) -> bool | str | None:
        """Decide the ``think`` value for this request.

        Precedence:
        1. If ``context.thinking_budget`` is set, it gates on/off:
           ``None``/``0`` → ``think=False``; ``>0`` → honors the
           provider config's effort string when set (so per-channel
           ``thinking_budget=4096`` preserves a ``think="high"``
           default), otherwise ``think=True``.
        2. Otherwise pass the provider config's ``think`` through
           verbatim — boolean or effort string.
        3. ``None`` lets the model decide.
        """
        budget = context.thinking_budget
        if budget is not None:
            if budget <= 0:
                return False
            # >0 means "thinking on" — honor a configured effort level,
            # otherwise plain True.
            if isinstance(self._config.think, str):
                return self._config.think
            return True
        return self._config.think

    def _build_kwargs(self, context: AIContext, stream: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": self._build_messages(context.messages, context.system_prompt),
            "stream": stream,
        }
        tools = self._build_tools(context.tools)
        if tools:
            kwargs["tools"] = tools
        options = self._build_options(context)
        if options:
            kwargs["options"] = options
        think = self._resolve_think(context)
        if think is not None:
            kwargs["think"] = think
        if self._config.keep_alive is not None:
            kwargs["keep_alive"] = self._config.keep_alive
        return kwargs

    # -- Error mapping ------------------------------------------------------

    def _wrap_error(self, exc: BaseException) -> ProviderError:
        if isinstance(exc, self._response_error):
            status = getattr(exc, "status_code", None)
            retryable = status in (429, 500, 502, 503)
            return ProviderError(
                str(exc),
                retryable=retryable,
                provider=self._provider_name,
                status_code=status,
            )
        # Connection / timeout / other transport errors are typically
        # retryable — let the upper RetryPolicy decide whether to act.
        return ProviderError(
            str(exc),
            retryable=True,
            provider=self._provider_name,
        )

    # -- Non-streaming ------------------------------------------------------

    async def generate(self, context: AIContext) -> AIResponse:
        kwargs = self._build_kwargs(context, stream=False)
        t0 = time.monotonic()
        try:
            response = await self._client.chat(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # ResponseError or transport error
            raise self._wrap_error(exc) from exc

        self._record_ttfb(t0)

        message = self._get_message(response)
        content = self._get_attr(message, "content", "") or ""
        thinking = self._get_attr(message, "thinking", "") or ""
        finish_reason = self._get_attr(response, "done_reason", None)
        usage = self._extract_usage(response)
        tool_calls = self._extract_tool_calls(message)

        return AIResponse(
            content=content,
            thinking=thinking or None,
            finish_reason=finish_reason,
            usage=usage,
            metadata={"model": self._get_attr(response, "model", self._config.model)},
            tool_calls=tool_calls,
        )

    # -- Streaming ----------------------------------------------------------

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text deltas (thinking content filtered out)."""
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events with thinking streamed separately.

        Ollama's native streaming emits one chunk per token-ish, each
        with ``message.thinking`` and/or ``message.content`` deltas
        plus an optional final ``message.tool_calls``. We pass these
        straight through as the corresponding ``StreamThinkingDelta``,
        ``StreamTextDelta``, and ``StreamToolCall`` events — no tag
        parsing, no field reordering.
        """
        kwargs = self._build_kwargs(context, stream=True)
        t0 = time.monotonic()
        first_token = True
        finish_reason: str | None = None
        usage: dict[str, int] = {}
        accumulated_tool_calls: list[StreamToolCall] = []

        try:
            stream = await self._client.chat(**kwargs)
            async for chunk in stream:
                message = self._get_message(chunk)
                thinking_delta = self._get_attr(message, "thinking", None)
                if thinking_delta:
                    if first_token:
                        self._record_ttfb(t0)
                        first_token = False
                    yield StreamThinkingDelta(thinking=thinking_delta)

                text_delta = self._get_attr(message, "content", None)
                if text_delta:
                    if first_token:
                        self._record_ttfb(t0)
                        first_token = False
                    yield StreamTextDelta(text=text_delta)

                # Tool calls arrive whole (Ollama doesn't fragment
                # arguments across chunks the way OpenAI does). Collect
                # them but defer the yield until the run finishes so
                # the consumer sees text-then-tools in the natural order.
                for tc in self._extract_tool_calls(message):
                    accumulated_tool_calls.append(
                        StreamToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                    )

                done = self._get_attr(chunk, "done", False)
                if done:
                    finish_reason = self._get_attr(chunk, "done_reason", None)
                    usage = self._extract_usage(chunk)

            for tc_event in accumulated_tool_calls:
                yield tc_event

            yield StreamDone(finish_reason=finish_reason, usage=usage)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _get_attr(obj: Any, name: str, default: Any) -> Any:
        """Read ``name`` from a Pydantic model, dataclass, or dict."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _get_message(self, response_or_chunk: Any) -> Any:
        """Resolve the message object from a chat response or chunk."""
        return self._get_attr(response_or_chunk, "message", None)

    def _extract_usage(self, chunk_or_response: Any) -> dict[str, int]:
        prompt = self._get_attr(chunk_or_response, "prompt_eval_count", None)
        completion = self._get_attr(chunk_or_response, "eval_count", None)
        usage: dict[str, int] = {}
        if prompt is not None:
            usage["input_tokens"] = int(prompt)
        if completion is not None:
            usage["output_tokens"] = int(completion)
        return usage

    def _extract_tool_calls(self, message: Any) -> list[AIToolCall]:
        raw_calls = self._get_attr(message, "tool_calls", None) or []
        result: list[AIToolCall] = []
        for tc in raw_calls:
            func = self._get_attr(tc, "function", None)
            if not func:
                continue
            name = self._get_attr(func, "name", "")
            arguments = self._get_attr(func, "arguments", {}) or {}
            if not isinstance(arguments, dict):
                arguments = {"raw": arguments}
            # Ollama doesn't issue stable tool-call ids — synthesize one
            # the consumer can pair calls with results by. A per-response
            # counter (``len(result)``) resets each turn and would collide
            # across turns of the same conversation; downstream code that
            # dedups START/END events by tool_id then collapses N pairs
            # into one. Use a uuid4 suffix so every synthesized id is
            # globally unique. The id is never echoed back to Ollama
            # (its API matches tool results by role+name, not by id).
            call_id = self._get_attr(tc, "id", None) or f"call_{name}_{uuid.uuid4().hex[:12]}"
            result.append(AIToolCall(id=str(call_id), name=str(name), arguments=arguments))
        return result

    async def close(self) -> None:
        """Release the underlying httpx client."""
        # ollama.AsyncClient owns an httpx.AsyncClient; close it if the
        # SDK version exposes it. Older versions are leak-tolerant.
        underlying = getattr(self._client, "_client", None)
        if underlying is not None and hasattr(underlying, "aclose"):
            await underlying.aclose()
