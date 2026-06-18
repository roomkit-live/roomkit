"""PolarGrid AI provider — generates responses via PolarGrid chat completions.

PolarGrid serves OpenAI-shaped chat completions from Canadian-hosted
edges (Toronto / Vancouver / Montreal). As of polargrid-sdk 0.8.4 the
chat-completions endpoint supports tool / function calling: ``context.tools``
are forwarded, and tool calls come back both non-streaming
(``message.tool_calls``) and streaming (fragmented ``delta.tool_calls``,
OpenAI-style). Tool arguments cross the wire as a JSON string and are
parsed back into a dict for RoomKit.

``tool_choice`` is left unset so the backend defaults to ``auto`` — note
that forcing a specific tool is *steered*, not hard-guaranteed, on
PolarGrid's backend.

polargrid-sdk 0.8.5+ exposes an ``enable_thinking`` request flag
(``PolarGridConfig.thinking``). When on, the qwen models surface their
reasoning inline as ``<think>...</think>`` tags in the message content
(the same convention vLLM/Ollama reasoning models use). We parse those
tags out and surface them as ``AIResponse.thinking`` (non-streaming) and
``StreamThinkingDelta`` (streaming), leaving the answer text clean —
reusing the OpenAI provider's tag parser. Thinking responses are larger
and slower (the reasoning counts toward latency and ``max_tokens``).
"""

from __future__ import annotations

import json
import logging
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
from roomkit.providers.openai.ai import _extract_think_tags, _ThinkTagParser
from roomkit.providers.polargrid.config import PolarGridConfig

logger = logging.getLogger("roomkit.providers.polargrid")


class PolarGridAIProvider(AIProvider):
    """AI provider using PolarGrid's chat completions API."""

    def __init__(self, config: PolarGridConfig) -> None:
        try:
            import polargrid as _pg
        except ImportError as exc:
            raise ImportError(
                "polargrid-sdk is required for PolarGridAIProvider. "
                "Install it with: pip install roomkit[polargrid]"
            ) from exc
        self._config = config
        self._sdk = _pg
        # Bind exception classes once so we can catch them without
        # re-importing in hot paths and the test suite can swap the
        # module out via sys.modules patching.
        self._auth_error = _pg.AuthenticationError
        self._validation_error = _pg.ValidationError
        self._rate_limit_error = _pg.RateLimitError
        self._network_error = _pg.NetworkError
        self._timeout_error = _pg.TimeoutError
        self._not_found_error = _pg.NotFoundError
        self._server_error = _pg.ServerError
        self._client: Any | None = None

    @property
    def _provider_name(self) -> str:
        return "polargrid"

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_vision(self) -> bool:
        # PolarGrid's current model catalog (qwen-3.5-9b/27b, kokoro,
        # whisper-large-v3-turbo) doesn't expose vision on the chat
        # endpoint. Revisit when a multimodal model lands.
        return False

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        # Emits StreamEvent objects — text deltas, tool calls, and done.
        return True

    # -- Client lifecycle ---------------------------------------------------

    async def _ensure_client(self) -> Any:
        """Lazily create the PolarGrid async client.

        Done on first call instead of in ``__init__`` because the
        auto-routing variant (``region=None``) is itself async — it
        pings edges to pick the fastest. Pinning a region uses the
        synchronous constructor.
        """
        if self._client is not None:
            return self._client

        kwargs: dict[str, Any] = {
            "api_key": self._config.api_key.get_secret_value(),
            "timeout": self._config.timeout,
            "max_retries": self._config.max_retries,
        }
        if self._config.debug:
            kwargs["debug"] = True

        if self._config.region:
            # Region pinned — synchronous constructor is fine.
            self._client = self._sdk.PolarGrid(region=self._config.region, **kwargs)
        else:
            # Auto-routing — discovers the fastest edge.
            self._client = await self._sdk.PolarGrid.create(**kwargs)
        return self._client

    # -- Message + tool conversion ------------------------------------------

    def _format_content(
        self,
        content: (
            str
            | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart]
        ),
    ) -> str:
        """Flatten message content to plain text.

        PolarGrid's chat endpoint takes string content only — no image
        parts. Tool-call and tool-result parts are handled structurally
        in :meth:`_render_message` and skipped here; images are dropped
        (``supports_vision`` is False).
        """
        if isinstance(content, str):
            return content

        parts: list[str] = []
        dropped: list[str] = []
        for part in content:
            if isinstance(part, AITextPart):
                parts.append(part.text)
            elif isinstance(part, AIThinkingPart):
                # Round-trip thinking back to the model as plain text;
                # PolarGrid has no dedicated thinking channel. The model
                # sees its own prior reasoning prefixed for context,
                # which is better than dropping it silently.
                parts.append(f"[thinking]\n{part.thinking}\n[/thinking]")
            elif isinstance(part, (AIToolCallPart, AIToolResultPart)):
                # Rendered as structured tool messages, not inline text.
                continue
            else:
                dropped.append(type(part).__name__)
        if dropped:
            logger.debug("Dropped unsupported content parts: %s", ", ".join(dropped))
        return "".join(parts)

    def _render_message(self, m: AIMessage) -> list[dict[str, Any]]:
        """Render one RoomKit message into PolarGrid chat message(s).

        Tool calls become an assistant message carrying ``tool_calls``
        (OpenAI-shaped, ``arguments`` as a JSON string). Tool results
        become one ``role="tool"`` message each, paired back to their
        call via ``tool_call_id``. Everything else flattens to a single
        string-content message (empty ones are skipped).
        """
        if isinstance(m.content, str):
            return [{"role": m.role, "content": m.content}] if m.content else []

        parts = m.content
        tool_calls = [p for p in parts if isinstance(p, AIToolCallPart)]
        if tool_calls:
            text = self._format_content(parts)
            return [
                {
                    "role": m.role,
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": p.id,
                            "type": "function",
                            "function": {
                                "name": p.name,
                                "arguments": json.dumps(p.arguments),
                            },
                        }
                        for p in tool_calls
                    ],
                }
            ]

        tool_results = [p for p in parts if isinstance(p, AIToolResultPart)]
        if tool_results:
            return [
                {
                    "role": "tool",
                    "content": r.result,
                    "tool_call_id": r.tool_call_id,
                    "name": r.name,
                }
                for r in tool_results
            ]

        text = self._format_content(parts)
        return [{"role": m.role, "content": text}] if text else []

    def _build_messages(
        self,
        messages: list[AIMessage],
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        for m in messages:
            result.extend(self._render_message(m))
        return result

    def _build_tools(self, tools: list[AITool]) -> list[dict[str, Any]] | None:
        """Convert RoomKit tools to PolarGrid's OpenAI-shaped tool list."""
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

    def _build_request(self, context: AIContext, *, stream: bool) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": self._config.model,
            "messages": self._build_messages(context.messages, context.system_prompt),
            "stream": stream,
        }
        tools = self._build_tools(context.tools)
        if tools:
            # No tool_choice in AIContext — leave it unset so PolarGrid
            # defaults to "auto". Forcing a tool is steered, not hard
            # guaranteed, on their backend anyway.
            req["tools"] = tools
        if self._config.thinking is not None:
            # polargrid-sdk 0.8.5+ exposes a real thinking toggle; qwen
            # then emits its reasoning inline as <think>...</think>, which
            # the streaming/non-streaming paths split out as thinking.
            req["enable_thinking"] = self._config.thinking
        max_tokens = context.max_tokens or self._config.max_tokens
        if max_tokens is not None:
            req["max_tokens"] = max_tokens
        if context.temperature is not None:
            req["temperature"] = context.temperature
        if self._config.top_p is not None:
            req["top_p"] = self._config.top_p
        if logger.isEnabledFor(logging.DEBUG):
            # Full outgoing payload (no API key — that lives on the client),
            # so the /think | /no_think switch in the messages is visible.
            # Handy to share with PolarGrid when debugging behavior.
            logger.debug("PolarGrid request: %s", json.dumps(req, ensure_ascii=False))
        return req

    # -- Error mapping ------------------------------------------------------

    def _retryable_for(self, exc: BaseException) -> bool:
        """Map an SDK exception to its retryable flag via dispatch table.

        Unknown errors default to retryable so RoomKit's RetryPolicy
        decides whether to back off or surface immediately.
        """
        retry_map: tuple[tuple[type[BaseException], bool], ...] = (
            (self._auth_error, False),
            (self._validation_error, False),
            (self._not_found_error, False),
            (self._rate_limit_error, True),
            (self._network_error, True),
            (self._timeout_error, True),
            (self._server_error, True),
        )
        for exc_type, retryable in retry_map:
            if isinstance(exc, exc_type):
                return retryable
        return True

    def _wrap_error(self, exc: BaseException) -> ProviderError:
        return ProviderError(
            str(exc),
            retryable=self._retryable_for(exc),
            provider=self._provider_name,
            status_code=getattr(exc, "status_code", None),
        )

    # -- Non-streaming ------------------------------------------------------

    async def generate(self, context: AIContext) -> AIResponse:
        client = await self._ensure_client()
        request = self._build_request(context, stream=False)

        t0 = time.monotonic()
        try:
            response = await client.chat_completion(request)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        self._record_ttfb(t0)

        choices = getattr(response, "choices", None) or []
        if not choices:
            return AIResponse(content="")
        choice = choices[0]
        message = getattr(choice, "message", None)
        raw_content = getattr(message, "content", "") or ""
        # qwen surfaces reasoning inline as <think>...</think>; split it out
        # so the answer text is clean and the reasoning rides on .thinking.
        thinking, content = _extract_think_tags(raw_content)
        finish_reason = getattr(choice, "finish_reason", None)
        usage = self._extract_usage(response)
        model = getattr(response, "model", self._config.model)
        tool_calls = self._extract_tool_calls(message)

        return AIResponse(
            content=content,
            thinking=thinking,
            finish_reason=finish_reason,
            usage=usage,
            metadata={"model": model},
            tool_calls=tool_calls,
        )

    # -- Streaming ----------------------------------------------------------

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield thinking + text deltas, then any tool calls, then done.

        Content deltas pass through a ``<think>`` tag parser so qwen's
        inline reasoning is emitted as :class:`StreamThinkingDelta` and
        the rest as :class:`StreamTextDelta`. Tool calls arrive
        OpenAI-style: fragmented across chunks as ``delta.tool_calls``
        with a stable ``index``, the ``id`` on the first fragment, and
        ``arguments`` concatenated from each fragment's ``function``
        dict. We accumulate by index and emit one :class:`StreamToolCall`
        per call after the text, so the consumer sees
        thinking-then-text-then-tools in natural order.
        """
        client = await self._ensure_client()
        request = self._build_request(context, stream=True)

        t0 = time.monotonic()
        first_token = True
        finish_reason: str | None = None
        usage: dict[str, int] = {}
        tool_accum: dict[int, dict[str, str]] = {}
        parser = _ThinkTagParser()

        try:
            stream = client.chat_completion_stream(request)
            async for chunk in stream:
                # Final chunk may carry usage with empty/no choices.
                chunk_usage = self._extract_usage(chunk)
                if chunk_usage:
                    usage = chunk_usage

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                tool_deltas = getattr(delta, "tool_calls", None)
                if tool_deltas:
                    if first_token:
                        self._record_ttfb(t0)
                        first_token = False
                    self._accumulate_tool_deltas(tool_accum, tool_deltas)

                text = getattr(delta, "content", None)
                if text:
                    for kind, segment in parser.feed(text):
                        if first_token:
                            self._record_ttfb(t0)
                            first_token = False
                        if kind == "thinking":
                            yield StreamThinkingDelta(thinking=segment)
                        else:
                            yield StreamTextDelta(text=segment)

            # Flush any buffered text held back for a partial tag.
            for kind, segment in parser.flush():
                if kind == "thinking":
                    yield StreamThinkingDelta(thinking=segment)
                else:
                    yield StreamTextDelta(text=segment)

            for event in self._finalize_tool_calls(tool_accum):
                yield event

            yield StreamDone(finish_reason=finish_reason, usage=usage)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _parse_arguments(raw: Any) -> dict[str, Any]:
        """Coerce PolarGrid's JSON-string tool arguments into a dict.

        RoomKit's ``AIToolCall.arguments`` is a dict; PolarGrid sends a
        JSON string. Already-dict inputs pass through; malformed or
        non-object JSON is preserved under a ``raw`` key so nothing is
        silently lost.
        """
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"raw": raw}
        return parsed if isinstance(parsed, dict) else {"raw": raw}

    def _extract_tool_calls(self, message: Any) -> list[AIToolCall]:
        """Read non-streaming ``message.tool_calls`` into AIToolCalls."""
        raw_calls = getattr(message, "tool_calls", None) or []
        result: list[AIToolCall] = []
        for tc in raw_calls:
            func = getattr(tc, "function", None)
            if func is None:
                continue
            name = getattr(func, "name", "") or ""
            arguments = self._parse_arguments(getattr(func, "arguments", ""))
            call_id = getattr(tc, "id", None) or f"call_{name}"
            result.append(AIToolCall(id=str(call_id), name=str(name), arguments=arguments))
        return result

    @staticmethod
    def _accumulate_tool_deltas(
        accum: dict[int, dict[str, str]],
        deltas: list[Any],
    ) -> None:
        """Fold streamed ``ToolCallDelta`` fragments into per-index slots."""
        for d in deltas:
            idx = getattr(d, "index", 0) or 0
            slot = accum.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            d_id = getattr(d, "id", None)
            if d_id:
                slot["id"] = d_id
            func = getattr(d, "function", None)
            if not isinstance(func, dict):
                continue
            name = func.get("name")
            if name:
                slot["name"] = name
            args = func.get("arguments")
            if args:
                slot["arguments"] += args

    def _finalize_tool_calls(self, accum: dict[int, dict[str, str]]) -> list[StreamToolCall]:
        """Turn accumulated tool-call slots into StreamToolCall events."""
        events: list[StreamToolCall] = []
        for idx in sorted(accum):
            slot = accum[idx]
            events.append(
                StreamToolCall(
                    id=slot["id"],
                    name=slot["name"],
                    arguments=self._parse_arguments(slot["arguments"]),
                )
            )
        return events

    @staticmethod
    def _extract_usage(obj: Any) -> dict[str, int]:
        usage_obj = getattr(obj, "usage", None)
        if not usage_obj:
            return {}
        prompt = getattr(usage_obj, "prompt_tokens", None)
        completion = getattr(usage_obj, "completion_tokens", None)
        result: dict[str, int] = {}
        if prompt is not None:
            result["input_tokens"] = int(prompt)
        if completion is not None:
            result["output_tokens"] = int(completion)
        return result

    async def close(self) -> None:
        """Release the underlying client if it exposes a close hook."""
        if self._client is None:
            return
        closer = getattr(self._client, "close", None) or getattr(self._client, "aclose", None)
        if closer is None:
            return
        result = closer()
        # SDK may expose sync or async close — handle both.
        if hasattr(result, "__await__"):
            await result
