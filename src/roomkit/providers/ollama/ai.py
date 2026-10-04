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

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from roomkit.providers.ai.base import (
    RETRYABLE_STATUS_CODES,
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
    ModelInfo,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    stream_call_of,
)
from roomkit.providers.ai.image_parts import image_part_base64
from roomkit.providers.ai.reasoning import nearest_level, thinking_switch
from roomkit.providers.ai.response_schema import (
    check_schema_answer,
    checked_stream,
    schema_for_generate,
)
from roomkit.providers.ai.tool_calls import CallIds, tool_arguments
from roomkit.providers.ai.tool_declaration import chat_tool_declarations
from roomkit.providers.ollama import sdk_patch
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.ollama.models import MODELS
from roomkit.providers.utils import _aclose_stream, http_timeout

# Bounded parallelism for the per-model ``/api/show`` fan-out in
# ``list_models``. Local Ollama serialises heavy work anyway; 8 keeps the
# model picker snappy without thundering the server.
_SHOW_CONCURRENCY = 8

# The ``think`` levels Ollama takes, on a model that takes any (gpt-oss).
_THINK_LEVELS = ("low", "medium", "high")


def _ollama_image_payload(part: AIImagePart, *, provider: str) -> str:
    """Reduce an image reference to what Ollama's SDK accepts.

    RoomKit carries images as ``data:<media_type>;base64,<data>`` URIs —
    the same convention the Anthropic and OpenAI providers consume.
    Ollama's ``Image`` type only accepts a raw base64 string or a file
    path; handed a full data URI it raises ``ValueError: Invalid image
    data, expected base64 string or path to image file``. A data URI is
    read by the reader every provider shares — payload validated, a
    malformed one refused before the request leaves — and handed over as
    canonical base64; a plain base64 string or path passes through.
    """
    if part.url.startswith("data:"):
        return image_part_base64(part, provider=provider)[1]
    return part.url


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
        # The module the client comes from, for the SDK patch (sdk_patch.py).
        self._sdk = _ollama
        self._client = _ollama.AsyncClient(
            host=config.host,
            timeout=http_timeout(config),
            headers=self._build_headers(config),
        )

    @staticmethod
    def _build_headers(config: OllamaConfig) -> dict[str, str] | None:
        """Merge configured headers with a Bearer token from ``api_key``.

        Returns ``None`` when nothing is configured so the SDK keeps its
        default behavior — including its own fallback to the
        ``OLLAMA_API_KEY`` environment variable. A configured ``api_key``
        wins over an ``Authorization`` entry supplied via ``headers``.
        """
        headers: dict[str, str] = dict(config.headers or {})
        if config.api_key is not None:
            headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
        return headers or None

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

    @property
    def supports_response_schema(self) -> bool:
        """Structured outputs, through ``format`` holding the schema."""
        return True

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline snapshot of popular public Ollama models."""
        return list(MODELS)

    async def list_models(self) -> list[ModelInfo]:
        """List models installed on the configured Ollama server.

        Reads ``/api/tags`` for the installed set, then probes ``/api/show``
        per model (bounded parallel fan-out) to attach ``capabilities`` —
        the tags the picker uses to keep completion-only models out of an
        embeddings channel and vice-versa. A per-model probe failure yields
        no capabilities for that model (older servers don't ship the field);
        consumers treat empty as "unknown, allow everywhere".
        """
        resp = await self._client.list()
        installed = self._get_attr(resp, "models", None) or []
        names = [name for m in installed if (name := self._get_attr(m, "model", None))]

        sem = asyncio.Semaphore(_SHOW_CONCURRENCY)

        async def _capabilities(name: str) -> list[str]:
            async with sem:
                try:
                    shown = await self._client.show(model=name)
                except Exception:  # noqa: BLE001 — any probe failure → unknown caps
                    return []
            caps = self._get_attr(shown, "capabilities", None) or []
            return [str(c) for c in caps]

        caps_per_model = await asyncio.gather(*(_capabilities(n) for n in names))
        live = [
            ModelInfo(id=name, capabilities=caps)
            for name, caps in zip(names, caps_per_model, strict=True)
        ]
        return self._listing(live)

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
            # so the model sees them paired with their calls in order. An
            # image result (screenshot tool) can't ride on the tool message:
            # Ollama honors the ``images`` field on user messages, not tool
            # ones, so the image is split onto a synthetic user message right
            # after the tool result(s) — the tool message stays text-only and
            # the call/result pairing valid. Text results are unchanged.
            tool_results = [p for p in m.content if isinstance(p, AIToolResultPart)]
            if tool_results:
                pending_images: list[str] = []
                for r in tool_results:
                    text, images = r.split_for_message()
                    result.append(
                        {
                            "role": "tool",
                            "content": text,
                            "tool_name": r.name,
                        }
                    )
                    pending_images.extend(
                        _ollama_image_payload(img, provider=self._provider_name) for img in images
                    )
                if pending_images:
                    result.append({"role": "user", "content": "", "images": pending_images})
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
                    images.append(_ollama_image_payload(part, provider=self._provider_name))
                elif isinstance(part, AIThinkingPart):
                    thinking_text += part.thinking
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
        return chat_tool_declarations(tools)

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
        if self._config.top_p is not None:
            options["top_p"] = self._config.top_p
        if self._config.top_k is not None:
            options["top_k"] = self._config.top_k
        if self._config.min_p is not None:
            options["min_p"] = self._config.min_p
        return options

    def _resolve_think(self, context: AIContext) -> bool | str | None:
        """Decide the ``think`` value for this request (RFC §6.7).

        The turn states whether the model thinks (:func:`thinking_switch`)
        and how much (``reasoning_effort``); the configured ``think``
        supplies what it leaves unstated, and ``None`` lets the model decide.
        A level goes only to a model configured with one: Ollama refuses a
        level on a model that takes none (``think value "low" is not
        supported for this model``, qwen3 on Ollama 0.17), and a configured
        level is the one sign the model takes levels. So with
        ``think="high"``, ``thinking_budget=4096`` sends ``"high"`` and
        ``reasoning_effort="low"`` sends ``"low"``.
        """
        configured = self._config.think
        levels = isinstance(configured, str)
        switch = thinking_switch(context, True if levels else configured)
        if switch is False:
            return False
        if levels:
            return nearest_level(context.reasoning_effort, _THINK_LEVELS) or configured
        return switch

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
        if context.response_schema is not None:
            kwargs["format"] = context.response_schema
        return kwargs

    # -- Error mapping ------------------------------------------------------

    def _wrap_error(self, exc: BaseException) -> ProviderError:
        if isinstance(exc, self._response_error):
            status = getattr(exc, "status_code", None)
            # No HTTP status (ollama reports -1) means the server aborted
            # mid-stream — e.g. its chat template failed to parse the
            # model's own tool-call output ("XML syntax error ... closed by
            # </function>"). That's a transient generation defect: a retry
            # regenerates with fresh sampling. Only definite HTTP client
            # errors stay non-retryable.
            retryable = status in (None, -1) or status in RETRYABLE_STATUS_CODES
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
        schema_for_generate(
            context,
            supported=self.supports_response_schema,
            with_tools=self.supports_response_schema_with_tools,
            provider=self._provider_name,
        )
        kwargs = self._build_kwargs(context, stream=False)
        t0 = time.monotonic()
        try:
            response = await sdk_patch.chat(self._sdk, self._client, **kwargs)
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
        if context.response_schema is not None and not tool_calls:
            check_schema_answer(
                content,
                schema=context.response_schema,
                provider=self._provider_name,
                finish_reason=finish_reason,
            )

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
        parsing, no field reordering. A response schema is checked before the
        done event (RFC §6.7).
        """
        stream = checked_stream(
            self._stream_events(context),
            context,
            provider=self._provider_name,
            refusal=lambda _done: None,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await _aclose_stream(stream)

    async def _stream_events(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """The streamed call itself."""
        schema_for_generate(
            context,
            supported=self.supports_response_schema,
            with_tools=self.supports_response_schema_with_tools,
            provider=self._provider_name,
        )
        kwargs = self._build_kwargs(context, stream=True)
        t0 = time.monotonic()
        first_token = True
        finish_reason: str | None = None
        usage: dict[str, int] = {}
        accumulated_tool_calls: list[StreamToolCall] = []
        ids = CallIds()
        model: str | None = None

        try:
            stream = await sdk_patch.chat(self._sdk, self._client, **kwargs)
            async for chunk in stream:
                model = self._get_attr(chunk, "model", None) or model
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
                accumulated_tool_calls.extend(
                    stream_call_of(tc) for tc in self._extract_tool_calls(message, ids)
                )

                done = self._get_attr(chunk, "done", False)
                if done:
                    finish_reason = self._get_attr(chunk, "done_reason", None)
                    usage = self._extract_usage(chunk)

            for tc_event in accumulated_tool_calls:
                yield tc_event

            metadata = {"model": model} if model else {}
            yield StreamDone(finish_reason=finish_reason, usage=usage, metadata=metadata)
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

    def _extract_tool_calls(self, message: Any, ids: CallIds | None = None) -> list[AIToolCall]:
        """The calls a message carries. Ollama doesn't issue stable tool-call
        ids: one is minted, unique across turns, for the consumer to pair calls
        with results by; it is never echoed back to Ollama (its API matches
        tool results by role+name, not by id). A stream hands every chunk the
        response's one *ids*, so no two calls of the response share an id."""
        raw_calls = self._get_attr(message, "tool_calls", None) or []
        ids = ids if ids is not None else CallIds()
        result: list[AIToolCall] = []
        for tc in raw_calls:
            func = self._get_attr(tc, "function", None)
            if not func:
                continue
            # A call whose name was lost keeps an empty one, for the loop to
            # refuse, as on every wire.
            name = str(self._get_attr(func, "name", None) or "")
            result.append(
                AIToolCall(
                    id=ids(self._get_attr(tc, "id", None), name),
                    name=name,
                    arguments=tool_arguments(self._get_attr(func, "arguments", None)),
                )
            )
        return result

    async def close(self) -> None:
        """Release the underlying httpx client."""
        # ollama.AsyncClient owns an httpx.AsyncClient; close it if the
        # SDK version exposes it. Older versions are leak-tolerant.
        underlying = getattr(self._client, "_client", None)
        if underlying is not None and hasattr(underlying, "aclose"):
            await underlying.aclose()
