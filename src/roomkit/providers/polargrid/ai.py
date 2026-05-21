"""PolarGrid AI provider — generates responses via PolarGrid chat completions.

PolarGrid serves OpenAI-shaped chat completions from Canadian-hosted
edges (Toronto / Vancouver / Montreal). Tool / function calling is
**not** exposed by the chat-completions endpoint at the time of
writing — ``context.tools`` is dropped with a warning so the caller
notices the degradation instead of getting silently text-only output
from a provider it expected to support tools.
"""

from __future__ import annotations

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
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.providers.polargrid.config import PolarGridConfig

logger = logging.getLogger("roomkit.providers.polargrid")


def _patch_pg_metadata_decoder(pg_module: Any) -> None:
    """Make ``PolarGrid._convert_pg_metadata`` tolerant of missing fields.

    polargrid-sdk 0.7.0 unconditionally indexes ``raw["latency_ms"]``,
    which crashes non-streaming chat completions whenever the edge
    omits that field (observed on the Toronto edge with
    ``qwen-3.5-27b``). Streaming is unaffected. We swap in a lenient
    version that uses ``.get()`` so the response can still deserialize
    — the pg_metadata block is informational and we don't surface it
    upstream. Idempotent via a sentinel so reloading providers is safe;
    remove when the SDK ships its own fix.
    """
    client_cls = getattr(pg_module, "PolarGrid", None)
    if client_cls is None or getattr(client_cls, "_roomkit_metadata_patched", False):
        return

    @staticmethod  # type: ignore[misc]
    def _lenient(raw: dict[str, Any] | None) -> Any:
        if not raw:
            return None
        from polargrid.types import PGMetadata

        return PGMetadata(
            region=raw.get("region", ""),
            latency_ms=raw.get("latency_ms", 0),
            model_load_time_ms=raw.get("model_load_time_ms"),
            queue_time_ms=raw.get("queue_time_ms"),
            inference_time_ms=raw.get("inference_time_ms"),
        )

    client_cls._convert_pg_metadata = _lenient
    client_cls._roomkit_metadata_patched = True


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
        _patch_pg_metadata_decoder(_pg)
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
        # Structured streaming here means "emits StreamEvent objects"
        # — text deltas + done. Tool-call events would require
        # endpoint support PolarGrid doesn't ship yet.
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

    # -- Message conversion -------------------------------------------------

    def _format_content(
        self,
        content: (
            str
            | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart]
        ),
    ) -> str:
        """Flatten message content to plain text.

        PolarGrid's chat endpoint only accepts string content — no
        image parts, no tool parts. Images, tool calls, and tool
        results are dropped (the channel layer shouldn't be sending
        them anyway since ``supports_vision`` is False and tools are
        rejected upstream).
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
                # PolarGrid has no dedicated thinking channel. The
                # model sees its own prior reasoning prefixed for
                # context, which is better than dropping it silently.
                parts.append(f"[thinking]\n{part.thinking}\n[/thinking]")
            else:
                dropped.append(type(part).__name__)
        if dropped:
            logger.debug("Dropped unsupported content parts: %s", ", ".join(dropped))
        return "".join(parts)

    def _build_messages(
        self,
        messages: list[AIMessage],
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        for m in messages:
            text = self._format_content(m.content)
            # Skip messages that flattened to nothing — sending an
            # empty user/assistant turn confuses the model.
            if not text:
                continue
            result.append({"role": m.role, "content": text})
        return result

    def _build_request(self, context: AIContext, *, stream: bool) -> dict[str, Any]:
        if context.tools:
            logger.warning(
                "PolarGrid does not support tool/function calling; "
                "%d tool(s) will be ignored.",
                len(context.tools),
            )

        req: dict[str, Any] = {
            "model": self._config.model,
            "messages": self._build_messages(context.messages, context.system_prompt),
            "stream": stream,
        }
        max_tokens = context.max_tokens or self._config.max_tokens
        if max_tokens is not None:
            req["max_tokens"] = max_tokens
        if context.temperature is not None:
            req["temperature"] = context.temperature
        if self._config.top_p is not None:
            req["top_p"] = self._config.top_p
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
        content = getattr(message, "content", "") or ""
        finish_reason = getattr(choice, "finish_reason", None)
        usage = self._extract_usage(response)
        model = getattr(response, "model", self._config.model)

        return AIResponse(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            metadata={"model": model},
        )

    # -- Streaming ----------------------------------------------------------

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        client = await self._ensure_client()
        request = self._build_request(context, stream=True)

        t0 = time.monotonic()
        first_token = True
        finish_reason: str | None = None
        usage: dict[str, int] = {}

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
                text = getattr(delta, "content", None) if delta else None
                if text:
                    if first_token:
                        self._record_ttfb(t0)
                        first_token = False
                    yield StreamTextDelta(text=text)

            yield StreamDone(finish_reason=finish_reason, usage=usage)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    # -- Helpers ------------------------------------------------------------

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
