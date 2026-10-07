"""Anthropic AI provider — generates responses via the Anthropic Messages API.

The provider owns the clients (the configured one and the per-request pool),
the call and the stream. Turning RoomKit messages and context into the
request lives in ``request.py``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from roomkit.providers.ai.base import (
    AIContext,
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
    provider_error,
    request_api_key,
    tool_call_of,
)
from roomkit.providers.ai.response_schema import checked_stream, schema_for_generate
from roomkit.providers.ai.thinking_blocks import ThinkingBlocks
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.anthropic.models import MODELS
from roomkit.providers.anthropic.request import build_kwargs
from roomkit.providers.anthropic.stream_events import done_event, is_first_output, stream_events
from roomkit.providers.anthropic.tool_blocks import ToolUseBlocks
from roomkit.providers.utils import _aclose_stream
from roomkit.providers.vendor_endpoint import ANTHROPIC_BASE_URL, is_vendor_endpoint

logger = logging.getLogger("roomkit.providers.anthropic.ai")

# Claude models that support vision (Claude 3 and later)
# Fallback only, for ids the catalog does not carry — a snapshot newer than
# this release, or a proxy exposing its own name. Every Claude model since
# Claude 3 accepts image input, so the family prefix is the honest default;
# a model that is genuinely in the catalog is answered from there instead.
_VISION_PREFIXES = ("claude-",)


def _refusal(done: StreamDone) -> str | None:
    return "refusal" if done.finish_reason == "refusal" else None


# How many per-request clients to keep alive alongside the configured one.
# Each holds an HTTP connection pool, so this is a memory/latency trade, not a
# correctness one: an evicted key simply builds a fresh client next turn. Sized
# for the realistic case — a handful of members in one room bringing their own
# credential — rather than for a whole tenant, which would pin a pool per person.
_MAX_PER_REQUEST_CLIENTS = 8


class AnthropicAIProvider(AIProvider):
    """AI provider using the Anthropic Messages API."""

    def __init__(self, config: AnthropicConfig) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for AnthropicAIProvider. "
                "Install it with: pip install roomkit[anthropic]"
            ) from exc
        self._config = config
        self._api_status_error = _anthropic.APIStatusError
        self._api_connection_error = _anthropic.APIConnectionError
        self._anthropic = _anthropic
        self._client = self._build_client(config.api_key.get_secret_value())
        # Clients for credentials supplied per request (see ``request_api_key``),
        # keyed by the credential itself. Insertion-ordered so the oldest is the
        # one evicted; never holds the configured key, which ``self._client``
        # already owns.
        self._per_request_clients: dict[str, Any] = {}
        # A cached client may serve several concurrent turns. Eviction can only
        # close clients whose last turn has released its lease; otherwise a
        # ninth credential could tear down an older stream mid-response.
        self._per_request_client_users: dict[str, int] = {}

    def _build_client(self, api_key: str) -> Any:
        """Build an Anthropic client for ``api_key`` with this provider's config."""
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            # The SDK's own ``Timeout``, with the connect/read split of
            # ``http_timeout_from``: anthropic 1.x runs on httpx2 and refuses
            # any object from the ``httpx`` package. Its check reads the class's
            # ``__module__``, which the openai SDK rewrites on import, so an
            # ``httpx.Timeout`` passes it only in a process that imported
            # openai first. This one needs no ``httpx`` installed at all.
            "timeout": self._anthropic.Timeout(
                self._config.timeout, connect=self._config.connect_timeout
            ),
            "max_retries": self._config.max_retries,
        }
        if self._config.base_url:
            client_kwargs["base_url"] = self._config.base_url
        if self._config.extra_headers:
            client_kwargs["default_headers"] = self._config.extra_headers
        return self._anthropic.AsyncAnthropic(**client_kwargs)

    async def _client_for(self, context: AIContext) -> tuple[Any, str | None]:
        """Lease the client this turn must use and return its cache key.

        The provider object is shared by every conversation it serves, so a turn
        carrying its own credential cannot be served by the shared client. Clients
        are cached per credential because building one per turn would throw away
        the connection pool on every message. The oldest idle entry is evicted
        past ``_MAX_PER_REQUEST_CLIENTS``; the cache may exceed that soft bound
        while every entry is serving an active turn.
        """
        api_key = request_api_key(context)
        if api_key is None or api_key == self._config.api_key.get_secret_value():
            return self._client, None

        cached = self._per_request_clients.get(api_key)
        if cached is None:
            cached = self._build_client(api_key)
            self._per_request_clients[api_key] = cached
        self._per_request_client_users[api_key] = (
            self._per_request_client_users.get(api_key, 0) + 1
        )

        await self._close_evicted(self._take_idle_evictions())
        return cached, api_key

    def _take_idle_evictions(self) -> list[Any]:
        """Remove oldest idle clients until the cache reaches its soft bound."""
        evicted: list[Any] = []
        while len(self._per_request_clients) > _MAX_PER_REQUEST_CLIENTS:
            idle_key = next(
                (
                    key
                    for key in self._per_request_clients
                    if self._per_request_client_users.get(key, 0) == 0
                ),
                None,
            )
            if idle_key is None:
                # Every client is in use. Temporarily exceeding the bound is
                # safer than terminating a live HTTP stream; release trims it.
                break
            evicted.append(self._per_request_clients.pop(idle_key))
            self._per_request_client_users.pop(idle_key, None)
        return evicted

    async def _release_client(self, api_key: str | None) -> None:
        """Release one per-request client lease and trim any temporary overflow."""
        if api_key is None or api_key not in self._per_request_client_users:
            # ``close()`` may have cleared the cache while a turn was winding
            # down. Do not recreate a dangling lease-counter entry afterwards.
            return
        users = self._per_request_client_users.get(api_key, 0)
        if users <= 1:
            self._per_request_client_users[api_key] = 0
        else:
            self._per_request_client_users[api_key] = users - 1
        await self._close_evicted(self._take_idle_evictions())

    @staticmethod
    async def _close_evicted(clients: list[Any]) -> None:
        """Close idle cache entries without breaking an unrelated live turn."""
        for client in clients:
            try:
                await client.close()
            except Exception:
                # The entry is already out of the cache; a stale pool failing
                # to close must not mask the response using another credential.
                logger.exception("Failed to close an evicted Anthropic client")

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    @property
    def supports_deferred_tools(self) -> bool:
        """Read from the catalogue: a model it does not carry is assumed not to.

        Never behind another ``base_url``: a proxy or gateway gets the request
        shape it always got, as the configuration's other shape defaults leave
        it.
        """
        if not is_vendor_endpoint(self._config.base_url, ANTHROPIC_BASE_URL):
            return False
        entry = self.catalog_entry()
        return entry is not None and "deferred_tools" in entry.capabilities

    @property
    def supports_vision(self) -> bool:
        """Whether the configured Claude model accepts image input.

        Read from the offline catalog, which states it per model, rather than
        from a prefix table that has to be remembered on every release — the
        prefix form silently reported the whole 4.5-and-later lineup as
        text-only, dropping images before they reached the wire.
        """
        entry = self.catalog_entry()
        if entry is not None and entry.supports_vision is not None:
            return entry.supports_vision
        return self._config.model.startswith(_VISION_PREFIXES)

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Claude models."""
        return list(MODELS)

    async def list_models(self) -> list[ModelInfo]:
        """List models the Anthropic API currently exposes for this key."""
        page = await self._client.models.list(limit=1000)
        live = [
            ModelInfo(id=m.id, display_name=getattr(m, "display_name", None)) for m in page.data
        ]
        return self._listing(live)

    @property
    def supports_response_schema(self) -> bool:
        """Structured outputs, through ``output_config.format``.

        A model older than the feature answers 400, surfaced as a
        :class:`ProviderError`.
        """
        return True

    @property
    def supports_response_schema_with_tools(self) -> bool:
        """``output_config.format`` and tools share a request: the model calls
        tools or answers in the schema."""
        return True

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events from the Anthropic Messages streaming API.

        When extended thinking is enabled, yields ``StreamThinkingDelta`` events
        before text deltas. A response schema is checked before the done event
        (RFC §6.7).
        """
        schema_for_generate(
            context,
            supported=self.supports_response_schema,
            with_tools=self.supports_response_schema_with_tools,
            provider="anthropic",
        )
        stream = checked_stream(
            self._events(context),
            context,
            provider="anthropic",
            refusal=_refusal,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await _aclose_stream(stream)

    async def _events(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """The streamed call itself, shared by :meth:`generate`."""
        kwargs = build_kwargs(self._config, context)
        # The one place the turn's credential is chosen. ``generate`` and
        # ``generate_stream`` both consume this stream, so they inherit it.
        client, leased_api_key = await self._client_for(context)
        first_token = FirstToken(self)

        try:
            blocks = ToolUseBlocks()

            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    for out in stream_events(event, blocks):
                        if is_first_output(out):
                            first_token.seen()
                        yield out
                final = await stream.get_final_message()

            for call in blocks.remaining(final):
                yield call
            yield done_event(final, self._config.model)
        except Exception as exc:
            raise self._wrap_error(exc) from exc
        finally:
            await self._release_client(leased_api_key)

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """The provider error an SDK failure reads as (:func:`provider_error`):
        an ``event: error`` of a 200 stream reads as the status its type
        names (``overloaded_error`` as the 529 it stands for), and the SDK's
        connection error is a lost connection."""
        return provider_error(
            exc, provider="anthropic", transport=isinstance(exc, self._api_connection_error)
        )

    async def generate(self, context: AIContext) -> AIResponse:
        """Generate by consuming the structured stream."""
        thinking = ThinkingBlocks()
        text_parts: list[str] = []
        tool_calls: list[AIToolCall] = []
        done_event: StreamDone | None = None

        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamThinkingDelta):
                thinking.add(event)
            elif isinstance(event, StreamTextDelta):
                text_parts.append(event.text)
            elif isinstance(event, StreamToolCall):
                tool_calls.append(tool_call_of(event))
            elif isinstance(event, StreamDone):
                done_event = event

        finish_reason = done_event.finish_reason if done_event else None
        return AIResponse(
            content="".join(text_parts),
            thinking=thinking.text if thinking.seen else None,
            thinking_signature=thinking.last_signature,
            thinking_parts=thinking.parts() if thinking.keyed else None,
            finish_reason=finish_reason,
            usage=done_event.usage if done_event else {},
            metadata=done_event.metadata if done_event else {},
            tool_calls=tool_calls,
        )

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text deltas as they arrive from the Anthropic Messages API."""
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text

    async def close(self) -> None:
        """Close the configured client and every per-request one still cached."""
        clients = [self._client, *self._per_request_clients.values()]
        self._per_request_clients.clear()
        self._per_request_client_users.clear()
        failures: list[Exception] = []
        for client in clients:
            try:
                await client.close()
            except Exception as exc:
                failures.append(exc)
                logger.exception("Failed to close an Anthropic client")
        if failures:
            raise ExceptionGroup(
                f"closing Anthropic clients failed for {len(failures)} client(s)", failures
            )
