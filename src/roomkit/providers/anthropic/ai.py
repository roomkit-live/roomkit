"""Anthropic AI provider — generates responses via the Anthropic Messages API.

The provider owns the clients (the configured one and the per-request pool),
the call and the stream. Turning RoomKit messages and context into the
request lives in ``request.py``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from roomkit.providers.ai.base import (
    RETRYABLE_STATUS_CODES,
    AIContext,
    AIProvider,
    AIResponse,
    AIToolCall,
    ModelInfo,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    StreamToolCallDelta,
    request_api_key,
)
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.anthropic.models import MODELS
from roomkit.providers.anthropic.request import build_kwargs
from roomkit.providers.utils import http_timeout

logger = logging.getLogger("roomkit.providers.anthropic.ai")

# Claude models that support vision (Claude 3 and later)
# Fallback only, for ids the catalog does not carry — a snapshot newer than
# this release, or a proxy exposing its own name. Every Claude model since
# Claude 3 accepts image input, so the family prefix is the honest default;
# a model that is genuinely in the catalog is answered from there instead.
_VISION_PREFIXES = ("claude-",)

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
            "timeout": http_timeout(self._config),
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
        return self._merge_curated(live)

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events from the Anthropic Messages streaming API.

        When extended thinking is enabled, yields ``StreamThinkingDelta`` events
        before text deltas.
        """
        kwargs = build_kwargs(self._config, context)
        # The one place the turn's credential is chosen. ``generate`` and
        # ``generate_stream`` both consume this stream, so they inherit it.
        client, leased_api_key = await self._client_for(context)
        t0 = time.monotonic()
        first_token = True

        try:
            # Track in-progress tool_use blocks for real-time streaming
            _tool_blocks: dict[int, dict[str, Any]] = {}  # index → {id, name, input_json}
            _yielded_tool_ids: set[str] = set()

            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if not hasattr(event, "type"):
                        continue

                    if event.type == "content_block_start":
                        cb = event.content_block
                        if hasattr(cb, "type") and cb.type == "tool_use":
                            _tool_blocks[event.index] = {
                                "id": cb.id,
                                "name": cb.name,
                                "input_json": "",
                            }
                            # The name is known here, before a single argument
                            # byte: emit it at once so a host can say what is
                            # being composed for the whole composition.
                            yield StreamToolCallDelta(
                                id=cb.id,
                                name=cb.name,
                                index=event.index,
                                arguments_delta="",
                            )

                    elif event.type == "content_block_delta" and hasattr(event.delta, "type"):
                        delta = event.delta
                        if delta.type == "thinking_delta":
                            if first_token:
                                self._record_ttfb(t0)
                                first_token = False
                            yield StreamThinkingDelta(thinking=delta.thinking)
                        elif delta.type == "signature_delta":
                            # The thinking block's opaque signature arrives as
                            # its own delta after the text. Surface it so the
                            # block can be echoed back in history (Anthropic
                            # 400s on a thinking block missing its signature).
                            yield StreamThinkingDelta(
                                thinking="",
                                signature=delta.signature,
                            )
                        elif delta.type == "text_delta":
                            if first_token:
                                self._record_ttfb(t0)
                                first_token = False
                            yield StreamTextDelta(text=delta.text)
                        elif delta.type == "input_json_delta":
                            idx = event.index
                            if idx in _tool_blocks:
                                tb = _tool_blocks[idx]
                                tb["input_json"] += delta.partial_json
                                yield StreamToolCallDelta(
                                    id=tb["id"],
                                    name=tb["name"],
                                    index=idx,
                                    arguments_delta=delta.partial_json,
                                )

                    elif event.type == "content_block_stop":
                        idx = event.index
                        if idx in _tool_blocks:
                            tb = _tool_blocks.pop(idx)
                            if tb["id"] not in _yielded_tool_ids:
                                _yielded_tool_ids.add(tb["id"])
                                try:
                                    args = json.loads(tb["input_json"]) if tb["input_json"] else {}
                                except json.JSONDecodeError:
                                    args = {}
                                yield StreamToolCall(
                                    id=tb["id"],
                                    name=tb["name"],
                                    arguments=args,
                                )

                final = await stream.get_final_message()

            # Yield any tool calls from final message not already yielded
            for block in final.content:
                if block.type == "tool_use" and block.id not in _yielded_tool_ids:
                    yield StreamToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )

            usage: dict[str, int] = {
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
            }
            if hasattr(final.usage, "cache_creation_input_tokens"):
                usage["cache_creation_input_tokens"] = final.usage.cache_creation_input_tokens or 0
            if hasattr(final.usage, "cache_read_input_tokens"):
                usage["cache_read_input_tokens"] = final.usage.cache_read_input_tokens or 0

            yield StreamDone(
                finish_reason=final.stop_reason,
                usage=usage,
                metadata={"model": final.model},
            )
        except self._api_status_error as exc:
            # Anthropic adds 529 "overloaded" to the shared retryable set.
            retryable = exc.status_code in RETRYABLE_STATUS_CODES or exc.status_code == 529
            raise ProviderError(
                str(exc),
                retryable=retryable,
                provider="anthropic",
                status_code=exc.status_code,
            ) from exc
        except Exception as exc:
            raise ProviderError(
                str(exc),
                retryable=False,
                provider="anthropic",
                status_code=None,
            ) from exc
        finally:
            await self._release_client(leased_api_key)

    async def generate(self, context: AIContext) -> AIResponse:
        """Generate by consuming the structured stream."""
        thinking_parts: list[str] = []
        thinking_signature: str | None = None
        text_parts: list[str] = []
        tool_calls: list[AIToolCall] = []
        done_event: StreamDone | None = None

        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamThinkingDelta):
                thinking_parts.append(event.thinking)
                if event.signature:
                    thinking_signature = event.signature
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
            thinking_signature=thinking_signature,
            finish_reason=done_event.finish_reason if done_event else None,
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
