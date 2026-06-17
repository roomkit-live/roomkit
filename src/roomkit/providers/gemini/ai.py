"""Google Gemini AI provider — generates responses via the Google Generative AI API."""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, cast
from uuid import uuid4

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
    ModelInfo,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
)
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.gemini.models import MODELS

logger = logging.getLogger(__name__)


def _parts_layout(parts: list[Any]) -> str:
    """Compact one-line summary of a streamed chunk's parts for diagnostics.

    Renders each part as ``kind(sig=yes|no)`` so a thought_signature that
    arrives on a different part (or a later chunk) than its function_call
    is visible. ``kind`` is fcall:<name> / thought / text / other.
    """
    out: list[str] = []
    for p in parts:
        has_sig = getattr(p, "thought_signature", None) is not None
        fc = getattr(p, "function_call", None)
        if fc is not None:
            kind = f"fcall:{getattr(fc, 'name', '?')}"
        elif getattr(p, "thought", False):
            kind = "thought"
        elif getattr(p, "text", None):
            kind = "text"
        else:
            kind = "other"
        out.append(f"{kind}(sig={'yes' if has_sig else 'no'})")
    return " ".join(out) or "(empty)"


class GeminiAIProvider(AIProvider):
    """AI provider using the Google Gemini API."""

    def __init__(self, config: GeminiConfig) -> None:
        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiAIProvider. "
                "Install it with: pip install roomkit[gemini]"
            ) from exc

        self._config = config
        self._genai = _genai
        self._types = _types
        self._client = _genai.Client(api_key=config.api_key.get_secret_value())

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
        """All Gemini models support vision."""
        return True

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        """Curated, offline catalog of Gemini models."""
        return list(MODELS)

    async def list_models(self) -> list[ModelInfo]:
        """List generate-content models the Gemini API currently exposes."""
        live: list[ModelInfo] = []
        pager = await self._client.aio.models.list()  # ty: ignore[unresolved-attribute]
        async for m in pager:
            name = (m.name or "").removeprefix("models/")
            actions = getattr(m, "supported_actions", None) or []
            if not name or (actions and "generateContent" not in actions):
                continue
            live.append(
                ModelInfo(
                    id=name,
                    display_name=getattr(m, "display_name", None),
                    context_window=getattr(m, "input_token_limit", None),
                )
            )
        return self._merge_curated(live)

    def _format_messages(self, messages: list[AIMessage]) -> list[Any]:
        """Convert AIMessage list to Gemini Content format."""
        contents = []
        for msg in messages:
            if isinstance(msg.content, list) and any(
                isinstance(p, AIToolCallPart) for p in msg.content
            ):
                # Model message with function calls
                parts = []
                for p in msg.content:
                    if isinstance(p, AITextPart):
                        parts.append(self._types.Part.from_text(text=p.text))
                    elif isinstance(p, AIToolCallPart):
                        sig = p.metadata.get("thought_signature")
                        if sig:
                            parts.append(
                                self._types.Part(
                                    function_call=self._types.FunctionCall(
                                        name=p.name,
                                        args=p.arguments,
                                    ),
                                    thought=True,
                                    thought_signature=base64.b64decode(sig),
                                )
                            )
                        else:
                            parts.append(
                                self._types.Part.from_function_call(
                                    name=p.name,
                                    args=p.arguments,
                                )
                            )
                contents.append(self._types.Content(role="model", parts=parts))
            elif isinstance(msg.content, list) and any(
                isinstance(p, AIToolResultPart) for p in msg.content
            ):
                # Function responses
                parts = []
                for p in msg.content:
                    if isinstance(p, AIToolResultPart):
                        parts.append(
                            self._types.Part.from_function_response(
                                name=p.name,
                                response={"result": p.result},
                            )
                        )
                contents.append(self._types.Content(role="user", parts=parts))
            else:
                role = "model" if msg.role == "assistant" else "user"
                parts = self._format_content(msg.content)
                contents.append(self._types.Content(role=role, parts=parts))
        return contents

    def _format_content(
        self,
        content: (
            str
            | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart]
        ),
    ) -> list[Any]:
        """Convert content to Gemini Parts."""
        if isinstance(content, str):
            return [self._types.Part.from_text(text=content)]

        parts = []
        for item in content:
            if isinstance(item, AITextPart):
                parts.append(self._types.Part.from_text(text=item.text))
            elif isinstance(item, AIImagePart):
                parts.append(self._image_part(item))
        return parts

    def _image_part(self, item: AIImagePart) -> Any:
        """Build a Gemini image Part from an ``AIImagePart``.

        RoomKit carries images as ``data:<media_type>;base64,<data>``
        URIs — the convention the Anthropic and OpenAI providers consume.
        Gemini's ``from_uri`` expects a fetchable file URI (``gs://`` or
        ``https://``); handed a data URI it ships a broken reference and
        the model never sees the image. Decode a data URI to inline bytes
        via ``from_bytes``; pass a real URI through to ``from_uri``.
        """
        url = item.url
        if url.startswith("data:"):
            header, _, b64data = url.partition(",")
            media_type = header[len("data:") :].split(";", 1)[0] or (
                item.mime_type or "image/jpeg"
            )
            return self._types.Part.from_bytes(
                data=base64.b64decode(b64data),
                mime_type=media_type,
            )
        return self._types.Part.from_uri(
            file_uri=url,
            mime_type=item.mime_type or "image/jpeg",
        )

    def _build_gen_config(self, context: AIContext) -> Any:
        """Build Gemini generation config from AIContext."""
        gen_config = self._types.GenerateContentConfig(
            temperature=context.temperature,
            max_output_tokens=context.max_tokens,
        )

        # Thinking config. ``include_thoughts=True`` is required for Gemini to
        # stream thought summaries — without it the model still reasons but the
        # reasoning never reaches the response. ``thinking_level`` targets Gemini
        # 3.x; ``thinking_budget`` (from the per-turn context) targets 2.5.
        thinking_level = self._config.thinking_level
        thinking_budget = context.thinking_budget
        if thinking_level:
            gen_config.thinking_config = self._types.ThinkingConfig(
                thinking_level=thinking_level,  # ty: ignore[invalid-argument-type]
                include_thoughts=True,
            )
        elif thinking_budget:
            gen_config.thinking_config = self._types.ThinkingConfig(
                thinking_budget=thinking_budget,
                include_thoughts=True,
            )

        if context.system_prompt:
            gen_config.system_instruction = context.system_prompt

        if context.tools:
            from roomkit.providers.gemini.schema import clean_gemini_schema

            func_decls = [
                self._types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=cast(Any, clean_gemini_schema(t.parameters))
                    if t.parameters
                    else None,
                )
                for t in context.tools
            ]
            gen_config.tools = [self._types.Tool(function_declarations=func_decls)]

        return gen_config

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """Wrap an SDK exception into a ProviderError."""
        status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
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
            provider="gemini",
            status_code=status_code,
        )

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        """Yield structured events from the Gemini streaming API."""
        gen_config = self._build_gen_config(context)
        contents = self._format_messages(context.messages)

        t0 = time.monotonic()
        first_token = True
        # Gemini can stream the same function call across several chunks — the
        # first carries its thought_signature, a later one re-emits the call
        # without it. Naively appending one tool call per part produced a
        # duplicate with no signature, which Gemini 3 then rejects with a 400
        # ("Function call is missing a thought_signature") on the next turn.
        # Accumulate by (name, args) and keep the signature from whichever
        # emission carried it, so each distinct call surfaces exactly once.
        fcalls: dict[str, dict[str, Any]] = {}
        fcall_order: list[str] = []
        usage: dict[str, int] = {}

        try:
            response_stream = await self._client.aio.models.generate_content_stream(  # ty: ignore[unresolved-attribute]
                model=self._config.model,
                contents=contents,
                config=gen_config,
            )
            async for chunk in response_stream:
                # Extract usage from each chunk (last one has the totals)
                if chunk.usage_metadata:
                    # Canonical key names (input_tokens / output_tokens) so the
                    # downstream usage tracker records Gemini like every other
                    # provider — the SDK calls them prompt/candidates counts.
                    usage = {
                        "input_tokens": chunk.usage_metadata.prompt_token_count or 0,
                        "output_tokens": chunk.usage_metadata.candidates_token_count or 0,
                    }

                if not chunk.candidates or not chunk.candidates[0].content:
                    continue

                parts = chunk.candidates[0].content.parts
                if not parts:
                    continue

                # Diagnostic: log the chunk's part layout whenever a function
                # call or a thought_signature is present, so we can see WHERE
                # Gemini puts the signature in streaming (on the function_call
                # part, on a thought part, or on a later chunk). Drives the
                # thought_signature round-trip fix.
                if any(
                    getattr(p, "function_call", None) is not None
                    or getattr(p, "thought_signature", None) is not None
                    for p in parts
                ):
                    logger.info("Gemini stream chunk parts: %s", _parts_layout(parts))

                for part in parts:
                    if hasattr(part, "text") and part.text:
                        if first_token:
                            self._record_ttfb(t0)
                            first_token = False
                        # Thought-summary parts are flagged thought=True.
                        if getattr(part, "thought", False):
                            yield StreamThinkingDelta(thinking=part.text)
                        else:
                            yield StreamTextDelta(text=part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        fc_name = fc.name or ""
                        fc_args = dict(fc.args) if fc.args else {}
                        # thought_signature lives on the Part (bytes), not the
                        # FunctionCall. Encode to a portable str for metadata.
                        raw_sig = getattr(part, "thought_signature", None)
                        sig = (
                            base64.b64encode(raw_sig).decode("ascii")
                            if isinstance(raw_sig, bytes)
                            else raw_sig
                        )
                        try:
                            fp = json.dumps(fc_args, sort_keys=True, default=str)
                        except (TypeError, ValueError):
                            fp = repr(fc_args)
                        key = f"{fc_name}::{fp}"
                        if key not in fcalls:
                            fcalls[key] = {
                                "id": f"call_{uuid4().hex[:12]}",
                                "name": fc_name,
                                "arguments": fc_args,
                                "signature": sig,
                            }
                            fcall_order.append(key)
                        elif fcalls[key]["signature"] is None and sig is not None:
                            # Re-emission carried the signature the first did not.
                            fcalls[key]["signature"] = sig

            if fcall_order:
                logger.info(
                    "Gemini tool calls finalized: %s",
                    [
                        f"{fcalls[k]['name']}({'sig' if fcalls[k]['signature'] else 'NOSIG'})"
                        for k in fcall_order
                    ],
                )
            for key in fcall_order:
                fc_data = fcalls[key]
                meta: dict[str, Any] = {}
                if fc_data["signature"] is not None:
                    meta["thought_signature"] = fc_data["signature"]
                else:
                    # After merging every emission of this call, still no
                    # signature — this is the one Gemini 3 will reject on the
                    # next turn. Logged so a residual 400 is attributable.
                    logger.warning(
                        "Gemini function_call '%s' has no thought_signature after "
                        "merge (thinking_budget=%s, args=%s) — Gemini 3 will reject "
                        "it on the next turn",
                        fc_data["name"],
                        context.thinking_budget,
                        fc_data["arguments"],
                    )
                yield StreamToolCall(
                    id=fc_data["id"],
                    name=fc_data["name"],
                    arguments=fc_data["arguments"],
                    metadata=meta,
                )

            yield StreamDone(usage=usage, metadata={"model": self._config.model})

        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def generate(self, context: AIContext) -> AIResponse:
        """Generate by consuming the structured stream."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[AIToolCall] = []
        done_event: StreamDone | None = None

        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamThinkingDelta):
                thinking_parts.append(event.thinking)
            elif isinstance(event, StreamTextDelta):
                text_parts.append(event.text)
            elif isinstance(event, StreamToolCall):
                tool_calls.append(
                    AIToolCall(
                        id=event.id,
                        name=event.name,
                        arguments=event.arguments,
                        metadata=event.metadata,
                    )
                )
            elif isinstance(event, StreamDone):
                done_event = event

        return AIResponse(
            content="".join(text_parts),
            thinking="".join(thinking_parts) if thinking_parts else None,
            usage=done_event.usage if done_event else {},
            tool_calls=tool_calls,
            metadata=done_event.metadata if done_event else {},
        )

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        """Yield text deltas as they arrive from the Gemini API."""
        async for event in self.generate_structured_stream(context):
            if isinstance(event, StreamTextDelta):
                yield event.text

    async def close(self) -> None:
        """Release the genai client reference."""
        self._client = None
