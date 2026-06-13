"""Google Gemini AI provider — generates responses via the Google Generative AI API."""

from __future__ import annotations

import base64
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
        tool_calls: list[StreamToolCall] = []
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
                    usage = {
                        "prompt_tokens": chunk.usage_metadata.prompt_token_count or 0,
                        "completion_tokens": chunk.usage_metadata.candidates_token_count or 0,
                    }

                if not chunk.candidates or not chunk.candidates[0].content:
                    continue

                parts = chunk.candidates[0].content.parts
                if not parts:
                    continue

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
                        fc_name: str = fc.name or ""
                        # thought_signature lives on the Part (bytes), not FunctionCall
                        tc_meta: dict[str, Any] = {}
                        raw_sig = getattr(part, "thought_signature", None)
                        if raw_sig is not None:
                            tc_meta["thought_signature"] = (
                                base64.b64encode(raw_sig).decode("ascii")
                                if isinstance(raw_sig, bytes)
                                else raw_sig
                            )
                        tool_calls.append(
                            StreamToolCall(
                                id=f"call_{uuid4().hex[:12]}",
                                name=fc_name,
                                arguments=dict(fc.args) if fc.args else {},
                                metadata=tc_meta,
                            )
                        )

            for tc in tool_calls:
                yield tc

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
