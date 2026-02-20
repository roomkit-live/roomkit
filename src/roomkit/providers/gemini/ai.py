"""Google Gemini AI provider — generates responses via the Google Generative AI API."""

from __future__ import annotations

import time
from typing import Any, cast

from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIProvider,
    AIResponse,
    AITextPart,
    AIToolCall,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
)
from roomkit.providers.gemini.config import GeminiConfig


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
    def supports_vision(self) -> bool:
        """All Gemini models support vision."""
        return True

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
        content: str | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart],
    ) -> list[Any]:
        """Convert content to Gemini Parts."""
        if isinstance(content, str):
            return [self._types.Part.from_text(text=content)]

        parts = []
        for item in content:
            if isinstance(item, AITextPart):
                parts.append(self._types.Part.from_text(text=item.text))
            elif isinstance(item, AIImagePart):
                parts.append(
                    self._types.Part.from_uri(
                        file_uri=item.url,
                        mime_type=item.mime_type or "image/jpeg",
                    )
                )
        return parts

    def _extract_tool_calls(self, response: Any) -> list[AIToolCall]:
        """Extract tool calls from Gemini response."""
        tool_calls: list[AIToolCall] = []
        if not response.candidates or not response.candidates[0].content:
            return tool_calls

        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append(
                    AIToolCall(
                        id=fc.name,  # Gemini doesn't provide separate IDs
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    )
                )
        return tool_calls

    async def generate(self, context: AIContext) -> AIResponse:
        # Build generation config
        gen_config = self._types.GenerateContentConfig(
            temperature=context.temperature,
            max_output_tokens=context.max_tokens,
        )

        if context.system_prompt:
            gen_config.system_instruction = context.system_prompt

        # Add tools if provided
        if context.tools:
            func_decls = [
                self._types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    # Cast to Any: Gemini SDK accepts dict as Schema at runtime
                    parameters=cast(Any, t.parameters) if t.parameters else None,
                )
                for t in context.tools
            ]
            gen_config.tools = [self._types.Tool(function_declarations=func_decls)]

        # Format messages
        contents = self._format_messages(context.messages)

        t0 = time.monotonic()
        try:
            response = await self._client.aio.models.generate_content(
                model=self._config.model,
                contents=contents,
                config=gen_config,
            )
        except Exception as exc:
            # Check for rate limit or server errors
            status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            retryable = (
                status_code in (429, 500, 502, 503)
                if status_code
                else any(
                    term in str(exc).lower()
                    for term in ["rate", "limit", "429", "500", "502", "503"]
                )
            )
            raise ProviderError(
                str(exc),
                retryable=retryable,
                provider="gemini",
                status_code=status_code,
            ) from exc

        ttfb_ms = (time.monotonic() - t0) * 1000
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        telemetry.record_metric(
            "roomkit.llm.ttfb_ms",
            ttfb_ms,
            unit="ms",
            attributes={"provider": "gemini", "model": self._config.model},
        )

        # Extract usage metadata
        usage: dict[str, int] = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
            }

        try:
            text = response.text or ""
        except (ValueError, AttributeError):
            text = ""

        return AIResponse(
            content=text,
            usage=usage,
            tool_calls=self._extract_tool_calls(response),
            metadata={"model": self._config.model},
        )

    async def close(self) -> None:
        """Release the genai client reference."""
        self._client = None  # type: ignore[assignment]
