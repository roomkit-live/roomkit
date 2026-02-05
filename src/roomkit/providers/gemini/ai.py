"""Google Gemini AI provider — generates responses via the Google Generative AI API."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIProvider,
    AIResponse,
    AITextPart,
    AIToolCall,
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
            role = "model" if msg.role == "assistant" else "user"
            parts = self._format_content(msg.content)
            contents.append(self._types.Content(role=role, parts=parts))
        return contents

    def _format_content(self, content: str | list[AITextPart | AIImagePart]) -> list[Any]:
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
        if not response.candidates:
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

        try:
            # Generate (sync API, wrap in executor for async)
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._config.model,
                contents=contents,
                config=gen_config,
            )
        except Exception as exc:
            # Check for rate limit or server errors
            exc_str = str(exc).lower()
            retryable = any(
                term in exc_str for term in ["rate", "limit", "429", "500", "502", "503"]
            )
            raise ProviderError(
                str(exc),
                retryable=retryable,
                provider="gemini",
                status_code=None,
            ) from exc

        # Extract usage metadata
        usage: dict[str, int] = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
            }

        return AIResponse(
            content=response.text or "",
            usage=usage,
            tool_calls=self._extract_tool_calls(response),
            metadata={"model": self._config.model},
        )
