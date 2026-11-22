"""Google Gemini vision provider for video frame analysis.

Uses the ``google-genai`` SDK to send video frames to Gemini
for analysis.  Gemini Flash is fast and cost-effective for
real-time frame analysis.

Usage::

    config = GeminiVisionConfig(api_key="...")
    provider = GeminiVisionProvider(config)
    result = await provider.analyze_frame(frame)
    print(result.description)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from roomkit.providers.ai.response_schema import check_schema_answer, check_schema_request
from roomkit.providers.gemini.errors import (
    REFUSAL_FINISH_REASONS,
    finish_reason_of,
    prompt_block_reason,
    reason_name,
)
from roomkit.providers.gemini.sdk import build_genai_client, close_genai_client
from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.base import DEFAULT_VISION_PROMPT, VisionProvider, VisionResult
from roomkit.video.vision.encode import frame_to_jpeg

logger = logging.getLogger("roomkit.video.vision.gemini")


@dataclass
class GeminiVisionConfig:
    """Configuration for GeminiVisionProvider.

    Attributes:
        api_key: Google AI API key.
        model: Gemini model name.
        prompt: System prompt for frame analysis.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
        extra_config: Extra ``GenerateContentConfig`` fields.
        timeout: HTTP timeout in seconds, the read budget of one frame analysis.
        connect_timeout: TCP connect timeout in seconds, apart from ``timeout``.
    """

    api_key: str = field(default="", repr=False)
    model: str = "gemini-3.8-flash"
    prompt: str = DEFAULT_VISION_PROMPT
    max_tokens: int = 1024
    temperature: float = 0.3
    extra_config: dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    connect_timeout: float = 5.0


class GeminiVisionProvider(VisionProvider):
    """Vision provider using Google Gemini.

    Sends video frames as inline JPEG data to the Gemini API
    and parses the response into a :class:`VisionResult`.

    Example::

        provider = GeminiVisionProvider(GeminiVisionConfig(api_key="AIza..."))
        result = await provider.analyze_frame(frame)
        print(result.description)
    """

    def __init__(self, config: GeminiVisionConfig | None = None) -> None:
        self._config = config or GeminiVisionConfig()
        self._client: Any = None
        self._http: Any = None
        self._types: Any = None
        self._thinking_budget_refused = False

    @property
    def name(self) -> str:
        return f"gemini-vision:{self._config.model}"

    def _get_client(self) -> Any:
        """Lazy-init the google-genai client."""
        if self._client is None:
            # The client carries the connect/read split; see ``build_genai_client``
            # for why it cannot go on the request.
            self._client, self._http, self._types = build_genai_client(
                self._config, provider="GeminiVisionProvider", api_key=self._config.api_key
            )
        return self._client

    @property
    def supports_response_schema(self) -> bool:
        """Controlled generation, through ``response_json_schema``."""
        return True

    async def analyze_frame(
        self,
        frame: VideoFrame,
        *,
        prompt: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> VisionResult:
        """Analyze a video frame via the Gemini API.

        Encodes the frame as JPEG and sends it as inline data.

        Args:
            frame: The video frame (raw_rgb24, raw_bgr24, or encoded).
            prompt: Optional prompt override (defaults to config prompt).
            response_schema: JSON Schema the description must satisfy.

        Returns:
            VisionResult with the model's description.
        """
        if response_schema is not None:
            check_schema_request(response_schema, supported=True, provider="gemini-vision")
        client = self._get_client()
        types = self._types
        jpeg_bytes = frame_to_jpeg(frame)

        image_part = types.Part.from_bytes(
            data=jpeg_bytes,
            mime_type="image/jpeg",
        )
        response = await self._generate(
            client, types, [prompt or self._config.prompt, image_part], response_schema
        )

        description = _answer_text(response)
        if response_schema is not None:
            _check_answer(response, description, response_schema)
        return VisionResult(description=description, metadata=self._metadata(response))

    def _metadata(self, response: Any) -> dict[str, Any]:
        """The model, and the token counts when the response carries them."""
        metadata: dict[str, Any] = {"model": self._config.model}
        usage = response.usage_metadata
        if usage:
            metadata["usage"] = {
                "prompt_tokens": usage.prompt_token_count,
                "completion_tokens": usage.candidates_token_count,
            }
            if hasattr(usage, "thoughts_token_count"):
                metadata["usage"]["thinking_tokens"] = usage.thoughts_token_count
        return metadata

    def _generation_config(
        self, types: Any, response_schema: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], bool]:
        """The config for one frame, and whether it carries the thinking-off
        setting this provider added (rather than the caller's own)."""
        gen_config: dict[str, Any] = {
            "max_output_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            **self._config.extra_config,
        }
        if response_schema is not None:
            gen_config["response_mime_type"] = "application/json"
            gen_config["response_json_schema"] = response_schema
        # Disable thinking for vision — we want direct descriptions,
        # not reasoning chains that consume the token budget.
        # Only models that support thinking_config (2.5+, 3.x).
        supports_thinking = any(
            self._config.model.startswith(p) for p in ("gemini-2.5", "gemini-3")
        )
        added_thinking = (
            "thinking_config" not in gen_config
            and supports_thinking
            and not self._thinking_budget_refused
        )
        if added_thinking:
            gen_config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        # No tools here, and the SDK's automatic function calling otherwise
        # logs a warning on every frame analysed.
        gen_config.setdefault(
            "automatic_function_calling", types.AutomaticFunctionCallingConfig(disable=True)
        )
        return gen_config, added_thinking

    async def _generate(
        self,
        client: Any,
        types: Any,
        contents: list[Any],
        response_schema: dict[str, Any] | None = None,
    ) -> Any:
        """Ask for the description, without the thinking-off setting where the
        model refuses it.

        No one setting minimises reasoning on every model: measured on
        2026-09-27, ``thinking_budget=0`` does it on 2.5 Flash, 3.1 Flash-Lite
        and 3.5/3.6 Flash, and is answered 400 by ``gemini-3.5-flash-lite`` and
        ``gemini-3.1-pro-preview``. So it is sent, and a 400 to the setting
        this provider added is answered by one retry without it; once that
        retry succeeds, the model is not sent it again. A failing retry raises
        its own error, the one about the request itself.
        """
        gen_config, added_thinking = self._generation_config(types, response_schema)
        try:
            return await client.aio.models.generate_content(
                model=self._config.model,
                contents=contents,
                config=types.GenerateContentConfig(**gen_config),
            )
        except Exception as exc:
            if not added_thinking or getattr(exc, "code", None) != 400:
                raise
        del gen_config["thinking_config"]
        response = await client.aio.models.generate_content(
            model=self._config.model,
            contents=contents,
            config=types.GenerateContentConfig(**gen_config),
        )
        self._thinking_budget_refused = True
        logger.info(
            "%s refuses thinking_budget=0; analysing frames with its own thinking default",
            self._config.model,
        )
        return response

    async def close(self) -> None:
        client, self._client = self._client, None
        http, self._http = self._http, None
        self._types = None
        await close_genai_client(client, http)


def _answer_text(response: Any) -> str:
    """The first candidate's answer parts, joined as written.

    Thought parts are left out, and nothing is inserted between parts: a JSON
    document split across two parts must come back whole.
    """
    parts: list[Any] = []
    if response.candidates:
        content = response.candidates[0].content
        parts = (content.parts or []) if content else []
    text = "".join(p.text for p in parts if p.text and getattr(p, "thought", None) is not True)
    if not text and response.text:
        text = response.text
    return text.strip()


def _check_answer(response: Any, description: str, schema: dict[str, Any]) -> None:
    """Refuse a constrained description that did not deliver its JSON document.

    A withheld answer is a refusal whether the model stopped on a safety reason
    or the prompt itself was blocked, which leaves no candidate at all.
    """
    finish = reason_name(response.candidates[0].finish_reason) if response.candidates else None
    check_schema_answer(
        description,
        schema=schema,
        provider="gemini-vision",
        refusal=finish if finish in REFUSAL_FINISH_REASONS else prompt_block_reason(response),
        finish_reason=finish_reason_of(finish),
    )
