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
    """

    api_key: str = ""
    model: str = "gemini-3.1-flash-lite-preview"
    prompt: str = DEFAULT_VISION_PROMPT
    max_tokens: int = 1024
    temperature: float = 0.3
    extra_config: dict[str, Any] = field(default_factory=dict)


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
        self._types: Any = None

    @property
    def name(self) -> str:
        return f"gemini-vision:{self._config.model}"

    def _get_client(self) -> Any:
        """Lazy-init the google-genai client."""
        if self._client is None:
            try:
                from google import genai as _genai
                from google.genai import types as _types
            except ImportError as exc:
                raise ImportError(
                    "google-genai is required for GeminiVisionProvider. "
                    "Install with: pip install roomkit[gemini]"
                ) from exc
            self._types = _types
            self._client = _genai.Client(api_key=self._config.api_key)
        return self._client

    async def analyze_frame(self, frame: VideoFrame) -> VisionResult:
        """Analyze a video frame via the Gemini API.

        Encodes the frame as JPEG and sends it as inline data.

        Args:
            frame: The video frame (raw_rgb24, raw_bgr24, or encoded).

        Returns:
            VisionResult with the model's description.
        """
        client = self._get_client()
        types = self._types
        jpeg_bytes = frame_to_jpeg(frame)

        image_part = types.Part.from_bytes(
            data=jpeg_bytes,
            mime_type="image/jpeg",
        )

        gen_config: dict[str, Any] = {
            "max_output_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            **self._config.extra_config,
        }
        # Disable thinking for vision — we want direct descriptions,
        # not reasoning chains that consume the token budget.
        # Only models that support thinking_config (2.5+, 3.x).
        supports_thinking = any(
            self._config.model.startswith(p) for p in ("gemini-2.5", "gemini-3")
        )
        if "thinking_config" not in gen_config and supports_thinking:
            gen_config["thinking_config"] = types.ThinkingConfig(
                thinking_budget=0,
            )

        response = await client.aio.models.generate_content(
            model=self._config.model,
            contents=[self._config.prompt, image_part],
            config=types.GenerateContentConfig(**gen_config),
        )

        # Extract text from all parts (Gemini 2.5 may include thinking parts)
        description = ""
        if response.candidates:
            content = response.candidates[0].content
            parts = (content.parts or []) if content else []
            text_parts = [p.text for p in parts if p.text]
            description = " ".join(text_parts).strip()
        # Fallback to response.text if parts extraction is empty
        if not description and response.text:
            description = response.text.strip()

        usage_meta: dict[str, Any] = {"model": self._config.model}
        if response.usage_metadata:
            usage_meta["usage"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
            }
            if hasattr(response.usage_metadata, "thoughts_token_count"):
                usage_meta["usage"]["thinking_tokens"] = (
                    response.usage_metadata.thoughts_token_count
                )

        return VisionResult(
            description=description,
            metadata=usage_meta,
        )

    async def close(self) -> None:
        self._client = None
        self._types = None
