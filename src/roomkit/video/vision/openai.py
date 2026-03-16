"""OpenAI-compatible vision provider for video frame analysis.

Works with any OpenAI-compatible API that supports vision:

- **OpenAI** — GPT-4o, GPT-4o-mini
- **Ollama** — qwen3.5, qwen3-vl, llava, llama3.2-vision
- **vLLM** — Qwen2.5-VL, InternVL, etc.

Usage::

    # OpenAI
    provider = OpenAIVisionProvider(api_key="sk-...", model="gpt-4o")

    # Ollama (local)
    provider = OpenAIVisionProvider(
        base_url="http://localhost:11434/v1",
        model="qwen3.5",
    )

    # vLLM (local)
    provider = OpenAIVisionProvider(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-VL-8B-Instruct",
    )

    result = await provider.analyze_frame(frame)
    print(result.description)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.base import DEFAULT_VISION_PROMPT, VisionProvider, VisionResult
from roomkit.video.vision.encode import frame_to_jpeg_base64

logger = logging.getLogger("roomkit.video.vision.openai")


@dataclass
class OpenAIVisionConfig:
    """Configuration for OpenAIVisionProvider.

    Attributes:
        api_key: API key. Use ``"ollama"`` for Ollama (ignored).
        base_url: API base URL. Defaults to OpenAI.
        model: Model name.
        prompt: System prompt for frame analysis.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
        timeout: HTTP timeout in seconds.
        detail: Image detail level (``low``, ``high``, ``auto``).
    """

    api_key: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3.5"
    prompt: str = DEFAULT_VISION_PROMPT
    max_tokens: int = 100
    temperature: float = 0.3
    timeout: float = 30.0
    detail: Literal["low", "high", "auto"] = "low"


class OpenAIVisionProvider(VisionProvider):
    """Vision provider using any OpenAI-compatible API.

    Sends video frames as base64 JPEG images to a chat completion
    endpoint and parses the response into a :class:`VisionResult`.

    Works with OpenAI, Ollama, vLLM, and any API that supports
    the ``image_url`` content type in chat messages.

    Example::

        # Ollama with qwen3.5 (default)
        provider = OpenAIVisionProvider()

        # Ollama with a different model
        provider = OpenAIVisionProvider(
            OpenAIVisionConfig(model="qwen3-vl:8b")
        )

        # OpenAI GPT-4o
        provider = OpenAIVisionProvider(
            OpenAIVisionConfig(
                api_key="sk-...",
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
            )
        )

        result = await provider.analyze_frame(frame)
        print(result.description)
    """

    def __init__(self, config: OpenAIVisionConfig | None = None) -> None:
        self._config = config or OpenAIVisionConfig()
        self._client: Any = None

    @property
    def name(self) -> str:
        return f"openai-vision:{self._config.model}"

    def _get_client(self) -> Any:
        """Lazy-init the AsyncOpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "openai is required for OpenAIVisionProvider. "
                    "Install with: pip install roomkit[openai]"
                ) from exc
            self._client = openai.AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout=self._config.timeout,
            )
        return self._client

    async def analyze_frame(
        self,
        frame: VideoFrame,
        *,
        prompt: str | None = None,
    ) -> VisionResult:
        """Analyze a video frame via the OpenAI-compatible vision API.

        Encodes the frame as JPEG, sends it as a base64 image in a
        chat completion request, and parses the response.

        Args:
            frame: The video frame (raw_rgb24, raw_bgr24, or encoded).
            prompt: Optional prompt override (defaults to config prompt).

        Returns:
            VisionResult with the model's description.
        """
        client = self._get_client()
        image_b64 = frame_to_jpeg_base64(frame)
        effective_prompt = prompt or self._config.prompt

        response = await client.chat.completions.create(
            model=self._config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": effective_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": self._config.detail,
                            },
                        },
                    ],
                }
            ],
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        description = response.choices[0].message.content or ""
        # Strip thinking blocks that leak through (Qwen3, DeepSeek-R1)
        description = re.sub(r"<think>.*?</think>", "", description, flags=re.DOTALL)
        description = description.strip()

        return VisionResult(
            description=description,
            metadata={
                "model": self._config.model,
                "usage": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                },
            },
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
