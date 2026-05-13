"""Provider builders for the screen assistant example."""

from __future__ import annotations

import os

from roomkit.video.vision.gemini import GeminiVisionConfig, GeminiVisionProvider
from roomkit.video.vision.openai import OpenAIVisionConfig, OpenAIVisionProvider


def build_vision_provider(
    tool_choice: str,
    google_api_key: str,
) -> GeminiVisionProvider | OpenAIVisionProvider:
    """Build a single vision provider used for both periodic and on-demand analysis."""
    if tool_choice == "openai":
        from roomkit.video.vision.base import VisionProvider

        vision: VisionProvider = OpenAIVisionProvider(
            OpenAIVisionConfig(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                base_url="https://api.openai.com/v1",
                model=os.environ.get("OPENAI_VISION_MODEL", "gpt-4o"),
                max_tokens=4096,
                detail="high",
            )
        )
    else:
        vision = GeminiVisionProvider(
            GeminiVisionConfig(
                api_key=google_api_key,
                model=os.environ.get(
                    "GEMINI_VISION_MODEL",
                    "gemini-3.1-flash-image-preview",
                ),
                max_tokens=4096,
                prompt=(
                    "Describe what is shown on this screen in 2-3 sentences. "
                    "Focus on the FOREGROUND application. "
                    "Include application name, visible text, URLs, "
                    "and what the user appears to be doing. Be concise."
                ),
            )
        )
    return vision


def build_voice_provider(voice_choice: str) -> object:
    """Build the realtime voice provider."""
    if voice_choice == "openai":
        from roomkit.providers.openai.realtime import OpenAIRealtimeProvider

        return OpenAIRealtimeProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ.get("OPENAI_MODEL", "gpt-realtime-1.5"),
        )
    from roomkit.providers.gemini.realtime import GeminiLiveProvider

    return GeminiLiveProvider(
        api_key=os.environ["GOOGLE_API_KEY"],
        model=os.environ.get(
            "GEMINI_MODEL",
            "gemini-3.1-flash-live-preview",
        ),
    )


def get_voice_name(voice_choice: str) -> str:
    """Get the voice preset for the chosen provider."""
    if voice_choice == "openai":
        return os.environ.get("OPENAI_VOICE", "alloy")
    return os.environ.get("GEMINI_VOICE", "Aoede")
