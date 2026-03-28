"""RoomKit -- Webcam vision: camera frames analyzed by AI.

Captures webcam frames via LocalVideoBackend, analyzes them with
a VisionProvider, and feeds the descriptions into an AIChannel so
the AI can "see" and respond to what's on camera.

Supports three vision modes:

- **Mock mode** (default): cycles through preset descriptions.
- **Gemini mode**: sends frames to Google Gemini (fast cloud API).
- **Ollama mode**: sends frames to a local Ollama model.

Prerequisites:
    pip install roomkit[local-video]

    # For Gemini mode:
    pip install roomkit[gemini]
    export GEMINI_API_KEY=AIza...

    # For Ollama mode:
    pip install roomkit[openai]
    ollama pull qwen3.5       # or qwen3-vl:8b, llava, etc.

Run with:
    uv run python examples/webcam_vision.py                  # mock mode
    uv run python examples/webcam_vision.py --gemini         # gemini 2.5 flash
    uv run python examples/webcam_vision.py --ollama         # ollama (qwen3.5)
    uv run python examples/webcam_vision.py --gemini --lang fr

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import asyncio
import logging

from shared import run_until_stopped, setup_logging

from roomkit import ChannelCategory, HookExecution, HookTrigger, RoomKit, VideoChannel
from roomkit.channels.ai import AIChannel
from roomkit.models.session_event import SessionStartedEvent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.video.backends.local import LocalVideoBackend
from roomkit.video.pipeline import VideoPipelineConfig
from roomkit.video.vision.base import VisionProvider
from roomkit.video.vision.gemini import GeminiVisionConfig, GeminiVisionProvider
from roomkit.video.vision.mock import MockVisionProvider
from roomkit.video.vision.openai import OpenAIVisionConfig, OpenAIVisionProvider

setup_logging("webcam_vision", level=logging.WARNING)

VISION_INTERVAL_MS = 3000

LANG_NAMES = {
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
}


def _build_prompt(lang: str | None) -> str:
    """Build the vision prompt, optionally in a specific language."""
    base = (
        "Describe what you see in this image briefly and precisely. "
        "Include key objects, people, actions, and any visible text."
    )
    if lang:
        lang_name = LANG_NAMES.get(lang, lang)
        base += f" Respond in {lang_name}."
    return base


def _build_vision_provider(args: argparse.Namespace) -> VisionProvider:
    """Build the vision provider based on CLI args."""
    prompt = _build_prompt(args.lang)
    if args.gemini:
        api_key = args.gemini_key
        if not api_key:
            import os

            api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise SystemExit("Gemini API key required. Set GEMINI_API_KEY or use --gemini-key.")
        return GeminiVisionProvider(
            GeminiVisionConfig(
                api_key=api_key,
                model=args.model or "gemini-3.1-flash-lite-preview",
                prompt=prompt,
            )
        )
    if args.ollama:
        return OpenAIVisionProvider(
            OpenAIVisionConfig(
                base_url=args.base_url,
                model=args.model or "qwen3.5",
                api_key="ollama",
                prompt=prompt,
                timeout=60.0,
            )
        )
    return MockVisionProvider(
        descriptions=[
            "A person sitting at a desk with a laptop",
            "A bright room with a window in the background",
            "Someone waving at the camera",
            "A coffee mug on the desk",
            "The person is typing on the keyboard",
            "A bookshelf visible behind the person",
        ],
        labels=[
            ["person", "desk", "laptop"],
            ["room", "window"],
            ["person", "gesture"],
            ["mug", "desk"],
            ["person", "keyboard"],
            ["bookshelf", "background"],
        ],
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam Vision Demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ollama", action="store_true", help="Use Ollama")
    group.add_argument("--gemini", action="store_true", help="Use Gemini")
    parser.add_argument("--model", default=None, help="Model name (auto per provider)")
    parser.add_argument("--gemini-key", default="", help="Gemini API key")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible API base URL (Ollama mode)",
    )
    parser.add_argument("--lang", default=None, help="Response language (e.g. fr, es, de)")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--fps", type=int, default=15, help="Capture FPS")
    parser.add_argument(
        "--interval", type=int, default=VISION_INTERVAL_MS, help="Vision interval ms"
    )
    args = parser.parse_args()

    kit = RoomKit()

    # --- Video backend: local webcam -----------------------------------------
    backend = LocalVideoBackend(device=args.device, fps=args.fps, width=640, height=480)

    # --- Vision provider -----------------------------------------------------
    vision = _build_vision_provider(args)

    # --- Video channel with pipeline ------------------------------------------
    # Vision runs inside the pipeline — frames are already raw (webcam),
    # so no decoder needed.  For RTP/SIP backends, add a decoder stage
    # to convert VP9/H.264 → raw pixels before vision analysis.
    video = VideoChannel(
        "video-main",
        backend=backend,
        pipeline=VideoPipelineConfig(vision=vision),
        vision_interval_ms=args.interval,
    )
    kit.register_channel(video)

    # --- AI channel (mock — responds based on what it "sees") ----------------
    ai = AIChannel(
        "ai",
        provider=MockAIProvider(
            responses=[
                "I can see you at your desk!",
                "The room looks nice with that pink light.",
                "Looks like you're waving at me!",
                "Is that a coffee mug? Nice choice.",
                "I see you're typing away.",
                "Nice bookshelf behind you!",
            ]
        ),
        system_prompt="You are a helpful assistant that can see a live camera feed.",
    )
    kit.register_channel(ai)

    # --- Room setup ----------------------------------------------------------
    await kit.create_room(room_id="webcam-demo")
    await kit.attach_channel("webcam-demo", "video-main")
    await kit.attach_channel("webcam-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # Vision results are auto-injected into AI channels in the same room.

    # --- Hooks: log video events ---------------------------------------------

    @kit.hook(HookTrigger.ON_VIDEO_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_session_started(event: SessionStartedEvent, ctx: object) -> None:
        print(f"  Video session started: {event.session.id[:8]}...")  # type: ignore[union-attr]

    @kit.hook(HookTrigger.ON_VIDEO_SESSION_ENDED, execution=HookExecution.ASYNC)
    async def on_session_ended(event: object, ctx: object) -> None:
        print("  Video session ended")

    # --- Framework event: vision results + AI context ------------------------
    frame_count = 0

    @kit.on("video_vision_result")
    async def on_vision(event: object) -> None:
        nonlocal frame_count
        frame_count += 1
        data = event.data  # type: ignore[attr-defined]
        elapsed = data.get("elapsed_ms", 0)
        desc = data["description"]
        if len(desc) > 500:
            desc = desc[:500] + "..."
        labels = ", ".join(data.get("labels", []))
        parts = [f"\n  [{frame_count}] ({elapsed}ms) {desc}"]
        if labels:
            parts.append(f"       Labels: {labels}")
        if data.get("text"):
            parts.append(f"       OCR: {data['text']}")
        print("\n".join(parts))

    # --- Connect and start capture -------------------------------------------
    session = await kit.connect_video("webcam-demo", "local-user", "video-main")

    if args.gemini:
        mode = f"Gemini ({args.model or 'gemini-3.1-flash-lite-preview'})"
    elif args.ollama:
        mode = f"Ollama ({args.model or 'qwen3.5'})"
    else:
        mode = "Mock"
    print("Webcam Vision Demo (with AI integration)")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Camera: device {args.device} at 640x480 @ {args.fps}fps")
    print(f"Vision analysis every {args.interval}ms")
    print("AI channel receives vision context via setup_video_vision()")
    if args.lang:
        print(f"Language: {args.lang}")
    print("Press Ctrl+C to stop.\n")

    await backend.start_capture(session)

    # --- Keep running until Ctrl+C -------------------------------------------
    async def cleanup() -> None:
        print("\nStopping...")
        await backend.stop_capture(session)
        await kit.disconnect_video(session)

    await run_until_stopped(kit, cleanup=cleanup)
    print(f"Done. Analyzed {frame_count} frames.")


if __name__ == "__main__":
    asyncio.run(main())
