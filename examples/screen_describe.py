"""RoomKit -- Screen Description: screen frames analyzed by AI.

Captures screen frames via ScreenCaptureBackend, analyzes them with
a VisionProvider, and feeds the descriptions into an AIChannel so
the AI can "see" your screen and describe what's happening.

Use case: AI-powered screen assistant that observes and explains
what software is displayed, guiding the user through tasks.

Supports three vision modes:

- **Mock mode** (default): cycles through preset descriptions.
- **Gemini mode**: sends frames to Google Gemini (fast cloud API).
- **Ollama mode**: sends frames to a local Ollama model.

Prerequisites:
    pip install roomkit[screen-capture]

    # For Gemini mode:
    pip install roomkit[gemini]
    export GEMINI_API_KEY=AIza...

    # For Ollama mode:
    pip install roomkit[openai]
    ollama pull qwen3.5       # or qwen3-vl:8b, llava, etc.

Run with:
    uv run python examples/screen_describe.py                  # mock mode
    uv run python examples/screen_describe.py --gemini         # gemini
    uv run python examples/screen_describe.py --ollama         # ollama
    uv run python examples/screen_describe.py --monitor 2      # secondary monitor
    uv run python examples/screen_describe.py --scale 0.5      # half resolution

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from roomkit import ChannelCategory, HookExecution, HookTrigger, RoomKit, VideoChannel
from roomkit.channels.ai import AIChannel
from roomkit.models.session_event import SessionStartedEvent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.video.ai_integration import setup_video_vision
from roomkit.video.backends.screen import ScreenCaptureBackend
from roomkit.video.vision.base import VisionProvider
from roomkit.video.vision.gemini import GeminiVisionConfig, GeminiVisionProvider
from roomkit.video.vision.mock import MockVisionProvider
from roomkit.video.vision.openai import OpenAIVisionConfig, OpenAIVisionProvider

logging.basicConfig(level=logging.WARNING)

SCREEN_VISION_PROMPT = (
    "Describe what software or application is shown on this screen. "
    "Include visible UI elements, menus, buttons, text, and what the user "
    "appears to be doing. Be concise and precise."
)


def _build_vision_provider(args: argparse.Namespace) -> VisionProvider:
    """Build the vision provider based on CLI args."""
    if args.gemini:
        api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise SystemExit("Gemini API key required. Set GEMINI_API_KEY or use --gemini-key.")
        return GeminiVisionProvider(
            GeminiVisionConfig(
                api_key=api_key,
                model=args.model or "gemini-3.1-flash-lite-preview",
                prompt=SCREEN_VISION_PROMPT,
            )
        )
    if args.ollama:
        return OpenAIVisionProvider(
            OpenAIVisionConfig(
                base_url=args.base_url,
                model=args.model or "qwen3.5",
                api_key="ollama",
                prompt=SCREEN_VISION_PROMPT,
                timeout=60.0,
            )
        )
    return MockVisionProvider(
        descriptions=[
            "A code editor showing a Python file with a class definition",
            "A terminal window with test output — 42 passed, 0 failed",
            "A web browser open to documentation with a sidebar menu",
            "A file manager showing the project directory structure",
            "An IDE with a diff view comparing two versions of a file",
            "A chat application with a conversation thread visible",
        ],
        labels=[
            ["code-editor", "python", "class"],
            ["terminal", "test-output", "pytest"],
            ["browser", "documentation", "sidebar"],
            ["file-manager", "directory", "project"],
            ["ide", "diff-view", "git"],
            ["chat", "conversation", "messages"],
        ],
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Screen Description Demo")
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
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (1=primary)")
    parser.add_argument("--fps", type=int, default=2, help="Capture FPS")
    parser.add_argument("--scale", type=float, default=0.5, help="Downscale factor")
    parser.add_argument("--diff", type=float, default=0.02, help="Diff threshold (0=disabled)")
    parser.add_argument("--interval", type=int, default=5000, help="Vision analysis interval ms")
    args = parser.parse_args()

    kit = RoomKit()

    # --- Screen capture backend ----------------------------------------------
    backend = ScreenCaptureBackend(
        monitor=args.monitor,
        fps=args.fps,
        scale=args.scale,
        diff_threshold=args.diff,
    )

    # --- Vision provider -----------------------------------------------------
    vision = _build_vision_provider(args)

    # --- Video channel -------------------------------------------------------
    video = VideoChannel(
        "video-screen",
        backend=backend,
        vision=vision,
        vision_interval_ms=args.interval,
    )
    kit.register_channel(video)

    # --- AI channel (mock — responds based on what it "sees") ----------------
    ai = AIChannel(
        "ai",
        provider=MockAIProvider(
            responses=[
                "I see you're working in a code editor. Need help with that code?",
                "Tests are passing — looks like the build is green!",
                "You're reading documentation. Want me to summarize that page?",
                "I can see the project files. Want me to explain the structure?",
                "You're reviewing a diff. Want me to describe the changes?",
                "I see a chat window. Anything I can help with?",
            ]
        ),
        system_prompt=(
            "You are a helpful screen assistant. You can see the user's screen "
            "and help them navigate software, explain what's visible, and guide "
            "them through tasks step by step."
        ),
    )
    kit.register_channel(ai)

    # --- Room setup ----------------------------------------------------------
    await kit.create_room(room_id="screen-demo")
    await kit.attach_channel("screen-demo", "video-screen")
    await kit.attach_channel("screen-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # Wire vision results into AI context
    setup_video_vision(kit, room_id="screen-demo", ai_channel_id="ai")

    # --- Hooks: log video events ---------------------------------------------

    @kit.hook(HookTrigger.ON_VIDEO_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_session_started(event: SessionStartedEvent, ctx: object) -> None:
        print(f"  Screen capture session started: {event.session.id[:8]}...")  # type: ignore[union-attr]

    @kit.hook(HookTrigger.ON_VIDEO_SESSION_ENDED, execution=HookExecution.ASYNC)
    async def on_session_ended(event: object, ctx: object) -> None:
        print("  Screen capture session ended")

    # --- Framework event: vision results -------------------------------------
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
    session = await kit.connect_video("screen-demo", "local-user", "video-screen")

    if args.gemini:
        mode = f"Gemini ({args.model or 'gemini-3.1-flash-lite-preview'})"
    elif args.ollama:
        mode = f"Ollama ({args.model or 'qwen3.5'})"
    else:
        mode = "Mock"
    print("Screen Description Demo (with AI integration)")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Monitor: {args.monitor} at scale {args.scale} @ {args.fps}fps")
    print(f"Vision analysis every {args.interval}ms")
    print(f"Diff threshold: {args.diff}")
    print("AI channel receives vision context via setup_video_vision()")
    print("Press Ctrl+C to stop.\n")

    await backend.start_capture(session)

    # --- Keep running until Ctrl+C -------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, stop.set)

    await stop.wait()

    # --- Cleanup -------------------------------------------------------------
    print("\nStopping...")
    await backend.stop_capture(session)
    await kit.disconnect_video(session)
    await kit.close()
    print(f"Done. Analyzed {frame_count} frames.")


if __name__ == "__main__":
    asyncio.run(main())
