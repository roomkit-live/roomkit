"""RoomKit -- Webcam vision: camera frames analyzed by AI.

Captures webcam frames via LocalVideoBackend, analyzes them with
a VisionProvider, and feeds the descriptions into an AIChannel so
the AI can "see" and respond to what's on camera.

Each AI channel attached to the room reads the latest vision result in
its turn's notes, as a ``<vision>`` block.  Every ``--ask-every`` seconds a text
"viewer" channel asks the AI what it sees; the reply is printed with the
camera view the AI was given.  The AI here is a ``MockAIProvider`` with
canned replies, so the printed view shows the wiring — swap in a real
AI provider for answers grounded in it.

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
    uv run python examples/webcam_vision.py --gemini         # gemini-3.8-flash
    uv run python examples/webcam_vision.py --ollama         # ollama (qwen3.5)
    uv run python examples/webcam_vision.py --gemini --lang fr
    uv run python examples/webcam_vision.py --ask-every 0    # never ask the AI

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import asyncio
import contextlib
import logging

from shared import require_env, run_until_stopped, setup_logging

from roomkit import (
    ChannelCategory,
    FrameworkEvent,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    VideoChannel,
    WebSocketChannel,
)
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
ASK_EVERY_S = 10.0
QUESTION = "What do you see on the camera right now?"

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
        api_key = args.gemini_key or require_env("GEMINI_API_KEY")["GEMINI_API_KEY"]
        return GeminiVisionProvider(
            GeminiVisionConfig(
                api_key=api_key,
                model=args.model or "gemini-3.8-flash",
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
    parser.add_argument(
        "--ask-every",
        type=float,
        default=ASK_EVERY_S,
        help="Seconds between questions to the AI (0 = never ask)",
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
    ai_provider = MockAIProvider(
        responses=[
            "I can see you at your desk!",
            "The room looks nice with that pink light.",
            "Looks like you're waving at me!",
            "Is that a coffee mug? Nice choice.",
            "I see you're typing away.",
            "Nice bookshelf behind you!",
        ]
    )
    ai = AIChannel(
        "ai",
        provider=ai_provider,
        system_prompt="You are a helpful assistant that can see a live camera feed.",
    )
    kit.register_channel(ai)

    # --- Viewer: a text channel that asks the AI about the camera ------------
    viewer = WebSocketChannel("viewer")
    kit.register_channel(viewer)

    async def on_ai_reply(_conn: str, event: RoomEvent) -> None:
        if isinstance(event.content, TextContent):
            print(f"  AI: {event.content.body}")

    viewer.register_connection("viewer-conn", on_ai_reply, room_id="webcam-demo")

    # --- Room setup ----------------------------------------------------------
    await kit.create_room(room_id="webcam-demo")
    await kit.attach_channel("webcam-demo", "video-main")
    await kit.attach_channel("webcam-demo", "viewer")
    await kit.attach_channel("webcam-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # The room's AI channels read the latest vision result in each turn's notes.

    # --- Hooks: log video events ---------------------------------------------

    @kit.hook(HookTrigger.ON_VIDEO_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_session_started(event: SessionStartedEvent, ctx: object) -> None:
        if event.session is not None:
            print(f"  Video session started: {event.session.id[:8]}...")

    @kit.hook(HookTrigger.ON_VIDEO_SESSION_ENDED, execution=HookExecution.ASYNC)
    async def on_session_ended(event: object, ctx: object) -> None:
        print("  Video session ended")

    # --- Framework event: vision results + AI context ------------------------
    frame_count = 0

    @kit.on("video_vision_result")
    async def on_vision(event: FrameworkEvent) -> None:
        nonlocal frame_count
        frame_count += 1
        data = event.data
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

    # --- Periodic question: the AI answers with the view in its turn's notes -------
    async def ask_ai_periodically() -> None:
        while True:
            await asyncio.sleep(args.ask_every)
            print(f"\n  Viewer: {QUESTION}")
            result = await kit.process_inbound(
                InboundMessage(
                    channel_id="viewer",
                    sender_id="viewer",
                    content=TextContent(body=QUESTION),
                ),
                room_id="webcam-demo",
            )
            if result.error is not None:
                print(f"  AI error: {result.error!r}")
            elif ai_provider.calls:
                turn = str(ai_provider.calls[-1].messages[-1].content)
                view = [line for line in turn.splitlines() if line.startswith("Description:")]
                print(f"  (AI was given: {view[-1].strip() if view else 'no camera view yet'})")

    # --- Connect and start capture -------------------------------------------
    session = await kit.join("webcam-demo", "video-main", participant_id="local-user")

    if args.gemini:
        mode = f"Gemini ({args.model or 'gemini-3.8-flash'})"
    elif args.ollama:
        mode = f"Ollama ({args.model or 'qwen3.5'})"
    else:
        mode = "Mock"
    print("Webcam Vision Demo (with AI integration)")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Camera: device {args.device} at 640x480 @ {args.fps}fps")
    print(f"Vision analysis every {args.interval}ms")
    print("The AI channel reads the latest vision result in each turn's notes")
    if args.ask_every > 0:
        print(f"Viewer asks the (mock) AI every {args.ask_every:g}s")
    if args.lang:
        print(f"Language: {args.lang}")
    print("Press Ctrl+C to stop.\n")

    await backend.start_capture(session)
    ask_task = asyncio.create_task(ask_ai_periodically()) if args.ask_every > 0 else None

    # --- Keep running until Ctrl+C -------------------------------------------
    async def cleanup() -> None:
        print("\nStopping...")
        if ask_task is not None:
            ask_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ask_task
        await backend.stop_capture(session)
        await kit.leave(session)

    await run_until_stopped(kit, cleanup=cleanup)
    print(f"Done. Analyzed {frame_count} frames.")


if __name__ == "__main__":
    asyncio.run(main())
