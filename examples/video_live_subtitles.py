"""Live translated subtitles on video.

Demonstrates the overlay system: user speaks French, STT transcribes,
AI translates to English, subtitles rendered on the video frame.

Uses mock providers — no external services required.

Run with:
    uv run python examples/video_live_subtitles.py
"""

from __future__ import annotations

import asyncio
import json
import logging

from roomkit import (
    HookExecution,
    HookTrigger,
    RoomKit,
)
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.video.pipeline.overlay import (
    Overlay,
    OverlayPosition,
    SubtitleManager,
)

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("example")


async def main() -> None:
    kit = RoomKit()

    # --- Overlay filter with subtitles -----------------------------------

    # Translation function (mock — in production use an AI provider)
    async def translate_to_english(text: str) -> str:
        return f"[EN] {text}"

    subtitle_mgr = SubtitleManager(
        kit,
        translate_fn=translate_to_english,
        max_lines=3,
        style={"font_scale": 0.9, "color": (255, 255, 0), "bg_color": (0, 0, 0)},
    )

    overlay_filter = subtitle_mgr.overlay_filter

    # Add a static title overlay on top
    overlay_filter.add_overlay(
        Overlay(
            id="title",
            content="Live Meeting — Translated",
            position=OverlayPosition.TOP_CENTER,
            z_order=50,
            style={"font_scale": 0.6, "color": (200, 200, 200)},
        )
    )

    # Add a dynamic data overlay (simulating a dashboard)
    table_data = json.dumps(
        {"headers": ["Metric", "Value"], "rows": [["Users", "42"], ["Latency", "12ms"]]}
    )
    overlay_filter.add_overlay(
        Overlay(
            id="stats",
            content=table_data,
            overlay_type="rich",
            position=OverlayPosition.TOP_RIGHT,
            z_order=30,
            style={"width": 200, "font_size": 14},
        )
    )

    # --- Hooks for observability -----------------------------------------

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def log_transcription(event: RoomEvent, ctx: RoomContext) -> None:
        text = getattr(event, "text", "") or getattr(getattr(event, "content", None), "body", "")
        logger.info("Transcription: %s", text)

    # --- Simulate transcription events -----------------------------------

    logger.info("Overlay filter ready with %d overlays", len(overlay_filter._overlays))
    logger.info("Subtitle position: %s", subtitle_mgr.overlay_filter.get_overlay("_subtitle"))

    # Simulate speech → subtitle flow
    subtitle_mgr.set_text("Bonjour, comment allez-vous?")
    ov = overlay_filter.get_overlay("_subtitle")
    logger.info("Subtitle content: %s", ov.content if ov else "(none)")

    # Update stats dynamically
    overlay_filter.update_overlay(
        "stats",
        content=json.dumps(
            {"headers": ["Metric", "Value"], "rows": [["Users", "43"], ["Latency", "11ms"]]}
        ),
    )

    logger.info("Done — overlay system works without a live video backend")
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
