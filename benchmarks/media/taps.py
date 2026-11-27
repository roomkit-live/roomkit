"""What the pipeline debug taps cost on the pipeline's path (RFC §12.3.15, RK-12.3.15-020).

The RFC says tap writes SHOULD be non-blocking, written off the pipeline's
path. roomkit writes each tapped frame where the pipeline processes it, in the
thread that calls ``process_inbound`` (the maintainer chose to leave it so,
2026-11-27). This measures what that costs: the time ``process_inbound`` takes
per 20 ms frame with the taps writing every stage, and without taps, through
the generic pipeline (no provider stage). The taps write into a directory on
the disk the results go to; the RFC gives no figure.
"""

from __future__ import annotations

import shutil
import struct
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from roomkit.voice import AudioFrame, AudioPipeline, AudioPipelineConfig, VoiceSession
from roomkit.voice.pipeline import PipelineDebugTaps

RATES = (8000, 16000)
FRAMES = 15000  # five minutes of 20 ms frames
CASES = tuple(f"{rate}-{taps}" for rate in RATES for taps in ("taps", "no-taps"))


def _frame(rate: int) -> AudioFrame:
    samples = rate // 50
    return AudioFrame(
        data=struct.pack(f"<{samples}h", *([1000] * samples)), sample_rate=rate, channels=1
    )


def _frame_costs(rate: int, frames: int, tap_dir: Path | None) -> list[float]:
    """Microseconds ``process_inbound`` takes for each frame."""
    taps = PipelineDebugTaps(output_dir=str(tap_dir)) if tap_dir is not None else None
    pipeline = AudioPipeline(AudioPipelineConfig(debug_taps=taps))
    session = VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="voice")
    pipeline.on_session_active(session)
    frame = _frame(rate)
    costs = []
    try:
        for _ in range(frames):
            started = time.perf_counter_ns()
            pipeline.process_inbound(session, frame)
            costs.append((time.perf_counter_ns() - started) / 1000)
    finally:
        pipeline.on_session_ended(session)
        pipeline.close()
    return costs


def measure(
    frames: int = FRAMES, disk: Path | None = None
) -> Callable[[], dict[str, list[float]]]:
    """One pass: each rate with the taps writing into *disk*, then without."""

    def run() -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for rate in RATES:
            tap_dir = Path(tempfile.mkdtemp(prefix="media-taps-", dir=disk))
            try:
                out[f"{rate}-taps"] = _frame_costs(rate, frames, tap_dir)
            finally:
                shutil.rmtree(tap_dir)
            out[f"{rate}-no-taps"] = _frame_costs(rate, frames, None)
        return out

    return run
