"""Regression: async pipeline callbacks survive running off the event loop.

With ``inbound_dsp_threads`` set, the stage chain runs on ``roomkit-dsp``
pool workers — threads with no running event loop. ``_maybe_schedule`` used
to *drop* any coroutine a callback returned there ("Async callback returned
outside event loop"), which unplugged exactly the consumers that matter in a
realtime session — the provider's audio feed and the audio-level hooks —
while every sync callback kept working, so the pipeline looked alive.

The fix: ``AudioPipeline`` captures its home loop at construction and
``_maybe_schedule`` sends off-loop coroutines there via
``run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.engine import AudioPipeline, _maybe_schedule


def _session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(id=sid, room_id="r1", participant_id=f"p-{sid}", channel_id="c1")


def _frame(value: int = 1000) -> AudioFrame:
    return AudioFrame(
        data=value.to_bytes(2, "little", signed=True) * 160,
        sample_rate=16000,
        channels=1,
        sample_width=2,
    )


# --- _maybe_schedule unit behaviour -------------------------------------------


async def test_offloop_coroutine_is_sent_to_the_home_loop() -> None:
    loop = asyncio.get_running_loop()
    landed = asyncio.Event()

    async def coro() -> None:
        landed.set()

    await asyncio.to_thread(_maybe_schedule, coro(), loop)
    await asyncio.wait_for(landed.wait(), timeout=2.0)


async def test_offloop_coroutine_without_home_loop_is_closed_not_leaked() -> None:
    ran = False

    async def coro() -> None:
        nonlocal ran
        ran = True  # pragma: no cover - must never execute

    # Pre-fix behaviour, still the only honest option with nowhere to send it.
    await asyncio.to_thread(_maybe_schedule, coro(), None)
    assert ran is False


async def test_onloop_coroutine_still_runs_as_a_task() -> None:
    landed = asyncio.Event()

    async def coro() -> None:
        landed.set()

    _maybe_schedule(coro())
    await asyncio.wait_for(landed.wait(), timeout=2.0)


# --- through the pipeline ------------------------------------------------------


async def test_async_processed_frame_callback_survives_a_worker_thread() -> None:
    """The exact realtime shape: process_inbound on a pool thread, async consumer."""
    pipeline = AudioPipeline(AudioPipelineConfig())  # captures this loop as home
    received = asyncio.Event()

    async def on_frame(session: VoiceSession, frame: AudioFrame) -> None:
        received.set()

    pipeline.on_processed_frame(on_frame)

    await asyncio.to_thread(pipeline.process_inbound, _session(), _frame())
    await asyncio.wait_for(received.wait(), timeout=2.0)


def test_pipeline_built_outside_async_context_has_no_home_loop() -> None:
    pipeline = AudioPipeline(AudioPipelineConfig())
    assert pipeline._home_loop is None


async def test_pipeline_built_on_the_loop_remembers_it() -> None:
    pipeline = AudioPipeline(AudioPipelineConfig())
    assert pipeline._home_loop is asyncio.get_running_loop()
