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

Sync callbacks go home too: the channels' handlers are loop code (asyncio
queues, tasks, ``get_running_loop``), so every callback a chain fires off the
loop runs on the home loop, in the order the chain fired them (RMK-392).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.engine import AudioPipeline, _maybe_schedule
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.pipeline.vad.mock import MockVADProvider


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


# --- sync callbacks of an off-loop chain ---------------------------------------


def _utterance_pipeline() -> AudioPipeline:
    """A pipeline whose three frames start, carry and end one utterance."""
    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x01\x00" * 160),
        ]
    )
    return AudioPipeline(AudioPipelineConfig(vad=vad))


def _record_callbacks(pipeline: AudioPipeline, log: list[tuple[str, Any]]) -> None:
    """Log each callback's kind and the thread it ran on."""

    def recorder(kind: str) -> Any:
        def record(session: VoiceSession, payload: Any) -> None:
            log.append((kind, threading.get_ident()))

        return record

    pipeline.on_vad_event(recorder("vad"))
    pipeline.on_speech_frame(recorder("speech_frame"))
    pipeline.on_speech_end(recorder("speech_end"))
    pipeline.on_processed_frame(recorder("processed"))


async def test_a_worker_chain_calls_back_on_the_home_loop_in_chain_order() -> None:
    inline_log: list[tuple[str, Any]] = []
    inline = _utterance_pipeline()
    _record_callbacks(inline, inline_log)
    for _ in range(3):
        inline.process_inbound(_session(), _frame())

    worker_log: list[tuple[str, Any]] = []
    pipeline = _utterance_pipeline()  # built here: this loop is its home
    _record_callbacks(pipeline, worker_log)
    frame_done = asyncio.Event()

    def processed(session: VoiceSession, frame: AudioFrame) -> None:
        frame_done.set()

    # Registered last, after the recorder. This frame's callbacks have now
    # reached the home loop in full when its processed notification arrives.
    pipeline.on_processed_frame(processed)
    for _ in range(3):
        frame_done.clear()
        await asyncio.to_thread(pipeline.process_inbound, _session(), _frame())
        await asyncio.wait_for(frame_done.wait(), timeout=2.0)

    loop_thread = threading.get_ident()
    assert [kind for kind, _ in worker_log] == [kind for kind, _ in inline_log]
    assert {thread for _, thread in worker_log} == {loop_thread}


def test_without_a_home_loop_a_worker_chain_calls_back_where_it_runs() -> None:
    pipeline = _utterance_pipeline()  # built outside async context
    log: list[tuple[str, Any]] = []
    _record_callbacks(pipeline, log)
    worker = threading.Thread(target=pipeline.process_inbound, args=(_session(), _frame()))
    worker.start()
    worker.join()

    assert [kind for kind, _ in log] == ["vad", "speech_frame", "processed"]
    assert {thread for _, thread in log} == {worker.ident}


def test_a_closed_home_loop_drops_the_callbacks_without_failing_the_chain() -> None:
    pipeline = _utterance_pipeline()
    closed = asyncio.new_event_loop()
    closed.close()
    pipeline._home_loop = closed  # the loop the pipeline was built on is gone
    log: list[tuple[str, Any]] = []
    _record_callbacks(pipeline, log)

    pipeline.process_inbound(_session(), _frame())  # must not raise

    assert log == []
