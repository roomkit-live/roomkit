"""Stress lane: the inbound DSP offload's ceiling claim, measured.

The offload exists for one reason — the stage chain used to run on a
single thread, so the session ceiling was one core. This bench feeds S
streams × F frames of GIL-releasing work (``time.sleep`` stands in for
native DSP, which releases the GIL the same way) through the offload and
through a plain serial loop, and asserts the pool actually buys the
parallelism it promises.
"""

from __future__ import annotations

import time

import pytest

from roomkit.voice.pipeline.offload import InboundFrameOffload

pytestmark = pytest.mark.stress

_STREAMS = 4
_FRAMES = 40
_FRAME_COST_S = 0.002  # 2 ms of "native DSP" per frame


def _work() -> None:
    time.sleep(_FRAME_COST_S)


class TestOffloadCeiling:
    def test_the_pool_parallelises_streams(self) -> None:
        serial_start = time.monotonic()
        for _ in range(_STREAMS * _FRAMES):
            _work()
        serial = time.monotonic() - serial_start

        offload = InboundFrameOffload(_STREAMS, max_queued_frames=_FRAMES + 1)
        pool_start = time.monotonic()
        for _ in range(_FRAMES):
            for s in range(_STREAMS):
                offload.submit(f"s{s}", _work)
        assert offload.wait_idle(timeout=30.0)
        pooled = time.monotonic() - pool_start
        offload.shutdown()

        # 4 streams on 4 workers should approach 4x; assert a lax 2x so a
        # loaded runner never flakes, and print the real ratio for the log.
        ratio = serial / pooled
        print(f"offload ceiling: serial={serial:.3f}s pooled={pooled:.3f}s ratio={ratio:.1f}x")
        assert ratio > 2.0, f"pool bought only {ratio:.1f}x over serial"

    def test_fifo_holds_under_sustained_load(self) -> None:
        offload = InboundFrameOffload(4, max_queued_frames=10_000)
        seen: dict[str, list[int]] = {f"s{s}": [] for s in range(8)}
        for i in range(500):
            for s in range(8):
                offload.submit(f"s{s}", seen[f"s{s}"].append, i)
        assert offload.wait_idle(timeout=30.0)
        offload.shutdown()
        for stream, order in seen.items():
            assert order == list(range(500)), f"{stream} lost FIFO"
