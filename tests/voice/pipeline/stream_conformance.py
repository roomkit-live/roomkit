"""Shared conformance check for per-stream pipeline stages.

A pipeline stage receives ``process(frame, stream)`` and MUST keep its state
under that key.  Nothing in the type system enforces it: a provider can accept
``stream``, ignore it, and silently mix two speakers into one detection state —
the exact defect this contract exists to remove.  This module is the only net
against that, so every stage in the repo is checked with it and third-party
implementers can run it against their own::

    from tests.voice.pipeline.stream_conformance import (
        assert_stage_keeps_state_per_stream,
    )

    def test_my_denoiser_is_per_stream():
        assert_stage_keeps_state_per_stream(MyDenoiser, _make_frame)

It verifies **one** thing: one stream's traffic — including its ``reset()`` —
must not change what another stream sees.  Concretely, a stream's output
sequence is identical whether it runs alone or interleaved with a second,
noisier stream.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class _Stage(Protocol):
    """The slice of a stage ABC this check relies on."""

    def process(self, frame: Any, stream: str) -> Any: ...

    def reset(self, stream: str) -> None: ...


def _fingerprint(result: Any) -> Any:
    """Reduce a stage result to something comparable across runs.

    Frames compare by payload; events are dataclasses, so ``repr`` covers
    every field without this helper needing to know the stage type.
    """
    if result is None:
        return None
    data = getattr(result, "data", None)
    if data is not None:
        return ("frame", bytes(data))
    return ("event", repr(result))


def assert_stage_keeps_state_per_stream(
    make_stage: Callable[[], _Stage],
    make_frame: Callable[[int], Any],
    *,
    frames: int = 6,
    noise_per_frame: int = 2,
) -> None:
    """Assert a stage keeps its state per stream.

    Runs the same frame sequence on stream ``"bob"`` twice: once alone, once
    interleaved with traffic and resets on stream ``"alice"``.  The two output
    sequences must match.  They diverge exactly when the stage shares state —
    when Alice's silence advances Bob's hangover, when her audio lands in his
    context window, or when resetting her drops his buffer.

    Args:
        make_stage: Builds a fresh stage instance. Called twice, so it must not
            hand back a shared singleton.
        make_frame: Builds the i-th frame. Vary the payload if the stage's
            output depends on it; a constant frame is fine otherwise.
        frames: How many frames stream "bob" sends.
        noise_per_frame: Frames stream "alice" sends before each of Bob's.
    """
    # Baseline: bob alone.
    alone = make_stage()
    expected = [_fingerprint(alone.process(make_frame(i), "bob")) for i in range(frames)]

    # Same sequence for bob, with alice talking over him and leaving twice.
    shared = make_stage()
    actual = []
    for i in range(frames):
        for n in range(noise_per_frame):
            shared.process(make_frame(1000 + i * noise_per_frame + n), "alice")
        if i in (2, 4):
            # Alice hangs up mid-call: releasing her state must not touch his.
            shared.reset("alice")
        actual.append(_fingerprint(shared.process(make_frame(i), "bob")))

    assert actual == expected, (
        "stage shares state between streams: stream 'bob' produced a different "
        "sequence once stream 'alice' was interleaved.\n"
        f"  alone:      {expected}\n"
        f"  interleaved:{actual}\n"
        "The stage accepts the `stream` argument but does not key its state on "
        "it — see AudioPipeline stage contract."
    )
