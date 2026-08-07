#!/usr/bin/env python3
"""Offline AEC bench — measure echo cancellation instead of guessing at it.

Consumes an ``aec-dump/1`` capture (see RoomKit UI's ``AECDumpRecorder``):
a JSONL event stream preserving, in arrival order, every reference frame
the echo canceller was fed and every capture frame with its processed
output.  Because ordering is preserved, a replay through a fresh provider
reproduces the live run bit-for-bit — and a replay with different settings
(a delay hint, noise suppression) measures a proposed fix on the exact
audio that exposed the problem.

Commands:

  report <dump-dir>                 metrics of the recorded (live) run
  replay <dump-dir> [--delay-ms N] [--ns]
                                    re-run the capture through a fresh
                                    WebRTC provider and compare with the
                                    recorded output side by side

The metric is per-window attenuation over 100 ms windows, reported only
for *echo-active* windows (trailing reference RMS above a floor): during
silence there is nothing to cancel, and during doubletalk low attenuation
is correct, so the summary also names the count of probable-doubletalk
windows (capture well above reference) rather than scoring them.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Windowing: score in buckets of this many capture frames (pipeline frames
# are typically 10 or 20 ms — durations are derived from the data length).
WINDOW_FRAMES = 10
# A window this close before an AEC deactivation toggle sits on a barge-in:
# the interrupt path deactivates cancellation when playback is flushed, so
# the user was speaking over the response — doubletalk by construction.
BARGE_IN_GUARD_S = 1.0
# Trailing reference RMS above this int16 level marks a window echo-active.
REF_ACTIVE_RMS = 200.0
# Echo-active windows whose capture is quieter than this carry nothing
# measurable (the acoustic echo already faded) — excluded from scoring.
IN_FLOOR_RMS = 150.0
# Capture this much louder than the dump's typical echo level smells like
# doubletalk (user speaking over playback).  Relative to the dump's own
# echo median because the reference RMS is a *digital* level while the
# capture RMS is *acoustic* — the two scales are incommensurable.
DOUBLETALK_RATIO = 1.6
# An echo-active window attenuated less than this is a leak.
LEAK_THRESHOLD_DB = 15.0
# Attenuation reported for a perfectly silenced window (log of zero).
ATTENUATION_CAP_DB = 90.0


@dataclass
class Event:
    kind: str  # "ref" | "cap" | "act"
    ns: int
    data: bytes = b""  # ref payload, or capture input
    out: bytes | None = None  # capture output (kind == "cap")
    active: bool | None = None  # activation toggle (kind == "act")


@dataclass
class Dump:
    sample_rate: int
    channels: int
    sample_width: int
    stream: str
    provider: str
    events: list[Event]


@dataclass
class Window:
    index: int
    start_s: float
    ref_rms: float
    in_rms: float
    out_rms: float
    near_barge_in: bool = False

    @property
    def attenuation_db(self) -> float:
        if self.in_rms <= 0:
            return 0.0
        if self.out_rms <= 0:
            return ATTENUATION_CAP_DB
        return min(ATTENUATION_CAP_DB, 20.0 * math.log10(self.in_rms / self.out_rms))

    @property
    def echo_active(self) -> bool:
        return self.ref_rms >= REF_ACTIVE_RMS


@dataclass
class Classified:
    """Windows sorted into the four regimes the summary reports."""

    active: list[Window]  # echo present, user (probably) silent — scored
    quiet: list[Window]  # no reference playing
    faint: list[Window]  # echo-active but capture too quiet to measure
    doubletalk: list[Window]  # user probably speaking over playback — unscored


def classify(windows: list[Window]) -> Classified:
    """Split windows by regime, self-calibrated on the dump's echo level.

    Doubletalk is recognized two ways.  Level: judged against the median
    capture level of the plainly echo-active windows — playback time
    vastly outweighs overlap time in a real session, so that median is
    the echo's acoustic level, and a window well above it carries the
    user's voice on top.  Structure: a window sitting just before an AEC
    deactivation toggle is a barge-in moment by construction (the
    interrupt path deactivates cancellation when playback is flushed), so
    the user was speaking whatever its level — low attenuation there is
    the canceller *protecting* the user's voice, not leaking echo.
    """
    echo = [w for w in windows if w.echo_active and w.in_rms >= IN_FLOOR_RMS]
    quiet = [w for w in windows if not w.echo_active]
    faint = [w for w in windows if w.echo_active and w.in_rms < IN_FLOOR_RMS]
    if not echo:
        return Classified(active=[], quiet=quiet, faint=faint, doubletalk=[])
    levels = sorted(w.in_rms for w in echo)
    echo_median = levels[len(levels) // 2]
    doubletalk = [w for w in echo if w.near_barge_in or w.in_rms > DOUBLETALK_RATIO * echo_median]
    active = [w for w in echo if w not in doubletalk]
    return Classified(active=active, quiet=quiet, faint=faint, doubletalk=doubletalk)


def load_dump(path: Path) -> Dump:
    """Parse an ``aec-dump/1`` events.jsonl (*path* is the file or its dir)."""
    if path.is_dir():
        path = path / "events.jsonl"
    with path.open(encoding="utf-8") as f:
        header = json.loads(f.readline())
        if header.get("format") != "aec-dump/1":
            raise SystemExit(f"not an aec-dump/1 file: {path}")
        events: list[Event] = []
        for line in f:
            row = json.loads(line)
            if row["t"] == "ref":
                events.append(Event("ref", row["ns"], base64.b64decode(row["d"])))
            elif row["t"] == "act":
                events.append(Event("act", row["ns"], active=bool(row["a"])))
            else:
                events.append(
                    Event(
                        "cap",
                        row["ns"],
                        base64.b64decode(row["i"]),
                        base64.b64decode(row["o"]),
                    )
                )
    return Dump(
        sample_rate=header["sample_rate"],
        channels=header["channels"],
        sample_width=header["sample_width"],
        stream=header.get("stream", "s"),
        provider=header.get("provider", "unknown"),
        events=events,
    )


def _rms(pcm: bytes) -> float:
    import numpy as np

    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0


def measure(dump: Dump, outputs: list[bytes] | None = None) -> list[Window]:
    """Score a run in 100 ms windows.

    *outputs* overrides the recorded capture outputs (a replay's results);
    ``None`` scores the recorded live run.  The trailing-reference RMS is
    carried along the event walk, so windows are judged against what was
    actually playing at that moment — interleaving is the whole point of
    the dump format.
    """
    windows: list[Window] = []
    cap_index = 0
    ref_tail: list[float] = []
    bucket_in: list[bytes] = []
    bucket_out: list[bytes] = []
    bucket_ref_rms: list[float] = []
    bucket_start_ns = 0
    # Wall-clock anchoring: the pipeline only routes capture through the
    # AEC while it is active, so frame counting compresses the session's
    # idle stretches.  The recorded ns timestamps are what line up with
    # the application's logs.
    origin_ns = dump.events[0].ns if dump.events else 0

    def close_bucket() -> None:
        nonlocal bucket_in, bucket_out, bucket_ref_rms
        if not bucket_in:
            return
        guard_ns = int(BARGE_IN_GUARD_S * 1e9)
        windows.append(
            Window(
                index=len(windows),
                start_s=(bucket_start_ns - origin_ns) / 1e9,
                ref_rms=max(bucket_ref_rms) if bucket_ref_rms else 0.0,
                in_rms=_rms(b"".join(bucket_in)),
                out_rms=_rms(b"".join(bucket_out)),
                near_barge_in=any(0 <= d - bucket_start_ns <= guard_ns for d in deactivations),
            )
        )
        bucket_in, bucket_out, bucket_ref_rms = [], [], []

    deactivations = [e.ns for e in dump.events if e.kind == "act" and e.active is False]

    refs_since_cap = 0
    for event in dump.events:
        if event.kind == "act":
            continue
        if event.kind == "ref":
            refs_since_cap += 1
            ref_tail.append(_rms(event.data))
            if len(ref_tail) > WINDOW_FRAMES:
                ref_tail.pop(0)
            continue
        # Reference and capture tick at the same 10 ms cadence: a capture
        # frame with no reference since the last one means playback went
        # silent — decay the tail, or a quiet stretch after a response
        # would stay "echo-active" forever.
        if refs_since_cap == 0:
            ref_tail.append(0.0)
            if len(ref_tail) > WINDOW_FRAMES:
                ref_tail.pop(0)
        refs_since_cap = 0
        out = outputs[cap_index] if outputs is not None else (event.out or b"")
        cap_index += 1
        if not bucket_in:
            bucket_start_ns = event.ns
        bucket_in.append(event.data)
        bucket_out.append(out)
        bucket_ref_rms.append(max(ref_tail) if ref_tail else 0.0)
        if len(bucket_in) >= WINDOW_FRAMES:
            close_bucket()
    close_bucket()
    return windows


@dataclass
class Score:
    active: int
    doubletalk: int
    leaks: int
    median_db: float
    worst: list[Window]

    def line(self, label: str) -> str:
        return (
            f"{label:<10} echo-active windows: {self.active:4d}  "
            f"median attenuation: {self.median_db:6.1f} dB  "
            f"leaks (<{LEAK_THRESHOLD_DB:.0f} dB): {self.leaks:3d}  "
            f"doubletalk (unscored): {self.doubletalk}"
        )


def score(windows: list[Window]) -> Score:
    c = classify(windows)
    attens = sorted(w.attenuation_db for w in c.active)
    median = attens[len(attens) // 2] if attens else 0.0
    leaks = [w for w in c.active if w.attenuation_db < LEAK_THRESHOLD_DB]
    worst = sorted(c.active, key=lambda w: w.attenuation_db)[:5]
    return Score(
        active=len(c.active),
        doubletalk=len(c.doubletalk),
        leaks=len(leaks),
        median_db=median,
        worst=worst,
    )


def replay(dump: Dump, *, delay_ms: int = 0, ns: bool = False) -> list[bytes]:
    """Re-run the recorded capture through a fresh WebRTC provider."""
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

    provider = WebRTCAECProvider(
        sample_rate=dump.sample_rate,
        channels=dump.channels,
        stream_delay_ms=delay_ms,
        enable_ns=ns,
    )
    # Dumps predating the "act" event replay always-active; dumps that
    # carry toggles replay the live pipeline's activation pairing exactly.
    if not any(e.kind == "act" for e in dump.events):
        provider.set_stream_active(dump.stream, True)

    def frame(data: bytes) -> AudioFrame:
        return AudioFrame(
            data=data,
            sample_rate=dump.sample_rate,
            channels=dump.channels,
            sample_width=dump.sample_width,
        )

    outputs: list[bytes] = []
    for event in dump.events:
        if event.kind == "act":
            provider.set_stream_active(dump.stream, bool(event.active))
        elif event.kind == "ref":
            provider.feed_reference(frame(event.data), dump.stream)
        else:
            outputs.append(bytes(provider.process(frame(event.data), dump.stream).data))
    provider.reset(dump.stream)
    return outputs


def _print_report(label: str, windows: list[Window]) -> Score:
    s = score(windows)
    print(s.line(label))
    for w in s.worst:
        print(
            f"    worst @ {w.start_s:7.2f}s  ref_rms={w.ref_rms:6.0f} "
            f"in_rms={w.in_rms:6.0f} out_rms={w.out_rms:6.0f} "
            f"attenuation={w.attenuation_db:6.1f} dB"
        )
    return s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_report = sub.add_parser("report", help="score the recorded live run")
    p_report.add_argument("dump", type=Path)
    p_replay = sub.add_parser("replay", help="re-run through a fresh provider and compare")
    p_replay.add_argument("dump", type=Path)
    p_replay.add_argument("--delay-ms", type=int, default=0)
    p_replay.add_argument("--ns", action="store_true")
    args = parser.parse_args(argv)

    dump = load_dump(args.dump)
    caps = [e for e in dump.events if e.kind == "cap"]
    n_ref = sum(1 for e in dump.events if e.kind == "ref")
    bytes_per_s = dump.sample_rate * dump.channels * dump.sample_width
    capture_s = sum(len(e.data) for e in caps) / bytes_per_s if bytes_per_s else 0.0
    print(
        f"dump: {dump.provider} @ {dump.sample_rate} Hz — "
        f"{len(caps)} capture frames, {n_ref} reference frames "
        f"({capture_s:.1f}s of capture)"
    )

    _print_report("recorded", measure(dump))
    if args.command == "replay":
        outputs = replay(dump, delay_ms=args.delay_ms, ns=args.ns)
        label = f"replay(delay={args.delay_ms},ns={args.ns})"
        _print_report(label, measure(dump, outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
