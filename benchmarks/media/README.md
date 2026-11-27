# Media measurements

Some RFC rules hold of a duration on the real clock, or of a signal's
quality, which no conformance scenario can read: a scenario on the virtual
clock shows what a timed path does once a duration passed, never how long the
work really took. These measurements give those numbers. Each drives mocks
that do no work, or the core's generic stages, never a provider: a vendor's
latency, queue or resampler is its provider's to measure, in its own
repository.

| Kind | RFC unit | What it times | The RFC's figure |
|---|---|---|---|
| `close` | RK-12.10.4-056 | `kit.close()` of a kit holding one piece of work it has to cut: a synchronous `BEFORE_BROADCAST` hook, or a model turn's tool call, that honours its cancellation or ignores it | the grace, plus the turn backend's budget, plus the channels' closes, each on its own budget, plus at most two unwinds of 1 s, which only work that ignores its cut spends |
| `turn` | RK-12.3.12-022 | from a transcript's `ON_TRANSCRIPTION` hook to its message's `BEFORE_BROADCAST` hook, with a turn detector that answers at once (`MockTurnDetector`) and without one | 200 ms at most added on the fast path |
| `taps` | RK-12.3.15-020 | `AudioPipeline.process_inbound` per 20 ms frame, with `PipelineDebugTaps` writing every stage and without | none |
| `resample` | RK-12.3.3-007 | the default resampler (`NumpyResamplerProvider`, and `LinearResamplerProvider` without NumPy), frame by frame: THD+N of a 1 kHz tone, the alias a downsampling leaves, the image an upsampling leaves, µs per frame | none, and no threshold is decided |

roomkit-rs runs the same `close` cases (its `benchmarks/`).

```bash
# Three passes, pinned to one performance core.
taskset -c 2 uv run python -m benchmarks.media close
taskset -c 2 uv run python -m benchmarks.media turn
taskset -c 2 uv run python -m benchmarks.media taps
taskset -c 2 uv run python -m benchmarks.media resample
```

Each run writes `results.json` (the environment, every pass and every sample)
and `summary.md` into a fresh `--output` directory, by default
`benchmark-results/media-<kind>-<time>/`, which git ignores, and exits
non-zero when a figure the RFC gives fails. A pass records the CPU time other
processes spent on the measured cores meanwhile (`/proc/stat`, as
`benchmarks/rtp` does), and runs again, twice at most, when it is above 10 %
of a core. Quantiles are nearest rank over the passes' samples.

- `close` gives the kit a 50 ms `close_grace`; the unwinds are the RFC's 1 s.
  The bound adds Python's budgets: the turn backend in the owner's process is
  given one unwind, an agent's close waits for its turns 5 s at most, the
  transport's close waits for nothing. The sum bounds the close's waits: its
  own work between them (releasing the store, closing a channel that waits for
  nothing) is allowed 50 ms on top (decided 2026-11-27). A case passes when its
  slowest close is within the bound so read and, for work that honours its cut,
  spends no unwind.
- `turn`: the detector path's slowest turn bounds what the detector adds,
  which is what passes or fails the 200 ms.
- `taps` writes the taps into a directory under the output, on the disk the
  results go to.

## Results

Compare runs from one machine only.

### 2026-11-27, i7-13700KF, Python 3.13.3

Commit b70b004a3, one performance core (`taskset -c 2`), Linux 7.0, three
passes each; other load on the core under 10 % of it in every pass, none run
again. A first run of `taps` and `resample` took twice the time per frame
with no other load counted on the core: its hyperthread sibling (CPU 3), which
the record does not read, was busy. Both were run again; their second run is
the one given.

**`close`**: pass.

| Case | n | p50 (s) | p99 (s) | max (s) | Bound (s) |
|---|---:|---:|---:|---:|---:|
| hook-honours | 60 | 0.050 | 0.053 | 0.053 | 3.05 |
| hook-ignores | 12 | 2.052 | 2.054 | 2.054 | 3.05 |
| tool-honours | 60 | 0.001 | 0.001 | 0.001 | 8.05 |
| tool-ignores | 6 | 7.057 | 7.060 | 7.060 | 8.05 |

Work that honours its cut ends with the grace (a model turn is cut at once),
spending no unwind; a hook that ignores it holds the close for the grace and
the two unwinds; a tool that ignores it, for the grace, two unwinds and the
agent's 5 s close. roomkit-rs, on the same cases (its `benchmarks/`), takes as
long where the work honours its cut, but does not wait for a hook that blocks
its thread (0.052 s), and waits one unwind more for a tool that does
(8.056 s, 6 ms of it the close's own work): each within the bound and the
50 ms of the close's own work. Aligning the two is carded (RMK-2162).

**`turn`**: pass, 2.471 ms at most against 200 ms.

| Case | n | p50 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| with the detector | 600 | 0.293 | 1.130 | 2.471 |
| without | 600 | 0.187 | 0.609 | 24.144 |

The detector adds 0.106 ms at the median.

**`taps`**: no figure. Per frame, written on the disk.

| Rate (Hz) | Taps | n | p50 (µs) | p99 (µs) | max (µs) |
|---|---|---:|---:|---:|---:|
| 8000 | every stage | 45000 | 12.0 | 17.6 | 264.6 |
| 8000 | none | 45000 | 1.2 | 1.5 | 22.9 |
| 16000 | every stage | 45000 | 12.1 | 18.4 | 185.5 |
| 16000 | none | 45000 | 1.2 | 1.5 | 38.2 |

The taps add about 11 µs per frame at the median, four stage writes on the
pipeline's path.

**`resample`**: no figure, no threshold. The quality figures are the same for
both resamplers (one algorithm) and in every pass.

| Pair (Hz) | THD+N (dB) | Alias (dB) | Image (dB) | NumPy p50 (µs/frame) | Pure Python p50 (µs/frame) |
|---|---:|---:|---:|---:|---:|
| 8000 → 16000 | -18.6 | | -31.2 | 10.4 | 65.2 |
| 16000 → 8000 | -18.7 | -3.0 | | 10.4 | 40.2 |
| 24000 → 16000 | -28.4 | -3.8 | | 10.9 | 79.4 |
| 48000 → 16000 | -22.4 | -2.1 | | 11.9 | 100.1 |
| 16000 → 48000 | -22.4 | | -41.4 | 12.8 | 197.1 |
| 44100 → 16000 | -22.8 | -2.3 | | 12.1 | 96.5 |

Each 20 ms frame is converted on its own, and nothing is filtered before a
downsampling: a tone above the target's Nyquist frequency comes out 2 to 4 dB
below its input level, as an alias inside the band.
