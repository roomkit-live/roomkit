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
