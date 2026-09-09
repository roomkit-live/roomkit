# Chat E2E benchmark

Run repeatable, assertion-based chat scenarios through RoomKit's inbound
pipeline, AIChannel, tool loop, store and WebSocket delivery callbacks. The
suite lives outside the installed package and uses synthetic data only.

From the repository root:

```bash
# Inspect the scenario catalog; no credential required.
uv run python -m benchmarks.chat --list

# Offline smoke checks; model-dependent scenarios are explicitly skipped.
uv run python -m benchmarks.chat --provider mock --output benchmark-results/mock

# Cerebras Qwen 3.8: five repetitions of every scenario.
uv run --extra cerebras python -m benchmarks.chat \
  --key-file ~/.secrets/cerebras --model qwen-3.8-27b \
  --repetitions 5 --output benchmark-results/qwen-memory

# Same suite against persistent SQLite (stdlib; no database server).
uv run --extra cerebras python -m benchmarks.chat \
  --key-file ~/.secrets/cerebras --store sqlite --repetitions 5 \
  --output benchmark-results/qwen-sqlite \
  --compare benchmark-results/qwen-memory/results.json

# Longer sampling of selected paths.
uv run --extra cerebras python -m benchmarks.chat \
  --key-file ~/.secrets/cerebras --repetitions 20 \
  --scenarios direct_text,direct_stream,chat_text,chat_stream,long_stream \
  --output benchmark-results/qwen-baselines
```

Credentials may instead come from `CEREBRAS_API_KEY`. A key file accepts a raw
key or `CEREBRAS_API_KEY=...`; it is read without shell evaluation. Keys and
their file paths are excluded from results. Outputs are gitignored. Choose a
fresh output directory for each run; previous results are never overwritten.
Real runs make billable API calls. The default uses one warm-up, temperature 0,
`reasoning_effort=none`, 1,024 maximum output tokens, no SDK retries, and a
90-second scenario deadline. The reasoning scenario explicitly overrides
effort to `low`; the retry scenario injects one recoverable failure.

`--provider openai --model MODEL --base-url URL` also supports services using
RoomKit's OpenAI adapter. Set `OPENAI_API_KEY` or pass `--key-file`. Provider
capabilities and accepted reasoning values vary; failures are recorded rather
than silently changing the request. Other native adapters can be added in
`make_provider()`.

## Outputs and measurements

- `results.json`: configuration, source fingerprint, package versions, warm-ups,
  every sample, checks, provider calls, usage, terminal markers and aggregates.
- `samples.jsonl`: redacted completed samples checkpointed during execution.
- `summary.csv`: per-scenario success counts, medians and interpolated p95.
- `report.md`: readable results, coverage, failures and optional comparison.

The command exits nonzero when any selected scenario fails. Mock mode skips
live-model scenarios and reports those skips. Latency statistics use successful
samples only; failures remain in the success denominator and token totals.
Warm-up usage is separate from scenario totals. Five samples provide a useful
smoke benchmark, not a stable production p95 or an SLA.

Each sample creates fresh RoomKit state and rooms, reusing one provider HTTP
client across the suite. Order is shuffled per repetition using `--seed`.
Initial setup and the final subscriber drain are excluded from timing; actions
and assertions inside a scenario are included. Concurrent-room setup is part
of that scenario. First text is measured at the WebSocket callback, per turn,
and excludes reasoning. Buffered responses have no first-text metric.

Provider wait intervals measure the async `generate()` call or each awaited
stream item. They exclude time a streamed item spends in the consumer. Their
union avoids double-counting concurrent requests. `residual_ms` subtracts the
union of provider waits and synthetic external-tool execution intervals from
elapsed time: it includes framework work, storage, hooks, scheduling and
instrumentation. Built-in skill tools and their script execution remain in the
residual. This is not a CPU profile. Direct-provider prompts also omit the
AIChannel system/history wrapper, so subtracting separate run medians is not
a causal measurement of RoomKit overhead.

The delivery boundary is an in-process WebSocket callback. There is no real
network WebSocket server, browser, or distributed load generator. Repeated
prompts may benefit from the provider's automatic cache. Raw cache usage is
retained; runs do not flush or control that cache. Compare equivalent versions,
configurations, hardware and sample sizes, and inspect upstream variability.

## Current coverage

| Area | Scenarios / checks |
|---|---|
| Generation | Direct buffered/streamed calls, full chat, 100-number stream, reasoning isolation |
| Tools | Parallel calls with overlapping execution, dependent receipt propagation, error recovery |
| Hooks | Inbound blocking, argument rewriting, result override, denied execution, response transcript/usage/counts |
| Skills | Activation, reference, gated inventory tool, real allowlisted Python script, follow-up reuse, room isolation |
| Discovery | Deferred 13-tool catalog through `find_tools` |
| Concurrency | Four rooms sharing one channel/provider, actor context and socket isolation |
| Storage | Memory or SQLite, sequential indices, four concurrent idempotent copies |
| Permissions | Read-only input, hidden input, muted buffered tools, muted-stream history |
| Memory | Sliding-window eviction of an old marker |
| Realtime | Tool composition hides arguments and terminates; start/end pairing; bounded previews; typing, presence, reactions, receipts |
| Resilience | Injected provider 503 with retry; explicit round cap |
| Path parity | Parallel tools, chained tools and hook overrides in buffered and streaming modes |

This initial 27-scenario catalog does **not** cover every chat feature.
Remaining work includes real WebSocket load and slow consumers; MCP; HITL;
orchestration, delegation and handoffs; cancellation, deadlines and fallback;
chain-depth limits; thread semantics; structured outputs and rich content;
context compaction; lifecycle/timers; transport retries/rate limits; identity
resolution backends; telemetry exporters; Postgres/Redis and multiple processes.
Voice, video and conference features require separate suites. Existing unit
and integration tests remain necessary; this benchmark is not an RFC
conformance certification.

## Extend the suite

For model evaluations, see the separate [quality suite](QUALITY.md). It tests
reasoning and task correctness through the same RoomKit pipeline, using
deterministic oracles and retaining incorrect answers for inspection.

Add an async function taking `Harness`, perform public RoomKit actions, and
record explicit contracts with `h.check(name, condition)`. Register a `Scenario`
in `scenarios.py` with feature tags and a description. Use `options_factory`
for stateful memory, skill or executor instances; it runs outside measurement.
Use `record_handler` for timed synthetic tools. Mark fault injection and
offline support honestly. Failed assertions are accumulated, not discarded.

The bundled skill executor accepts only its own calculation fixture and runs
it with an empty environment and a three-second deadline. A model cannot
select arbitrary executable files. Real MCP/external integration fixtures
should receive similarly scoped capabilities.

Validate framework changes with `make all`. The benchmark's offline contracts
also run in `tests/test_chat_benchmark.py` without credentials or network.
