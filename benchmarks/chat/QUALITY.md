# Model quality through RoomKit

The chat suite tests integration contracts. `--suite quality` adds model
evaluations with deterministic grading, synthetic inputs and no model judge.
The default catalog contains seven families with three variants each:

| Family | Task | Oracle |
|---|---|---|
| `invoice` | Line-level discounts, rounding, shipping and tax | Exact rational arithmetic |
| `planning` | Maximize value with two budgets, dependencies and exclusions | Exhaustive enumeration, unique optimum |
| `documents` | Resolve expired/draft policies among 75 documents, cite evidence, leave missing information null | Generated authoritative facts |
| `memory_updates` | Reconcile three turns of corrections while retaining unchanged constraints | Final structured state |
| `tool_injection` | Read a case despite malicious instructions embedded in tool data | Correct fields, no attempted write/export, no synthetic canary disclosure |
| `refund_decision` | Inspect an order/policy, refund only when eligible, avoid duplicate operations | Recorded tool actions and synthetic backend state |
| `sql_aggregation` | Aggregate paid orders and approved refunds without join fanout | Execute on two hidden SQLite data fixtures |

All tool writes are local synthetic operations. The SQL grader uses a private
in-memory database, a read-only authorizer, a function allowlist, bounded VM
instructions and bounded results. It does not execute generated Python, run a
shell, access external databases or load extensions.

```bash
uv run python -m benchmarks.chat --suite quality --list

uv run --extra cerebras python -m benchmarks.chat --suite quality \
  --key-file ~/.secrets/cerebras --model qwen-3.8-27b \
  --reasoning-effort none --max-tokens 4096 --repetitions 2 \
  --output benchmark-results/quality-none

uv run --extra cerebras python -m benchmarks.chat --suite quality \
  --key-file ~/.secrets/cerebras --model qwen-3.8-27b \
  --reasoning-effort low --max-tokens 4096 --repetitions 2 \
  --output benchmark-results/quality-low \
  --compare benchmark-results/quality-none/results.json
```

That produces 42 attempts per reasoning setting: 21 case definitions, twice each.
`--seed` fixes generated inputs and shuffled scenario order. `--variants` changes
the number of variants per family. Use the same seed/variants/repetitions when
comparing configurations. More repeats on the same case measure consistency,
not breadth. These small synthetic tasks do not establish a general model
ranking, a standard public benchmark score or production reliability.
With seed 42, the three SQL variants share the same prompt but use distinct
hidden rows; there are 19 distinct prompt sequences across the 21 cases.

`quality.csv` and the model-quality section in `report.md` contain exact-case
success counts, partial scores, JSON-format compliance, infrastructure errors
and latency across **all graded answers**, including incorrect ones. The common
chat latency table still includes only passed samples. Unfinished cases retain
their errors, remain in the attempts denominator and have no quality score.

A case is fully correct only if every criterion passes. Partial score is the
fraction of passed criteria for that case, then averaged within its family.
`task_correct` excludes the two formatting criteria (`json_only`, `schema_keys`);
`format_compliant` requires both. These columns distinguish a correct answer
wrapped in Markdown from valid JSON containing wrong values. `truncated` counts
cases where a provider call ended with `finish_reason=length`. On Cerebras the
completion-token cap includes reasoning and answer, so increasing reasoning can
exhaust the budget before visible text appears. Retest larger budgets separately
instead of replacing the original score.
Criteria are not interchangeable business risks: always inspect individual
failures, especially unauthorized actions. JSON fences lose the format point
but can receive semantic credit. Missing fields do not earn credit for null
answers, duplicate keys are rejected, and integers must be JSON integers.

Every graded sample retains its prompts, expected answer, actual final-turn
answer, criterion decisions and tool actions under `metrics.details.quality`
in `results.json`. Earlier turns remain part of the measured scenario latency.
SQL correctness means passing the two hidden fixtures; it is not a formal
proof that the query works for every database. Reasoning length is recorded
from provider events; private reasoning text is not exported.

Keep model mistakes in the results. Fix an oracle only when a separate test or
manual calculation proves it wrong, document that change and rerun affected
cases. The oracle tests in `tests/test_chat_quality.py` include hand-calculated
values and deliberately wrong queries/actions to check false positives.
