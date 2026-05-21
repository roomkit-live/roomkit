# Ollama Provider

`OllamaAIProvider` talks to the native Ollama `/api/chat` endpoint
through the official `ollama-python` SDK. Use it when you want
features the OpenAI-compatible shim hides — chiefly the `think`
parameter and streamed `thinking` deltas.

## Install

```bash
pip install roomkit[ollama]
```

## When to use this vs OpenAI-compat

Ollama also exposes an OpenAI-compatible endpoint at
`http://host:11434/v1/chat/completions`, which works fine with
`OpenAIAIProvider` (or `create_vllm_provider`). Pick `OllamaAIProvider`
when **any** of these matter:

| You want | OpenAI-compat | `OllamaAIProvider` |
|---|---|---|
| Disable reasoning on a thinking model (`think=False`) | Silently ignored | Honored |
| Force reasoning on a non-default model (`think=True`) | Silently ignored | Honored |
| Stream reasoning tokens to a UI as they arrive | Returned in a non-streamed `reasoning` field that the SDK consumer doesn't split out | Streamed as `StreamThinkingDelta` events token-by-token |
| Pass `keep_alive`, `num_ctx`, or `num_predict` cleanly | Awkward via `extra_body` | First-class config |

For plain non-reasoning models (`llama3.2`, `gemma2`, etc.) either
provider works.

## Quick start

```python
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig

provider = OllamaAIProvider(OllamaConfig(
    host="http://localhost:11434",
    model="qwen3:8b",
))

# Use exactly like any other AIProvider — pass it to AIChannel, etc.
```

## Config knobs

```python
OllamaConfig(
    host="http://localhost:11434",   # Ollama server
    model="qwen3:8b",                # any pulled model
    max_tokens=None,                 # → options.num_predict
    temperature=0.7,                 # → options.temperature
    timeout=120.0,                   # long: cold-start + reasoning is slow
    think=None,                      # None = model default, True/False = explicit
    keep_alive="5m",                 # how long the model stays loaded
    num_ctx=8192,                    # context window override
)
```

## How `think` is resolved

Per-request precedence (highest first):

1. **`AIContext.thinking_budget`** — if set, this wins:
   - `None` or `0` → `think=False`
   - any `>0` → `think=True`
2. **`OllamaConfig.think`** — fallback when `thinking_budget` is unset.
3. **Omit `think`** — let the model decide its default (reasoning
   models think, others don't).

This makes the "Leave empty to disable" semantics in higher-level
agent configs actually honor themselves: a missing/zero
`thinking_budget` at the agent layer reaches the provider as
`think=False` and the model genuinely skips the reasoning phase
instead of just having its reasoning silently discarded.

## Streamed events

`generate_structured_stream()` yields:

- `StreamThinkingDelta(thinking="...")` — one per `message.thinking`
  delta chunk from Ollama. Arrives **token-by-token**, which is the
  whole reason this provider exists.
- `StreamTextDelta(text="...")` — one per `message.content` delta.
- `StreamToolCall(id, name, arguments)` — collected from
  `message.tool_calls` and yielded after the text/thinking deltas
  for the same chunk (Ollama doesn't fragment tool-call arguments
  across chunks the way OpenAI does).
- `StreamDone(finish_reason, usage)` — terminator.

## History round-trip

The provider preserves `AIThinkingPart` in the message history by
sending it back to Ollama as a top-level `thinking` field on the
assistant message — no `<think>...</think>` tag wrapping needed.
This keeps reasoning models honest across tool-loop rounds: they
see their own prior chain-of-thought when computing the next turn.

## Tool calls

Ollama's tool-call format is essentially OpenAI's, so the standard
`AITool` definitions pass through unchanged. The one quirk: Ollama
doesn't issue stable `id` fields on tool calls, so the provider
synthesizes ones like `call_<name>_<index>` so consumers can pair
calls with their results.

## Interactive test bed

`examples/ollama_cli.py` exercises every knob in one place — `--think`,
`--no-think`, `--stream`, `--no-stream`, `--mcp <url>` for MCP tool
discovery — with a Rich-powered display that labels thinking output
in italics and answer output in bold.

```bash
# Default: model decides whether to think, streams tokens, no tools
uv run python examples/ollama_cli.py --model qwen3:8b

# Force thinking off — fast, single-pass response
uv run python examples/ollama_cli.py --model qwen3:8b --no-think

# Wire in MCP tools
uv run python examples/ollama_cli.py --model qwen3:8b \
    --mcp http://localhost:8080/mcp
```

At the prompt, `/think on|off` and `/stream on|off` toggle the
behavior mid-session; `/tools` lists what's available; `/reset`
clears history; `/quit` exits.

## Errors

`ollama.ResponseError` maps to `ProviderError(retryable=...)` where
`retryable=True` for `429`, `500`, `502`, `503`. Transport/connection
errors (timeouts, refused connections) get marked retryable so the
calling `RetryPolicy` decides whether to act.

## What this provider doesn't do

- **Image input**: `supports_vision` reports `True` and image parts
  reach the wire, but it's the server-side model that decides whether
  to honor them. Unsupported models 400 the request — RoomKit
  surfaces that as a non-retryable `ProviderError`.
- **Embeddings**: out of scope. Use `ollama.AsyncClient().embeddings()`
  directly if you need them.
