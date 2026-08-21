# AI Channels

AIChannel connects rooms to LLM providers. When a message is broadcast to an AI channel, it generates a response using conversation history and re-enters it through the inbound pipeline.

## Basic Setup

```python
from roomkit import RoomKit, ChannelCategory
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

kit = RoomKit()

ai = AIChannel(
    "ai-assistant",
    provider=AnthropicAIProvider(AnthropicConfig(
        api_key="sk-ant-...",
        model="claude-opus-5",
    )),
    system_prompt="You are a helpful customer support agent.",
    temperature=0.7,
)
kit.register_channel(ai)

await kit.create_room(room_id="support")
await kit.attach_channel("support", "ai-assistant", category=ChannelCategory.INTELLIGENCE)
```

## AI Providers

| Provider | Class | Config | Extra |
|----------|-------|--------|-------|
| Anthropic (Claude) | `AnthropicAIProvider` | `AnthropicConfig` | `roomkit[anthropic]` |
| OpenAI (GPT) | `OpenAIAIProvider` | `OpenAIConfig` | `roomkit[openai]` |
| Google Gemini | `GeminiAIProvider` | `GeminiConfig` | `roomkit[gemini]` |
| Gemini on Vertex AI | `GeminiVertexProvider` | `GeminiVertexConfig` | `roomkit[gemini]` |
| Mistral | `MistralAIProvider` | `MistralConfig` | `roomkit[mistral]` |
| Azure OpenAI | `AzureAIProvider` | `AzureAIConfig` | `roomkit[azure]` |
| OpenRouter (300+ models) | `OpenRouterAIProvider` | `OpenRouterConfig` | `roomkit[openrouter]` |
| LiteLLM proxy (self-hosted gateway) | `LiteLLMAIProvider` | `LiteLLMConfig` | `roomkit[litellm]` |
| xAI (Grok) | `XAIAIProvider` | `XAIConfig` | `roomkit[xai]` |
| PolarGrid (Canadian-hosted) | `PolarGridAIProvider` | `PolarGridConfig` | `roomkit[polargrid]` |
| vLLM (local) | `create_vllm_provider()` | `VLLMConfig` | `roomkit[vllm]` |
| Ollama (local) | `OllamaAIProvider` | `OllamaConfig` | `roomkit[ollama]` |
| Mock (testing) | `MockAIProvider` | — | built-in |

Provider notes:

- **Explicit model selection** — `model=` is required by `OpenAIConfig` and
  `AnthropicConfig`; upgrading RoomKit therefore cannot silently change cost,
  latency, or model behavior. For the selected model, `OpenAIConfig`
  automatically uses `max_completion_tokens` and omits custom temperature for
  current GPT-5 and o-series ids, while `AnthropicConfig` uses adaptive thinking
  and omits temperature for current Claude reasoning ids. Explicit flags take
  precedence, and a custom `base_url` keeps conservative legacy behavior.

- **Gemini on Vertex AI** (`roomkit.providers.gemini.vertex`) — subclass of `GeminiAIProvider` serving the same Gemini models through a Google Cloud project with a pinned region (`GeminiVertexConfig` requires `project` and `location`, e.g. `"northamerica-northeast1"`; no API key — the identity is `impersonate_service_account`, else `service_account_json`, else ADC). Use it when data residency matters (Québec Law 25 / PIPEDA). Generation, streaming, thinking, and the model catalog are inherited unchanged.
- **OpenRouter** (`roomkit.providers.openrouter`) — subclass of `OpenAIAIProvider` pointed at `https://openrouter.ai/api/v1`; `OpenRouterConfig` subclasses `OpenAIConfig` (adds `site_url`/`app_name` attribution headers) and `model` is a required slug like `"anthropic/claude-sonnet-4.5"`. Reasoning is forwarded to any upstream model via OpenRouter's unified `reasoning` object. Model-listing nuance: OpenRouter's `/models` items omit the `object`/`owned_by` fields the OpenAI SDK expects, so `list_models()` reads the raw JSON instead.
- **LiteLLM proxy** (`roomkit.providers.litellm`) — subclass of `OpenAIAIProvider` pointed at a self-hosted LiteLLM gateway (default `http://localhost:4000`); the extra installs the `openai` SDK, deliberately **not** the `litellm` package (the gateway keeps keys/budgets/routing server-side). `model` is the deployment's public alias and `api_key` a virtual or master key, both required. Reasoning rides LiteLLM's cross-provider normalisation: `reasoning_effort` passes through, `thinking_budget>0` maps to a `thinking` token budget, and the trace comes back in `reasoning_content`; `thinking_budget=0` sends no reasoning params at all (LiteLLM has no disable token every upstream translator accepts — force-off belongs in the proxy's per-model config or `extra_body`). `available_models()` is empty (the model list is the operator's config); `list_models()` reads the proxy's `/model/info` — context window, vision flag, and per-token costs per alias, with load-balanced deployments collapsed and LiteLLM's `0`-for-unknown costs mapped to "unpriced" rather than free.
- **xAI (Grok)** (`roomkit.providers.xai`) — subclass of `OpenAIAIProvider` pointed at `https://api.x.ai/v1`; defaults to `max_completion_tokens` and stream usage. `XAIRealtimeProvider` (Grok speech-to-speech) is a separate import from `roomkit.providers.xai.realtime`.
- **PolarGrid** (`roomkit.providers.polargrid`) — Canadian-hosted inference network (edges in Toronto, Vancouver, Montréal) via the official `polargrid-sdk` async client; OpenAI-shaped chat-completions surface. Supports tool calling, thinking (`PolarGridConfig(thinking=True)` sets the `enable_thinking` request flag; reasoning is surfaced as `AIResponse.thinking` / `StreamThinkingDelta`), and vision (`image_url` content parts). Pin `region` in production when residency matters.

Pick **Ollama** over the OpenAI-compat shim (`OpenAIAIProvider` pointed at `http://host:11434/v1` or `create_vllm_provider()` with an Ollama URL) whenever the model is a reasoning model (DeepSeek-R1, Qwen 3 thinking variants, etc.) — only the native API exposes the `think` parameter and streams the `thinking` field separately from `content`. See `docs/c7/ollama-provider.md` for the full rundown.

```python
# OpenAI
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig

provider = OpenAIAIProvider(OpenAIConfig(api_key="sk-...", model="gpt-4o"))

# Gemini
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig

provider = GeminiAIProvider(GeminiConfig(api_key="...", model="gemini-2.0-flash"))

# Mock (for testing)
from roomkit.providers.ai.mock import MockAIProvider

provider = MockAIProvider(responses=["Hello!", "How can I help?"])
```

## Model Catalog and Pricing

`provider.catalog_entry()` returns the configured model's offline `ModelInfo`.
When it carries `pricing`, `pricing.cost_for(response.usage)` prices fresh
input, output, cache reads and represented cache writes. A `None` cache rate
means no separate per-token charge is represented; it contributes zero rather
than falling back implicitly. Catalogs repeat the input rate explicitly when a
vendor bills a cache counter as ordinary input.

Tiered entries also carry a long-context threshold plus input/output
multipliers. `cost_for()` applies them automatically to GPT-5.6, Gemini Pro and
current Grok usage after total input crosses the vendor's threshold.

## Agent Class

`Agent` extends `AIChannel` with role, description, greeting, and memory support — designed for multi-agent orchestration:

```python
from roomkit import Agent
from roomkit.providers.ai.mock import MockAIProvider

agent = Agent(
    "support-agent",
    provider=MockAIProvider(responses=["I can help with that."]),
    role="Customer support specialist",
    description="Handles billing and account questions",
    system_prompt="You are a support specialist. Be concise and helpful.",
    greeting="Hi! How can I help you today?",
)
```

## Tool Calling

Define tools as JSON schema and attach them to the AI channel:

```python
from roomkit import ChannelCategory
from roomkit.channels.ai import AIChannel
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig

ai = AIChannel(
    "ai-assistant",
    provider=OpenAIAIProvider(OpenAIConfig(api_key="sk-...", model="gpt-4o")),
    system_prompt="You help users check the weather.",
    tools=[
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    ],
)
```

### Tool Handler

Register a handler to execute tool calls via the constructor:

```python
async def handle_tools(name: str, arguments: dict) -> str:
    if name == "get_weather":
        city = arguments["city"]
        return f'{{"temperature": 22, "condition": "sunny", "city": "{city}"}}'
    return '{"error": "Unknown tool"}'

ai = AIChannel(
    "ai-assistant",
    provider=provider,
    tools=[...],
    tool_handler=handle_tools,
)
```

### Tool Protocol (Tool ABC)

For structured tool definitions, use the `Tool` base class:

```python
from roomkit.tools.base import Tool

class GetWeather(Tool):
    name = "get_weather"
    description = "Get current weather for a city"
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
        },
        "required": ["city"],
    }

    async def execute(self, arguments: dict) -> str:
        return '{"temperature": 22, "condition": "sunny"}'

ai = AIChannel("ai", provider=provider, tools=[GetWeather()])
```

### MCP Tool Provider

Integrate Model Context Protocol servers:

```python
from roomkit.tools.mcp import MCPToolProvider

mcp = MCPToolProvider(server_command=["uvx", "mcp-server-sqlite", "--db", "data.db"])
await mcp.initialize()

ai = AIChannel("ai", provider=provider, tools=mcp.tools())
```

## Tool Search (Progressive Tool Disclosure)

When an agent has dozens of tools, sending every schema to the model on
every turn burns context and makes smaller models hallucinate tool names.
**Tool Search** hides the catalogue behind two discovery tools and lets the
model reveal only what it needs:

- `find_tools(query)` — search the catalogue by natural language; the
  matches become directly invocable for the rest of the turn.
- `list_tools(category=None)` — list the catalogue (name + short description).

```python
ai = AIChannel(
    "ai",
    provider=provider,
    tool_handler=handle_tools,
    tools=big_catalogue,              # e.g. 60+ MCP tools
    tool_search=None,                 # None = auto, True/False = force
    tool_search_threshold_pct=10.0,   # auto-enable above this % of the window
    tool_search_threshold=20,         # fallback tool count when window unknown
    tool_search_pinned=["get_help"],  # always visible, never searched for
)
```

How it works:

1. The model first sees only `find_tools`/`list_tools` plus the pinned set —
   the discretionary catalogue is hidden.
2. It calls `find_tools("send a text message")`; the matches are scored and
   returned, and their names are recorded for the turn.
3. On the **next** tool-loop round the matched tools are visible and directly
   callable. The text loop re-sends its (re-filtered) tool list every round,
   so no provider reconfigure is needed — this works on **any** text/HTTP
   provider. (The realtime voice channel offers the same feature via
   `provider.reconfigure`.)

Notes:

- **Activation auto-tunes to the model.** In `auto` mode it defers when the
  *deferrable* (non-pinned) tools would cost more than `tool_search_threshold_pct`
  % of the model's context window (default 10%) — a large model is a no-op, a
  small one defers early. When the window is unknown (custom / local model ids
  absent from the provider catalog) it falls back to the `tool_search_threshold`
  tool count. Below the threshold Tool Search is a no-op and every tool is sent.
- **Pinned tools** stay visible without a search. The discovery tools always
  pass tool-policy and skill gating, so they work even under a restrictive
  `tool_policy`.
- A second `find_tools` call **swaps** the revealed window (keeping the visible
  surface small); `list_tools` reveals nothing — it is purely informational.

See `examples/ai_tool_search.py` for a runnable, no-API-key walkthrough.

## Agent Skills

A skill is a directory holding a `SKILL.md` (YAML frontmatter + an instructions
body) and optional `scripts/` and `references/` — the
[Agent Skills](https://agentskills.io) standard. It packages knowledge the model
loads **on demand** instead of knowledge every system prompt has to carry.

```python
from roomkit.skills import SkillRegistry

registry = SkillRegistry()
count = registry.discover("./skills")     # returns how many were found

ai = AIChannel(
    "ai",
    provider=provider,
    skills=registry,
    skills_in_prompt=True,          # False = the host renders its own manifest
    script_executor=my_executor,    # omit and run_skill_script is not offered
)
```

`discover()` takes one or more directories and commits only once every
candidate has parsed, so a failure leaves the registry as it was rather than
half filled. It raises on a malformed skill by default (`strict=False` skips it):
a bad skill is a deployment error, and skipping it silently leaves an agent that
quietly cannot do something it was configured to do.

Three tools are registered automatically alongside the host's own:

| Tool | Offered | Effect |
|------|---------|--------|
| `activate_skill(name)` | always | Loads the skill for the conversation |
| `read_skill_reference(skill_name, filename)` | always | Reads one file from `references/` |
| `run_skill_script(skill_name, script_name, arguments)` | only with `script_executor` | Runs a script through the integrator's executor |

### Activation lifecycle

**An activation lasts the conversation, not the turn that made it.** Re-sending
a 9 KB body on every turn costs more than the skill is worth, so the body moves
to where the per-turn rebuild carries it for free:

| | First `activate_skill` in a room | Later calls |
|---|---|---|
| Tool result | Full `instructions` + the `scripts`/`references` listing | `{"ok": true, "already_active": true, ...}` — no body |
| System prompt | *(nothing yet — composed before the call)* | `# Active skill instructions (binding rules)` carrying the body |
| Gated tools | Revealed for the rest of the turn | Still revealed, no re-activation |

- The ack is safe because the prompt carries the rules. Lose the record (process
  restart, channel object replaced) and the prompt block goes with it, so the
  next activation returns the body again — it degrades to reloading, never to an
  ack with no rules.
- The record is **hydrated** from the room's persisted tool-call history, so a
  channel swapped mid-conversation does not restart amnesic.
- **Bodies are never evicted.** Large tool results are normally replaced by a
  `read_stored_result` pointer; an `activate_skill` result is exempt, because
  binding rules cut to a head/tail preview are not rules. References still
  evict — those are data, and paginating data is what eviction is for.
- **Four skills stay active per room**, by recency; a fifth retires the least
  recently used one, whose next `activate_skill` returns the body again.

### Reading the active set

`skills_in_prompt=False` hands the manifest to the host, and a catalogue is only
half of what a manifest needs. `active_skill_names(room_id)` supplies the other
half — the skills that room is already carrying:

```python
active = ai.active_skill_names("support-room")     # {"code-review"}
rows = [
    f"- {m.name} ({'loaded' if m.name in active else 'available'}): {m.description}"
    for m in registry.all_metadata()
]
```

Without it every row reads *available*, including the skill whose instructions
the prompt already carries, so the manifest asks the model to load rules that
are in front of it — one wasted round, answered by an ack. Keyed on the room,
empty for a room that activated nothing and for `None`. Active bodies are
injected whether or not `skills_in_prompt` is set: that is runtime state a host
cannot know.

### Tool gating

A skill's `allowed_tools` frontmatter (a YAML list or a comma-separated scalar)
names the tools it unlocks. Entries are `ToolPolicy` **globs** (RFC §24.2), so
`search_*` covers every tool whose name starts with `search_` — match them, never
test membership. A gated tool is hidden from the catalogue *and* refused at
execution, because a model that saw the name before the skill was deactivated can
still call it. The skill and Tool Search infrastructure tools are exempt from
gating in both places: gating `find_tools` or `activate_skill` would tell the
model to activate a skill it has no way left to name.

### Visibility states

```python
registry.mark_unlisted("legacy-csv-import")       # activatable, absent from the manifest
registry.mark_unavailable("deploy-helper", "needs a ScriptExecutor")
```

`registry.skill_names` is what can be activated, `registry.listed_names` what the
manifest advertises, and `registry.unavailable_skills` maps a name to the reason
the model can quote instead of guessing. `registry.to_prompt_xml()` renders the
manifest RoomKit would otherwise inject.

### Script execution

There is **no default executor** — sandboxing, timeouts and allowed interpreters
are the integrator's call. Implement `ScriptExecutor` and pass it as
`script_executor`. The script name comes from the model, so RoomKit resolves it
first: use `skill.resolve_script(name)` rather than joining
`skill.path / "scripts" / name` yourself. A name that escapes the skill —
including through a symlink planted in `scripts/` — raises `SkillPathError` and
never reaches your executor.

### Realtime voice

`RealtimeVoiceChannel` runs the same lifecycle per **session**, with
`skill_delivery_mode` deciding how a body reaches the model:

- `"on_demand"` — metadata only in the prompt; `activate_skill` loads the body
  through `provider.reconfigure`.
- `"inline_full"` — every body baked into the initial `system_instruction`;
  `activate_skill` becomes a declarative ack and no reconfigure is needed.

It defaults to `"inline_full"` when the provider reports
`supports_mid_session_reconfigure=False` (e.g. Gemini 3.x Flash Live), and to
`"on_demand"` otherwise.

Runnable, no API key needed for the last one: `examples/agent_skills.py`
(discovery and activation), `examples/skill_visibility.py` (the three states plus
a recommender hook), `examples/skill_active_manifest.py` (a host manifest reading
the active set).

## Per-Room Configuration

Override AI settings per room via binding metadata:

```python
await kit.attach_channel(
    "billing-room",
    "ai-agent",
    category=ChannelCategory.INTELLIGENCE,
    metadata={
        "system_prompt": "You are a billing specialist.",
        "temperature": 0.3,
        "tools": [...],
    },
)
```

## Per-Turn Configuration

Binding metadata is a snapshot taken at attach time. When the configuration
changes underneath you — admin edits, per-user gating, feature flags — that
snapshot becomes a second source of truth that goes stale. `config_provider`
resolves the config fresh at the start of every generation instead:

```python
from roomkit import AIChannel, AIChannelTurnConfig

async def per_turn(binding, context) -> AIChannelTurnConfig | None:
    settings = await load_settings(context.room.id)
    return AIChannelTurnConfig(
        system_prompt=settings.prompt,
        temperature=settings.temperature,
        enable_thinking=settings.thinking,
        reasoning_effort=settings.effort,
    )

ai = AIChannel("ai-agent", provider=provider, config_provider=per_turn)
```

`AIChannelTurnConfig` (exported from `roomkit`) carries `system_prompt`,
`tools`, `temperature`, `max_tokens`, `thinking_budget`, `enable_thinking`
and `reasoning_effort`. Each setting resolves from the most specific source
that has an opinion: binding metadata (per-room operator intent, always
wins), then the `config_provider` result, then the `AIChannel` constructor
default, then the provider config. `None` at a tier means "not set here" and
defers outward, so an unset knob never overrides with a default.

## Streaming

AIChannel supports streaming responses to WebSocket clients:

```python
from roomkit import WebSocketChannel

ws = WebSocketChannel("ws-user")

# Register with stream support
ws.register_connection("conn-1", on_recv, stream_send_fn=on_stream)

async def on_stream(conn_id: str, msg) -> None:
    # StreamStart, StreamChunk, StreamEnd
    print(f"Stream: {msg}")
```

### Why a streaming tool loop stopped

The streaming tool loop ends on rules of its own — the round cap, the
wall-clock deadline, a round truncated at the output cap, a model that
answered nothing after its tools, the anti-loop ripcord, a cancellation. It
yields a final
`LoopEndMarker(reason, rounds)` on **every** exit, `completed` included, so
the end of the stream is never itself the signal and no consumer has to
re-derive the cause by counting tool calls and reading a clock.

Read it at the source, by subclassing `AIChannel` and wrapping
`ChannelOutput.response_stream`:

```python
from roomkit.channels.ai import AIChannel
from roomkit.models.streaming import LoopEndMarker


class ObservingAIChannel(AIChannel):
    async def on_event(self, event, binding, context):
        output = await super().on_event(event, binding, context)
        if output.response_stream is None:
            return output
        return output.model_copy(
            update={"response_stream": self._observe(output.response_stream)}
        )

    async def _observe(self, inner):
        async for delta in inner:
            if isinstance(delta, LoopEndMarker):
                if delta.reason != "completed":
                    logger.warning(
                        "agent stopped: %s after %d rounds", delta.reason, delta.rounds
                    )
                continue          # keep the terminal marker out of the stream
            yield delta
```

`reason` is one of `completed`, `max_rounds`, `timeout`, `truncated`,
`empty_response`, `force_stopped`, `cancelled`, `error`. The limits each
reason refers to are the caller's own configuration and are not repeated on
the marker.

`force_stopped` is the one worth special attention, because it is the only
non-`completed` exit that ends **with text**: the anti-loop guard pulled the
ripcord on a model re-issuing an already-blocked call, which strips the tools
and asks for a plain-text answer. That text summarises an interrupted turn, so
a consumer treating "the model produced prose" as "the model answered" will
deliver a cut run as a finished one — which is precisely why the reason is
named rather than folded into `completed`. `error` ends with text on the same
terms: a mid-loop provider failure salvaged as a partial answer (non-streaming
only — a streaming failure reaches the consumer as the exception itself).

### Why a non-streaming tool loop stopped

The non-streaming loop returns `RoomEvent`s rather than a stream, so its
reason travels on them: every response MESSAGE event's metadata carries
`loop_end_reason` with the same values, next to `ai_usage` (which sums every
generation round of the turn, not just the last).

```python
reply = output.response_events[-1]
if reply.metadata["loop_end_reason"] != "completed":
    logger.warning("agent stopped: %s", reply.metadata["loop_end_reason"])
```

The framework's inbound streaming path forwards text deltas and the tool-call
and thinking markers to a channel's `deliver_stream`, but **not** the terminal
marker — it would reach a renderer as noise. Overriding `deliver_stream` on a
WebSocket or CLI channel will therefore not see it; wrap the AI channel's own
`response_stream` instead.

Additive by construction: the streaming protocol is a mixed
`str | StreamMarker` whose consumers already dispatch on the markers they
know, so a text-only consumer filtering on `isinstance(chunk, str)` is
unaffected. Streaming only — the non-streaming loop returns an `AIResponse`
the caller already holds.

Two bounds keep a degenerate model from running away with a turn:
`max_tool_rounds` caps how many rounds run, and a **32-call ceiling per
round** caps how wide one round may be. The per-round cap is applied before
the assistant message is assembled, so a dropped call is absent from the
transcript as well as from the results — no provider sees a tool call with no
matching result. Both loops enforce it from the shared loop rules.

## AI Thinking/Reasoning

Some providers support extended thinking:

```python
ai = AIChannel(
    "ai",
    provider=AnthropicAIProvider(AnthropicConfig(
        api_key="...",
        model="claude-opus-5",
    )),
    system_prompt="Think step by step.",
    thinking_budget=4096,  # Setting a budget enables thinking mode
)
```

Per-provider mechanisms: Anthropic uses `thinking_budget` as above; OpenAI-family providers (OpenAI, Azure, OpenRouter, LiteLLM, xAI) use the `reasoning_effort` config field — OpenRouter translates it into its unified `reasoning` object for any upstream model, and a LiteLLM gateway normalises it (plus an explicit `thinking` budget when `thinking_budget>0`) for every upstream it fronts; Gemini (and Vertex) use `GeminiConfig.thinking_level`; Ollama exposes the native `think` parameter (see `docs/c7/ollama-provider.md`); PolarGrid uses `PolarGridConfig(thinking=True)` (the `enable_thinking` flag, surfaced as `AIResponse.thinking` / `StreamThinkingDelta`).

`thinking_budget`, `enable_thinking` and `reasoning_effort` all ride the
per-turn chain described under *Per-Turn Configuration*, so reasoning is
steerable per room and per turn rather than only per provider instance:

```python
ai = AIChannel(
    "ai",
    provider=provider,
    enable_thinking=True,
    reasoning_effort="low",
)
```

A thinking model costs two to three times the tokens and the latency of a
direct answer, and that trade differs between an agent's tool loop — where the
model mostly shapes results it already has — and a chat turn where the
reasoning is the value.

Reasoning competes with the answer for the same `max_tokens`. A round that
spends its whole budget thinking returns empty `content` with a truncation
finish reason; RoomKit recognises that across every provider's spelling
(`length`, `max_tokens`, `MAX_TOKENS`, case-insensitively) and skips the
empty-response retry, which would only truncate again under the same cap.

`AIContext.max_tokens` defaults to `None`, not a number: every provider reads
`context.max_tokens or self._config.max_tokens`, so a non-`None` default here
would shadow the provider config and make a configured cap unreachable.
Providers whose config also defaults it to `None` (Ollama, PolarGrid) send no
cap at all, letting the server pick its own.

### vLLM reasoning and sampling

vLLM renders the model's chat template server-side, so reasoning is steered
through `chat_template_kwargs` rather than a top-level sampling parameter.
`VLLMConfig.enable_thinking` and `VLLMConfig.reasoning_effort` map onto them;
both default to `None`, leaving the model's own default (current Qwen builds
think at their most verbose effort unless told otherwise). An explicit
`extra_body["chat_template_kwargs"]` entry still wins.

`VLLMConfig` also types five sampling knobs that previously required
`extra_body`: `top_p`, `top_k`, `min_p`, `presence_penalty` and
`repetition_penalty`. Each defaults to `None` ("the server decides"); an
explicit `0` is sent, since `min_p=0.0` and `presence_penalty=0.0` are values
rather than absences.

```python
from roomkit.providers.vllm import VLLMConfig, create_vllm_provider

provider = create_vllm_provider(VLLMConfig(
    model="Qwen/Qwen3-8B",
    enable_thinking=False,
    presence_penalty=1.5,   # Qwen3's guidance for non-thinking mode
    top_p=0.8,
    top_k=20,
))
```

## Vision Support

AI providers that support vision can process images sent as `MediaContent`:

```python
from roomkit.models.event import MediaContent

await kit.process_inbound(
    InboundMessage(
        channel_id="ws-user",
        sender_id="user",
        content=MediaContent(url="https://example.com/chart.png", mime_type="image/png"),
    )
)
# AI sees the image and responds with analysis
```

OpenAI-family providers (OpenAI, Azure, OpenRouter, LiteLLM, xAI) and PolarGrid send images as OpenAI-shaped `image_url` content parts (remote URL or `data:` URI); PolarGrid and xAI gate this on the model's `supports_vision` from their curated catalogs — whether the model actually reads the image is the deployed model's capability.

## Agentic Features

### Dangling Tool Call Recovery

When a user sends a new message while the AI is mid-tool-execution (barge-in), tool calls can be left without matching results. AIChannel automatically detects these orphaned calls and injects synthetic cancellation results before the next AI turn, preventing provider API rejections.

This is fully automatic — no configuration needed.

### Large Output Eviction

When tool results are very large (database queries, file dumps, API responses), they consume significant context budget. AIChannel can evict large results to a side buffer and replace them with a preview:

```python
ai = AIChannel(
    "ai-agent",
    provider=provider,
    system_prompt="You are a data analyst.",
    evict_threshold_tokens=5000,  # default: 5000 tokens
    tools=[QueryDatabase()],
)
```

When a tool result exceeds the threshold:
1. The full result is stored in a FIFO-bounded buffer (max 50 entries)
2. A head/tail preview (first 5 + last 5 lines) replaces the result in context
3. A `_read_tool_result` tool is auto-injected so the agent can paginate through the full output

### Planning Tools

Enable structured task planning so agents can break down complex work and track progress:

```python
ai = AIChannel(
    "ai-agent",
    provider=provider,
    system_prompt="You are a research assistant.",
    enable_planning=True,
)
```

When enabled, the AI gets a `plan_tasks` tool that accepts up to 100 tasks with a title of at most 500 characters and a `status` (`pending`, `in_progress`, `completed`, `blocked`). Undeclared task fields are discarded. The current plan is:
- Injected into the system prompt on each turn (so the AI sees its progress)
- Published as an ephemeral `CUSTOM` event with `data.type = "plan_updated"` for real-time UI rendering

Subscribe to plan updates for UI:

```python
await kit.subscribe_room("room-1", my_callback)

# Callback receives ephemeral event with:
# type: "custom", data: {"type": "plan_updated", "tasks": [...]}
```

The ephemeral event is the only plan-update signal — the `HookTrigger.ON_PLAN_UPDATED` enum value is reserved and not currently fired.

### SummarizingMemory

For long conversations, use `SummarizingMemory` to proactively manage context budget with two tiers:

```python
from roomkit.memory import SummarizingMemory, SlidingWindowMemory

ai = AIChannel(
    "ai-agent",
    provider=main_provider,
    memory=SummarizingMemory(
        inner=SlidingWindowMemory(max_events=100),
        provider=summary_provider,       # lightweight model (e.g. Haiku)
        max_context_tokens=128_000,
        tier1_ratio=0.50,                # truncate old events at 50%
        tier2_ratio=0.85,                # LLM summarization at 85%
    ),
)
```

- **Tier 1** (~50% capacity): Truncates large text bodies in older events to 2000 chars. No LLM call — cheap and fast.
- **Tier 2** (~85% capacity): Calls the summary provider to summarize older events into a concise paragraph. Keeps recent events at full fidelity. Supports chained summaries (prior summary is incorporated into the new one).

### Knowledge Retrieval (RAG)

Enrich AI context with external knowledge sources using `RetrievalMemory`:

```python
from roomkit.knowledge import KnowledgeSource, KnowledgeResult
from roomkit.memory import RetrievalMemory, SlidingWindowMemory

# Implement your own knowledge source (vector store, search engine, etc.)
class FAQSource(KnowledgeSource):
    async def search(self, query, *, room_id=None, limit=5):
        results = await my_vector_db.search(query, top_k=limit)
        return [KnowledgeResult(content=r.text, score=r.score, source="faq") for r in results]

ai = AIChannel(
    "ai-agent",
    provider=provider,
    memory=RetrievalMemory(
        sources=[FAQSource()],
        inner=SlidingWindowMemory(max_events=50),
        max_results=5,
    ),
)
```

`RetrievalMemory` searches all sources concurrently, deduplicates results, and prepends relevant knowledge as a context message. When `ingest()` is called (automatic on every inbound event), it also indexes content in all sources.

#### Built-in: PostgreSQL Full-Text Search

For production use without a vector database, use `PostgresKnowledgeSource`:

```python
from roomkit.knowledge.postgres import PostgresKnowledgeSource

source = PostgresKnowledgeSource(dsn="postgresql://localhost/mydb")
await source.init()

# Or share the pool with PostgresStore:
source = PostgresKnowledgeSource(pool=store._pool, source_name="faq")
await source.init()
```

Uses PostgreSQL `tsvector` with `ts_rank_cd` for relevance scoring. Auto-creates schema, supports room-scoped queries, and upserts on conflict.

### Response Scoring

Score AI responses automatically using the `ScoringHook`:

```python
from roomkit.scoring import ScoringHook, ConversationScorer, Score

class QualityScorer(ConversationScorer):
    async def score(self, *, response_content, query, room_id, channel_id, **kwargs):
        # Your scoring logic (LLM-as-judge, rules, heuristics)
        return [Score(value=0.9, dimension="relevance", reason="On topic")]

hook = ScoringHook(scorers=[QualityScorer()])
hook.attach(kit)

# Scores are stored as Observations and accessible via hook.recent_scores
```

### User Feedback

Collect user quality ratings:

```python
await kit.submit_feedback("room-1", rating=0.9, comment="Very helpful", dimension="helpfulness")
# Stored as Observation in ConversationStore, fires ON_FEEDBACK hook
```

## Tool Call Events

AIChannel automatically broadcasts ephemeral `TOOL_CALL_START` and `TOOL_CALL_END` events when executing tools. Subscribe to these for UI indicators:

```python
await kit.subscribe_room("room-1", my_callback)

# Callback receives:
# TOOL_CALL_START: {tool_calls: [{id, name, arguments}], round, channel_id}
# TOOL_CALL_END: {tool_calls: [{id, name, result}], round, channel_id, duration_ms}
```
