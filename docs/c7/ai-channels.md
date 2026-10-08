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
- **LiteLLM proxy** (`roomkit.providers.litellm`) — subclass of `OpenAIAIProvider` pointed at a self-hosted LiteLLM gateway (default `http://localhost:4000`); the extra installs the `openai` SDK, deliberately **not** the `litellm` package (the gateway keeps keys/budgets/routing server-side). `model` is the deployment's public alias and `api_key` a virtual or master key, both required. Reasoning rides LiteLLM's cross-provider normalisation: `reasoning_effort` passes through, `thinking_budget>0` maps to a `thinking` token budget, and the trace comes back in `reasoning_content`; `thinking_budget=0` sends no reasoning params at all (LiteLLM has no disable token every upstream translator accepts — force-off belongs in the proxy's per-model config or `extra_body`). `available_models()` is empty (the model list is the operator's config); `list_models()` reads the proxy's `/model/info` — context window, vision flag, and per-token costs per alias, with a load-balanced group merged into the one entry every deployment can honour (smallest window, vision and price only when unanimous) and LiteLLM's `0`-for-unknown costs mapped to "unpriced" rather than free.
- **Outbound policy** (`transport=`) — `OpenAIAIProvider(config, transport=...)`, and every provider built on it (`AzureAIProvider`, `OpenRouterAIProvider`, `create_vllm_provider(config, transport=...)`, the inherited xAI, DeepSeek, Qwen, LiteLLM, Cerebras, Meta), sends every request through an `httpx.AsyncBaseTransport` placed inside the SDK's own default client (redirects followed, each hop through the transport; per-request timeout kept). The transport owns its connection pool and limits, and httpx reads no environment proxy beside a transport. The seam for a policy that judges the address actually dialled when users name the endpoint, or a `MockTransport` in a test; closed with the provider. See `examples/openai_outbound_policy.py`.
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

provider = GeminiAIProvider(GeminiConfig(api_key="...", model="gemini-3.8-flash"))

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

`role`, `description`, `scope` and `language` are appended to the system
prompt as an `--- Agent Identity ---` block, in a turn, a handoff and a
realtime pipeline alike. A host that renders the agent's identity in its own
prompt passes `identity_in_prompt=False`: no block is written anywhere, and
the fields stay readable on the agent.

## Steering a Running Turn

`steer()` hands a running tool loop a directive (`Cancel`, `InjectMessage`,
`UpdateSystemPrompt`, RFC §21.3) and returns how many loops it reached. One
channel object serves every room it is bound to, so a host acting for one
room addresses that room:

```python
from roomkit.models.steering import Cancel, InjectMessage

agent.steer(Cancel(reason="stop pressed"), room_id="room-a")   # every loop of room-a
agent.steer(InjectMessage(content="Also check the logs."), room_id="room-a")  # its latest
agent.steer(Cancel(reason="timeout"), loop_id="loop-abc123")   # one loop
```

Without `loop_id` or `room_id`, a directive reaches the channel's most
recent loop, whatever its room. A loop is reachable once its turn has started
(its response stream is read): a directive that comes before reaches none, and
`steer()` returns 0. See `examples/ai_shared_agent_rooms.py`.

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

#### Refusing a call

What the handler returns is the tool's answer, so a refusal returned as a body
reads as work that was done: the tool-call event says `completed`, and an audit
trail records a call that never ran. Raise `ToolRefusedError` instead.

```python
from roomkit import ToolRefusedError

async def handle_tools(name: str, arguments: dict) -> str:
    if name not in MY_TOOLS:
        raise ToolRefusedError(f"Error: the tool '{name}' does not exist.")
    ...
```

The message reaches the model verbatim, which is the point: any other exception
reads as `{"error": "Tool '<name>' failed (<ExceptionClass>)"}`, its message
withheld from the model (it can hold a password or a path) and handed to the
log and to `ON_TOOL_CALL` observers as `event.error_detail`. The call is marked
failed, observers see `event.is_error`, and the stored `TOOL_CALL_END` carries
what the model read.

A tool that ran and failed, with words for the model, raises `ToolFailedError`:
the message reaches the model verbatim and the observers as `error_detail`, and
the call is failed, not refused (`event.refused` is false). An MCP tool whose
result says `isError` raises it.

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

async with MCPToolProvider.from_command("uvx", ["mcp-server-sqlite", "--db", "data.db"]) as mcp:
    ai = AIChannel("ai", provider=provider, tools=mcp.get_tools(), tool_handler=mcp.as_tool_handler())
```

Beside the model's tools, a host reads what an MCP App needs from the same
connection: `mcp.tool_meta()` (each discovered tool's `_meta`, `ui.resourceUri`
and `ui.csp` among it, from the listing made at connection),
`await mcp.read_resource(uri)` (the server's `ReadResourceResult`, the app's
HTML) and `await mcp.call_tool_result(name, arguments)` (the server's
`CallToolResult` as it is, `isError` included, for a frame's own calls). Like
`call_tool`, `call_tool_result` calls any tool the server has: `tool_filter`
shapes what discovery offers the model, not what the host may call, so the
host authorizes a frame's call itself.

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
    tool_search_threshold_tokens=8000,  # auto-enable above this many schema tokens
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

On a provider that can hold a tool declared but unseen (Anthropic, `AIProvider.supports_deferred_tools`), the hidden catalogue is declared that way (`defer_loading`) from the first round, and `find_tools` makes its matches callable by reference: the tool list does not change within the turn, so the provider's prompt cache holds across the reveal (RFC §6.4). A skill's gated tools are held the same way until `activate_skill` opens them. A tool used in one turn is visible from the next, as on any provider.

Notes:

- **Activation weighs cost and fit.** In `auto` mode it defers when the
  schemas of the tools it can hide (not pinned, not the channel's own, not orchestration's) pass `tool_search_threshold_tokens`
  (8,000 by default, whatever the window; `None` drops this cap), or would cost
  more than `tool_search_threshold_pct` % of the model's context window
  (default 10%). The cap is for cost: measured on `claude-sonnet-5`, hiding a
  catalogue pays for its discovery round from about 15 tools, prompt caching
  included (60 tools over two turns: $0.074 sent whole, $0.018 behind Tool
  Search; 73 % less on `gpt-4.1-mini`). When the window is unknown (custom / local model ids absent from the
  provider catalog) the cap or the `tool_search_threshold` tool count decides.
  Below the thresholds Tool Search is a no-op and every tool is sent.
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
still call it. The tools that only read or unlock (`activate_skill`,
`read_skill_reference`, `read_stored_result`, `find_tools`, `list_tools`) are
exempt from gating in both places: gating `find_tools` or `activate_skill` would
tell the model to activate a skill it has no way left to name. `run_skill_script`
acts, and is gated like any other tool.

### Visibility states

```python
registry.mark_unlisted("legacy-csv-import")       # activatable, absent from the manifest
registry.mark_unavailable("deploy-helper", "needs a ScriptExecutor")
```

`registry.skill_names` is what can be activated, `registry.listed_names` what the
manifest advertises, and `registry.unavailable_skills` maps a name to the reason
the model can quote instead of guessing. `registry.to_prompt_xml()` renders the
manifest RoomKit would otherwise inject.

A host that builds skills elsewhere (a store, a marketplace) registers them with
`registry.add(skill)`, and one that narrows a registry for an agent copies it:
`registry.copy(["reports", "invoices"])` keeps each skill's path, so a skill
discovered but not yet loaded is still found, and keeps the source's marks;
`marks=False` drops them (a hand-picked set, every skill advertised).

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

When the skills belong to another agent (a reasoning backend's), the realtime
channel still runs their scripts behind its own gate: pass
`RunSkillScriptTool(skills, executor)` in its `tools=`. It declares the one
`run_skill_script` schema and runs the script through the one handler every
channel uses, so a script outside its skill is refused there too.
`RunSkillScriptTool.name` is the name it is declared and called under.

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
`tools`, `temperature`, `max_tokens`, `thinking_budget`, `enable_thinking`,
`reasoning_effort`, `response_schema`, `turn_budget_tokens` and
`turn_budget_usd`. Each setting resolves from the most specific source
that has an opinion: binding metadata (per-room operator intent, wins when
it sets a value), then the `config_provider` result, then the `AIChannel`
constructor default, then the provider config. `None` at a tier means "not
set here" and defers outward, an explicit `null` in the binding metadata
included, so an unset knob never overrides with a default. An
`enable_thinking: False` set at a tier turns off a `thinking_budget` a less
specific tier set.

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
`LoopEndMarker` on **every** exit, `completed` included, so
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
                if delta.reason == "timeout":
                    logger.warning(
                        "agent stopped at its %ss deadline", delta.timeout_seconds
                    )
                elif delta.reason != "completed":
                    logger.warning(
                        "agent stopped: %s after %d rounds", delta.reason, delta.rounds
                    )
                continue          # keep the terminal marker out of the stream
            yield delta
```

`reason` is one of `completed`, `max_rounds`, `timeout`, `budget_exceeded`,
`truncated`, `empty_response`, `unfinished`, `force_stopped`, `cancelled`, `error`.
`rounds` is how many tool rounds ran, as `ON_AI_RESPONSE` counts them
(`round_count`). The marker also states the limits the turn ran under, so the
consumer names the one a `max_rounds`, `timeout` or `budget_exceeded` end hit
without reading the channel: `max_rounds`, `timeout_seconds`, `budget_tokens`
and `budget_usd`, `None` for no such limit. They ride the marker only, not
`ON_AI_RESPONSE`. The budget is resolved per turn (the binding, then the turn's
config, then the channel), which only the marker can say.

`force_stopped` is the one worth special attention, because it is the one
non-`completed` exit that usually ends **with text**: the anti-loop guard pulled the
ripcord on a model re-issuing an already-blocked call, and one last generation
is asked for a plain-text answer (its tools stay declared, so the provider's
prompt cache holds, and none of its calls runs). That text summarises an interrupted turn, so
a consumer treating "the model produced prose" as "the model answered" will
deliver a cut run as a finished one — which is precisely why the reason is
named rather than folded into `completed`. `error` is a turn the provider
interrupted after a tool round: the rounds are kept, each round's text as its
own message, `ON_AI_RESPONSE` reports it with the usage of its rounds, and it
is an error too, surfaced through `ON_ERROR` and the caller's
`InboundResult.error`. No message is added to say so: the loop yields its
`LoopEndMarker` with `reason="error"`, then the exception reaches the
consumer, and a delegated turn it ends fails. A stored `[Response interrupted]`
message (`metadata["interruption_marker"] = True`,
`roomkit.models.event.is_interruption_marker`) solicits no agent and no
strategy or delegation takes it for an answer. `LoopEndMarker.usage` is what the turn's
generations used, summed over its rounds; a response without tools yields the
marker too, with `rounds=0`, a provider that streams text alone included.

### Why a turn stopped, read from its reply

The loop records the reason on the turn's reply: its last MESSAGE event
carries `loop_end_reason` with the same values, next to `ai_usage` (which sums
every generation round of the turn, not just the last). That is the final
answer, or, for a turn that ended without one (a cancellation between
rounds, an interruption), the last round's text. A streamed turn
learns its end after that message was written: its stored row is updated once
the turn's deliveries are done (`ON_EVENT_UPDATED` fires), so read it from the
store or the `InboundResult`, not from the frames a channel was streamed.

```python
from roomkit.models.enums import EventType

result = await kit.process_inbound(message)
reply = [e for e in result.response_events if e.type == EventType.MESSAGE][-1]
if reply.metadata["loop_end_reason"] != "completed":
    logger.warning("agent stopped: %s", reply.metadata["loop_end_reason"])
```

The framework's inbound streaming path forwards text deltas and the tool-call
and thinking markers to a channel's `deliver_stream`, but **not** the terminal
marker — it would reach a renderer as noise. Overriding `deliver_stream` on a
WebSocket or CLI channel will therefore not see it; wrap the AI channel's own
`response_stream` instead.

A round the loop tries again without a call, a continuation the channel's
policy asked for or a call the provider could not parse, ends on a
`SegmentBreakMarker`: the round's text is a segment of its own, as text
before a call is, and the room writes it as its own message, so the next
round's text never runs on from it.

Additive by construction: the streaming protocol is a mixed
`str | StreamMarker` whose consumers already dispatch on the markers they
know, so a text-only consumer filtering on `isinstance(chunk, str)` is
unaffected. Every turn ends on one, whatever its provider streams: a
provider read through its `generate()` runs the same loop.

Two bounds keep a degenerate model from running away with a turn:
`max_tool_rounds` (50 by default) caps how many rounds run, and a **32-call
ceiling per round** caps how wide one round may be. The per-round cap is applied before
the assistant message is assembled, so a dropped call is absent from the
transcript as well as from the results — no provider sees a tool call with no
matching result. The loop enforces it from its round rules.

A turn can also be capped by what it spends: `turn_budget_tokens` counts every token the provider bills for the turn (input, cache reads and writes, output), `turn_budget_usd` prices each generation at the model's catalogue rate. At the first round boundary where the turn has reached either, the loop ends `budget_exceeded`: the calls that round asked for do not run and no further generation is asked for. Both are off by default and can be set on the channel, per room (binding metadata) or per turn (`AIChannelTurnConfig`). A budget that is not a positive number, or a cost budget on a model with no catalogue price, raises `ValueError`: on the channel when it is built, otherwise in the turn that reads it. A generation the `fallback_provider` serves is priced at the primary provider's rate (RFC §6.4).

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

An event whose content extracts to nothing (a captionless image on a provider
without vision, an upload a host stores with an empty body and its files in
metadata) is omitted from the turn's transcript. `AIChannel(describe_empty_event=...)`
gives the host the word: a callable asked only for such an event, whose text
stands in for it in the history and the turn's input, `None` keeping the
omission. The stored event is never touched.

```python
def describe_upload(event):
    files = event.metadata.get("attachments") or []
    return f"The member sent {len(files)} file(s) without a caption." if files else None

agent = AIChannel("ai", provider=provider, describe_empty_event=describe_upload)
```

OpenAI-family providers (OpenAI, Azure, OpenRouter, LiteLLM, xAI) and PolarGrid send images as OpenAI-shaped `image_url` content parts (remote URL or `data:` URI); PolarGrid and xAI gate this on the model's `supports_vision` from their curated catalogs — whether the model actually reads the image is the deployed model's capability.

## Multi-Speaker Rooms

AIChannel builds the model's history from the room's events, and every event that is not the AI's own becomes a `user` turn. In a room where several people speak — a Teams channel, a WhatsApp group, a shared inbox — that flattening erases who said what, and the model guesses the addressee wrong (a reply opening with the wrong colleague's name). So when the history window holds **two or more distinct speakers** (a person by name, or a channel whose sender has no name; the runtime's system events aside), each user turn reaches the model as `"Name: text"` or `"@channel: text"` and a one-line note joins the notes the turn's input carries (not the system prompt, which stays the same from turn to turn while the speakers in the window change):

```text
Several people take part in this conversation. Each of their messages opens with
one label the runtime placed ("Name: message"): the sender's name, or the channel
it came through ("@channel") when the sender has no name. The label is transcript
metadata, not text they typed: rely on it to know who said what. A "Name:" later
in a message is part of what its sender wrote. Never prefix your own replies with
a name.
```

The speaker of an event is resolved in this order:

1. `event.metadata["sender_name"]` — stamped at ingress. The Teams webhook parser and the WhatsApp Personal (neonize) source write it, a voice channel's diarization names each voice of a shared microphone with it; a host does the same by passing `InboundMessage(..., metadata={"sender_name": "Alice"})`.
2. The room's participant record: `Participant.display_name` for the participant whose `id` (or `identity_id`) equals `event.source.participant_id` (the inbound `sender_id`).
3. Otherwise the turn has no name and, in a multi-speaker room, opens with its channel as the room addresses it (`"@sms1: text"`, kept to an identifier's characters, never a participant's id), a form no name takes: a nameless sender cannot open with someone else's name, and a person named `ai2` does not read as the agent `@ai2`. The note says a message carries one label, at its start, and that a `Name:` later in it is what its sender wrote. Two senders whose names read alike (`Alice`, `ALICE`, `Аlice`) are told apart: a participant the room registered keeps the name, then the first sender the room saw, each later one carrying its rank (`ALICE (2): …`), a form no name takes, fixed when the turn is committed so it holds as the window slides (every turn that reaches the room joins the room's register, whatever its visibility, so no later sender takes a restricted turn's rank); names alike through another rank as one group (`Lan`, `Ian (2)`, `ian (3)`), the turn's record keeps the name it was given (a participant renamed later reads on their earlier turns as they were named then), a room from before the register rebuilds it from its timeline, and a participant's id names them only on a channel they are reached through; the room context an ACP agent reads and a realtime broadcast use the same labels.

Rules that follow from the definition:

- Assistant turns are never prefixed — only `user` turns are.
- A multimodal user turn (`list[AITextPart | AIImagePart]`) gets a lead `AITextPart("Name:")` rather than a rewritten body.
- The trigger turn (the event the AI is replying to) is attributed like the history.
- The note joins the turn's notes once per turn; the channel's own `system_prompt` stays the same.
- A single-speaker room — a 1:1 DM — builds a byte-identical prompt: no prefixes, no note. Attribution is decided per turn from the window the memory provider returns, so it switches on the moment a second speaker (a name, or a nameless channel) appears in it and off when the window no longer holds one.

There is nothing to configure. See `examples/ai_multi_speaker.py`.

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
1. The full result is stored under an id no other result is given while it is among the last 10,000 the store issued: each room keeps its 50 most recently stored or read results, and the store's bounds for memory (200 results, 64 Mi characters of text) take from the room holding the most first, so a quiet room keeps its results while others evict
2. A head/tail preview (first 5 + last 5 lines) replaces the result in context. The preview is also bounded in characters: at most 8000, or twice `evict_threshold_tokens` when that is smaller. A line too long for it is clipped with a `[... N chars truncated ...]` marker, and the lines left out are counted in `[... N lines omitted ...]`
3. A `read_stored_result` tool lets the agent paginate through the full output, or search it: with `query`, one line of text, it returns the lines that contain it (case aside, never a pattern) with two lines around each and their numbers, bounded like a page, a long line cut around the match. A search reads every line whole, so no match means the text is absent (RFC §21.5); an answer holding only part of the matches says so. It is declared from the first round of any turn that declares a tool, after the others, so the tool list does not change when a result is stored: a tool list that changes mid-turn invalidates a provider's prompt cache (RFC §6.4). A `BEFORE_AI_GENERATION` hook sees it once the room holds a stored result, and may withdraw it

Every outcome the model reads goes through the same eviction: a result made of content parts (text and images, such as a screenshot with its page text) has its text evicted and its images kept in place, and a refusal, an error or an `ON_TOOL_CALL` override is evicted like a result. An `ON_TOOL_CALL` hook sees the tool's whole result, before eviction, and what it hands back is what gets stored: a hook that redacts covers everything `read_stored_result` can page back, not only the preview. It sees the shape the model reads (for a model without vision, the flattened text). Declare such a hook `fail_closed=True`: `ON_TOOL_CALL` otherwise fails open, and a hook that times out on a large result would let the raw text through. A provider without vision gets the text of a content-part result, its images marked `[image]`. An `activate_skill` body is the one exception: a skill's instructions are never reduced to a preview. The `TOOL_CALL_END` event that records a result keeps at most 512 KB of its images and notes the rest, whatever their shape (image parts, image, audio or blob blocks in JSON such as an ACP agent's output, data URIs), in the result and its structured copy together (the structured copy first, being the one UI surfaces render); the model's copy keeps them all.

When the provider refuses a round's context as too long, the channel compacts it once and replays the round. The turn's input and its notes stay whole. When the input falls in the older half of the messages, the history before it is summarized and the long results of the turn's older tool rounds are stored like an evicted result, a short preview in their place, readable with `read_stored_result` (a skill's instructions and a page already read back stay whole); otherwise the older half is summarized. Every call keeps its result, and a summary (a compaction's, or one a memory provider returns) joins the user message that follows it instead of forming a second one in a row (RFC §6.4).

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
- Carried with each turn's input, after the user's words and marked as the runtime's notes, so the AI sees its progress; not in the system prompt, which stays the same from turn to turn so a provider's prompt cache holds (RFC §6.4)
- Published as an ephemeral `CUSTOM` event with `data.type = "plan_updated"` for real-time UI rendering

Subscribe to plan updates for UI:

```python
await kit.subscribe_room("room-1", my_callback)

# Callback receives ephemeral event with:
# type: "custom", data: {"type": "plan_updated", "tasks": [...]}
```

`HookTrigger.ON_PLAN_UPDATED` fires too, as an async hook carrying a `PlanUpdatedEvent` (room, channel, tasks): the signal for a host that records or acts on plan changes without a realtime backend.

### The turn's footprint

What the history may take is what the window leaves once the rest of the turn
is in it. Before it reads its memory, the AI channel measures that rest as the
first round sends it and exposes it for the turn as `current_turn_footprint()`,
a `TurnFootprint` (RFC §20): `input_tokens` (the system prompt with an
`Agent`'s identity, the tools declared under Tool Search and the tool policy,
the room's plan, the digest of the tools already used, the speaker attribution)
and `reply_tokens` (the turn's `max_tokens`, 0 when the provider applies its
own). It is `None` in a tool handler and outside an AI channel's turn.

`BudgetAwareMemory` reserves the larger of its `reserved_tokens` and the
measured input: `reserved_tokens` is a floor on that input, not an addition, so
a host passes 0 unless it knows a part the channel cannot. For the reply it
reserves the larger of its safety margin and `reply_tokens`, never both. What a
`BEFORE_AI_GENERATION` hook adds after the memory read is not measured: a host
that adds a large block there lowers `max_context_tokens` by it.
`CompactingMemory` and `SummarizingMemory` do not read the footprint.

```python
from roomkit.memory import BudgetAwareMemory, SlidingWindowMemory

memory = BudgetAwareMemory(SlidingWindowMemory(max_events=100), max_context_tokens=128_000)
```

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

`RetrievalMemory` searches all sources concurrently, deduplicates results, and returns the passages as the turn's note (`MemoryResult.notes`), each set apart in a `<knowledge>` block: they follow the turn's input, so the history before it stays the same, and cached, when the passages change at the next question (RFC §20.2). When `ingest()` is called (automatic on every inbound event), it also indexes content in all sources.

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

AIChannel automatically broadcasts ephemeral `TOOL_CALL_DELTA`,
`TOOL_CALL_START`, and `TOOL_CALL_END` events when executing tools. Subscribe
to these for UI indicators:

```python
await kit.subscribe_room("room-1", my_callback)

# Callback receives:
# TOOL_CALL_DELTA: {tool_calls: [{id, name, arguments_chars}], round, channel_id}
# TOOL_CALL_START: {tool_calls: [{id, name, arguments}], round, channel_id}
# TOOL_CALL_END: {tool_calls: [{id, name, result}], round, channel_id, duration_ms}
```

`TOOL_CALL_DELTA` is streaming-only. It reports the running argument size while
the model composes a call, but never exposes the argument content. An empty
`tool_calls` is the terminal frame for that composition, including cancellation,
provider failure, retry, and fallback. Each retry is a new attempt, so its
cumulative `arguments_chars` counts restart from zero. These projection events
are not persisted; the complete arguments still arrive in `TOOL_CALL_START`.
