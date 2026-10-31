# AI Channels

AIChannel connects rooms to LLM providers. When a message is broadcast to an AI channel, it generates a response using conversation history and re-enters it through the inbound pipeline.

## Basic Setup

```python
from roomkit import AIChannel, ChannelCategory, RoomKit
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

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

- **Gemini on Vertex AI** (`roomkit.providers.gemini.vertex`) — subclass of `GeminiAIProvider` serving the same Gemini models through a Google Cloud project with a pinned region (`GeminiVertexConfig` requires `project` and `location`, e.g. `"northamerica-northeast1"`; no API key — the identity is `impersonate_service_account`, else `service_account_json`, else ADC). Use it when data residency matters (Québec Law 25 / PIPEDA). Generation, streaming, thinking, and the model catalog are inherited; the one addition is billing labels: `GeminiVertexConfig.labels` (`{"tenant": "acme"}`) rides every request as `GenerateContentConfig.labels`, so Cloud Billing splits one project's spend per tenant or partner: metadata only, 24 to 48 h behind in the report, never a source of truth; Google's label rules (64 labels, 63 characters, lowercase letters, digits, `_`, `-`, a key starts with a letter) are enforced when the config is built, and the field is Vertex-only because the Developer API refuses it.
- **OpenRouter** (`roomkit.providers.openrouter`) — subclass of `OpenAIAIProvider` pointed at `https://openrouter.ai/api/v1`; `OpenRouterConfig` subclasses `OpenAIConfig` (adds `site_url`/`app_name` attribution headers) and `model` is a required slug like `"anthropic/claude-sonnet-4.5"`. Reasoning is forwarded to any upstream model via OpenRouter's unified `reasoning` object. Model-listing nuance: OpenRouter's `/models` items omit the `object`/`owned_by` fields the OpenAI SDK expects, so `list_models()` reads the raw JSON instead.
- **LiteLLM proxy** (`roomkit.providers.litellm`) — subclass of `OpenAIAIProvider` pointed at a self-hosted LiteLLM gateway (default `http://localhost:4000`); the extra installs the `openai` SDK, deliberately **not** the `litellm` package (the gateway keeps keys/budgets/routing server-side). `model` is the deployment's public alias and `api_key` a virtual or master key, both required. Reasoning rides LiteLLM's cross-provider normalisation: `reasoning_effort` passes through, `thinking_budget>0` maps to a `thinking` token budget, and the trace comes back in `reasoning_content`; `thinking_budget=0` sends no reasoning params at all (LiteLLM has no disable token every upstream translator accepts — force-off belongs in the proxy's per-model config or `extra_body`). `available_models()` is empty (the model list is the operator's config); `list_models()` reads the proxy's `/model/info` — context window, vision flag, and per-token costs per alias, with a load-balanced group merged into the one entry every deployment can honour (smallest window, vision and price only when unanimous) and LiteLLM's `0`-for-unknown costs mapped to "unpriced" rather than free.
- **Outbound policy** (`transport=`) — `OpenAIAIProvider(config, transport=...)`, and every provider built on it (`AzureAIProvider`, `OpenRouterAIProvider`, `create_vllm_provider(config, transport=...)`, the inherited xAI, DeepSeek, Qwen, LiteLLM, Cerebras, Meta), sends every request through an `httpx.AsyncBaseTransport` placed inside the SDK's own default client (redirects followed, each hop through the transport; per-request timeout kept). The transport owns its connection pool and limits, and httpx reads no environment proxy beside a transport. The seam for a policy that judges the address actually dialled when users name the endpoint, or a `MockTransport` in a test; closed with the provider. LiteLLM's and OpenRouter's `list_models()` go through it too. The image and vision clients on the `openai` SDK take the same `transport=` keyword: `OpenAIImageProvider`, `AzureImageProvider`, `XAIImageProvider`, `MetaImageProvider` (edits included) and `OpenAIVisionProvider(config, transport=...)`, whose client is built on first use and closes with it; so does `OpenRouterImageProvider`, on httpx alone. `OpenAIClassifier` takes a `client=` of the caller's instead. See `examples/openai_outbound_policy.py`.
- **xAI (Grok)** (`roomkit.providers.xai`) — subclass of `OpenAIAIProvider` pointed at `https://api.x.ai/v1`; defaults to `max_completion_tokens` and stream usage. `XAIRealtimeProvider` (Grok speech-to-speech) is a separate import from `roomkit.providers.xai.realtime`.
- **PolarGrid** (`roomkit.providers.polargrid`) — Canadian-hosted inference network (edges in Toronto, Vancouver, Montréal) via the official `polargrid-sdk` async client; OpenAI-shaped chat-completions surface. Supports tool calling, thinking (`PolarGridConfig(thinking=True)` sets the `enable_thinking` request flag; reasoning is surfaced as `AIResponse.thinking` / `StreamThinkingDelta`), and vision (`image_url` content parts). Pin `region` in production when residency matters.

Pick **Ollama** over the OpenAI-compat shim (`OpenAIAIProvider` pointed at `http://host:11434/v1` or `create_vllm_provider()` with an Ollama URL) whenever the model is a reasoning model (DeepSeek-R1, Qwen 3 thinking variants, etc.) — only the native API exposes the `think` parameter and streams the `thinking` field separately from `content`. See `docs/c7/ollama-provider.md` for the full rundown.

```python
# OpenAI
from roomkit.providers.openai import OpenAIAIProvider, OpenAIConfig

provider = OpenAIAIProvider(OpenAIConfig(api_key="sk-...", model="gpt-4o"))

# Gemini
from roomkit.providers.gemini import GeminiAIProvider, GeminiConfig

provider = GeminiAIProvider(GeminiConfig(api_key="...", model="gemini-3.8-flash"))

# Mock (for testing)
from roomkit.providers.ai import MockAIProvider

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
from roomkit.providers.ai import MockAIProvider

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

## An Agent the Kit Holds by Its Declaration

The kit that owns the rooms needs no live agent object (RFC §10.6.2): it
registers a `ChannelDeclaration`, and a `TurnExecutor` builds the agent where
its turns run, from the host's factory:

```python
from roomkit import AgentIdentity, ChannelDeclaration, RoomKit
from roomkit.turns import InMemoryTurnBackend, TurnExecutor

backend = InMemoryTurnBackend()
executor = TurnExecutor(build_agent, hosts=lambda channel_id: channel_id == "meteo", store=store)
await backend.serve(executor)  # in this process or another, on a shared backend

kit = RoomKit(store=store, turn_backend=backend)
kit.register_channel(
    ChannelDeclaration(
        channel_id="meteo",
        identity=AgentIdentity(name="Meteo", greeting="Hi, ask me the weather."),
        recent_events_window=50,
    )
)
await kit.attach_channel(room_id, "meteo")  # an intelligence binding
```

- A field left out reads as an `AIChannel` built with its defaults (text,
  rich and media capabilities, a 50-event history); `agent.declaration()`
  gives a live agent's own.
- Give the host's executor `hosts=`: without it, it takes the turns of every
  channel the kit does not run itself. The kit's own executor alone takes the
  turns of the agents the kit holds live (RFC §10.6.7).
- The kit holds a `DeclaredChannel` under the id: a binding takes its
  `capabilities`, routing plans its turns, `delegate` / `start_task` name it,
  `send_greeting` and auto-greet read its identity (`auto_greet` defaults to
  true, as on `Agent`).
- Its turns run on an executor that takes the id; the kit's own executor never
  builds it, so with none they fail at once with `no_executor`.
- `agent.declaration()` gives any channel's declaration; registering it reads
  on the kit as registering the agent does. An intelligence channel whose
  declaration is invalid (an empty id, a negative window) is never registered.
- Strategies and the owner's tools take a declared agent by its id:
  `Supervisor(supervisor="desk", workers=["meteo"])`, `Loop(agent="writer",
  reviewer="critic")`, `Swarm(agents=[...])`, `Pipeline(agents=[...])`,
  `Discussion(agents=[...])`, `ConversationRouter.install(kit, [...])`,
  `ConversationPipeline.install(kit, [...])`, `setup_handoff("ana", handler)`,
  `setup_delegation("ana", handler)`, `kit.setup_background("ana")`,
  `kit.setup_task_tools("ana")`. The kit keeps the tools beside the
  declaration and lists them in each turn; a strategy reads the agent's
  identity (`name`, `role`, `description`, `purpose`, `voice`, `greeting`)
  from it. An id the kit does not hold raises `ChannelNotRegisteredError`.
- `takes_owner_tools` (true by default) says whether the channel's turns
  declare the tools its owner serves; an `ACPChannel`'s declaration says
  false. The owner refuses to set a tool up for one that takes none, and a
  delegation to it returns its text rather than asking for `submit_result`.
  `unregister_channel` drops the tools set up for the id.
- Refused: a declaration that is not an intelligence channel's, a binding of
  it with another category, a strategy given a live agent whose id the kit
  holds by declaration (give it the id), and a realtime `ConversationPipeline`
  or a `HandoffHandler` serving a realtime channel given a declared agent or
  another non-`Agent` intelligence channel, its `set_language()` included
  (the session runs each agent's own prompt).

Runnable: `examples/declared_agent.py`, `examples/declared_supervisor.py`.

## A Room's Turns One After the Other

By default an agent answers a room's messages side by side: two tasks handed
to one agent in one room run at once, and the second turn reads the first
task's order still unanswered, so it may do that task too. With
`serialize_room_turns=True` the agent's turns in a room run one after the
other, in the order the messages arrived:

```python
worker = Agent("worker", provider=provider, serialize_room_turns=True)
```

- A turn takes its place when its message is delivered, and starts once the
  turn before it has written its answer (a cut answer's kept text included).
- It reads the room as it is when it starts, less what others wrote after
  its message: never a message that came after it, which waits its own turn,
  nor one deleted meanwhile. Three tasks A, B, C: B reads A and its answer,
  not C.
- `kit.cancel_delivery(key)` (or the deferred handle's `cancel()`) on a
  waiting turn's delivery: it never starts, the running one goes on. Cutting
  the running turn lets the next one start.
- A message the running turn sends into its own room that asks the same
  agent (a tool calling `kit.process_inbound`, `kit.send_event`, `kit.post`)
  is answered after that turn: the call, or its handle's `wait()`, returns
  without the answer rather than wait on the turn that made it.
- A turn whose agent left the room while it waited never starts. A muted
  agent reads each message as it comes, its answer withheld, as any muted
  channel does.
- The order is kept by the room's owner, per process, as an ACP agent keeps
  its own; two processes each handed a message of the room run them side by
  side. A strategy that takes the agent's turns (`Loop`, `Supervisor`) orders
  them itself. A memory provider that reads the store itself, rather than the
  history it is handed, still sees later messages.
- An agent the kit holds by its declaration says so with
  `ChannelDeclaration(..., serializes_room_turns=True)`.

Runnable: `examples/agent_turns_in_order.py`.

## Steering a Running Turn

`steer()` hands a running tool loop a directive (`Cancel`, `InjectMessage`,
`UpdateSystemPrompt`, RFC §21.3) and returns how many loops it reached. One
channel object serves every room it is bound to, so a host acting for one
room addresses that room:

```python
from roomkit.models import Cancel, InjectMessage

agent.steer(Cancel(reason="stop pressed"), room_id="room-a")   # every loop of room-a
agent.steer(InjectMessage(content="Also check the logs."), room_id="room-a")  # its latest
agent.steer(Cancel(reason="timeout"), loop_id="loop-abc123")   # one loop
```

Without `loop_id` or `room_id`, a directive reaches the channel's most
recent loop, whatever its room. A loop is reachable once its turn has started
(its response stream is read): a directive that comes before reaches none, and
`steer()` returns 0. See `examples/ai_shared_agent_rooms.py`.

The agent's turns run through the kit's turn backend (RFC §10.6), so a host
that may run them in another process steers through the kit, which reaches
the turns it started wherever they run:

```python
reached = await kit.steer(Cancel(reason="stop pressed"), "agent", room_id="room-a")
```

`kit.steer(directive, channel_id, room_id=..., turn_id=...)` follows the same
rules (a `Cancel` reaches every turn of the room, another directive the most
recent one) and answers the loops reached. A channel without steering (an ACP
agent) takes a `Cancel` as its turn's cancel, counted as one. With
`kit.steer(..., organization_id=...)` (RFC §17.2), a `room_id` of another
organization raises `RoomNotFoundError`, and only the turns of that
organization's rooms are reached. `agent.steer()` reaches the loops of the
process it is called in, whatever room's: the channel holds no store, and
the kit's `steer()` is the door that reads the room.

## Tool Calling

Define tools as JSON schema and attach them to the AI channel:

```python
from roomkit import AIChannel, ChannelCategory
from roomkit.providers.openai import OpenAIAIProvider, OpenAIConfig

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

### A call's intent for the person (`tool_call_intents`)

With `tool_call_intents=True`, the model writes, with each call, one short
sentence saying what the call does, for a UI to show ("Je recherche les
fichiers texte du projet."). Off by default.

```python
ai = AIChannel("ai", provider=provider, tools=[shell], tool_handler=run, tool_call_intents=True)

@kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
async def show(event: ToolCallEvent, ctx) -> None:
    print(event.name, event.intent)  # shell Je recherche les fichiers texte du projet.
```

- Each tool's schema is shown to the model with one optional string property
  more, `_roomkit_intent`. The tool's own schema is unchanged; validation and
  hooks read it.
- The loop takes the property out of the call's arguments before anything
  reads them: the handler, `BEFORE_TOOL_USE` and the policy see the tool's
  arguments only, and a closed schema (`additionalProperties: false`) does not
  refuse the call.
- The value, normalized, is the call's `intent`: on `ToolCallEvent`
  (`BEFORE_TOOL_USE`, `ON_TOOL_CALL`, refusals, cut calls), on the
  `TOOL_CALL_START` and `TOOL_CALL_END` ephemeral events (`intent` beside each
  call's `id`, only when it has one) and on the stored rows
  (`ToolCallContent.intent`). Normalized: a string only, control characters
  as spaces, bidirectional controls removed, whitespace collapsed, trimmed,
  cut at 200 characters; anything else is `None`, and never fails the call.
- A tool whose own schema declares `_roomkit_intent` keeps it as its argument
  and its calls have no intent.
- The model reads its calls back with their intent, in the turn's next rounds
  and in later turns while the option is on.
- It is display text the model wrote: it grants nothing, and says nothing
  about the call's outcome. A hook cannot change it.
- Not covered: a realtime session's calls, an ACP agent's calls, and calls a
  provider serves itself carry no intent. An agent a realtime session
  delegates to takes the property out of its calls' arguments, but reports
  them through the session, whose reports carry none.

See `examples/tool_call_intent.py`.

### Tool Protocol (Tool ABC)

For structured tool definitions, use the `Tool` base class:

```python
from roomkit.tools import Tool

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
from roomkit.tools import MCPToolProvider

async with MCPToolProvider.from_command("uvx", ["mcp-server-sqlite", "--db", "data.db"]) as mcp:
    ai = AIChannel("ai", provider=provider, tools=mcp.get_tools(), tool_handler=mcp.as_tool_handler())
```

Entering a provider that starts its server waits `connect_timeout` seconds at
most (`from_command(..., connect_timeout=60.0)`): the start, the handshake and
the listing. Past it the server is stopped and entering raises `TimeoutError`.

Beside the model's tools, a host reads what an MCP App needs from the same
connection: `mcp.tool_meta()` (each discovered tool's `_meta`, `ui.resourceUri`
and `ui.csp` among it, from the listing made at connection),
`await mcp.read_resource(uri)` (the server's `ReadResourceResult`, the app's
HTML) and `await mcp.call_tool_result(name, arguments)` (the server's
`CallToolResult` as it is, `isError` included, for a frame's own calls). Like
`call_tool`, `call_tool_result` calls any tool the server has: `tool_filter`
shapes what discovery offers the model, not what the host may call, so the
host authorizes a frame's call itself.

#### Several servers: `label=`

A model may call an MCP tool under the alias prompts written for Claude Code
use, `mcp__<server>__<tool>`; the handler runs `<tool>`. `label=` (on the
constructor, `from_url` and `from_command`) is the host's name for the server,
the `<server>` of that alias: letters, digits and `_ . -`, 64 at most, no `__`
and no final `_`, either of which would make an alias read another server (a
`ValueError` otherwise). A labelled provider's handler serves only the aliases
naming its server; one naming another server is not served
(`UnservedToolCallError`), so the next handler of a `compose_tool_handlers`
chain takes it. Give `label=` wherever a host composes several servers: a
provider without one serves an alias whatever server it names, so the first
of the chain that has the tool runs it, and a policy that names a server
cannot hold a call to it.

A policy judges an alias under both its names, the alias and the tool's
(RFC §21.1): a call passes only when both do. To keep one server out, deny its
aliases; to admit one server only, allow its aliases and its tools' names.
Either holds only when the tools are declared under their aliases alone: a
call under a bare name (`delete_records`) names no server, so a policy cannot
tell which server it reaches, and the first handler of the chain that serves
it runs it. Declared under aliases alone, a bare name is refused as not
declared:

```python
async with (
    MCPToolProvider.from_url(CRM_URL, label="crm") as crm,
    MCPToolProvider.from_url(ARCHIVE_URL, label="archive") as archive,
):
    aliases = [
        tool.model_copy(update={"name": f"mcp__{label}__{tool.name}"})
        for label, server in (("crm", crm), ("archive", archive))
        for tool in server.get_tools()
    ]
    ai = AIChannel(
        "ai",
        provider=provider,
        tools=aliases,  # the aliases alone: a bare name is not declared
        tool_handler=compose_tool_handlers(archive.as_tool_handler(), crm.as_tool_handler()),
        tool_policy=ToolPolicy(deny=["mcp__archive__*"]),  # keep the archive out
    )
    # Or admit the CRM only: ToolPolicy(allow=["mcp__crm__*", "delete_records", "search_records"])
```

`allow=["mcp__crm__*"]` alone admits nothing: the tool's own name fails it.
A host that declares the bare names instead puts first in the chain the
server its bare names must reach. Runnable: `examples/mcp_server_labels.py`.

#### Server instructions

A server may publish, when it connects, how to use its tools
(`InitializeResult.instructions`). `mcp.instructions` keeps that text (`None`
when the server sent none) and `mcp.server_info` what the server says of itself
(`name`, `version`). Hand the instructions to the channel, under a label of
your own for each server:

```python
async with MCPToolProvider.from_command("gamemaster-mcp") as gm:
    ai = AIChannel(
        "ai",
        provider=provider,
        system_prompt="You are the game master.",
        tools=gm.get_tools(),
        tool_handler=gm.as_tool_handler(),
        server_instructions={"gm": gm.instructions},
    )
```

The channel adds them to every system prompt it composes, after its own prompt
and before the skills, so a prompt a binding or a `config_provider` sets for the
room keeps them, Tool Search does not hide them, and no eviction or compaction
reaches them. Each server's text is that server's data: it sits in a
`<server_instructions>` block it cannot close, under `## MCP server: <label>`,
is cut at 8192 characters with a warning, and is introduced as not outranking
the system's or the user's instructions. The label must be an identifier
(letters, digits, `_ . -`). A server mapped to `None` is left out, which is how
a host keeps one out. `RealtimeVoiceChannel(server_instructions=...)` does the
same for a realtime session whose provider takes tools, and the `Agent` an
`AgentReasoningBackend` drives reads its own. A realtime pipeline refuses an
agent that carries some: its sessions read the voice channel's.

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
   returned, and their names are recorded for the turn. A query that is a
   tool's name (`find_tools("send_sms")`) returns that tool first. Otherwise
   a query word counts once per tool, more in its name than in its tags and
   more in its tags than in its description, weighed by how rare it is in
   the catalogue.
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
- A second `find_tools` call **swaps** the search window while retaining tools
  already called in this turn. A call rejected solely for its model-supplied
  arguments also retains its schema, so the provider can parse the next attempt
  correctly; it is not remembered as an executed call. Access refusals and
  skill gates still apply. Realtime sessions retain called schemas until their
  catalogue is replaced or the session ends. `list_tools` reveals nothing.

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

### Skills served over MCP

An MCP server may serve skills beside its tools (the MCP Skills extension,
`io.modelcontextprotocol/skills`, SEP-2640). `add_mcp_server` registers them in
the same registry, beside the host's own (RFC §24.7). Requires
`pip install roomkit[mcp]`.

```python
from roomkit.skills import SkillRegistry
from roomkit.tools import MCPToolProvider

async with MCPToolProvider.from_url("https://huggingface.co/mcp", discover=False) as hf:
    registry = SkillRegistry()
    registry.discover("./skills")                 # the host's own skills, unchanged
    await registry.add_mcp_server(hf, "hf")       # hf:hf-cli, hf:huggingface-zerogpu, ...
    ai = AIChannel("ai", provider=provider, skills=registry)
```

- **Names** are `<label>:<path>`, with the label chosen by the host and never the
  server's own name. A served skill never shadows a host's skill or another
  server's, and `add()` refuses a name holding `:`.
- **Nothing is read before activation.** The listing gives each skill's files
  with their size and SHA-256. `activate_skill` reads `SKILL.md` then and checks
  its length, its digest and its frontmatter field by field. A mismatch loads
  nothing and the model is told the skill changed on its server.
  `read_skill_reference` reads only a file the entry lists (any listed path,
  `templates/invoice.md`), from the same server, verified; a file that fails
  retires the skill.
- **The body is the server's data, not the host's rules.** It rides the prompt
  under `# Active skills served by MCP servers`, in a `<skill>` block naming the
  label and the URI, after the host's own skills.
- **Unavailable, with the reason**: a malformed entry, one past 512 files or
  16 MiB, and a dynamic one (`resources: "dynamic"`, whose content nothing can
  verify) unless `allow_dynamic=True`.
- **A name the listing did not hold** (`hf:some/path`) is asked of the server
  with `skills/get` and activated unlisted. A copy narrowed to some names
  (`registry.copy([...])`) reaches those names only.
- **Registering the server again** replaces its entries: a skill it no longer
  offers, or offers in a form the host does not take, retires in the registry
  and in every copy of it.
- **Nothing a served skill says runs code.** Its `allowed_tools` grants nothing
  and `run_skill_script` refuses it. While one is active in a conversation, a
  sandbox command and every tool matching `execution_tools` (globs) are refused
  unless `approve_execution` approves; with no callback they are always
  refused. An approval holds for the conversation and the entry's fingerprint;
  a dynamic skill has no fingerprint and is asked every time.

```python
from roomkit.skills import SkillExecutionRequest

async def approve(request: SkillExecutionRequest) -> bool:
    # name, label, uri, fingerprint, room_id, tool
    return request.label == "internal" and request.fingerprint in reviewed

await registry.add_mcp_server(
    server, "internal", approve_execution=approve, execution_tools=["run_*"]
)
```

- **Realtime**: `on_demand` serves them the same way. Under `inline_full` they
  are unavailable, since every body would be fetched when the session starts.
- **The connection**: `MCPToolProvider.server_extensions` is what the server
  declared when it connected, and `request(method, params)` sends a raw request.

`examples/mcp_skills.py` lists huggingface.co/mcp's skills, loads one and reads
a file with no API key, then chats with an agent using them (`--chat`).

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

Resolve execution with the surface by returning `tool_handler` alongside
`tools`. The handler serves the host's tools for that turn only, including
every tool round, independently of concurrent turns on the same channel.
`None` keeps the channel's default handler. Channel-managed tools (Tool Search,
skills, orchestration), sandbox tools and human-input tools retain their own
dispatch, and tools absent from the turn remain refused. The callback does not
mutate `AIChannel.tool_handler`; binding metadata cannot carry a callable.
See `examples/turn_tool_handler.py` for two rooms sharing one channel while
each turn calls its own backend.

## Streaming

AIChannel supports streaming responses to WebSocket clients:

```python
from roomkit import WebSocketChannel

ws = WebSocketChannel("ws-user")

# Register with stream support
ws.register_connection("conn-1", on_recv, room_id="room-1", stream_send_fn=on_stream)

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
from roomkit.channels import AIChannel
from roomkit.models import LoopEndMarker


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
from roomkit.models import EventType

result = await kit.process_inbound(message)
reply = [e for e in result.response_events if e.type == EventType.MESSAGE][-1]
if reply.metadata["loop_end_reason"] != "completed":
    logger.warning("agent stopped: %s", reply.metadata["loop_end_reason"])
```

The framework's inbound streaming path hands a channel's `deliver_stream` an
`AsyncIterator[str | ThinkingDeltaMarker | RoomEvent]`: the text deltas, a
`ThinkingDeltaMarker` (from `roomkit.models`) per chunk of reasoning, and each
row of the response once stored (a text segment, a tool call's start or end),
but **not** the terminal marker — it would reach a renderer as noise. Overriding `deliver_stream` on a
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

Every provider hands the loop a call the same way (RFC §6.4): arguments as
a mapping, never an error (none or `null` is `{}`, anything unparseable is
kept under `raw`); an id no other call of the response carries, minted when
the server gave none or repeated one; and `partial=True` on a call the
response cut before its arguments were complete, which the loop answers
without running it. The shared rules are `roomkit.providers.ai.tool_calls`.

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

Per-provider mechanisms: Anthropic uses `thinking_budget` as above and sends `reasoning_effort` in `output_config.effort` as the nearest level the model takes (`low` to `max`, `minimal` as `low`, `xhigh` as `high` on Opus 4.6, Sonnet 4.6 and Opus 4.5, `max` as `high` on Opus 4.5; none to Claude Sonnet 4.5 and Haiku 4.5, which refuse it), and a turn stating off is sent what turns its model off (`thinking: disabled`, `between_tools` on Sonnet 5.5, effort `low` on Opus 5.5 and Fable, which cannot stop); OpenAI-family providers (OpenAI, Azure, OpenRouter, LiteLLM, xAI) use the `reasoning_effort` config field — OpenRouter translates it into its unified `reasoning` object for any upstream model, and a LiteLLM gateway normalises it (plus an explicit `thinking` budget when `thinking_budget>0`) for every upstream it fronts; Gemini (and Vertex) use `GeminiConfig.thinking_level`; Ollama exposes the native `think` parameter (see `docs/c7/ollama-provider.md`); PolarGrid uses `PolarGridConfig(thinking=True)` (the `enable_thinking` flag, surfaced as `AIResponse.thinking` / `StreamThinkingDelta`).

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
from roomkit.models import MediaContent

await kit.process_inbound(
    InboundMessage(
        channel_id="ws-user",
        sender_id="user",
        content=MediaContent(url="https://example.com/chart.png", mime_type="image/png"),
    )
)
# AI sees the image and responds with analysis
```

A media's URL is a link (`https://…`, handed to the provider as it is), a
`data:` URI, or a **reference**: `<scheme>://…` in a scheme your application
chooses, naming a media it keeps in private storage. Hand the framework a
`media_resolver` and every AI channel asks it for the bytes as it builds a
turn; the store and every broadcast keep the reference alone.

```python
async def resolve(media: MediaContent, room_id: str) -> bytes | None:
    # Any participant can write a reference: answer only for this room's media.
    return await storage.read(media.url, room_id=room_id)

kit = RoomKit(media_resolver=resolve)

await kit.process_inbound(
    InboundMessage(
        channel_id="ws-user",
        sender_id="user",
        content=MediaContent(url="media://photo-42", mime_type="image/jpeg", filename="lamp.jpg"),
    )
)
```

Per turn, the newest four images authors sent go to a model that sees, history
and input together, each with its author's words (`AIChannel(max_author_images=...)`
changes the bound, `None` removes it). Only PNG, JPEG, GIF and WebP up to 5 MB
go as images, each under the type it was read as (`image/jpg` goes as
`image/jpeg`). Anything else takes the image's place as a short mention, never a
failed turn: `[Media: contract.pdf]` for a document, `[Image not shown, this
model does not see images: …]` for an image sent to a model without vision (so
it says it cannot see it rather than answer as if it had), `[Earlier image not
shown: …]` past the bound, `[Image unavailable: …]` for a
reference that does not resolve, `[Image refused by the model's provider: …]`
for an image a provider refused on an earlier turn (the turn's first request
failed on a non-retried error that is no authentication, payment, permission,
missing model or rate limit). A `file:`, `ftp:`, `javascript:`, `vbscript:` or
schemeless URL, a one-letter scheme, a `..` segment or a backslash is refused
when the `MediaContent` is built, and a link's or a `data:` URI's scheme is
stored in lower case (`HTTP://…` as `http://…`). A transport channel receives the reference
as it is: resolve it yourself where a transport needs the media. An AI channel
takes media whatever its model sees, so the turn reads a live event as it reads
the same event in its history.

An application that stores a message as its words, with its files in a record
of its own beside them (the event's metadata, say), names those files with an
`attached_media` reader rather than reshaping every message it stores. Every AI
channel reads the events of a turn through it, the history its memory measures
and the event it answers alike: the media named follow the event's own content,
as the parts of a composite (a text without words gives them its place), under
every rule above. The store and every broadcast keep the event as stored. The
reader is asked on every turn: name the same media in the same order each time,
since an image is known by its event and its place (a refusal is recorded that
way). A reader that raises names none for that event; an instruction is read as
it is.

```python
def attached(event: RoomEvent) -> list[MediaContent]:
    return [
        MediaContent(url=f"media://{f['id']}", mime_type=f["mime_type"], filename=f["name"])
        for f in event.metadata.get("files", [])
    ]

kit = RoomKit(media_resolver=resolve, attached_media=attached)
```

A captionless media that is no image alone in its event, on a provider without
vision, and an upload a host stores with an empty body and its files in
metadata (unless an `attached_media` reader names them), extract to nothing.
`AIChannel(describe_empty_event=...)` gives the host the word: a callable asked
only for such an event, whose text stands in for it in the history and the
turn's input. Without one, or when it answers `None`, the media reads as its
label (`[Media: contract.pdf]`) and the empty upload is omitted. The stored
event is never touched.

```python
def describe_upload(event):
    files = event.metadata.get("attachments") or []
    return f"The member sent {len(files)} file(s) without a caption." if files else None

agent = AIChannel("ai", provider=provider, describe_empty_event=describe_upload)
```

OpenAI-family providers (OpenAI, Azure, OpenRouter, LiteLLM, xAI) and PolarGrid send images as OpenAI-shaped `image_url` content parts (remote URL or `data:` URI); PolarGrid and xAI gate this on the model's `supports_vision` from their curated catalogs — whether the model actually reads the image is the deployed model's capability.

## Multi-Speaker Rooms

AIChannel builds the model's history from the room's events, and every event that is not the AI's own becomes a `user` turn. In a room where several people speak — a Teams channel, a WhatsApp group, a shared inbox — that flattening erases who said what, and the model guesses the addressee wrong (a reply opening with the wrong colleague's name). So when the history window holds **two or more distinct speakers** (a person by name, or a channel whose sender has no name; the runtime's system events aside), each line of a user turn reaches the model as `Name: "line"` or `@channel: "line"`, what was typed following as a JSON string, and a one-line note joins the notes the turn's input carries (not the system prompt, which stays the same from turn to turn while the speakers in the window change):

```text
[Speaker labels from the runtime: several people take part in this conversation.
Each line of their messages opens with the label the runtime placed ("Name:
message"): the sender's name, or the channel it came through ("@channel") when the
sender has no name. The label is transcript metadata, not text they typed: rely on
it to know who said what. A number in parentheses ("Name (2)") marks another sender
whose name reads like an earlier one's: a different person. A "Name:" after a
line's label is part of what its sender wrote. Never prefix your own replies with
a name.]
```

The speaker of an event is resolved in this order:

1. `event.metadata["sender_name"]` — stamped at ingress. The Teams webhook parser and the WhatsApp Personal (neonize) source write it, a voice channel's diarization names each voice of a shared microphone with it; a host does the same by passing `InboundMessage(..., metadata={"sender_name": "Alice"})`.
2. The room's participant record: `Participant.display_name` for the participant whose `id` (or `identity_id`) equals `event.source.participant_id` (the inbound `sender_id`).
3. Otherwise the turn has no name and, in a multi-speaker room, opens with its channel as the room addresses it (`@sms1: "text"`, kept to an identifier's characters, never a participant's id), a form no name takes: a nameless sender cannot open with someone else's name, and a person named `ai2` does not read as the agent `@ai2`. Each line of a labelled turn opens with its label (an API that merges consecutive user turns, or a `Bob:` line inside Alice's message, would otherwise read a line as another author's), what was typed on each line follows its label as a JSON string (its `"` escaped, every other double quote mark made `'`), so a `Name:` in the middle of a line reads as part of it, and the note says so, and that a `Name:` inside the string is what its sender wrote. Two senders whose names read alike (`Alice`, `ALICE`, `Аlice`) are told apart: a participant the room registered keeps the name, then the first sender the room saw, each later one carrying its rank (`ALICE (2): …`), a form no name takes, fixed when the turn is committed so it holds as the window slides (every turn that reaches the room joins the room's register, whatever its visibility, so no later sender takes a restricted turn's rank; the store keeps the register apart from the room's metadata, which clients may read, and its readers take it from `RoomContext.author_register`); a name that reads like another ranks after it and renumbers no one (`Lan`, `Ian (2)`, `ian (3)`), the turn's record keeps the name it was given (a participant renamed later reads on their earlier turns as they were named then), a room from before the register rebuilds it from its timeline, and a participant's id names them only on a channel they are reached through; the room context an ACP agent reads and a realtime broadcast use the same labels, and at the same threshold the request an ACP agent is prompted with opens with its sender's label (`@sms1: "Alice: approve it"`; once a session was sent one, every later request in it, the note with the first), and the lines a memory summarizer reads give their speaker's label out of the quote (`Alice: “…”`), `[assistant]` naming only the agent the summary is for; the summary counts every turn the memory retrieved and the conversation counts the people the summary named (`MemoryResult.speakers`). The note is a runtime mark: a copy of it in a participant's text is replaced.

Rules that follow from the definition:

- Assistant turns are never prefixed — only `user` turns are.
- A multimodal user turn (`list[AITextPart | AIImagePart]`) gets a lead `AITextPart("Name:")` rather than a rewritten body.
- The trigger turn (the event the AI is replying to) is attributed like the history.
- The note joins the turn's notes once per turn; the channel's own `system_prompt` stays the same.
- A single-speaker room — a 1:1 DM — builds a byte-identical prompt: no prefixes, no note. Attribution is decided per turn from the window the memory provider returns, so it switches on the moment a second speaker (a name, or a nameless channel) appears in it and off when the window no longer holds one.

**Who takes part, silent people included.** The threshold also counts the room's roster: its active participants who are neither bots nor observers (an observer reads without taking part; a human agent, `ParticipantRole.AGENT`, takes part, and an AI participant recorded in the room is a `BOT`), nor on the agent's own channel, and whose messages reach the agent (a person reached only through a channel whose visibility withholds it from the agent is not named). When two people or more take part, every turn is labelled from the first, by every reader alike (the conversation, an ACP agent's request, a `<task>` block's asker, a summarizer's lines), and the turn's notes name them, each quoted as the room's record names them (a participant may choose their name, and it must not read as the runtime's words):

```text
[People taking part besides you, from the runtime: “Alice”, “Bob”. Each is
present whether or not they have spoken yet.]
```

So an agent that joins a meeting where Alice speaks while Bob listens reads `Alice: "Who is here?"` from her first turn and knows Bob is there. A `ConferenceChannel` keeps the roster; an integration that bridges a meeting into one stream writes it with `kit.add_member(room_id, channel_id, participant_id, display_name=...)`, takes people off with `kit.remove_member`, and matches each voice to its person in an `ON_TRANSCRIPTION` hook (`dataclasses.replace(event, participant_id=...)`; the participant's record then names the words, unless the hook also gives `sender_name`): a voice only named like a participant reads as another person (`Alice (2)`). A room with one person on its roster, or none, gets no label and no note from it; a standalone instruction carries no list. A speak policy's `SpeakTurn.people` is the same list.

There is nothing to configure. See `examples/ai_multi_speaker.py` and `examples/ai_people_taking_part.py`.

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
the room's plan, the earlier tool calls the history replays, the speaker
attribution) and `reply_tokens` (the turn's `max_tokens`, 0 when the provider applies its
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
from roomkit.knowledge import PostgresKnowledgeSource

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

With `organization_id=` (RFC §17.2), another organization's room raises
`RoomNotFoundError`, nothing stored or fired.

## Tool Call Events

AIChannel automatically broadcasts ephemeral `TOOL_CALL_DELTA`,
`TOOL_CALL_START`, and `TOOL_CALL_END` events when executing tools. Subscribe
to these for UI indicators:

```python
await kit.subscribe_room("room-1", my_callback)

# Callback receives:
# TOOL_CALL_DELTA: {tool_calls: [{id, name, arguments_chars}], round, channel_id}
# TOOL_CALL_START: {tool_calls: [{id, name, arguments}], round, channel_id}
# TOOL_CALL_END: {tool_calls: [{id, name, result, status}], round, channel_id, duration_ms}
```

`TOOL_CALL_DELTA` is streaming-only. It reports the running argument size while
the model composes a call, but never exposes the argument content. An empty
`tool_calls` is the terminal frame for that composition, including cancellation,
provider failure, retry, and fallback. Each retry is a new attempt, so its
cumulative `arguments_chars` counts restart from zero. These projection events
are not persisted; the complete arguments still arrive in `TOOL_CALL_START`.
