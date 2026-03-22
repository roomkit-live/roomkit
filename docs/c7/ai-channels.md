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
        model="claude-sonnet-4-20250514",
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
| Mistral | `MistralAIProvider` | `MistralConfig` | `roomkit[mistral]` |
| Azure OpenAI | `AzureAIProvider` | `AzureAIConfig` | `roomkit[azure]` |
| vLLM (local) | `create_vllm_provider()` | `VLLMConfig` | `roomkit[vllm]` |
| Mock (testing) | `MockAIProvider` | — | built-in |

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

## AI Thinking/Reasoning

Some providers support extended thinking:

```python
ai = AIChannel(
    "ai",
    provider=AnthropicAIProvider(AnthropicConfig(
        api_key="...",
        model="claude-sonnet-4-20250514",
    )),
    system_prompt="Think step by step.",
    thinking_budget=4096,  # Setting a budget enables thinking mode
)
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

When enabled, the AI gets a `_plan_tasks` tool that accepts a list of tasks with `title` and `status` (`pending`, `in_progress`, `completed`, `blocked`). The current plan is:
- Injected into the system prompt on each turn (so the AI sees its progress)
- Published as an ephemeral `CUSTOM` event with `data.type = "plan_updated"` for real-time UI rendering

Subscribe to plan updates for UI:

```python
await kit.subscribe_room("room-1", my_callback)

# Callback receives ephemeral event with:
# type: "custom", data: {"type": "plan_updated", "tasks": [...]}
```

Hook into plan updates:

```python
kit.hook(
    HookTrigger.ON_PLAN_UPDATED,
    execution=HookExecution.ASYNC,
    fn=lambda event, ctx: save_plan_to_db(event),
)
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
