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

Register a handler to execute tool calls:

```python
@ai.tool_handler
async def handle_tools(name: str, arguments: dict) -> str:
    if name == "get_weather":
        city = arguments["city"]
        return f'{{"temperature": 22, "condition": "sunny", "city": "{city}"}}'
    return '{"error": "Unknown tool"}'
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
    thinking=True,
    thinking_budget=4096,
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

## Tool Call Events

AIChannel automatically broadcasts ephemeral `TOOL_CALL_START` and `TOOL_CALL_END` events when executing tools. Subscribe to these for UI indicators:

```python
await kit.subscribe_room("room-1", my_callback)

# Callback receives:
# TOOL_CALL_START: {tool_calls: [{id, name, arguments}], round, channel_id}
# TOOL_CALL_END: {tool_calls: [{id, name, result}], round, channel_id, duration_ms}
```
