# ACP Agent Channel and CLI Channel

`ACPChannel` makes RoomKit an **Agent Client Protocol client**: an external coding agent (Claude Code, Codex CLI, Gemini CLI, any registry-listed ACP agent) speaks ACP v1 — spawned here as a subprocess over stdio by default, or reached through a caller-supplied `ACPTransport`; each Room becomes one session of that agent. The reverse direction — exposing a RoomKit agent as an ACP *server* — is out of scope. `CLIChannel` is the interactive terminal transport playing the human side in local sessions.

```bash
pip install "roomkit[acp]"      # agent-client-protocol>=0.11.0,<0.12
pip install "roomkit[console]"  # rich>=13.0, for CLIChannel(markdown=True)
```

The SDK is imported lazily: `import roomkit` works without the extra; the first connection raises an actionable `ImportError`. RoomKit pins stable ACP wire protocol v1 and rejects any other negotiated `protocolVersion`.

## ACPChannel

```python
from roomkit import ACPChannel

agent = ACPChannel(
    "coding-agent",
    command=["npx", "-y", "@agentclientprotocol/claude-agent-acp@0.61.0"],
    transport=None,                    # or an ACPTransport, instead of command
    cwd="/srv/workspaces/my-project",  # required, absolute — on the AGENT's host
    additional_directories=None,       # extra absolute dirs for the session
    env=None,                          # added to the SDK's restricted env
    mcp_servers=None,                  # ACP MCP-server descriptors (SDK types)
    authentication_method=None,        # optional ACP auth method id
    external_tool_handler=None,        # permission policy; None = reject all
)
```

`command` is an argument vector executed directly, **no shell**; a bare string, empty/non-string args, or non-absolute `cwd`/`additional_directories` raise `ValueError`. Exactly one of `command` / `transport` is required (neither or both → `ValueError`), and `env`/`inherit_env` next to a `transport` raise rather than being ignored — they configure the spawn. Class attrs: `channel_type = ChannelType.AI`, `category = ChannelCategory.INTELLIGENCE`, `direction = BIDIRECTIONAL`; capabilities TEXT + RICH. `handle_inbound()` raises `NotImplementedError` — the channel reacts to Room events via `on_event()`.

### Transports

`ACPTransport` (ABC, `channels/acp_transport.py`) is the pipe, and nothing more: `open(client, *, queue) -> ClientSideConnection` (build it with `acp.connect_to_agent(client, writer, reader, queue=queue)` — the protocol only needs a reader/writer pair), `close()` (must not raise; called on teardown *and* on a failed handshake), `is_alive()` (default `True`), and a `name` property surfaced as `info["transport"]`. `StdioACPTransport(command, cwd=…, env=…, inherit_env=…)` is the default, constructed for you from `command=`, and owns the spawn: argument-vector validation, `_resolve_spawn_env`, the stderr drain, and `returncode`-based liveness. Everything protocol-level — `initialize`, version negotiation, `authenticate`, sessions, prompts, permissions, config options, event mapping — stays on the channel, so a transport inherits it.

### Process and session model

One connection per channel, opened lazily on the first prompt; one ACP **session** per Room, created on demand and tagged with extension key `roomkit.live/roomId`. Prompts are serialized per Room (per-room lock); different Rooms progress concurrently through the same connection. When the transport reports the connection dead (for stdio: the subprocess exited), the next prompt reconnects and clears all session mappings — a reconnect never resumes sessions. The client declares no fs/terminal capabilities: `fs/*`, `terminal/*`, and `session/request_input` (elicitation) requests get `method_not_found` — `ON_USER_INPUT_REQUIRED` never fires from ACP.

Methods: `session_id(room_id)` returns the process-local session id or `None`; `cancel(room_id)` cancels the active turn (`True` if a cancel was sent); `close_session(room_id)` closes and forgets one Room's session; `close()` cancels turns, closes sessions, closes the transport and stops the handler. The `info` property reports `{transport, protocol_version, sdk_version, connected, agent, session_count}`.

### How agent output enters the room

For each non-self text event, `on_event()` returns `ChannelOutput(responded=True, response_stream=...)` (TOOL_CALL_START/END events are skipped). The prompt (tagged `roomkit.live/eventId`) yields a `StreamDelta` stream consumed by the inbound-streaming pipeline:

- `agent_message_chunk` → `str` deltas → streamed to transports, persisted as the response event.
- `agent_thought_chunk` → `ThinkingDeltaMarker` in the stream, plus ephemeral `THINKING_START` / `THINKING_DELTA` (thinking truncated to 1000 chars) / `THINKING_END`.
- `tool_call` / `tool_call_update` → `ToolCallStartMarker` / `ToolCallEndMarker` in the stream (persisted as `TOOL_CALL_START`/`TOOL_CALL_END` RoomEvents) plus matching ephemeral events (`result` truncated to 500 chars, `duration_ms`); non-terminal progress → ephemeral `CUSTOM` `{"type": "acp_tool_progress"}`.
- `plan` / `plan_update` / `plan_removed` → ephemeral `CUSTOM` `{"type": "acp_plan_update", session_id, update}`. **Not** `ON_PLAN_UPDATED` — that hook belongs to AIChannel's `plan_tasks` tool.
- `usage_update` → ephemeral `CUSTOM` `{"type": "acp_usage"}`.

`register_channel()` wires `kit`'s realtime backend in automatically. A failed prompt surfaces as `ProviderError(provider="acp")`; cancellation ends the stream silently and closes any open thinking block.

Every output starts a live response record containing
`{"acp": {"protocol_version": "..."}}`. When the prompt returns for a reason
other than `end_turn`, the channel adds `stop_reason`; if it never returns
because of an exception or cancellation, it adds `interrupted: true`. A clean
turn adds neither marker. The final record is available on
`InboundResult.response_metadata`, even when the last activity was a tool call
and no final `MESSAGE` was persisted. With `defer_delivery=True`, await
`result.delivery.wait()` before reading it.

A turn never outlives its tool calls. Whichever way it ends — error, cancellation, or a stop the user asked for, which returns through the ordinary end of a prompt — every tool started without a terminal `tool_call_update` is closed first: a `ToolCallEndMarker` with `status="failed"` and an `error` saying the turn ended before the tool reported, emitted into the stream so the stored `TOOL_CALL_END` exists, plus the matching ephemeral. A turn whose tools all reported emits nothing extra. One gap remains by construction: a stream closed from the outside (its consumer cancelled, a muted binding) is past yielding, so only the ephemeral goes out and the stored row stays pending.

ACP fixes the envelope and leaves the payload to the agent, so `CLIChannel(console=True)` unwraps rather than prints (`roomkit.console._tool_preview`): ACP `text`/`diff` blocks, MCP `content`, and `raw_output` wrappers (`formatted_output`+`exit_code` from Codex, `output`, `result`/`error`) all reduce to their text; a `terminal` block carries no text, so the preview falls back to `raw_output`; `image`/`audio`/`resource` blocks are named, never dumped as base64; 5 lines per result, 200 chars per line; unknown shapes render as compact JSON.

### Permission flow

Every agent `session/request_permission` goes to the `external_tool_handler` (`ExternalToolHandler` ABC from `roomkit.tools`, with `PolicyExternalToolHandler` and `ToolDecision`):

1. `process_tool_call(tool_name, tool_input, *, tool_call_id, session_id, room_id, ...)` → `ToolDecision(approved=...)`. Calling `self._fire_before_hook(...)` fires **`BEFORE_TOOL_USE`** sync hooks (callbacks injected at `register_channel`).
2. Approved → RoomKit selects the agent's `allow_once`/`allow_always` option; denied → `reject_once`/`reject_always`; no matching option → `DeniedOutcome(outcome="cancelled")`.
3. **No handler ⇒ every permission request is rejected.** `ToolDecision.modified_input`/`result` overrides cannot be applied over ACP — setting them logs a warning and rejects the call.
4. On tool completion, `on_tool_result(...)` runs; via `_fire_on_tool_hook` it fires **`ON_TOOL_CALL`** hooks.

## CLIChannel

Interactive terminal transport (`channel_type = ChannelType.CLI`, TEXT only): reads stdin, prints agent output to stdout with ANSI colors.

```python
cli = CLIChannel(
    "cli",                      # channel_id, default "cli"
    prompt="You: ",
    user_color="\033[33m",      # yellow
    agent_color="\033[36m",     # cyan
    thinking_color="\033[2;3m", # dim italic
    use_color=None,             # None = auto-detect TTY
    agent_label=None,           # channel_id -> display name ("agent-researcher" -> "Researcher")
    show_thinking=False,        # render ThinkingDeltaMarker chunks
    markdown=False,             # live Markdown rendering, requires roomkit[console]
)
```

- `sender_is_participant = True`: `run()`'s `sender_id` is a room **Participant ID**, not an address — identity resolution is skipped for typed lines.
- `deliver()` skips self-echo and prints `Label: text`. `supports_streaming_delivery` is `True`; `deliver_stream()` renders text deltas as they arrive, thinking above the answer, tool events inline (`🔧 name {args}`, `✓`/`✗ name (N ms)`).
- `run(kit, room_id, *, sender_id="user", welcome=None, content_factory=None)` — input loop in a worker thread; each line becomes `kit.process_inbound(...)` with `TextContent(body=line)`, or `content_factory(line)` (`None` skips the line). `quit`/`exit`/`q` or Ctrl+D exits.

## Example: terminal → Claude Code

Condensed from `examples/acp_claude_code.py` (requires `roomkit[acp,console]`, Node.js 22+):

```python
from roomkit import ACPChannel, ChannelCategory, CLIChannel, RoomKit
from roomkit.tools import ExternalToolHandler, ToolDecision

class TerminalPermissionHandler(ExternalToolHandler):
    async def process_tool_call(self, tool_name, tool_input, *, tool_call_id="",
                                room_id=None, **kwargs) -> ToolDecision:
        if not await self._fire_before_hook(tool_name, tool_input,
                                            tool_call_id=tool_call_id, room_id=room_id):
            return ToolDecision(approved=False, reason="Denied by BEFORE_TOOL_USE hook")
        answer = await asyncio.to_thread(input, f"Allow {tool_name}? [y/N] ")
        return ToolDecision(approved=answer.strip().casefold() in {"y", "yes"})

    async def on_tool_result(self, tool_name, tool_input, result, **kwargs) -> None:
        await self._fire_on_tool_hook(tool_name, tool_input, result)  # ON_TOOL_CALL hooks

kit = RoomKit()
cli = CLIChannel("you", show_thinking=True, markdown=True,
                 agent_label=lambda _cid: "Claude Code")
claude = ACPChannel(
    "claude-code",
    command=["npx", "-y", "@agentclientprotocol/claude-agent-acp@0.61.0"],
    cwd=workspace,  # absolute Path to the project
    env={"ANTHROPIC_API_KEY": api_key, "MAX_THINKING_TOKENS": "1024"},
    external_tool_handler=TerminalPermissionHandler(),
)
kit.register_channel(cli)
kit.register_channel(claude)
await kit.create_room(room_id="claude-code-cli")
await kit.attach_channel("claude-code-cli", "you")
await kit.attach_channel("claude-code-cli", "claude-code",
                         category=ChannelCategory.INTELLIGENCE)
await cli.run(kit, room_id="claude-code-cli")  # blocks until quit/Ctrl+D
await kit.close()
```

Flow: terminal line → `CLIChannel` inbound → Room broadcast → `ACPChannel.on_event` → ACP prompt to Claude Code; deltas, thinking, and tool activity stream back through the Room and render live in the terminal, each permission approved once at the prompt.
