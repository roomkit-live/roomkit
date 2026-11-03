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
    mcp_servers=None,                  # ACP MCP-server descriptors, or room_id -> them
    session_config=None,               # {"mode": "default"}: set on every session opened
    authentication_method=None,        # optional ACP auth method id
    external_tool_handler=None,        # permission policy; None = reject all
    name=None, role=None, description=None,  # its identity, as an Agent's
    instructions=None,                 # standing direction, sent once per session
)
```

`instructions` open the first prompt of each session the channel opens for a
room (again after a session is opened again), as the application's block;
they are not an `INSTRUCTION`, which directs one turn. In a `Discussion` an
ACPChannel takes turns like an AI channel: the turn's notes close its prompt
under the runtime's header (copies elsewhere replaced), and `listen_only`
while it speaks cancels its prompt, through `kit.steer` wherever the turn runs.

`command` is an argument vector executed directly, **no shell**; a bare string, empty/non-string args, or non-absolute `cwd`/`additional_directories` raise `ValueError`. Exactly one of `command` / `transport` is required (neither or both → `ValueError`), and `env`/`inherit_env` next to a `transport` raise rather than being ignored — they configure the spawn. Class attrs: `channel_type = ChannelType.AI`, `category = ChannelCategory.INTELLIGENCE`, `direction = BIDIRECTIONAL`; capabilities TEXT + RICH. `handle_inbound()` raises `NotImplementedError` — the channel reacts to Room events via `on_event()`.

### Transports

`ACPTransport` (ABC, `channels/acp_transport.py`) is the pipe, and nothing more: `open(client, *, queue) -> ClientSideConnection` (build it with `acp.connect_to_agent(client, writer, reader, queue=queue)` — the protocol only needs a reader/writer pair; `queue` is `None` under SDK ≥0.12.1, which removed it and rejects the keyword, so forward it only when set), `close()` (must not raise; called on teardown *and* on a failed handshake), `is_alive()` (default `True`), and a `name` property surfaced as `info["transport"]`. `StdioACPTransport(command, cwd=…, env=…, inherit_env=…)` is the default, constructed for you from `command=`, and owns the spawn: argument-vector validation, `_resolve_spawn_env`, the stderr drain, and `returncode`-based liveness. Everything protocol-level — `initialize`, version negotiation, `authenticate`, sessions, prompts, permissions, config options, event mapping — stays on the channel, so a transport inherits it.

### Recovering a refused room prompt

A custom transport's connection may raise
`ACPSessionInvalidatedError(reason, recovery_authorized=True)` from `prompt()`
only after proving the session is unusable **and no part of the prompt ran**.
Authorization belongs to the host: reserve any durable retry identity before
raising, and keep the refusal's receipt. An ordinary error or the default
`recovery_authorized=False` authorizes nothing. This is a local Python contract,
not a new ACP wire error or a general retry policy.

RoomKit keeps the room's turn lock, closes the failed stream, forgets that
session and its catch-up cursor, then opens a session normally on the same
connection. The same event id, contributed blocks and request are prompted once
more, with the visible room tail recomposed under `room_history`. Opening uses
the usual cwd, MCP servers and transport-provided instructions/configuration;
RoomKit does not restore private agent state or copy obsolete session options.
A successful recovery reports one response without a stale `interrupted` mark.

Any observed update (including a plan, usage or thinking) or permission request
makes the refusal terminal. So do a second refusal, an opening failure and a
standalone turn. Cancellation/abandonment during recovery discards the replacement
session and releases the lock. Existing transports do not opt in implicitly.
The host must enforce its retry budget across redelivery, restarts and workers;
RoomKit limits only this one stream to two attempts.

Runnable without a remote agent: `examples/acp_session_recovery.py`.

### Session setup

What a session is opened with is set per room (RFC §6.4, ACP agent channel, *Session setup*). `mcp_servers` is a list given to every room's sessions, or a function of the room id (sync or async), called each time a session opens for that room: a stdio server can be told its room in its `env`. `session_config` maps ACP config ids to values (`mode`, `model`, an agent's own options) and is set on every session the channel opens, before its first prompt: the room's first, one rebuilt after a recovery, one opened after the agent restarted, a standalone turn's, a task's child room's. A value the session already announces is not set again. A value the host sets with `set_config_option(room_id, id, value)` belongs to the room: its next session is set to it too, over `session_config`, until `close_session(room_id)` forgets it. Set through `kit.channel_call(channel_id, room_id, "set_config_option", config_id=..., value=...)`, it is stored with the room (room metadata `roomkit:channel_state:<channel_id>`, RFC §6.4), so the session another executor opens for the room after this one is lost is set to it as well; a detach forgets it. A value the agent refuses fails the opening closed: the session is released, never prompted, and the turn fails (or `set_config_option` raises) with a `RuntimeError` naming the option. A `set_config_option` that opens the room's session sets its value in place of the room's earlier one, so a value a restarted agent no longer accepts is replaced by setting another (or forgotten with `close_session`).

```python
from acp.schema import EnvVariable, McpServerStdio

agent = ACPChannel(
    "claude-code",
    command=["npx", "--no-install", "@agentclientprotocol/claude-agent-acp@0.61.0"],
    cwd="/srv/workspace",
    mcp_servers=lambda room_id: [
        McpServerStdio(name="notes", command="notes-server", args=[],
                       env=[EnvVariable(name="NOTES_ROOM", value=room_id)]),
    ],
    session_config={"mode": "default"},  # every MCP call and write is asked of RoomKit
    # An MCP call is judged under Claude Code's alias and its own name: allow both.
    external_tool_handler=PolicyExternalToolHandler(
        policy=ToolPolicy(allow=["mcp__notes__*", "add_note", "list_notes"]),
    ),
)
```

Example: `examples/acp_room_mcp.py`.

### Process and session model

One connection per channel, opened lazily on the first prompt; one ACP **session** per Room, created on demand; its `session/new` `_meta` carries `roomkit.live/roomId` and `roomkit.live/sessionScope: "room"`. A standalone `INSTRUCTION` (RFC §10.1.1 step 7) runs in a session of its own, opened for that turn with `roomkit.live/sessionScope: "turn"` and closed after it; the room's session is neither prompted nor told, and catches up on the reply at its next turn. A relay that keeps one remote session per room must file a `"turn"` session apart and close only that one; one that answers the turn with an open session fails the turn with `RuntimeError`. Prompts are serialized per Room (per-room lock); different Rooms progress concurrently through the same connection. When the transport reports the connection dead (for stdio: the subprocess exited), the next prompt reconnects and clears all session mappings — a reconnect never resumes sessions. The client declares no fs/terminal capabilities: `fs/*`, `terminal/*`, and `session/request_input` (elicitation) requests get `method_not_found` — `ON_USER_INPUT_REQUIRED` never fires from ACP.

Methods: `session_id(room_id)` returns the process-local session id or `None`; `cancel(room_id)` cancels the active turn (`True` if a cancel was sent), a turn whose session is still being opened or set up included: it then ends without prompting, marked `interrupted`; a room's session is opened by one caller at a time (a first prompt, a rebuild, a host call), the others find it; `close_session(room_id)` closes and forgets one Room's session and never raises for a refused close (`session/close` is optional in ACP: it is sent only to an agent whose `initialize` announces `sessionCapabilities.close`, or to a connection whose `initialize` reply carries no `agent_capabilities` at all; an agent that does not announce it keeps its sessions until the connection closes, with a warning per standalone turn); `close()` cancels turns, closes sessions, closes the transport and stops the handler. The `info` property reports `{transport, protocol_version, sdk_version, connected, agent, session_count}`.

Through the kit, wherever the agent's turns run (RFC §10.6.3): `await kit.channel_call(channel_id, room_id, verb, *, organization_id=None, **arguments)` runs `set_config_option` (`config_id`, `value`), `close_session`, `session_config` or `config_options` on the executor holding the room's session and returns what the method returns (another verb raises `ValueError`; a channel not attached to the room raises `ChannelNotFoundError`); `await kit.steer(Cancel(reason=...), channel_id, room_id=...)` cancels the room's running turn as `cancel` does and answers 1: the channel's `steer(directive, *, turn_id)` takes a `Cancel` only, reaching the turn it names, and a turn that has not reached the agent yet is cut by its owner instead (`steer` answers `None`). Detaching the agent from a room closes the room's session and forgets the values set on it, never waiting for the agent: in the background once the room's running turn ended (a standalone turn's included), or by that turn itself; a turn started before the detach that has not prompted yet ends without prompting, marked `interrupted`. The room's channel state (`roomkit:channel_state:<channel_id>`) is written by `channel_call` alone: `update_room_metadata`, `create_room` and a turn's `TurnHost.update_room_metadata` drop it with a warning, as every `roomkit:` key, and write the rest. A value set by calling the channel object directly is this process's copy: a later `kit.channel_call` stores it only when that call changes the room's values, never a read.

`acp_event_text(event)` (exported from `roomkit`) is the text the channel gives the agent for an event, its prompt and each line of room context: a `RichContent` is read as its `plain_text`, where `extract_event_text` keeps the markup body. A host that builds an ACP prompt of its own (a relay on the user's machine, say) reads events through it to send what the channel sends.

### How agent output enters the room

For each non-self text event, `on_event()` returns `ChannelOutput(responded=True, response_stream=...)` (TOOL_CALL_START/END events are skipped). The prompt (tagged `roomkit.live/eventId`) yields a `StreamDelta` stream consumed by the inbound-streaming pipeline:

- `agent_message_chunk` → `str` deltas → streamed to transports, persisted as the response event.
- `agent_thought_chunk` → `ThinkingDeltaMarker` in the stream, plus ephemeral `THINKING_START` / `THINKING_DELTA` (`thinking_length`, and the text truncated to 1000 chars with `publish_thinking_text=True`) / `THINKING_END`.
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

The handler hears only of the calls the agent submits, and the agent's own mode decides which: Claude Code in `auto` mode (a user's settings, inherited with `HOME`) acts without asking; `default` asks before any action that is not a read; `plan` refuses them itself. Set it on every session with `session_config={"mode": "default"}`, or for one room with `await kit.channel_call(channel_id, room_id, "set_config_option", config_id="mode", value="default")`, which the room keeps across a recovery, an agent restart or another executor.

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
    agent_label=None,           # channel_id -> display name; default: the agent's discussion handle, else "agent-researcher" -> "Researcher"
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
        # ON_TOOL_CALL hooks; the keywords (is_error, tool_call_id, ...) are its own
        await self._fire_on_tool_hook(tool_name, tool_input, result, **kwargs)

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
