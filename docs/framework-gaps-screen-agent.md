# Framework Gaps — Screen Agent Use Case

Identified during development of `screen_assistant_ia.py` and `screen_agent_orchestrated.py`. These are improvements needed in the RoomKit framework for robust AI-driven screen/browser automation.

## Fixed (committed)

### 1. AIChannel missing `tools` constructor parameter
- **File**: `src/roomkit/channels/ai.py`
- **Issue**: `AIChannel` accepted `tool_handler` but not `tools` (definitions). Tools could only be injected via internal `_extra_tools` or binding metadata.
- **Fix**: Added `tools: list[AITool] | None = None` parameter. Commit `2f7c492`.

### 2. Streaming tool loop didn't capture token usage
- **File**: `src/roomkit/channels/ai.py`
- **Issue**: Non-streaming `on_event` captured usage from `AIResponse.usage` into telemetry spans. Streaming path (`_run_streaming_tool_loop`) ignored `StreamDone` events which carry the same usage data. Result: execution agent tokens always showed 0.
- **Fix**: Added `StreamDone` handling in the streaming loop that records `roomkit.llm.input_tokens` / `roomkit.llm.output_tokens` metrics. Commit `32d01e5`.

### 3. Gemini FunctionResponse rejected string tool results
- **File**: `src/roomkit/providers/gemini/realtime.py`
- **Issue**: `json.loads('"some string"')` returns a Python `str`, not a `dict`. Gemini's `FunctionResponse(response=...)` requires a dict. When the hook-based tool handler returned a plain string result, Pydantic rejected it.
- **Fix**: Check `isinstance(parsed, dict)` and wrap non-dict values in `{"result": parsed}`. Commit `b0c16e1`.

### 4. OpenAI/Gemini realtime providers didn't track token usage
- **File**: `src/roomkit/providers/openai/realtime.py`, `src/roomkit/providers/gemini/realtime.py`
- **Issue**: `response.done` (OpenAI) and `usage_metadata` (Gemini) carry token counts but the providers ignored them. No way to track realtime voice costs.
- **Fix**: Extract usage, log it, record via telemetry metrics, attach to turn spans. Commit `39a72a1`.

### 5. Vision click_element crashes on malformed Gemini JSON
- **File**: `src/roomkit/video/vision/screen_input.py`
- **Issue**: Gemini sometimes returns garbled bounding box JSON (missing keys, stray quotes). `KeyError: 'y1'` crashes the click handler. Also, coordinates outside image bounds (dock at y=957 on 900px screen) caused off-screen clicks.
- **Fix**: Validate all box keys before access, bounds check before clicking, fallback regex JSON parser. Commit `39a72a1`.

## Open Gaps (not yet fixed)

### 6. No hook-based tool execution for RealtimeVoiceChannel
- **Impact**: High
- **Issue**: `RealtimeVoiceChannel` supports `tool_handler` callback OR `ON_REALTIME_TOOL_CALL` hook, but not both. The hook path returns results via `HookResult.modify(event)` with metadata, which is fragile (requires constructing a full `RoomEvent`). The callback path is simpler but bypasses the hook pipeline.
- **Suggestion**: Add a simpler hook return path — let the hook return a plain string/dict result instead of requiring a full `RoomEvent`. Or support both callback AND hook (callback first, hook as fallback).

### ~~7. Delegation doesn't work with RealtimeVoiceChannel natively~~ ✅ Fixed
- **Fix**: Added `setup_realtime_delegation()` that injects the delegate tool dict into `_tools` and wraps `_tool_handler` with session-aware room_id resolution via `get_current_voice_session()`.

### ~~8. Streaming tool loop usage not in telemetry spans~~ ✅ Fixed
- **Fix**: Added `_total_input_tokens`/`_total_output_tokens` accumulators across streaming rounds. Moved `end_span` from `else` to `finally` block (async generator exits via `return`, skipping `else`). Span attributes now contain summed tokens across all tool rounds.

### ~~9. No framework-level screen observation for RealtimeVoiceChannel~~ ✅ Fixed
- **Fix**: Added `setup_realtime_vision()` in `video/ai_integration.py` — listens for `video_vision_result` events and calls `channel.inject_text(session, context, silent=True)` on all active sessions. Includes dedup to skip unchanged descriptions.

### 10. `press_key` tool description not OS-aware
- **Impact**: Low (fixed in screen_input.py but could be framework-level)
- **Issue**: Tool descriptions say `ctrl+l` on all platforms. On macOS, the correct key is `command+l`. The LLM sends `ctrl+t` which does nothing on Mac.
- **Fix applied**: OS-aware `_press_key_description()` in `screen_input.py`. Could be promoted to the framework if more tools need OS-aware descriptions.

### ~~11. Background task result delivery to RealtimeVoiceChannel~~ ✅ Fixed
- **Fix**: Both `WaitForIdleDelivery` and `ImmediateDelivery` now detect `RealtimeVoiceChannel` and deliver via `inject_text()` on active sessions instead of `process_inbound()`.

### ~~12. Playwright MCP tool schemas incompatible with Gemini~~ ✅ Fixed
- **Fix**: Added `clean_gemini_schema()` in `providers/gemini/schema.py` that recursively strips `$schema`, `additionalProperties`, `default`, `title`. Applied in both `gemini/ai.py` and `gemini/realtime.py` when building `FunctionDeclaration`.

## Discovered during orchestration testing (2026-03-16)

### ~~13. Delegation re-spawns after task completion~~ ✅ Fixed
- **Fix**: Added `CompletedTaskCache` in `tasks/cache.py` with TTL-based expiry. `DelegateHandler` accepts optional `cache` parameter — checks cache before delegating and returns cached result for matching `(room_id, agent_id, task_hash)` keys.

### ~~14. Multiple concurrent delegations create screen conflicts~~ ✅ Fixed
- **Fix**: Added `serialize_per_room` option to `DelegateHandler` — uses per-room `asyncio.Lock` to ensure only one delegation runs at a time per room.

### ~~15. Delegated agent context doesn't include previous task results~~ ✅ Fixed
- **Fix**: When `CompletedTaskCache` is provided, `DelegateHandler` injects `previous_tasks` (list of recent task descriptions) into the delegation context.

### ~~16. `pyautogui.typewrite()` fails on non-US keyboard layouts~~ ✅ Fixed
- **Fix**: Replaced `typewrite()` with `_clipboard_paste()` helper: uses `pbcopy`+`cmd+v` on macOS, `clip`+`ctrl+v` on Windows, `xclip`+`ctrl+v` on Linux. Falls back to `typewrite` on failure.

### ~~17. No shared status bus between agents~~ ✅ Fixed
- **Fix**: `StatusBus` is now a first-class framework component. `RoomKit(status_bus=...)` accepts an optional custom bus; defaults to `StatusBus()` with `InMemoryStatusBackend`. Available as `kit.status_bus`. Posts emit `status_posted` framework events via `kit.on("status_posted")`. `ON_STATUS_POSTED` hook trigger added. Bus is closed automatically on `kit.close()`.

### 18. No standard audit interface for tool execution
- **Impact**: Medium
- **Issue**: Tool call auditing (input, output, timing, status) is implemented ad-hoc in each example. Every example reinvents JSONL logging, summary printing, and cost tracking. There's no framework ABC that tool handlers can implement to get automatic auditing.
- **Current workaround**: Manual `_audit()` function, `_audit_entries` list, `_print_audit()` in each example.
- **Proposal**: Add `ToolAudit` ABC and built-in implementations:
  ```python
  class ToolAuditEntry(BaseModel):
      ts: datetime
      agent_id: str
      tool_name: str
      arguments: dict[str, Any]
      result: str
      status: str  # ok | failed | error
      duration_ms: float
      metadata: dict[str, Any] = Field(default_factory=dict)

  class ToolAuditor(ABC):
      """ABC for tool execution audit logging."""

      @abstractmethod
      def record(self, entry: ToolAuditEntry) -> None: ...

      @abstractmethod
      def summary(self) -> str: ...

  class JSONLToolAuditor(ToolAuditor):
      """Writes audit entries to JSONL file + prints summary."""
      def __init__(self, path: Path): ...

  class ConsoleToolAuditor(ToolAuditor):
      """Prints audit entries to console in real-time."""
      ...
  ```
  - Integration point: `AIChannel` wraps `tool_handler` calls with audit recording automatically
  - `Agent(auditor=JSONLToolAuditor("/tmp/audit.jsonl"))` — one line to enable
  - Captures timing, input/output, status without any per-tool boilerplate
  - Integrates with telemetry: `ToolAuditEntry` → telemetry span attributes
  - `ON_REALTIME_TOOL_CALL` hook also feeds into the auditor for voice channel tools

### 19. High-level tools dramatically improve agent efficiency
- **Impact**: Observation
- **Finding**: Replacing 8 low-level tools (open_app, press_key, type_text, etc.) with 3 high-level tools (search_google, click_result, navigate) reduced token usage from 290k to 40k and steps from 65+ to ~10. The exec agent follows the prescribed workflow instead of freelancing.
- **Recommendation**: For computer-use agents, always provide task-level tools that encapsulate multi-step sequences. Let the framework handle the low-level orchestration (open browser, new tab, type, enter), not the LLM.
