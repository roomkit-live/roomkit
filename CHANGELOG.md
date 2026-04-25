# Changelog

All notable changes to RoomKit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`SIPVoiceBackend.set_auth_resolver()`** — runtime-installable callback for digest-auth credential lookups. The resolver receives the username from the `Authorization` header and returns the matching password (or `None` to deny). Consulted on every authenticated INVITE, so the application owns credential storage — no need to hold every tenant's credentials in process memory or rebuild the backend when one is added/rotated/revoked. Takes precedence over the static `auth_users` dict when both are set; falls through to the dict when the resolver returns `None`. Resolver exceptions are caught and treated as denial so a buggy callback can't crash the SIP message loop. Driving use case: multi-tenant deployments where each SIP trunk has its own credentials and tenants come and go without restarting the backend.
- **`AuthResolver` type alias** in `roomkit.voice.backends.sip_auth` — `Callable[[str], str | None]`, exported alongside `SIPAuthMixin`.
- **`SIPVoiceBackend.has_auth()`** — returns `True` when at least one credential source (the static `auth_users` dict or a resolver) is configured. Used internally by `_handle_invite` to gate the auth challenge; surfaced publicly for apps that need to make their own decisions before an INVITE arrives.
- **RFC 3326 BYE `Reason` exposed on SIP sessions** — `SIPVoiceBackend._handle_bye` now parses the carrier `Reason: Q.850 ;cause=N ;text="…"` header on every BYE and stashes the result on `session.metadata["bye_reason"]` (`{"cause": int, "text": str}`). A canonical Q.850 cause→text map fills in `text` when the carrier omits it. The same dict is attached to the inbound BYE `ProtocolTrace` metadata. Lets dialer orchestrators distinguish "user rejected" from "no circuits" from "normal hangup" without re-parsing the wire — the SIP layer just exposes what it sees; consumers decide what to do with it.
- **`parse_bye_reason()` helper** in `roomkit.voice.backends._sip_types` — accepts `str | bytes | None`, returns the parsed `{"cause", "text"}` dict or `None`.

### Fixed

- **`SIPVoiceBackend.disconnect()` silently dropped the BYE on inbound sessions when the dialog hadn't reached `CONFIRMED` yet.** For inbound calls the dialog only confirms once the carrier ACK lands — usually within one RTT after our `200 OK`. An application that calls `disconnect()` from the `on_call` callback (e.g. routing decided the call is unwanted right after accept) would beat the ACK to the dispatch queue, find the dialog still in `EARLY`, and the BYE branch's `if call.dialog.state == DialogState.CONFIRMED` check would silently no-op. The carrier never saw a BYE and held the call open until its own timeout. `disconnect()` now polls dialog state for up to 500 ms before sending the BYE; if the ACK still hasn't arrived after the wait it logs a warning and skips the BYE rather than sending it into an un-confirmed dialog (which the carrier would reject with `481 Call/Transaction Does Not Exist`).
- **Inbound auth gate ignored an empty `auth_users` dict.** `_handle_invite` previously checked `if self._auth_users` — truthy on a populated dict, falsy on `{}` or `None`. That meant an app that wanted to start with no credentials and add them at runtime via `set_auth_resolver` (or by mutating the dict) would skip the entire auth path until at least one entry was present. Replaced the gate with `self.has_auth()` so a resolver alone is enough to enable the challenge flow, and a deliberately-empty dict-plus-resolver setup behaves predictably.
- **`RealtimeVoiceChannel.start_session()` swallowed `CancelledError` without cleanup.** The bare `except Exception:` around the long `provider.connect()` await didn't catch `asyncio.CancelledError` (Python 3.8+), so when an orchestrator (e.g. SIP dialer on remote BYE) cancelled the in-flight handshake, the cancellation propagated without running resampler / idle-event / skill-state teardown — leaving partial-state leaks on the transport and provider. The handler now catches `(Exception, asyncio.CancelledError)` together, runs cleanup unconditionally, and branches the log path: real exceptions still log at ERROR with a stack, deliberate cancellations log a single INFO line so dashboards stay quiet.

## [0.7.0a16] — 2026-04-23

### Fixed

- **`OpenAIRealtimeProvider.connect()` silently ignored `input_sample_rate` and `output_sample_rate`.** Input format was hardcoded to `{type: 'audio/pcm', rate: 24000}` and output format was never rebuilt from the parameter. A caller passing the ABC default of 16 kHz got 24 kHz on the wire, so the API played their audio back 1.5× faster than intended. The provider now honours both rates — but per the GA API, PCM is only accepted at 24 kHz, so invalid rates now raise `ValueError` up-front instead of silently mis-routing.

### Added

- **`OpenAIRealtimeProvider` G.711 telephony support.** Pass `input_sample_rate=8000, output_sample_rate=8000` and optionally `provider_config={"codec": "pcmu"}` (default) or `"pcma"` to emit `audio/pcmu` / `audio/pcma` formats. Lets SIP backends at 8 kHz skip a resampler. (PCM is only accepted at 24 kHz by the API.)
- **`OpenAIRealtimeProvider` additional `provider_config` keys**: `speed` (output playback rate), `idle_timeout_ms` (server_vad), `language` and `transcription_prompt` (passed to `audio.input.transcription`).

### Changed

- **`prime_realtime_input()` → `start_audio_stream()`** and hoisted to the `RealtimeVoiceProvider` ABC as a default no-op. OpenAI/xAI inherit the no-op; Gemini overrides with the 20 ms silence + interleave-safe flag flip. Renames the Gemini-internal term (`realtime_input`) out of the public surface.
- **`RealtimeVoiceChannel.inject_text(..., start_audio_stream=True)`** — one-shot way to open the realtime audio path and inject the first greeting in a single call, instead of calling `start_audio_stream()` + `inject_text()` separately. Intended for outbound-dial flows where the app speaks first. The channel-level `start_audio_stream()` method remains as a low-level escape hatch for openings without a text inject.

### Removed

- **`GeminiLiveProvider.prime_realtime_input()`** — replaced by `start_audio_stream()` (see above).
- **`kit.connect_realtime_voice()` and `kit.disconnect_realtime_voice()`** — deprecated shims that forwarded to `kit.join()` / `kit.leave()`. The 0.7.0a1 changelog announced their removal but the code only emitted `DeprecationWarning`; the shims are now actually gone. Use `kit.join(room_id, channel_id, participant_id=..., connection=...)` and `kit.leave(session)` instead.

## [0.7.0a15] — 2026-04-23

### Added

- **PSTN-compatibility for outbound SIP dial** — three opt-in knobs on `SIPVoiceBackend` / `OutboundAudioPacer` that make Gemini-Live (and other realtime) calls viable over carrier trunks:
  - `send_silence_on_answer` (seconds, default `0.0`) — one-shot PCM silence burst right after `200 OK` so carriers doing symmetric-RTP learning latch our stream before their ~8 s RTP-timeout drops the call.
  - `outbound_silence_fill` / `OutboundAudioPacer.fill_with_silence_when_idle` — the pacer emits a 20 ms silence frame whenever its queue is empty, keeping RTP flowing at a steady 50 pps regardless of TTS chunk cadence (PSTN endpoints have no packet-loss concealment, so gaps become audible stutter).
  - `GeminiLiveProvider.prime_realtime_input()` — pre-sends a 20 ms silence frame to flip the internal `realtime_input_sent` flag, so the first `inject_text` uses the audio-interleave-safe path and avoids the 1008 disconnect seen on some Gemini Live preview models.
- **`examples/voice_sip_dial.py` wiring** — silence priming, jitter prefetch, outbound silence fill, `inject_text`-based greeting trigger, and `SIP_DEBUG` env var for a working outbound PSTN demo end-to-end.
- **`send_event(..., created_at=)`** — optional override lets callers stamp emitted `RoomEvent`s with a chosen time instead of always "now". Needed so realtime voice transcriptions can carry the actual turn-start time.
- **`ON_TOOL_CALL` hook for realtime skill-infra tools** — `activate_skill` and friends now fire the tool-call hook so audit and downstream broadcast hooks observe them identically to regular tools.

### Fixed

- **Choppy / cut-off audio on SIP realtime calls** — `RealtimeVoiceChannel` hardcoded `SincResamplerProvider` (pure-Python sin/cos loop, ~17 % of real-time at 24 k→8 k, ~30 % at 24 k→16 k) for per-session transport resamplers. A 100-200 ms Gemini/OpenAI Realtime burst blocked the event loop long enough to drain the `OutboundAudioPacer` 60 ms jitter headroom. Switched to `NumpyResamplerProvider` (vectorized `np.interp`, 6-15× faster) with a Sinc fallback when NumPy is absent — same preference order as `voice/bridge.py`. WebRTC was unaffected (no `transport_sample_rate` set).
- **Realtime transcription ordering vs. mid-turn tool calls** — user transcriptions now carry the VAD `SPEECH_START` timestamp as `created_at`, so they sort before any tool calls Gemini fired mid-turn (which finalize earlier than transcription). Introduces `_user_turn_start_at` capture on `SPEECH_START`, cleared on session end.
- **Muted sessions hanging deliveries** — `WaitForIdle` in `core/delivery` now degrades gracefully on timeout: if voice never falls silent (e.g. a muted session where audio can't drain), it delivers anyway instead of silently dropping. A WARN log surfaces the event.
- **Pacer underrun noise** — `OutboundAudioPacer` only counts/logs an underrun when actually behind wall-clock. Empty-queue polls while the stream is ahead are silent.
- **`FastRTCStreamHandler.send_message` LSP violation** — suppress the `ty` `invalid-method-override` diagnostic on the sync override of FastRTC's async base method. The override stays sync because `aiortc`'s `RTCDataChannel.send` is itself sync and existing call sites don't await the handler method.
- **`TwilioWebSocketBackend` dropped first ~120 ms of every call** — `soxr.ResampleStream` at the default `"VHQ"` quality buffers six 20 ms Twilio frames before emitting any output, silently swallowing the opening words of every mu-law → PCM path. Switched to `quality="QQ"` (Quick), which emits a full chunk immediately and is still well above telephony-band fidelity for 8↔16 kHz. Resurfaces the 4 pre-existing resampler test failures as passes.

### Observability

- **Resampler selection logged at session start** — `RealtimeVoiceChannel` now logs which resampler was chosen (NumPy vs. Sinc) and the in/out sample rates, making the audio path visible in production logs.
- **Resample-slow WARN guard** — inbound and outbound resample calls log at WARN when they exceed a single RTP frame (20 ms), surfacing future regressions as pipeline logs rather than user-reported jitter.
- **Pacer end-of-response summary includes `max_behind_ms`** — call-quality signal stays observable even when `underruns == 0`.

## [0.7.0a14] — 2026-04-17

### Added

- **`kit.status_bus` lifecycle posts across every orchestration strategy** — `post_agent_lifecycle` helper in `roomkit/orchestration/status_bus.py` with shared conventions (`agent_id` = observed agent; `action` in `task | handoff | iteration | review | pipeline`; detail capped at 200 chars):
  - **Pipeline & Swarm** post via `HandoffHandler.handle` — `INFO` on every accepted handoff, `FAILED` on every rejected one.
  - **Loop** posts `PENDING` / `COMPLETED` / `FAILED` around each producer iteration and each reviewer review, in both sequential and parallel modes. Reviewer turns that don't approve stay at `INFO` so subscribers can distinguish "reviewed" from "approved".
  - **Supervisor** posts worker lifecycle events (pending / completed / failed) from every delegation path — sequential, parallel, and per-worker tools — plus a terminal pipeline-level entry under `agent_id="orchestration"`.
- **`async_delivery=True` in Supervisor strategy-tool mode** — no longer voice-only. With `strategy="sequential" | "parallel"`, workers dispatch as a background task and the supervisor returns `{"status": "dispatched", ...}` immediately; their combined output arrives back in the room via `kit.deliver()` when done, re-triggering the supervisor. This prevents the 300 s `tool_loop_timeout_seconds` from aggregating worker wall-clock time — each agent's timeout now covers only its own reasoning turn.

### Fixed

- **Supervisor `_running` / `_dedup_cache` atomicity on background failures** — if `asyncio.create_task` raised mid-dispatch (shutdown race), `_running` stayed set forever and the room was permanently marked busy. Both the strategy-tool path and the voice `auto_delegate` path now wrap `create_task` in `try/except BaseException` and discard `_running` on failure.
- **Stale dedup cache on pipeline failure** — when the background `_async_run_and_deliver` itself failed, its cached "dispatched" response survived for the 30 s dedup window and silently swallowed retries. A success flag threaded through the `on_done` callback now evicts the dedup entry on failure in the strategy-tool path.
- **Supervisor `agents()` / `install()` attach rules** — `async_delivery` now only skips attaching the supervisor in voice `auto_delegate` mode; strategy-tool mode keeps the supervisor attached so it can continue driving the conversation.

### Chores

- **`chore(release): publish only the current version's artifacts`** — `scripts/release.sh` now uploads exactly the current version's `*.tar.gz` + `*-py3-none-any.whl` instead of the whole `dist/` directory, which was failing when older wheels from prior releases were still sitting there.

## [0.7.0a13] — 2026-04-16

### Added

- **`inject_image()` on RealtimeProvider** — multimodal image injection for voice sessions. Gemini Live implementation sends images via `inline_data` Part. Exposed on `RealtimeVoiceChannel` for voice agents analyzing conversation attachments.
- **Tool-call-in-text recovery** — `RealtimeToolRecoveryMixin` detects when Gemini Live speaks tool calls as text (e.g. `call:send_to_agent{task:...}`) instead of using the function calling API, parses arguments, and dispatches through the normal tool handler pipeline.
- **Server-side RTCConfiguration passthrough** — `mount_fastrtc_realtime()` now forwards `rtc_configuration` to FastRTC as `server_rtc_configuration`, enabling TURN server credentials and relay candidate gathering.

### Fixed

- **Gemini `inject_text`/`inject_image` 1007 disconnect** — route text and image injection through `send_realtime_input` when audio is already flowing, avoiding `send_client_content` interleaving that causes WebSocket 1007 disconnects. Adds `realtime_input_sent` flag, pending tool call guards, and queued text injection flushing on `submit_tool_result`.
- **Gemini image injection during pending tool calls** — queue image injections when tool responses are pending (Gemini rejects `send_client_content` in this state) and flush the queue after all tool results are submitted.
- **`inject_text` sanitization** — strip control characters, null bytes, and unpaired surrogates from `inject_text`/`inject_image` payloads that were causing Gemini 1007 disconnects on conversation switches.
- **AI context polluted with non-message events** — `_build_context` now uses `get_conversation()` (MESSAGE events only) instead of `list_events()`, preventing channel attachment and tool call events from consuming the 50-event context limit.
- **OpenAI/vLLM/Azure provider resilience** — lower default timeout from 120s to 30s, add `max_retries` config (default 0, defers to RoomKit RetryPolicy), and make `APIConnectionError` retryable so RetryPolicy handles backoff and fallback. Previously, unreachable vLLM/Ollama would hang for 360s.
- **Cancel directive ignored during streaming** — `cancel_event` is now checked between every stream event in the streaming tool loop, interrupting mid-generation immediately instead of waiting for the full LLM stream.
- **Non-str deltas in delegation and supervisor streaming** — guard against non-string delta values.
- **PostgresStore `idx_participants_channel` non-unique** — allow multiple participants to share the same channel in group rooms. Includes migration to convert existing UNIQUE indexes to regular indexes.
- **Gemini `usage_metadata` field** — `candidates_token_count` → `response_token_count`.
- **CI: Python 3.13 test failures** — add `APIConnectionError` stub to OpenAI/Azure/vLLM test mock modules (Python 3.13 rejects MagicMock in `except` clauses) and align Azure test expectations with new timeout/retry defaults.

### Changed

- **RealtimeVoiceProvider callback dispatch refactored** — callback list initialization, `on_*` registration, and generic `_fire()` dispatcher lifted from 6 individual providers (OpenAI, xAI, ElevenLabs, Anam, PersonaPlex, Gemini) into the shared base class, eliminating ~280 LOC of boilerplate.

### Performance

- **Skip hook dispatch when no hooks registered** — short-circuit `_build_context` and audio level callbacks when no hooks are registered for voice/realtime triggers, avoiding 4+ DB queries per event.

## [0.7.0a12] — 2026-04-08

### Fixed

- **PostgresStore v1→v2 auto-migration** — detect old JSONB blob schema (`data` column on `rooms`) and drop v1 tables before creating v2 relational schema. Handles CI environments and existing deployments transparently.
- **PostgresStore test mocks aligned with v2 schema** — row-builder helpers replace stale `{"data": json}` mocks with proper relational column dicts.

## [0.7.0a11] — 2026-04-04

### Added

- **Activity persistence with interleaved tool calls** — AI responses are persisted as individual events per segment (text, tool call start, tool call end) with shared `correlation_id` and sequential indexing, replacing the single concatenated text blob.
- **`ToolCallContent`** — new content type for tool call events (name, id, args, result, status, duration, error).
- **`EventFilter`** — rich query filter (event types, source, time range, correlation_id, visibility) for `list_events`.
- **`PersistencePolicy`** — write-side control (`persist_types` / `exclude_types`) checked before every `add_event` call.
- **`get_conversation()`** / **`get_timeline()`** — convenience methods on `ConversationStore` for AI context rebuilding and full activity logs.
- **`deliver_stream` interleaved events** — stream generator yields `str | RoomEvent`, delivering text segments and tool call events inline during streaming with correct chronological order.
- **Human-in-the-loop tool handler** — `HumanInputToolHandler` pauses tool execution awaiting user input, with `PendingInput` model for tracking pending questions.
- **`tool_definitions` support on `HumanInputToolHandler`** — `AITool` definitions are auto-injected into the AI context with deduplication.
- **`organization_id` parameter on `create_room`** — set the org/tenant ID at room creation time for multi-tenant isolation.

### Fixed

- **Tool call events broadcast to transport channels** — removed broadcast blocking for `TOOL_CALL_START`/`TOOL_CALL_END`; the AI channel's self-loop guard already prevents re-responses.
- **Tool call events delivered to streaming channels** — `exclude_delivery` now only applies to `MESSAGE` events; tool calls are delivered to all channels.
- **All segment events delivered inline during streaming** — text segments and tool call events are both delivered during the stream, not deferred.
- **`segment_stream` yield guard** — track persisted event count to avoid yielding stale events when persist is a no-op.
- **PostgresStore JSONB codec** — register `json.dumps`/`json.loads` codec on pool init for proper JSONB serialization.
- **Multi-agent tool call guard** — `AIChannel.on_event` skips `TOOL_CALL_START`/`TOOL_CALL_END` to prevent spurious responses to another agent's tool calls.
- **`model_dump(mode='json')` in PostgresStore** — datetime fields serialized as ISO strings before JSONB encoding.
- **Stream consumer `RoomEvent` filtering** — `deliver_stream` consumers in `base.py`, `cli.py`, `_voice_tts.py` filter `RoomEvent` items from the `str | RoomEvent` stream.
- **Session started/ended messages over DataChannel** — `RealtimeVoiceChannel` now notifies the connected client via DataChannel for session lifecycle events.
- **Clear `_barge_in_active` on speech end** — prevents stale barge-in state when speech detection is a false positive.
- **Mock TTS audio padded to even byte length** — fixes PCM validation for 16-bit samples.

### Changed

- **PostgresStore relational schema (v2)** — all tables use proper indexed columns instead of JSONB blobs. Events, rooms, bindings, participants, identities, tasks, and observations have individual columns with B-tree indexes. Schema version bumped to 2.

## [0.7.0a10] — 2026-04-03

### Added

- **`BEFORE_TOOL_USE` hook** — pre-execution gate for local tools. Fires before tool execution in `_execute_tools_parallel`. Hooks can block to deny the tool call.
- **`ExternalToolHandler` ABC** — control and observe tools executed by an external provider (e.g. Claude Code sandbox). Framework injects hook callbacks on `register_channel` so the handler can fire `BEFORE_TOOL_USE` and `ON_TOOL_CALL` hooks.
- **`PolicyExternalToolHandler`** — concrete implementation with `ToolPolicy`-based auto-approve for standalone/testing.
- **`AnthropicConfig` `base_url` + `extra_headers`** — allows pointing the Anthropic SDK at a proxy and injecting custom headers.

### Fixed

- **Realtime voice barge-in** — multiple fixes across Gemini provider, channel layer, and transport backends for reliable interruption handling: immediate `clear_audio` on speech detection, `_user_speaking` gate on outbound audio, per-session `_has_pipeline_vad`, and `_rt_interrupted` flag on `LocalAudioBackend`.

## [0.7.0a9] — 2026-04-01

### Added

- **Sandbox tool schemas: write, edit, delete** — three new file modification tools for sandbox executors.
- **Docker and SmolBSD sandbox examples** — `sandbox_docker.py` (container-based) and `sandbox_smolbsd.py` (VM-isolated).
- **vLLM + HuggingFace example** — French-language example using Chocolatine-2-4B-Instruct with `SlidingWindowMemory`.

## [0.7.0a8] — 2026-04-01

### Added

- **Face touch detection filter** — MediaPipe-based `FaceTouchFilter` detects hand-to-face contact with zone geometry, false-positive filtering (proximity, z-depth, confirmation, cooldown), and sensitivity presets. Uses generic `FilterEvent` mechanism and `ON_VIDEO_DETECTION` hook trigger.
- **Supervisor `share_channels` parameter** — allows parent room channels to be shared with every child room during delegation. Threaded through all delegation paths.
- **`SandboxExecutor` ABC** — sandboxed command execution for AI agents with 7 reference tool schemas (read, ls, grep, find, git, diff, bash), system prompt preamble, and `AIChannel` integration via `sandbox` constructor parameter.

### Fixed

- **Face touch filter review fixes** — video pipeline close on channel teardown, model filename mismatch, thread-safe model init, partial download cleanup, 3D distance for z-depth filtering.
- **Supervisor `_running` race** — `asyncio.Lock` in `async_delivery` path, `_dedup_cache` eviction.

## [0.7.0a7] — 2026-03-27

### Added

- **`BEFORE_AI_GENERATION` hook** — new sync hook that fires after context building but before AI provider invocation. Hooks receive an `AIGenerationEvent` containing the full `AIContext` (messages, system prompt, tools, temperature, metadata) and can mutate it in-place or block generation entirely. Fires on all three generation paths (non-streaming, streaming, streaming with tools). Enables budget gating, PII redaction, knowledge injection, dynamic model routing, and compliance audit trails — all without touching provider code.
- **`AIGenerationEvent`** dataclass and **`BeforeGenerationCallback`** type alias for the new hook.
- **12 tests** for BEFORE_AI_GENERATION covering block, modify, streaming, priority ordering, and framework integration.

### Fixed

- **3 additional fire-and-forget `create_task` sites** missed in the v0.7.0a6 audit: SIP pacer start (`sip_audio.py`), SIP cancel_audio (`sip_transport.py`), and mock backend session ready callback (`mock.py`).
- **Inline import violation** in `_ai_generation.py` — moved `AIGenerationEvent` import to top-level per project conventions.

## [0.7.0a6] — 2026-03-27

### Added

- **`BEFORE_AI_GENERATION` hook** — new sync hook that fires after context building but before AI provider invocation. Hooks receive an `AIGenerationEvent` containing the full `AIContext` (messages, system prompt, tools, temperature, metadata) and can mutate it in-place or block generation entirely. Enables budget gating, PII redaction, knowledge injection, dynamic model routing, and compliance audit trails — all without touching provider code.
- **`AIGenerationEvent`** dataclass — carries `ai_context`, `channel_id`, `room_id`, and `provider_name` for the hook.
- **`BeforeGenerationCallback`** type alias — callback signature for the hook.
- **Shared `log_task_exception` callback** (`core/task_utils.py`) — done-callback for `asyncio.create_task()` that logs unhandled exceptions. Replaces 4 duplicate implementations across `webtransport`, `sip_calling`, `status_bus`, and `tasks/memory`.
- **Scoring module tests** — 31 tests covering `Score`, `MockScorer`, `ScoringHook`, and `QualityTracker` (was 0% coverage).
- **RoomKit Console** — full-screen terminal dashboard for voice agents with audio meters, transcription, voice activity timeline, barge-in indicators, and streaming text via Rich.
- **Unified voice pipeline** — extracted `VoicePipelineMixin` shared by `VoiceChannel` and `RealtimeVoiceChannel`. Pipeline creation, backend audio wiring, AEC reference feeding, and session lifecycle are now in one place.
- **Protocol contracts for all 34 mixins** — explicit host interface declarations via class-level type annotations and companion Protocol classes. Eliminates `# type: ignore[attr-defined]` on cross-mixin dependencies.
- **VAD model selection** — `VAD` env var selects energy, silero, or ten VAD. Falls back to energy VAD when sherpa-onnx is unavailable.
- **Manual VAD mode for RealtimeVoiceChannel** — local VAD drives `activityStart`/`activityEnd` signals to Gemini, OpenAI, and xAI realtime providers.
- **Smart-turn ONNX helper** — `build_turn_detector()` factory for the ONNX turn detection model.

### Fixed

- **Fire-and-forget task exception tracking** — ~20 `asyncio.create_task()` call sites across voice backends, realtime transports, orchestration strategies, and providers now have `add_done_callback(log_task_exception)`. Previously, exceptions in these tasks were silently dropped.
- **Protocol contract gaps** — type erasure, dead declarations, and weak annotations fixed across mixin boundaries.
- **Release script uses ty instead of mypy** — `scripts/release.sh` updated after the mypy-to-ty migration.

### Changed

- **mypy replaced with ty** for type checking (`ty check src/roomkit/`). Pre-commit hooks updated.
- **All examples refactored** to use shared helpers from `examples/shared/` (`setup_logging`, `run_until_stopped`, `require_env`, `build_pipeline`). Console mode added to voice examples.
- **Deprecated `connect/disconnect_video` migrated** to `join`/`leave` across all examples.

## [0.7.0a5] — 2026-03-26

### Added

- **Persistent delivery backend** — `DeliveryBackend` ABC decouples enqueue from execution so delivery requests survive process restarts and can be distributed across workers. `kit.deliver()` transparently enqueues when a backend is configured; a background worker loop dequeues and executes deliveries with retry and dead-letter support.
- **`InMemoryDeliveryBackend`** — asyncio.Queue-based backend for single-process deployments. Bounded dead-letter queue, backpressure-safe `nack()` and `close()`, re-enqueues in-flight items on shutdown.
- **`RedisDeliveryBackend`** — Redis Streams backend with consumer groups for multi-worker deployments. At-least-once delivery via PEL, bounded dead-letter stream (`MAXLEN ~`), injected client support for connection pooling. Install with `pip install roomkit[redis]`.
- **`DeliveryItem`** — Pydantic model for serializable delivery requests with retry tracking, status lifecycle, and strategy serialization.
- **`RoomKit(delivery_backend=...)`** constructor parameter with `start()`/`close()` lifecycle wired into `__aenter__`/`close()`.
- **`delivery_backend`** property on `RoomKit` (matches other backend properties).
- **Worker-side `BEFORE_DELIVER`/`AFTER_DELIVER` hooks** — hooks now fire during worker execution, not just in-process delivery. Shared `build_delivery_hook_event()` ensures consistent metadata across both paths.
- **`_cancel_worker_task()`** — shared helper on `DeliveryBackend` ABC for clean worker shutdown (DRY across backends).
- **Double-start guard** on both backends prevents orphaned worker task leaks.
- **Auto-delegate test coverage** — 3 new tests for `refine_instruction`, `delegation_message`, and `async_delivery` background delegation.
- **`delivery_backend.py` example** — InMemory backend with mock AI (no external deps).
- **`delivery_redis.py` example** — Redis backend with Anthropic AI.

- **Rich video overlays** — `OverlayFilter` renders dynamic content (text, images, tables) onto live video frames. Plugs into `VideoPipelineConfig.filters` as a `VideoFilterProvider`.
- **`TextOverlayRenderer`** — OpenCV-based text overlay with multi-line support, cached patch rendering, and 9 named positions + custom x/y. No extra dependencies.
- **`ImageOverlayRenderer`** — blit PNG/JPEG images onto frames with alpha blending, optional resize, and caching.
- **`RichOverlayRenderer`** — Pillow-based styled text and table rendering. Requires `pip install roomkit[video-overlay]`.
- **`SubtitleManager`** — wires `ON_TRANSCRIPTION` hook to an overlay for live subtitles. Optional `translate_fn` for real-time translation (e.g. French speech → English subtitles).
- **`subtitle_overlay()`** — one-liner factory for live subtitles on video.
- **`video_live_subtitles.py` example** — demonstrates the full subtitle + overlay system.

### Changed

- **`orchestration_supervisor_parallel_tasks.py`** — updated to use `auto_delegate=True, refine_task=False` (was `auto_delegate=False`).
- **Strategy metadata format standardized** — both in-process and backend delivery paths now use the serialized type key (`"immediate"`, `"wait_for_idle"`, `"queued"`) instead of class names.

### Removed

- **`tests/tasks/test_delivery.py`** — stale test file referencing deleted `roomkit.tasks.delivery` module.

## [0.7.0a4] — 2026-03-25

### Added

- **`TwilioWebSocketBackend`** — voice backend for Twilio Media Streams WebSocket audio. Bridges JSON-framed mu-law 8 kHz audio to/from the pipeline's PCM format. Dedicated writer task prevents outbound sends from blocking inbound receives on the same WebSocket.
- **Stateful soxr stream resampler** for `TwilioWebSocketBackend` inbound/outbound audio — high-quality resampling between 8 kHz (Twilio) and pipeline rate (default 24 kHz) with no inter-frame discontinuities. Falls back to pure-Python linear interpolation when soxr is unavailable.
- **Pure-Python G.711 mu-law codec** (`_mulaw.py`) — `pcm16_to_mulaw()` and `mulaw_to_pcm16()` with precomputed lookup tables. Replaces the deprecated `audioop` module (removed in Python 3.13). Shared by `TwilioWebSocketBackend` and `FastRTCVoiceBackend`.
- **`RecordingChannelMode.ALL`** — new recording channel mode that outputs all three files: `*_inbound.wav`, `*_outbound.wav`, and `*_mixed.wav` in a single recording session.
- **Configurable SIP jitter buffer** — new `SIPVoiceBackend` constructor parameters `jitter_capacity`, `jitter_prefetch`, and `skip_audio_gaps` for tuning the RTP jitter buffer per deployment. Previously hardcoded in `sip_calling.py`.
- **SIP + ElevenLabs Conversational AI example** — incoming SIP calls routed to an ElevenLabs agent with real-time transcription logging and protocol tracing.

### Fixed

- **SIP port leak on `call_session.start()` failure** — if RTP session startup fails after accepting an inbound INVITE, the allocated port is now released and BYE is sent to tear down the call. Previously the port leaked and the call was left in a zombie state.
- **SIP `_handle_bye` close-before-cleanup race** — `call_session.close()` is now awaited before releasing the RTP port. Previously the port could be reallocated while the close was still running as a background task.
- **SIP inactivity timeout close race** — same fix applied to the RTP inactivity timeout path in `_audio_stats_loop`.
- **WavFileRecorder silence gap insertion** — silence is now only inserted for gaps exceeding 30ms (processing jitter threshold), preventing spurious silence from frame scheduling variance. First frame in each direction no longer gets leading silence from the gap between `start()` and first audio arrival.
- **TwilioWebSocketBackend disconnect callback** — renamed `on_transport_disconnect` to `on_client_disconnected` to match the `VoiceBackend` ABC. Previously the disconnect callback was silently never registered by `VoiceChannel`.
- **TwilioWebSocketBackend stale state on reconnect** — write queue, WebSocket reference, and resampler state are now cleared on disconnect, preventing stale filter artifacts and memory leaks when the backend handles sequential calls.
- **SIP dial test failures** — added missing `_jitter_capacity`, `_jitter_prefetch`, `_skip_audio_gaps` attributes to test fixture (broken since a2 refactor).

### Changed

- **`audioop` dependency removed** — replaced with pure-Python G.711 codec and linear interpolation resampler. No C extensions or `audioop-lts` package needed on Python 3.13+.

## [0.7.0a3] — 2026-03-24

### Added

- **ElevenLabs Conversational AI realtime provider** — `ElevenLabsRealtimeProvider` for speech-to-speech AI via ElevenLabs' server-side STT, LLM, TTS, and turn detection. Uses the official SDK `AsyncConversation` class with async audio I/O. Supports tool calling, custom voices, and system prompt overrides. Install with `pip install roomkit[realtime-elevenlabs]`.
- **ElevenLabs tool-calling example** — demonstrates AI agent with weather tool via ElevenLabs Conversational AI.
- **ElevenLabs local voice example** — local microphone + speaker voice agent using `LocalAudioBackend` with ElevenLabs.

### Fixed

- Updated ElevenLabs provider for SDK v2.40 API changes.
- Suppressed unused `type: ignore` comments in CI for ElevenLabs provider.

## [0.7.0a2] — 2026-03-24

### Changed

- **SIPVoiceBackend refactored into focused modules** — split the 1600-line monolith into `sip.py` (facade + session lifecycle), `sip_audio.py` (RTP + codec + audio pipeline), `sip_calling.py` (outbound dialing + call state machine), `sip_auth.py` (SIP digest authentication), and `_sip_types.py` (shared types). Public API unchanged.

### Fixed

- Include `roomkit.tasks` module in wheel distribution.

## [0.7.0a1] — 2026-03-24

### Added

- **SIP NAT traversal (`advertised_ip`)** — `SIPVoiceBackend` and `SIPVideoBackend` accept `advertised_ip` to advertise a public IP in SDP `c=`/`o=` lines and SIP Contact/Via headers while binding RTP sockets to a private address. Requires `aiosipua>=0.4.1`.
- **`AICousticsDenoiserProvider`** — new denoiser provider using ai|coustics Quail speech enhancement models (neural noise suppression, dereverberation, Voice Focus speaker isolation). Install with `pip install roomkit[aicoustics]`. Requires `AIC_SDK_LICENSE` env var or `license_key` config.
- **`kit.join()` / `kit.leave()`** — unified session lifecycle API. `join(room_id, channel_id)` creates and starts a session (pull model); `join(room_id, channel_id, session=session)` binds an externally-created session (push model, e.g. SIP); `join(..., backend=other_backend)` supports cross-transport bridging; `join(..., connection=ws)` supports RealtimeVoiceChannel. `leave(session)` stops, unbinds, and disconnects.
- **Auto-start on `attach_channel`** — `VoiceBackend.auto_connect` property (default `False`). When `True` (e.g. `LocalAudioBackend`), `attach_channel` automatically calls `join()` to create a session, eliminating manual connect/bind/start_listening boilerplate for single-user backends.
- **Opt-out recording** — room-level recording now captures all channels by default when a room has recorders. `ChannelRecordingConfig` is only needed to *disable* recording on specific channels (e.g. `ChannelRecordingConfig(audio=False)`). No per-channel opt-in required.
- **Outbound TTS recording** — room-level recording now captures both inbound (mic) and outbound (TTS) audio, mixed into a single track via a thread-safe ring buffer with sample-by-sample clamping. Previously only inbound audio was recorded.
- **`VoiceChannel.add_outbound_media_tap()`** — register a tap on outbound TTS audio after pipeline processing, for room-level recording or other consumers.
- **`VideoBridge`** — 1:1 video forwarding between participants in the same room, mirroring `AudioBridge`. Supports frame filter/processor callbacks, `BEFORE_BRIDGE_VIDEO` hook trigger, and per-session backends. Wired into `VideoChannel` (via `bridge=True`) and `AudioVideoChannel` (via `video_bridge=True`).
- **`send_video_sync()`** on `VideoBackend` — synchronous frame send for bridge forwarding from callback threads
- **Unified `ON_TOOL_CALL` hook** — replaces `ON_REALTIME_TOOL_CALL`. Fires from both `AIChannel` and `RealtimeVoiceChannel` with a channel-agnostic `ToolCallEvent` carrying `channel_type`, `session`, `room_id`. `tool_handler` and hooks now coexist (handler runs first, hook observes/overrides). Simplified result return: `HookResult(action="allow", metadata={"result": "..."})` — no `RoomEvent` construction needed.
- **`ToolCallEvent`** dataclass and **`ToolCallCallback`** type — exported from `roomkit` and `roomkit.models`.
- **`Tool` protocol** — pass tool objects directly to channels via `tools=[my_tool]`. Any object with `.definition` (dict) and `.handler(name, args) -> str` works. All built-in tools (`DescribeScreenTool`, `DescribeWebcamTool`, `ListWebcamsTool`, `ScreenInputTools`) implement it.
- **`get_current_voice_session()`** — contextvar accessor for voice tool handlers that need session access
- **Webcam vision tools** — `DescribeWebcamTool`, `ListWebcamsTool`, `capture_webcam_frame`, `save_frame` for AI agents to capture and analyze webcam frames on demand
- **Webcam assistant example** — terminal chat with Claude + OpenAI vision via webcam
- **Video subsystem** — vision AI, video pipeline engine, decoder/resizer/filter/transform stages
- **Screen capture backend** with screen assistant example
- **Vision providers** — OpenAI and Gemini vision analysis with `ON_VISION_RESULT` hook
- **Video recording** — OpenCV, PyAV (H.264/VP9/NVENC), room-level media recording with A/V sync
- **Avatar providers** — MuseTalk lip-sync, WebSocket avatar, HTTP avatar, Anam AI cloud provider
- **Video filters** — WatermarkFilter, YOLO object detection, censor filter, 8 visual effects
- **Video pipeline** — `VideoPipelineConfig`, `VideoFilterProvider`, `VideoTransformProvider`
- **RealtimeAVBridge** — generic audio/video bridge for speech-to-speech + avatar
- **ScreenInputTools** — mouse/keyboard control, vision-based `click_element`
- **StatusBus** — shared status bus for multi-agent coordination with pluggable backends; wired into `RoomKit` as `kit.status_bus` with `status_posted` framework events via `kit.on("status_posted")`
- **`JSONLSessionAuditor`** — full conversation auditing that captures speech turns, tool calls, vision events, and interruptions in a unified JSONL timeline. Auto-attaches to `RoomKit` via `auditor.attach(kit)` using `ON_TRANSCRIPTION`, `ON_VISION_RESULT`, `ON_BARGE_IN`, and `ON_SESSION_STARTED` hooks. Produces readable conversation transcript via `summary()`. Drop-in replacement for `JSONLToolAuditor` via `.tool_auditor` bridge property.
- **`examples/shared/`** — reusable helpers for examples: `setup_logging()`, `run_until_stopped()`, `build_aec()`, `build_denoiser()`, `build_pipeline()`, `build_debug_taps()`, `os_info()`, `auto_select_provider()`.
- **JSONLToolAuditor** — tool execution auditing ABC with JSONL recording
- **Token usage tracking** — streaming tool loop usage, OpenAI/Gemini realtime token tracking
- **`setup_realtime_delegation()`** — one-call delegation wiring for RealtimeVoiceChannel (resolves room_id from voice session context)
- **`setup_realtime_vision()`** — wire video vision results into RealtimeVoiceChannel via `inject_text()` with dedup
- **`CompletedTaskCache`** — TTL-based dedup cache for delegation results, prevents re-spawning completed tasks
- **`DelegateHandler` enhancements** — `cache` for dedup (gap 13), `serialize_per_room` lock (gap 14), previous task context injection (gap 15)
- **Dangling tool call recovery** — `AIChannel` now detects orphaned tool calls (from barge-in interruptions) and injects synthetic cancellation results before the next AI turn. Prevents provider API rejections caused by `AIToolCallPart` entries without matching `AIToolResultPart`.
- **Large output eviction** — tool results exceeding `evict_threshold_tokens` (default 5000) are stored in a side buffer and replaced with a head/tail preview. A `_read_tool_result` tool is auto-injected so the agent can paginate through the full output on demand. FIFO-bounded to 50 entries.
- **Planning tools** — opt-in `enable_planning=True` on `AIChannel` gives the AI a `_plan_tasks` tool to create and track structured task plans. Plans are injected into the system prompt and published as ephemeral `CUSTOM` events for real-time UI rendering. New `ON_PLAN_UPDATED` hook trigger.
- **`SummarizingMemory`** — two-tier memory provider that proactively manages context budget. Tier 1 truncates large event bodies in older messages at ~50% capacity (no LLM call). Tier 2 summarizes older events via a lightweight AI provider at ~85% capacity with chained summaries and TTL caching.
- **`KnowledgeSource` ABC** — pluggable knowledge retrieval backend with `search()` and optional `index()`/`close()`. Backends can be vector stores, search engines, or any relevance system. Includes `MockKnowledgeSource` for testing.
- **`PostgresKnowledgeSource`** — production-ready full-text search knowledge source using PostgreSQL `tsvector`. Auto-creates schema, supports room-scoped queries, relevance ranking via `ts_rank_cd`, and upsert-on-conflict indexing. Shares the connection pool with `PostgresStore` via the `pool` parameter. No new dependencies (reuses `asyncpg`).
- **`RetrievalMemory`** — memory provider that enriches AI context with knowledge from pluggable sources. Searches all sources concurrently, deduplicates by content, and auto-indexes on `ingest()`.
- **`ON_AI_RESPONSE` hook** — fires after AI generation completes (streaming and non-streaming) with response content, usage metrics, latency, and tool call counts. Enables evaluation and scoring integrations.
- **`MemoryProvider.ingest()` wired** — `AIChannel` now calls `ingest()` on every inbound event, enabling stateful memory providers (vector stores, search indexes) to update as events arrive.
- **`ConversationScorer` ABC** — pluggable quality scoring for AI responses with `Score` dataclass (value, dimension, reason). Includes `MockScorer` for testing.
- **`ScoringHook`** — attaches to `ON_AI_RESPONSE` hook to run scorers automatically. Stores scores as `Observation` objects in the ConversationStore and buffers recent scores in memory.
- **`kit.submit_feedback()`** — submit user quality ratings for conversations. Stores feedback as `Observation` in the store and fires the new `ON_FEEDBACK` hook trigger.
- **`QualityTracker`** — aggregates scores and feedback into quality reports with per-dimension breakdowns, trend detection (first-half vs second-half comparison), and worst/best dimension identification. Reads from the store with optional time-window filtering. Supports multi-room reports via `report_multi()`.
- **AIChannel `tools` parameter** — pass tools directly to constructor
- **Room-level audio recording** for RealtimeVoiceChannel sessions
- **WebTransport backend** using QUIC datagrams
- **Cursor-based pagination** — `after_index`/`before_index` on ConversationStore
- **`output_muted` on ChannelBinding** with `mute_output`/`unmute_output` ops
- **Configurable `response_modalities`** for Gemini realtime provider
- SECURITY.md with vulnerability reporting contact
- PyPI metadata: keywords and author email
- Version floors for `fastrtc`, `sounddevice`, `anam`, `numpy` dependencies
- **Grok TTS provider** — `GrokTTSProvider` for xAI's text-to-speech API with REST, HTTP chunked streaming, and bidirectional WebSocket (`text.delta`/`audio.delta`) modes. 5 voices (eve, ara, rex, sal, leo), 20 languages, PCM/WAV/MP3/mulaw/alaw codecs. Includes voice agent example with Deepgram STT + Claude Haiku + Grok TTS.

### Fixed

- **Hook engine: ASYNC hooks on sync-only triggers** — `HookEngine.run_sync_hooks()` now fires ASYNC observer hooks after the sync pipeline completes. Previously, ASYNC hooks registered on triggers like `ON_TRANSCRIPTION`, `ON_VISION_RESULT`, and `ON_TOOL_CALL` (which are only invoked via `run_sync_hooks`) were silently ignored.
- **Recorder A/V sync** — wall-clock-aligned PTS, silence injection, late track handling, drift prevention
- Gemini: wrap non-dict tool results for `FunctionResponse`
- Watermark: use local timezone instead of UTC for timestamp
- FastRTC: handle WebSocket send race on client disconnect
- Gemini realtime: include sample rate in audio/pcm MIME type
- CI: resolve formatting, mypy, smoke test, and test failures
- Replace `print()` with `logger.info()` in StatusBus and ToolAuditor
- **Streaming telemetry spans** — `_run_streaming_tool_loop` now accumulates tokens across rounds and attaches summed totals to the `LLM_GENERATE` span (was only recording last round). Also fixed span not being ended in async generator due to `else` clause being skipped by `return`.
- **Task delivery for RealtimeVoiceChannel** — `WaitForIdleDelivery` and `ImmediateDelivery` now detect RealtimeVoiceChannel and deliver via `inject_text()` instead of `process_inbound()`
- **Gemini schema cleaning** — `clean_gemini_schema()` recursively strips `$schema`, `additionalProperties`, `default`, `title` from tool parameter schemas; applied automatically in both Gemini AI and Gemini Live providers
- **Clipboard paste** — `ScreenInputTools._type_text()` uses clipboard paste (`pbcopy`/`xclip`/`clip`) instead of `pyautogui.typewrite()`, fixing non-US keyboard layouts

### Changed

- **BREAKING: `parse_voicemeup_webhook()` and `configure_voicemeup_mms()` module-level functions removed.** MMS aggregation state is now per-instance on `VoiceMeUpSMSProvider`. Use `provider.parse_inbound(payload, channel_id)` and `provider.configure_mms(timeout_seconds=..., on_timeout=...)` instead. This enables multi-tenant deployments where each tenant has isolated MMS buffers.
- **BREAKING: `connect_voice`, `disconnect_voice`, `connect_video`, `disconnect_video`, `bind_voice_session`, `connect_realtime_voice`, `disconnect_realtime_voice` removed.** Use `kit.join()` / `kit.leave()` instead.
- **BREAKING: `stt`, `tts`, `voice` parameters removed from `RoomKit()` constructor.** Pass providers directly to `VoiceChannel(stt=..., tts=..., backend=...)`. The `kit.stt`, `kit.tts`, `kit.voice` properties now look up from registered VoiceChannels. `kit.transcribe()` and `kit.synthesize()` find providers the same way.
- **BREAKING: Top-level exports slimmed from 399 to 66.** Only core types (`RoomKit`, channels, enums, models, errors, tools) remain at `from roomkit import`. All providers, voice/video types, mocks, recording, orchestration, and telemetry now import from subpackages (e.g. `from roomkit.providers.anthropic.ai import AnthropicAIProvider`, `from roomkit.voice.backends.mock import MockVoiceBackend`).
- **BREAKING: `ON_REALTIME_TOOL_CALL` renamed to `ON_TOOL_CALL`.** The hook trigger `HookTrigger.ON_REALTIME_TOOL_CALL` is removed. Use `HookTrigger.ON_TOOL_CALL` instead. Hook event is now a `ToolCallEvent` (not `RealtimeToolCallEvent`). Return results via `HookResult(action="allow", metadata={"result": ...})` instead of `HookResult.modify(RoomEvent(..., metadata={"result": ...}))`.
- **BREAKING: `Tool` protocol is now the standard way to register tools.** Pass tool objects directly to `tools=[my_tool]` on `AIChannel`, `RealtimeVoiceChannel`, or `Agent` — definitions and handlers are extracted automatically. The `tool_handler` parameter still exists but is reserved for advanced use cases only (MCP server bridging, auditing middleware). **Migration:** replace `AIChannel(tools=[AITool(...)], tool_handler=my_fn)` with a class that has `.definition` and `.handler()`, then pass it via `tools=[MyTool()]`.
- **BREAKING: Unified `ToolHandler` signature** — all tool handlers now use `async (name: str, arguments: dict) -> str` across `AIChannel`, `RealtimeVoiceChannel`, and all tool classes. The old 3-arg `(session, name, arguments)` signature is removed. Use `get_current_voice_session()` contextvar for session access in voice tool handlers.
- **`audit_realtime_tool_handler` removed** — use `audit_tool_handler` instead (same signature now)
- `click_element` made generic via `VisionProvider` instead of hardcoded Gemini
- `print_summary()` methods now log via `logger.info()` instead of `print()`

## [0.6.13] — 2026-03-05

### Added

- `concurrency_limit` parameter to `mount_fastrtc_voice`
- Live AI analyst on bridged call example

## [0.6.12] — 2026-03-05

### Added

- **PyroscopeProfiler** for continuous CPU profiling with example
- **Multi-transport bridge** — SIP + WebRTC + WebSocket bridging
- **Cross-transport bridging** with numpy resampler
- Raw PCM WebSocket format for FastRTC backend
- WebRTC transport support for FastRTC backend
- `send_audio_sync` for efficient thread-safe audio in FastRTC
- `BEFORE_BRIDGE_AUDIO` hook with bridge + AI tests and example
- **N-party mixing** with cross-rate resampling and `MixerProvider` ABC
- **Audio bridging** — `TranscriptionEvent`, SIP metadata, human-to-human calls
- Outbound DTMF support for SIP and RTP backends
- Modern voice agent UI example

### Fixed

- Thread-safe `send_audio_sync` and WebRTC transcriptions
- Mypy override for pyroscope and flaky ws disconnect test

## [0.6.11] — 2026-03-03

### Added

- Cache `cache_read_input_tokens` extraction from OpenAI `prompt_tokens_details`
- FastRTC voice backend example and browser client

### Fixed

- FastRTC realtime transport tests for new API
- Audio overlap and interim transcriptions in FastRTC browser client
- Deepgram streaming STT sample rate and browser audio overlap
- Usage key assertions normalized to match token names
- CORS middleware for realtime FastRTC example

## [0.6.10] — 2026-03-03

### Added

- Binary `audio_format` option to `WebSocketRealtimeTransport`

## [0.6.9] — 2026-03-02

### Added

- Greeting gate for text channels — decouple send_greeting from TTS

### Fixed

- Three greeting gate bugs: LRU eviction, hook blocking, partial failure
- FastRTC: suppress gradio/huggingface telemetry on import

## [0.6.8] — 2026-03-02

### Added

- **`response_visibility`** to control AI response delivery scope
- **Handoff farewell prompt** and task delivery interrupt mode
- **TTS text filter** to strip internal prompt markers before synthesis
- **`BackgroundTaskDeliveryStrategy`** ABC for proactive task result delivery

### Fixed

- Auto-disconnect SIP sessions and guard farewell TTS block
- SIP re-INVITE race and task event index invariant
- Voice: enforce permissions on streaming delivery and prevent drain-period barge-in
- Handle stray `[/internal]` tags split across streaming chunks
- Prevent double delivery when proactive strategy is active
- SIP race, pacer stall, handoff timing, streaming dedup, and task delegation

## [0.6.7] — 2026-02-28

### Added

- **`ON_SESSION_STARTED`** unified hook (replaces `ON_VOICE_SESSION_READY`)
- **`Agent.auto_greet`** — direct TTS greeting via Agent
- `send_greeting()` API and LLM-generated greeting pattern

### Fixed

- Review findings in greeting and session-ready

## [0.6.6] — 2026-02-28

### Fixed

- Voice: return `None` from `emit()` to stop sending silence frames

## [0.6.5] — 2026-02-28

### Fixed

- Voice: throttle FastRTC emit loop to prevent 100% CPU spin

## [0.6.4] — 2026-02-28

### Added

- Pluggable transport auth and inbound rate limiting

## [0.6.3] — 2026-02-27

### Added

- AEC bypass mode, post-denoiser barge-in, continuous STT improvements
- `include_stream_usage` option for OpenAI/vLLM/Azure streaming token tracking

## [0.6.1] — 2026-02-26

### Added

- **Mistral AI provider** and Gemini streaming support
- **AI thinking/reasoning abstraction** unified across providers with example and guide

### Fixed

- Use event visibility for routing, not only source binding
- Visibility assertion — event visibility is preserved, not overridden

## [0.6.0] — 2026-02-24

### Added

- **Multi-agent orchestration** — `ConversationState`, `ConversationRouter`, handoff protocol, `ConversationPipeline`
- **Autonomous agent runtime** — uncapped tool loop, retry/fallback, context management
- **Mid-run steering** for AI channel tool loops
- **`kit.delegate()`** API for background agent delegation via child rooms
- **Agent class** with `greeting`, `language`, and `handler.set_language()` for voice orchestration
- **Streaming tool calls** — inline XML tool call events, `StreamError` message, `ON_ERROR` hook
- Tool calls broadcast as ephemeral events instead of inline XML
- Certificate-based authentication to Teams Bot Framework provider
- Proactive 1:1 personal conversation support for Teams
- Threading and reaction support for Teams provider
- Azure AI Studio provider
- Outbound SIP calling via `SIPVoiceBackend.dial()`
- `VoiceChannel.play()` accepts WAV files with format validation

### Fixed

- 11 critical, 19 high, and dozens of medium production-readiness issues
- Concurrency and safety issues from 4 rounds of deep code review
- SIP Contact header resolution and handoff TTS blocking
- Deepgram STT WebSocket staying open after call ends
- MCP tool handler prefix stripping for cross-context tool calls

### Changed

- README rewritten to reflect orchestration framework positioning

## [0.5.3] — 2026-02-17

### Added

- Structured streaming events and streaming tool loop for AIChannel

## [0.5.2] — 2026-02-16

### Added

- Streaming text delivery for WebSocketChannel

## [0.5.1] — 2026-02-16

### Added

- **MCPToolProvider** and `compose_tool_handlers` for MCP tool integration

## [0.5.0] — 2026-02-15

### Added

- **Provider-agnostic telemetry** — span tracing and metrics across all providers, backends, store, event routing, voice channels, hooks, and pipeline engine
- **MemoryProvider** ABC for pluggable AI context construction
- Speaker diarization with audio pipeline moved from channel to transport

### Fixed

- Audio crackling in LocalAudioBackend on macOS with AEC enabled
- ElevenLabs v3 streaming and Gemini realtime debug logging

### Changed

- Unified `VoiceBackend` and `RealtimeAudioTransport` into single ABC

## [0.4.18] — 2026-02-13

### Added

- Session resumption, context compression, and keepalive tuning for Gemini provider

### Fixed

- ElevenLabs TTS sample rate for `pcm_24000` output format
- Barge-in destroying new STT stream; rewrite Gradium turn detection

## [0.4.17] — 2026-02-13

### Added

- Agent Skills integration for AIChannel

## [0.4.16] — 2026-02-12

### Fixed

- NeuTTS Perth watermarker crash; add `neutts` optional extra

## [0.4.15] — 2026-02-12

### Added

- Gemini Live reconnection resilience and NeuTTS voice cloning provider

### Fixed

- ndarray type annotations for mypy 1.19+ with numpy 2.x
- NeuTTS streaming crackling by disabling per-chunk watermarking

## [0.4.14] — 2026-02-11

### Added

- `ON_INPUT_AUDIO_LEVEL` and `ON_OUTPUT_AUDIO_LEVEL` hooks
- Cross-thread scheduling for audio level hooks with VU meter example

## [0.4.13] — 2026-02-11

### Added

- AI tool calling loop for AIChannel
- Async SMS notification example for cross-channel coordination
- ChannelBinding access/muted enforcement on voice audio paths

### Fixed

- WebRTC AEC `AttributeError` when `process()` called after `close()`

## [0.4.12] — 2026-02-11

### Fixed

- `batch_mode` not disabling continuous STT

## [0.4.11] — 2026-02-11

### Added

- Whisper translate task support for SherpaOnnxSTTProvider
- Resampler caching in SherpaOnnxDenoiserProvider for non-native rates

## [0.4.10] — 2026-02-11

### Added

- Manual batch STT mode for VoiceChannel
- NeMo Parakeet TDT support for sherpa-onnx STT

### Fixed

- `sed -i` portability in release script for Linux

## [0.4.9] — 2026-02-10

### Added

- Public `set_input_muted()` and `send_event()` API

## [0.4.8] — 2026-02-10

### Fixed

- macOS audio crackling with stream diagnostics
- Release script `sed -i` for macOS compatibility

## [0.4.7] — 2026-02-10

### Added

- `say()` and `play()` public API on VoiceChannel
- OutboundAudioPacer for SIP TTS streaming
- Real-time RTP pacing for SIP outbound stream
- SIP + local agent example (sherpa-onnx STT/TTS + local LLM)
- CLAUDE.md project guide

### Fixed

- Slow TTS playback in SIP local agent example
- Long text truncation in sherpa-onnx TTS

## [0.4.6] — 2026-02-10

### Added

- Unified `process_inbound`, protocol traces, and `EventSource.provider`

### Changed

- Removed `ON_ERROR` hook; wire `ON_DELIVERY_STATUS` through hook engine

## [0.4.5] — 2026-02-10

### Added

- **SIPVoiceBackend** for incoming SIP call handling via aiosipua
- **Windowed sinc resampler**
- G.722 codec awareness with resampling moved to RealtimeVoiceChannel
- Deferred STT connection, Gradium pre-buffer warmup

### Fixed

- AEC double-feeding when backend and pipeline share same instance
- TTS echo leaking into STT transcription
- Post-TTS echo transcriptions in continuous STT mode
- WAV recorder -6dB amplitude loss
- Production hardening: input validation, path traversal, task tracking, SSRF

### Changed

- Split VoiceChannel (1650 lines) into 4 mixins for maintainability

## [0.4.4] — 2026-02-09

### Added

- **Gradium STT/TTS provider** with STT stream tracing and VAD pre-roll fix
- **Qwen3-TTS provider** with zero-shot voice cloning
- **Streaming AI → TTS pipeline** for low-latency voice responses
- Streaming STT support with Gradium provider
- Continuous STT mode for VAD and Deepgram

### Fixed

- Deepgram streaming close, ElevenLabs null audio, AEC shutdown race
- STT reconnection by signaling audio queue on turn complete
- VAD speech-end latency

## [0.4.3] — 2026-02-08

### Added

- **Telegram Bot API provider** with example
- GitHub Release creation in release script
- CI and mypy checks to release script

## [0.4.2] — 2026-02-08

### Fixed

- AEC pipeline regression with regression tests
- Barge-in interruption in local ONNX example
- Release script to read PyPI credentials from `~/.pypirc`
- VAD debug logging, audio trace diagnostics, lower default threshold

## [0.4.1] — 2026-02-07

### Added

- **WebRTC AEC3** — transport-level echo cancellation with examples
- **RTP voice backend** for PBX/SIP gateway integration with docs and example
- Release script and Makefile target

### Fixed

- All CI failures: mypy, ruff, bandit, smoke test, and STT test loop
- Pre-commit hook versions and ruff formatting on 29 files

## [0.4.0] — 2026-02-07

### Added

- **Audio processing pipeline** (RFC §12.3) — VAD, AEC, AGC, denoiser, recorder, resampler, DTMF, diarization, backchannel, turn detection
- **SherpaOnnxVADProvider** for neural speech detection
- **SherpaOnnxDenoiserProvider** (GTCRN) for neural speech enhancement
- **EnergyVADProvider** for energy-based voice activity detection
- **SpeexAECProvider** using libspeexdsp via ctypes
- **RNNoiseDenoiserProvider** using librnnoise via ctypes
- **SmartTurnDetector** for audio-native turn detection
- **WavFileRecorder** for debug audio capture
- **PipelineDebugTaps** for diagnostic audio capture at stage boundaries
- Pluggable `ResamplerProvider` replacing hardcoded config
- Bandit security scanner in CI, Makefile, and pre-commit

### Fixed

- Pipeline data models and defaults aligned with RFC (Phase 1+2)
- Error handling gaps, thread safety, and test coverage
- Onboarding DX: broken `HookTrigger` refs, smoke test, PyPI metadata

### Changed

- Pipeline reorganized into subdirectories per provider
- `STTProvider.transcribe()` returns `TranscriptionResult` (Phase 3.1)
- Framework event names enriched with payloads (Phase 4)

[0.7.0a15]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a14...HEAD
[0.7.0a14]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a13...v0.7.0a14
[0.7.0a13]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a12...v0.7.0a13
[0.7.0a12]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a11...v0.7.0a12
[0.7.0a11]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a10...v0.7.0a11
[0.7.0a10]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a8...v0.7.0a10
[0.7.0a8]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a7...v0.7.0a8
[0.7.0a7]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a6...v0.7.0a7
[0.7.0a6]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a5...v0.7.0a6
[0.7.0a5]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a4...v0.7.0a5
[0.7.0a4]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a3...v0.7.0a4
[0.7.0a3]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a2...v0.7.0a3
[0.7.0a2]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a1...v0.7.0a2
[0.7.0a1]: https://github.com/roomkit-live/roomkit/compare/v0.6.13...v0.7.0a1
[0.6.13]: https://github.com/roomkit-live/roomkit/compare/v0.6.12...v0.6.13
[0.6.12]: https://github.com/roomkit-live/roomkit/compare/v0.6.11...v0.6.12
[0.6.11]: https://github.com/roomkit-live/roomkit/compare/v0.6.10...v0.6.11
[0.6.10]: https://github.com/roomkit-live/roomkit/compare/v0.6.9...v0.6.10
[0.6.9]: https://github.com/roomkit-live/roomkit/compare/v0.6.8...v0.6.9
[0.6.8]: https://github.com/roomkit-live/roomkit/compare/v0.6.7...v0.6.8
[0.6.7]: https://github.com/roomkit-live/roomkit/compare/v0.6.6...v0.6.7
[0.6.6]: https://github.com/roomkit-live/roomkit/compare/v0.6.5...v0.6.6
[0.6.5]: https://github.com/roomkit-live/roomkit/compare/v0.6.4...v0.6.5
[0.6.4]: https://github.com/roomkit-live/roomkit/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/roomkit-live/roomkit/compare/v0.6.1...v0.6.3
[0.6.1]: https://github.com/roomkit-live/roomkit/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/roomkit-live/roomkit/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/roomkit-live/roomkit/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/roomkit-live/roomkit/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/roomkit-live/roomkit/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/roomkit-live/roomkit/compare/v0.4.18...v0.5.0
[0.4.18]: https://github.com/roomkit-live/roomkit/compare/v0.4.17...v0.4.18
[0.4.17]: https://github.com/roomkit-live/roomkit/compare/v0.4.16...v0.4.17
[0.4.16]: https://github.com/roomkit-live/roomkit/compare/v0.4.15...v0.4.16
[0.4.15]: https://github.com/roomkit-live/roomkit/compare/v0.4.14...v0.4.15
[0.4.14]: https://github.com/roomkit-live/roomkit/compare/v0.4.13...v0.4.14
[0.4.13]: https://github.com/roomkit-live/roomkit/compare/v0.4.12...v0.4.13
[0.4.12]: https://github.com/roomkit-live/roomkit/compare/v0.4.11...v0.4.12
[0.4.11]: https://github.com/roomkit-live/roomkit/compare/v0.4.10...v0.4.11
[0.4.10]: https://github.com/roomkit-live/roomkit/compare/v0.4.9...v0.4.10
[0.4.9]: https://github.com/roomkit-live/roomkit/compare/v0.4.8...v0.4.9
[0.4.8]: https://github.com/roomkit-live/roomkit/compare/v0.4.7...v0.4.8
[0.4.7]: https://github.com/roomkit-live/roomkit/compare/v0.4.6...v0.4.7
[0.4.6]: https://github.com/roomkit-live/roomkit/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/roomkit-live/roomkit/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/roomkit-live/roomkit/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/roomkit-live/roomkit/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/roomkit-live/roomkit/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/roomkit-live/roomkit/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/roomkit-live/roomkit/releases/tag/v0.4.0
