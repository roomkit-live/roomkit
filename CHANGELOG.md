# Changelog

All notable changes to RoomKit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Configurable jitter buffer for SIPVoiceBackend** — new constructor parameters `jitter_capacity`, `jitter_prefetch`, and `skip_audio_gaps` allow tuning the RTP jitter buffer per deployment. Defaults remain unchanged (capacity 32, prefetch 0, skip gaps enabled).

## [0.7.0a4] — 2026-03-25

### Added

- **`TwilioWebSocketBackend`** — voice backend for Twilio Media Streams WebSocket audio. Bridges JSON-framed mu-law 8 kHz audio to/from the pipeline's PCM format. Dedicated writer task prevents outbound sends from blocking inbound receives on the same WebSocket.
- **Stateful soxr stream resampler** for `TwilioWebSocketBackend` inbound audio — high-quality upsampling from 8 kHz to pipeline rate (default 24 kHz) with no inter-frame discontinuities. Falls back to stateful `audioop.ratecv` when soxr is unavailable.
- **`RecordingChannelMode.ALL`** — new recording channel mode that outputs all three files: `*_inbound.wav`, `*_outbound.wav`, and `*_mixed.wav` in a single recording session.
- **SIP + ElevenLabs Conversational AI example** — incoming SIP calls routed to an ElevenLabs agent with real-time transcription logging and protocol tracing.

### Fixed

- **WavFileRecorder silence gap insertion** — silence is now only inserted for gaps exceeding 30ms (processing jitter threshold), preventing spurious silence from frame scheduling variance. First frame in each direction no longer gets leading silence from the gap between `start()` and first audio arrival.
- **TwilioWebSocketBackend disconnect callback** — renamed `on_transport_disconnect` to `on_client_disconnected` to match the `VoiceBackend` ABC. Previously the disconnect callback was silently never registered by `VoiceChannel`.
- **TwilioWebSocketBackend stale state on reconnect** — write queue, WebSocket reference, and resampler state are now cleared on disconnect, preventing stale filter artifacts and memory leaks when the backend handles sequential calls.
- Removed dead `_create_resampler()` method from `TwilioWebSocketBackend`.
- Moved inline `import asyncio` and `import contextlib` to module-level in `TwilioWebSocketBackend`.

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

[0.7.0a4]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a3...HEAD
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
