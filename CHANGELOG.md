# Changelog

All notable changes to RoomKit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Every timeline write is a single atomic store commit.** The pipeline
  previously assigned the event index with `get_event_count()`, then wrote the
  event and the room counters in separate calls — so two processes without an
  advisory lock could compute the same index (one write then failing on the
  `UNIQUE(room_id, index)` constraint), and a crash between the writes left
  `events` and `rooms.event_count` / `latest_index` divergent (RFC §8.1, §10.1,
  §14.3). The new `ConversationStore.commit_event()` assigns the authoritative
  index, inserts the event, and bumps the room counters as one transaction
  (`SELECT … FOR UPDATE` on the room row in Postgres). **Every** path that adds
  to a room's timeline now goes through it — the trigger message, AI reentry /
  tool responses and regenerated responses (previously stored `PENDING` and
  never counted), streamed AI segments, chain-depth-blocked, injected, greeting,
  child-room (delegated agent) trace, and system events (e.g. `channel_attached`,
  which was `DELIVERED` yet uncounted) — so the timeline and the counters can
  never diverge, and the post-broadcast counter reconcile is gone (RFC §10.1
  step 13/15). Injected, child-room, and regenerated events are committed
  `DELIVERED` (not left `PENDING`), and an event injected by a reentry's hook is
  now committed **after** the response that produced it, so it takes the higher
  index (causal order). End-to-end tests drive two `RoomKit` instances and an AI
  reentry through the real pipeline to prove it.
- **A `PersistencePolicy` that excludes an event no longer creates a phantom
  `latest_index`.** An excluded event is delivered but not stored, so it consumes
  no index; the room counters are left untouched instead of being advanced to the
  unstored event's provisional index.
- **`ON_ERROR` hooks run after the room lock is released.** A failing
  intelligence channel previously fired `ON_ERROR` while still holding the room
  lock, so a slow error hook (up to the hook timeout) blocked every following
  message for that room. `ON_ERROR` is now deferred past the lock, like
  `AFTER_BROADCAST`.

### Added

- **`PostgresAdvisoryLockManager` and `PostgresStore` are exported from
  `roomkit.store`**, and `RoomLockManager` / `InMemoryLockManager` from the
  top-level `roomkit` package.

### Changed

- **`scripts/release.sh` generates and validates the SBOM before any Git
  mutation**, pins the CycloneDX generator (`cyclonedx-bom==7.3.0`), and is
  re-runnable end to end. The clean-tree check tolerates an already-applied
  version bump; the commit, tag, and GitHub-Release steps are idempotent;
  `uv publish --check-url` skips files already on PyPI (so a partial upload
  resumes and uploads only what is missing); a local tag lets the PyPI safety
  check tell a resume from a fresh release; and a run that already published and
  opened the next dev cycle re-pushes and exits instead of aborting.
- **The Level 0 conformance matrix no longer overstates its guarantee.** Its
  docstring now distinguishes behavioural checks from structural (API-surface)
  ones and points to the feature suites that own the end-to-end coverage; the
  timers auto-pause/close, chain-depth blocking, and transcoder-fallback checks
  are now behavioural.

## [0.28.0] — 2026-07-11

Hardening release addressing a production-readiness review: the three critical
blockers plus tool-authorization, privacy, and supply-chain fixes.

### Changed

- **BREAKING — `PostgresStore.init()` never drops tables.** It previously ran a
  schema that `DROP … CASCADE`-ed every table when it detected a v1 (JSONB-blob)
  schema, so a routine connect after an upgrade could wipe rooms, events,
  participants, and identities. `init()` now runs additive, idempotent DDL only
  and raises `PostgresSchemaError` when a v1 schema is present. The destructive
  v1→v2 migration moved to an explicit, opt-in
  `PostgresStore.migrate(dry_run=True, confirm=False)` serialized by a PostgreSQL
  advisory lock.
- **BREAKING — WebRTC `/webrtc/offer` is authenticated before a peer connection
  is created.** The auth callback previously ran only for connections carrying a
  WebSocket object, so HTTP WebRTC offers were unauthenticated and an
  `RTCPeerConnection` was allocated for any caller. `mount_fastrtc_voice` and
  `mount_fastrtc_av` now authenticate the offer (and ICE candidates) at the HTTP
  layer and require an explicit `allow_anonymous=True` when no `auth` callback is
  given.
- **BREAKING — `process_timeout` is scoped to the pre-commit phase.** The whole
  locked pipeline (persist → broadcast → counters) was wrapped in a single
  timeout, so a slow broadcast could leave an event stored `DELIVERED` while the
  caller received `blocked=process_timeout` and room counters went unset. The
  inbound pipeline now splits at the commit point — pre-commit is timeout-bounded
  with no durable write before commit, and the post-commit broadcast runs
  unbounded — and the event persist and room-counter bump commit atomically, so
  the timeline and counters never diverge. (RFC §10.1 / §13.6 / §14.3.)

### Added

- **`newest_first` offset pagination** on `list_events` /
  `get_activity_timeline` — return the most recent `limit` events (still
  ascending) for reconnect snapshots.
- **`ConversationStore.close()`** (default no-op, idempotent), called by
  `RoomKit.close()` so a PostgreSQL connection pool is released on shutdown.
- **Central content-redaction policy** — `set_content_logging()` /
  `content_logging_enabled()` (and the `ROOMKIT_LOG_CONTENT` env var); message
  content is redacted from logs by default.
- **Blocking `pip-audit` CI job** on the core dependency set, plus a Dependabot
  configuration (uv + github-actions).

### Fixed

- **Tool authorization fails closed.** A context-build failure for the
  `BEFORE_TOOL_USE` hook now denies the call (was: allowed by default). Tool
  arguments are validated against the declared schema before execution, and
  realtime voice runs authorization before the handler so a block prevents the
  side effect rather than only hiding the result.
- **PII is no longer logged in clear** — STT transcripts, TTS/AI responses, and
  screen-agent typed text moved to DEBUG behind the redaction gate.
- **Inbound audio decode is size-capped** (Twilio and realtime WebSocket) before
  base64 decoding.
- **`InMemoryStore` reads return deep copies** — mutating a nested field of a
  read object no longer mutates the stored object.
- README: the WhatsApp Personal extra is `roomkit[whatsapp-personal]` (was
  incorrectly documented as `roomkit[neonize]`).

## [0.27.0] — 2026-07-10

### Changed

- Development status promoted to Beta.

### Documentation

- Corrected the hook-trigger count to 65 and fixed the trigger listings.
- Added a runnable room-membership example under `examples/`.

## [0.26.0] — 2026-07-10

### Added

- **Message threading (flat two-level, Slack/Teams style).** Replies now form
  threads on the existing `RoomEvent.parent_event_id` field. A reply carries the
  id of its thread **root**; a root or non-threaded message is `None`. Set it via
  `InboundMessage.parent_event_id` or the new `send_event(..., parent_event_id=)`
  argument. The locked pipeline **normalises** any parent reference to the thread
  root (replying to a reply collapses to the same thread; a dangling/cross-room
  parent drops to top level with a warning), so the invariant "`parent_event_id`
  is always a root" is enforced by the framework rather than the caller. The
  parent is applied **centrally** in the inbound pipeline, so every channel
  (WebSocket, SMS, email, …) threads without per-channel wiring. An AI channel's
  response **inherits the trigger's thread root** on both the streaming and
  non-streaming paths, so an `@`-mention inside a thread is answered in-thread.
  New reads: `EventFilter.top_level_only` (roots + standalone, replies excluded),
  `EventFilter.parent_event_id` (one thread's replies), and
  `ConversationStore.get_thread_summaries()` (per-root reply count + last-reply
  time, returning `ThreadSummary`). The PostgreSQL store adds a partial index on
  `events(parent_event_id)`. Distinct from `ChannelData.thread_id`, which remains
  the provider-native thread reference. The in-app WebSocket channel now
  advertises `supports_threading`. See `examples/message_threading.py`.
- **Explicit room membership (join/leave).** Member-level join/leave on top of
  the participant model, distinct from `ensure_participant` (which lazily
  materialises a sender the first time they speak). `add_member()` is a
  deliberate, idempotent join — safe to call on every room open: joining an
  already-`ACTIVE` member is a no-op (no write, no event), while a brand-new
  member or a re-join (someone who previously left) upserts them `ACTIVE` and
  preserves the original `joined_at`. `remove_member()` is a soft leave — it
  flips `status` to `LEFT` (or `BANNED`) rather than deleting the row, so
  membership history and read markers survive. `list_members()` returns the
  active roster (`include_left=True` for the full history) and `is_member()`
  tests active membership by identity. Each transition emits a
  `PARTICIPANT_JOINED` / `PARTICIPANT_LEFT` system event and fires the new
  `ON_PARTICIPANT_JOINED` / `ON_PARTICIPANT_LEFT` hooks. No schema migration —
  `ParticipantStatus`, `participants.status` and the `read_markers` table
  already existed.
- **Read-marker aggregation ("seen by").** New
  `ConversationStore.list_read_markers(room_id)` (on the ABC, PostgreSQL and
  in-memory stores) and `RoomKit.list_read_markers()` return every channel's
  read high-water-mark as `channel_id -> event index`. With one channel per
  member, this is the raw material for aggregating per-member "seen by"
  receipts. `read_markers` is now documented as the single source of truth for
  read position; `ChannelBinding.last_read_index` is an explicitly
  non-authoritative per-binding hint that the read API does not advance.

## [0.25.0] — 2026-07-09

### Added

- **Image tool results across every vision-capable provider.** An image tool
  result (`AIToolResultPart.result` carrying an `AIImagePart` — e.g. a screenshot
  tool) now reaches the model as a real image on **Ollama, OpenAI, Gemini,
  Mistral, and PolarGrid**, not just Anthropic. Unlike Anthropic — whose Messages
  API accepts image blocks inside a `tool_result` — these providers reject images
  in a tool/function-response message, so the tool message is kept text-only and
  the image is split onto a synthetic `user` message right after it, in each
  provider's native shape (Ollama `images`, OpenAI/Mistral/PolarGrid `image_url`,
  Gemini inline-bytes `Part`). A new `AIToolResultPart.split_for_message()` (a
  format-agnostic peer to `as_text()`) does the text/image split; each provider
  renders the images itself. Fully backward compatible: string and text-only-list
  results render exactly as before, and a non-vision model still can't see the
  image (vision is the model's capability, not RoomKit's — the image is simply no
  longer dropped before it gets there).
- **PolarGrid image input (vision).** `polargrid-sdk` 0.9.0 added multimodal chat
  (`Message.content` accepts OpenAI-shaped `image_url` parts), so an `AIImagePart`
  in a user turn now crosses the wire to PolarGrid instead of being dropped.
  `PolarGridAIProvider.supports_vision` is model-driven from the curated catalog:
  `qwen-3.6-35b-a3b` (yul-02) reads images (verified live), while `qwen-3.5-27b`
  accepts the request but does not — so only the former is flagged vision-capable.
  Vision is the deployed model's capability, not the SDK's.
- **`CLIChannel.run(content_factory=…)`.** Optional hook mapping a raw input line
  to inbound content (default `TextContent`); returning `None` skips the line.
  Lets an example accept richer input — the PolarGrid example uses it for an
  `/image <path> [question]` command — without reimplementing the input loop.

### Changed

- Updated the PolarGrid optional dependency from `polargrid-sdk>=0.8.5` to
  `polargrid-sdk>=0.9.0` (multimodal chat / image input).

## [0.24.0] — 2026-07-08

### Added

- **Public provider-lifecycle control on `VoiceChannel`.** New keyword-only
  constructor flag `close_providers` (default `True`, backward compatible).
  When `False`, `close()` leaves the injected STT/TTS providers open so the
  caller owns their lifecycle — reusing cached models across sessions, or
  closing them itself to avoid a double-`aclose` hang (e.g. ElevenLabs's httpx
  client). The backend is always closed by `close()`. Replaces callers reaching
  into `channel._stt` / `channel._tts` to null them before teardown.
- **`AIChannel.set_system_prompt(prompt)` + `system_prompt` property.** The
  supported way to swap the system prompt (persona/attitude) mid-conversation:
  the channel rebuilds request context from it each turn, so the change takes
  effect next turn with no reconnect and no loss of memory or tool state.
  (When a `config_provider` is set it still wins per turn.) Replaces writing to
  the private `AIChannel._system_prompt` slot.
- **`DiarizationProvider.clear_speakers()`.** Forgets every enrolled speaker
  (distinct from `reset()`, which only clears transient clustering state), so a
  provider reused across sessions doesn't carry speakers between conversations.
  Implemented for `SherpaOnnxDiarizationProvider` (clears the embedding manager
  and the debug-scoring cache); a documented no-op default on the base class.
  Replaces callers reaching into `_manager` / `_enrolled_embeddings`.
- **Image content in tool results.** `AIToolResultPart.result` now accepts a
  list of content parts (`AITextPart` / `AIImagePart`) alongside a plain string,
  so a tool can return an image (e.g. a screenshot) to the model. The Anthropic
  provider renders these as `tool_result` content blocks — the Messages API
  accepts `image` blocks inside a `tool_result` — while the other providers
  flatten to text via the new `AIToolResultPart.as_text()`. Tool handlers may
  now return `str | list[AITextPart | AIImagePart]`. Fully backward compatible:
  string results are unchanged everywhere.

## [0.23.0] — 2026-07-07

### Fixed

- **Turn errors now surface on the no-streaming-targets path.** When an agent's
  streaming send fn is withheld — a PII-locked or edge agent driven through the
  hooked "locked" delivery path — a failure during the turn (a context-window
  overflow, a provider error) used to propagate raw and vanish: the branch
  consumed the segment stream with a bare `async for`, so `ON_ERROR` never
  fired, the error hooks that classify and surface it never ran, and the user
  saw only a typing indicator that stopped. That branch now runs the same error
  contract as the streaming branch above — persist partial text, build the
  error event, fire `ON_ERROR`.
- **polargrid: an unknown pinned region is rejected at config construction**
  instead of surfacing later.

## [0.22.0] — 2026-07-06

### Added

- **Anti-loop guard in the tool loop.** A model that re-issues the *same*
  tool call with identical arguments is short-circuited: `find_tools` /
  `list_tools` (pure within a turn) on the 2nd identical call, other tools on
  the 3rd, with an explicit "stop repeating" result. When the model ignores
  the advisory and keeps hammering the same call, the guard pulls the
  ripcord — tools are stripped and a final plain-text answer is forced, so the
  turn ends instead of burning rounds (observed: `sandbox_bash({})` called
  37×). Small local models were the main offender.
- **`activate_skill` on an unknown skill that names TOOLS redirects.** Small
  models confuse skills with tools ("activate the Spotify skill" when
  `SpotifySearch`/… are tools). Instead of a dead-end "not found", the
  matching tools are revealed into the tool list with a hint to call one
  directly.
- **`tool_search_miss_hint`** on `AIChannel` — host-supplied steering appended
  to a `find_tools` no-match result, so a query only a *pinned* tool would
  satisfy (pinned tools are excluded from search results by design) points the
  model at the right pinned entry point instead of a dead end.

## [0.20.0] — 2026-07-03

### Added

- **Ephemeral tool-call events.** The tool loops publish `TOOL_CALL_START` /
  `TOOL_CALL_END` events so callers can surface tool activity live.
- **Anthropic prompt caching.** Explicit cache breakpoints on the stable
  request prefix cut input-token cost on multi-turn conversations.
- **Gemini cached-token usage.** Implicitly-cached input tokens are now
  reported in usage.

### Changed

- **Vendored, gradio-free WebRTC transport.** The WebRTC transport is
  vendored under `roomkit.webrtc` (extracted from fastrtc 0.0.34); the
  `fastrtc` extra now pulls the transport's own deps (aiortc, av, librosa,
  pydub, anyio) instead of the upstream `fastrtc` package and its gradio 5.x
  / pillow<12 constraints, so the default install is gradio-free.

### Fixed

- **OpenAI Realtime reconfigure is in-band.** `reconfigure` sends a partial
  `session.update` instead of tearing down and reconnecting, so the
  conversation and the in-flight tool call survive — Tool Search and skill
  activation work over OpenAI Realtime.
- **Gemini parallel tool calls** are replayed signed, never as thought parts.
- **ICE connection timeout** raised 30s → 60s so a client reachable only over
  a slow TURN relay (strict NAT) can connect before the timeout fires.
- **`read_stored_result` paging.** Pages carry more content per round while
  staying under the re-eviction bound even for worst-case JSON escaping, so a
  large evicted result reads back in a few rounds without looping.

## [0.19.0] — 2026-06-26

### Added

- **Discord bot channel.** A first-class Discord integration over the gateway
  (`discord.py`), wired as a source + REST provider sharing one `discord.Client`.
  Inbound messages (text, attachments, replies) and reactions arrive through the
  gateway; outbound supports text, embeds (`RichContent`), media uploads, and
  replies. `pip install roomkit[discord]`.
- **Supervised orchestration (hub-and-spoke).** In synchronous sequential mode
  the supervisor acts as a reviewer between every worker: it frames each task,
  reviews the worker's output with a strict APPROVE/REJECT verdict, sends rework
  with feedback up to `max_revisions`, and carries the validated result into the
  next worker's brief. On exhaustion the chain stops and reports an honest
  failure rather than presenting unreviewed work. New `Supervisor` parameters
  `task_timeout` (per-worker budget, default 120s) and `max_revisions` (default 3).
- **Structured-result handoff.** `kit.delegate(require_structured_result=True)`
  forces a delegated worker to return its work by calling a `submit_result`
  tool — a structured, parseable handoff and a guaranteed result (the worker
  can't punt with a question). A completion guard re-prompts the worker and, on
  exhaustion, submits an orchestration-level failure carrying its last output.
  Capture is delivery-agnostic (a function-calling tool call, or a `claude_code`
  worker's persisted trace).
- **Per-conversation tool memory.** `AIChannel` keeps a per-room record of tool
  usage and uses it two ways: a compact "what you did" digest injected into the
  system prompt, and sticky re-exposure of recently-used tool names so a tool
  used once stays callable while Tool Search hides the rest of the catalogue.
- **Parent → child delegation context.** A delegated child room inherits the
  parent room's context envelope, cascading verbatim through nested delegations.
  The worker's full trace (tool calls + messages) is persisted in its child room.
- **Worker capabilities for the supervisor.** The supervisor is given each
  worker's role and concise purpose, so it frames tasks knowing what each worker
  does rather than from a bare label.
- **Telegram Rich Messages.** Opt-in `TelegramConfig(rich_messages=True)` for Bot
  API 10.1 native tables and headings, with automatic fallback to entity
  formatting. Outbound Markdown is rendered into Telegram entities via
  telegramify-markdown (bundled in `roomkit[telegram]`).
- **Ollama sampling options.** `OllamaConfig` gains `temperature`, `num_ctx`,
  `top_p`, `top_k`, `min_p`, and `keep_alive` — with numeric-string coercion so a
  unit-less `"-1"` / `"0"` isn't rejected as a malformed Go duration.
- **Agent display name.** Optional `Agent(name=...)` — a human-readable label,
  distinct from `channel_id` and `role`, for attributing a step in orchestration
  timelines.

### Fixed

- **Realtime tool schema.** Strip non-API tool keys (e.g. Tool Search `tags`)
  from the OpenAI / xAI realtime `session.tools` payload, which the API rejects
  as unknown parameters.
- **Supervisor recursion.** `delegate_workers` no longer re-fires from inside a
  delegated sub-task room (delegate-within-delegate), in both strategy-tool and
  supervised-review paths.
- **Supervisor stuck / hang.** The supervisor runs dispatch/review without its
  own `delegate_workers` tool; a worker infra failure aborts the chain instead of
  waiting forever; and the completion hook fires when a delegation is cancelled
  or times out, so a consumer's step doesn't stay stuck on "running".
- **`submit_result` trace scan** caps its cursor to the int32 range (the Postgres
  store binds `before_index` as int4).

## [0.18.0] — 2026-06-21

### Fixed

- **`list_tools` is a compact inventory, not a catalogue re-dump.** It returned
  every tool with a full (200-char) description — re-sending the whole catalogue
  and defeating Tool Search (a small model that called `list_tools` instead of
  `find_tools` filled its context with ~3.4k tokens in one result). Each entry is
  now name + a one-line gist; the model uses `find_tools` for details and to act.
- **`find_tools` result no longer overflows and gets evicted.** Inlining each
  match's full parameter schema (0.17.1) blew up the result when the matches were
  verbose multi-action tools (`outlook`, `gmail`, …): a few of them exceeded the
  tool-result size limit, so the search result was evicted to `read_stored_result`
  — the model never saw its matches and gave up. `find_tools` is compact again
  (name + a truncated description); the matched tools' full schemas reach the
  model the proper way — the text loop re-sends them in the next round's tool
  list, realtime via `provider.reconfigure`.

### Changed

- **Relevance-ranked `find_tools` matching.** The matcher now scores candidates
  with **IDF weighting** — a query word is weighted by how rare it is in the
  catalogue, so ubiquitous words (`on`, `the`, `de`, `la`) contribute little and
  a discriminating word (`spotify`) dominates. No stopword list, language-
  agnostic, self-tuning to the catalogue (smoothed so it never collapses on a
  tiny catalogue). Tool names are also split on camelCase/PascalCase boundaries
  (`SpotifySearch` → `spotify` + `search`) so edge / device tools match by name,
  and only matches within 50% of the best score are kept. Fixes naive
  token-overlap surfacing unrelated tools (e.g. "play music on spotify" returned
  `scheduled_tasks`/`colleagues` merely because their text contained "on").
- **Stronger Tool Search preamble.** The system-prompt instruction now leads with
  "your visible tools are only a SMALL SUBSET" and a hard rule — never tell the
  user you lack a capability until you've called `find_tools` for the task. Small
  / local models were concluding "that's outside my skillset" from the visible
  tools without ever searching; the directive targets that failure mode.
- **`find_tools` returns matched tools' parameter schemas inline on text/HTTP
  channels.** Previously each match carried only name + description, so a model
  reading the result knew a tool existed but not how to call it — weak/local
  models then stalled or guessed arguments. The text path now includes each
  match's `parameters` JSON schema (the realtime path stays compact, since it
  delivers schemas via `provider.reconfigure`). This makes the tool's advertised
  "best matches with their schemas" actually true for the text loop.

### Added

- **Tool Search observability on text/HTTP channels.** When Tool Search defers a
  large catalogue, `AIChannel` now logs one line per turn (parity with the
  realtime channel, which already logged it): `Tool Search active: N tools
  deferred behind find_tools/list_tools (pinned=M, window=W)`. Makes the
  deferral visible in production logs; the text path was previously silent.
- **Cross-lingual tool search via English tags.** `AITool` gains an optional
  `tags: list[str]` of English keywords, scored by `search_catalogue` alongside
  the name (same weight) and description. A query normalized to English now
  matches a tool whose name/description are written in another language —
  fixing French/Spanish `find_tools` queries that previously returned nothing
  (e.g. « liste mes fichiers » → a tool named/described only in French). Tags
  propagate through both the text and realtime catalogues and are read from MCP
  tools' `_meta.fastmcp.tags`. The Tool Search preamble now instructs the model
  to phrase its `find_tools` query in English so both sides meet in one
  language-invariant space.

## [0.17.0] — 2026-06-20

### Added

- **Tool Search on text/HTTP agents (`AIChannel`).** Progressive tool
  disclosure — previously realtime-only — now works on any text provider.
  `AIChannel` gains `tool_search` (`None` = auto, `True`/`False` = force),
  `tool_search_pinned`, `tool_search_threshold_pct` (default 10) and
  `tool_search_threshold` (default 20). In `auto` mode it self-tunes to the
  model: it hides the catalogue when the deferrable (non-pinned) tools would
  cost more than `tool_search_threshold_pct` % of the model's context window
  (resolved from the provider catalog), falling back to the
  `tool_search_threshold` tool count when the window is unknown (custom / local
  model ids). The model then sees only `find_tools` / `list_tools` plus the
  pinned set; calling `find_tools(query)` reveals the matched tools on the next
  tool-loop round. Unlike the realtime channel (which pushes matches via
  `provider.reconfigure`), the text loop re-sends its re-filtered tool list
  every round, so no provider capability is required — the same mechanism as
  skill gating. The discovery tools bypass `tool_policy` and skill gating so
  they always work; a restrictive policy still governs the revealed tools. The
  scoring + result rendering is shared with the realtime path via
  `roomkit.channels._tool_search`. Also adds `AIProvider.context_window`
  (resolves the active model's window from the offline catalog) and
  `token_estimator.estimate_tool_tokens`. Backward compatible — Tool Search is a
  no-op below the threshold and when `tool_search=False`. See
  `examples/ai_tool_search.py` and `docs/c7/ai-channels.md`.

## [0.16.0] — 2026-06-19

### Added

- **Ollama endpoint authentication.** `OllamaConfig` now accepts `api_key`
  (a `SecretStr`, sent as `Authorization: Bearer <key>`) and `headers` (extra
  proxy / non-Bearer headers), so the native `OllamaAIProvider` can reach a
  protected endpoint — Ollama Cloud/Turbo, or a self-hosted server behind a
  Bearer-checking reverse proxy. `api_key` takes precedence over an
  `Authorization` entry in `headers`; when both are unset the SDK still falls
  back to the `OLLAMA_API_KEY` environment variable. Backward compatible —
  both default to `None`.
- **Custom headers and `extra_body` passthrough for OpenAI-compatible
  providers.** `OpenAIConfig` gains `default_headers` (custom proxy / non-Bearer
  auth headers, forwarded to the SDK) and `extra_body` (merged into every Chat
  Completions request body) for server-specific params the OpenAI schema omits —
  vLLM guided decoding (`guided_json`/`guided_choice`) and extra sampling
  (`top_k`, `repetition_penalty`). `VLLMConfig` exposes these as `headers` /
  `extra_body`; `AzureAIConfig` gains `extra_body`; `OpenRouterConfig` inherits
  both, with `default_headers` layered on top of its attribution headers.
  `extra_body` is merged rather than replaced, so static config never clobbers a
  per-turn value such as OpenRouter's `reasoning`. vLLM's `api_key` already mapped
  to a Bearer token. Backward compatible — all new fields default to `None`.

## [0.15.0] — 2026-06-18

### Added

- **Configurable WebRTC concurrency limit for realtime voice.**
  `mount_fastrtc_realtime()` now accepts a `concurrency_limit` argument,
  forwarded to the underlying FastRTC `Stream`. Previously the limit was left at
  FastRTC's default of 1, so a single shared transport could host only one
  simultaneous voice session platform-wide; further offers were rejected with
  `concurrency_limit_reached`. `None` (the default) preserves the old behavior,
  so this is backward compatible.

### Changed

- **Gemini Live fails fast on permanent disconnects.** When the Live API closes
  with a non-retryable code (`1007` invalid argument — e.g. a tool schema it
  won't accept, `1008` policy, `1011` quota), the receive loop now ends the
  session immediately and fires the error callback as `ws_<code>` instead of
  burning five doomed reconnect attempts (~10 s). Transient closes still
  reconnect as before. This lets embedders surface the precise reason to users
  right away rather than after a silent stall.

## [0.14.0] — 2026-06-18

### Added

- **Room lifecycle timers can be set directly.** `create_room()` now accepts a
  `timers=RoomTimers(...)` argument, and a new `kit.set_room_timers(room_id,
  timers)` method sets or replaces the timers on an existing room — replacing
  the previous `model_copy` + `store.update_room` boilerplate. Both entry
  points fill in `last_activity_at` automatically when it is omitted, so the
  idle clock starts immediately. `set_room_timers()` preserves an existing
  activity timestamp when only thresholds change, so adjusting a window
  mid-conversation never resets the idle clock. Backward compatible: the new
  `create_room` parameter is optional and defaults to `None`.

## [0.13.0] — 2026-06-17

### Added

- **PolarGrid provider supports tool / function calling.** Requires
  `polargrid-sdk>=0.8.5` (was `>=0.1`). `context.tools` are now forwarded
  to the chat-completions endpoint (OpenAI-shaped `tools`), and tool
  calls are surfaced back both non-streaming (`AIResponse.tool_calls`)
  and streaming (`StreamToolCall`, accumulated from the SDK's fragmented
  `delta.tool_calls`). PolarGrid sends tool arguments as a JSON string;
  the provider parses them into a dict for RoomKit, preserving malformed
  payloads under a `raw` key. Multi-turn tool loops render
  `AIToolCallPart`/`AIToolResultPart` back into structured messages
  instead of flattening them to text. `tool_choice` is left unset so the
  backend defaults to `auto` — forcing a specific tool is steered, not
  hard-guaranteed, on PolarGrid's backend. The SDK 0.8.4 release also
  fixes the non-streaming `latency_ms` decode crash, so the provider's
  `_patch_pg_metadata_decoder` monkeypatch was removed.
- **PolarGrid provider surfaces qwen reasoning (thinking).** A new
  `PolarGridConfig.thinking` flag drives the `enable_thinking` request
  field (polargrid-sdk 0.8.5+): `True` turns reasoning on, `False` off,
  `None` (default) leaves it unset. qwen then emits reasoning inline as
  `<think>...</think>` tags, which the provider parses (reusing the
  OpenAI provider's tag parser): `generate()` returns it on
  `AIResponse.thinking` with clean `content`, and
  `generate_structured_stream()` emits `StreamThinkingDelta` (handling
  tags split across chunks) ahead of the text; `generate_stream()`
  filters thinking out. Validated end-to-end on `qwen-3.6-35b-a3b`.
  Thinking responses are larger and slower, so raise `timeout` and
  `max_tokens` when enabling it.
- **PolarGrid model discovery.** `PolarGridAIProvider.available_models()`
  returns a curated, offline catalog of the chat models (`qwen-3.5-27b`,
  `qwen-3.6-35b-a3b`), and `list_models()` queries the connected edge via
  the SDK — returning the region-specific set (also the STT/TTS models),
  with display names backfilled from the catalog. Added to
  `examples/list_models.py` and the provider guide (with the per-edge
  availability table). Reasoning-capable `qwen-3.6-35b-a3b` is `yul-02`-only.
  `available_regions()` returns the curated catalog of all nine edges
  (`PolarGridRegion` id + name + location), and `connected_region()` reports
  the edge a provider is actually routed to (location backfilled from the
  catalog) — useful for data residency under auto-routing, where the
  `location` carries the Canada/US split (Law 25 / PIPEDA). PolarGrid serves
  no live full-region list (the `/v1/status` endpoint 404s on edges), so the
  catalog is a static snapshot of PolarGrid's regions guide.

## [0.12.0] — 2026-06-17

### Added

- **OpenRouter AI provider** — `OpenRouterAIProvider` / `OpenRouterConfig`
  (`roomkit[openrouter]`), a thin subclass of `OpenAIAIProvider` giving
  OpenAI-compatible access to 300+ models behind one key. `OpenRouterConfig`
  subclasses `OpenAIConfig`, inheriting every request field (so the two can't
  drift), and adds the routing `base_url` plus optional `site_url`/`app_name`
  app-attribution headers (`HTTP-Referer`/`X-Title`). `available_models()`
  ships a curated snapshot of current flagships; `list_models()` reads
  OpenRouter's rich `/models` endpoint as raw JSON — its entries omit the
  `object`/`owned_by` fields the OpenAI SDK's `Model` type requires — and maps
  context windows and vision support. Thinking is requested through
  OpenRouter's unified `reasoning` parameter (gated by `thinking_budget`), so
  Claude, Gemini, and DeepSeek all surface a reasoning trace via
  `StreamThinkingDelta`. See `examples/openrouter_ai.py` and the OpenRouter
  guide.
- **Gemini on Vertex AI** — `GeminiVertexProvider` / `GeminiVertexConfig` (in
  the existing `roomkit.providers.gemini` package, no new dependency). A thin
  subclass of `GeminiAIProvider` that builds the `google-genai` client in
  Vertex mode (`vertexai=True, project, location`) with Application Default
  Credentials instead of an API key — same models, processed in a pinned region
  with no training-data retention (data residency for Québec Law 25 / PIPEDA).
  `location` is required (no default) so requests can't silently route out of
  region; `GeminiVertexConfig` subclasses `GeminiConfig` so generation fields
  can't drift. See `examples/gemini_vertex_ai.py` and the Vertex guide.

### Changed

- **Provider examples follow the `<provider>_ai.py` convention.** `ai_azure.py`
  → `azure_ai.py`, and it is rewritten on the current `process_inbound` /
  `attach_channel` API (the old version still called the removed
  `kit.join`/`kit.send`/`Room.room_id` surface and no longer ran). The new
  OpenRouter example is `openrouter_ai.py`. The `ai_*` prefix is reserved for
  AI *feature* demos (memory, thinking, planning, …).

## [0.11.0] — 2026-06-13

### Added

- **Model discovery on every AI provider** — `AIProvider.available_models()`
  (a curated, offline classmethod — no API key, network, or SDK needed) and
  `list_models()` (a live query against the provider's models endpoint that
  backfills curated metadata). Both return `ModelInfo` (`id`, `display_name`,
  `context_window`, `supports_vision`, `deprecated`, `capabilities`). Curated
  catalogs ship for Anthropic, OpenAI, Gemini, Mistral, and Ollama; Ollama's
  `list_models()` probes `/api/show` per installed model to attach capability
  tags. See `examples/list_models.py`.
- **Voice discovery on every realtime provider** — `RealtimeVoiceProvider.available_voices()`
  / `list_voices()` returning `VoiceInfo` (`id`, `name`, `language`, `gender`,
  `description`, `deprecated`). Curated catalogs for OpenAI Realtime (10),
  Gemini Live (30), xAI Grok (5), PersonaPlex (18), and ElevenLabs (21, with a
  live `client.voices` query). `VoiceInfo.id` is exactly the `connect(voice=…)`
  value. See `examples/list_voices.py`.
- **Reasoning / thinking surfaced across all AI providers.** Providers emit
  `StreamThinkingDelta` when reasoning is enabled, so the trace renders inline
  (💭) through `CLIChannel(show_thinking=True)`:
  - Mistral reads structured `ThinkChunk` content (modern reasoning models no
    longer use inline `<think>` tags); `MistralConfig.reasoning_effort` maps
    from `thinking_budget`.
  - Gemini requests thought summaries (`include_thoughts`) and surfaces
    `thought=True` parts.
  - OpenAI surfaces the dedicated `reasoning_content` delta alongside the
    `<think>` parser; `OpenAIConfig` gains `reasoning_effort`,
    `supports_custom_temperature`, and `use_max_completion_tokens`.
  - Anthropic adds adaptive thinking and round-trips the thinking-block
    signature.
  - `examples/mistral_ai.py` is now an interactive `CLIChannel` REPL that
    streams reasoning live.

### Changed

- **Provider SDKs updated to current releases:** mistralai `>=2.0` (PEP 420
  namespace package — the client import moved to `mistralai.client`),
  google-genai `>=2.0`, websockets `>=14.0`, plus refreshed anthropic, openai,
  twilio, neonize, and protobuf (`>=7`) locks.
- **Image inputs decode `data:` URIs to inline bytes** for Gemini and Ollama
  rather than shipping a broken file reference.

### Fixed

- **neonize 0.3.18 compatibility** — the `event_global_loop` workaround is
  guarded by `hasattr` (0.3.18 binds the loop internally and dropped the field).
- **Azure inherits OpenAI's sampling config** — `AzureAIConfig` gained
  `reasoning_effort`, `supports_custom_temperature`, and
  `use_max_completion_tokens`, which the inherited OpenAI request builder reads.
- **Canonical usage tokens** — Mistral and Gemini report
  `input_tokens`/`output_tokens` consistently.
- **Order-dependent event-loop tests** — sync tests moved off the deprecated
  `asyncio.get_event_loop()` to `asyncio.run()` / `asyncio.get_running_loop()`.

## [0.10.0] — 2026-06-11

### Added

- **`playout` / `playout_max_delay_ms` on `SIPVoiceBackend`** (default off /
  200 ms) — adaptive clocked playout for inbound audio, via aiortp's
  AdaptivePlayout through aiosipua 0.7.0. Buffer depth tracks the measured
  network jitter (EWMA) with deadline-based concealment, replacing the
  static `jitter_prefetch` guess — the inbound defense for jittery links
  (WiFi callers, congested paths). `jitter_prefetch` only applies when
  playout is off.
- **`cn` / `cn_payload_type` on `SIPVoiceBackend` (default off) — RFC 3389
  comfort noise.** With `cn=True`, outbound silence (between TTS responses,
  while the LLM thinks) carries comfort-noise packets via aiortp instead of
  dead air, so carriers and handsets don't read the pause as a dead call.
  Talkspurt resumption is marked on the RTP stream for clean jitter-buffer
  resync. See `examples/voice_sip_comfort_noise.py`.
- **`duplicate_tx` on `SIPVoiceBackend` (default off) — outbound TX
  redundancy.** Every outbound RTP datagram is sent twice, the duplicate
  riding the next frame's send ~20 ms later (via aiortp). Receivers dedupe
  by sequence number, so no negotiation is needed; RTP bandwidth doubles.
  The outbound defense for lossy links.
- **RTCP Receiver Report observability in SIP audio stats.** The periodic
  and final stats lines now carry the remote endpoint's view of our
  outbound stream — cumulative packets lost, last-interval loss %, and
  interarrival jitter in ms (`RR lost=… loss=…% jitter=…ms`; `RR none`
  until a report arrives). Outbound degradation was previously invisible:
  local stats only measure the inbound leg.

### Changed

- **Outbound SIP registration delegates to `aiosipua.Registration`.** The
  hand-rolled REGISTER transaction machinery (~250 lines: message building,
  response interception, MD5-only digest, 80% renewal loop) is replaced by
  the upstream client: challenges are now answered per RFC 7616 (`qop`,
  MD5 **and SHA-256** — registrars requiring qop previously failed), 423
  Min-Expires is honoured, and the binding refreshes itself before expiry.
  The `register()` contract is unchanged (awaits the first outcome, raises
  on rejection, 5 s timeout) and a lost registration still retries every
  30 s. `close()` still unregisters with `Expires: 0`.
- **Dependency floors: `aiortp>=0.7.0`, `aiosipua[rtp]>=0.7.0`.** The
  playout wire-clock fix for RFC 3551 G.722 senders, `duplicate_tx`, and
  the Receiver Report stats keys all live in 0.7.0 of both.

## [0.9.1] — 2026-06-11

### Added

- **`RoomKit.unregister_channel(channel_id)`** — the missing inverse of
  `register_channel`. Pops the channel from the registry, resets the
  router cache, and returns the channel so the caller can
  `await channel.close()` explicitly. Integrators creating per-session
  channels (e.g. one `RealtimeVoiceChannel` per outbound call) previously
  had no removal API: channels accumulated in the registry and their
  provider sessions outlived the call — a hung-up Gemini Live session
  kept its receive loop alive and burned five reconnect attempts on a
  dead websocket before erroring out.

- **`plc` on `SIPVoiceBackend` (default `True`) — packet loss concealment.**
  RTP packets confirmed lost in transit are replaced with concealment PCM
  before delivery to the pipeline (via aiortp / aiosipua): native
  libopus PLC for Opus, last-frame repetition fading to silence over 60 ms
  for G.711/G.722/L16, silence fill beyond that. The inbound stream stays
  temporally continuous, so recordings keep their duration and AEC reference
  alignment no longer drifts under loss — previously the lost 20 ms frames
  were silently skipped and the timeline compressed. Loss detection is
  sequence-number based: VAD/DTX sender pauses are never concealed, and
  RFC 4733 telephone-events (which consume sequence numbers) are marked as
  received in the jitter buffer so DTMF digits are neither read as loss nor
  concealed. The per-session `concealed_frames` counter is synced into the
  audio stats and appears in the periodic (DEBUG) and final (INFO) stats
  log lines as `concealed=N`. `plc=False` restores skip-silently behavior.
  Validated end to end with controlled loss injection (aiosipua's
  `lossy_caller` example): `concealed` matches the sender's dropped count
  exactly, with and without DTMF interleaved.

### Changed

- **SIP/RTP extras require aiosipua >= 0.6.0 and aiortp >= 0.6.0.** aiosipua
  0.5/0.6 bring an RFC conformance overhaul (RFC 7616 digest, RFC-compliant
  CANCEL, dialog validation, 2xx retransmission), REGISTER/PRACK/REFER/session
  timers, hardened parsing, and a comfort-noise passthrough backed by aiortp
  0.6.0 (RFC 3389). RoomKit's SIP backends are source-compatible with the new
  versions — the aiosipua breaking changes (`send_cancel(call)`, `body: bytes`)
  touch APIs RoomKit does not call.

- **Realtime outbound audio: one resident send worker per session.** Provider
  audio chunks and the end-of-response flush now travel through a per-session
  FIFO queue drained by a single worker task, replacing one task creation per
  20 ms chunk (50/s, with task tracking and traceback capture under debug
  instrumentation). Audio → flush → RESPONSE_END ordering becomes structural —
  it no longer depends on task-creation FIFO surviving awaits inside the
  transport — and a barge-in drops queued stale chunks at queue speed instead
  of paying the resample for each. Public behavior is unchanged; covered by
  an adversarial yielding-transport ordering test.

## [0.9.0] — 2026-06-10

Realtime voice audio-quality release. A field investigation of intermittent
audio drop-outs on the speech-to-speech path traced three concurrent root
causes — speaker-buffer starvation, AEC reference desync, and event-loop
contention — all fixed and validated by before/after measurement: zero
underruns over a full session, first-second AEC attenuation after each
response start at -21.5 to -31.9 dB (was -3.8 to -19 dB), steady state
improved to -28/-38 dB, user-speech passthrough unchanged. The same pass
vectorised the SIP/RTP codec layer (via aiortp 0.3.2) and coalesced AI
thinking-stream publishes off the shared event loop.

### Added

- **`rt_prebuffer_ms` on `LocalAudioBackend` (default `120`).** The realtime
  speaker path now primes ~120 ms of audio before starting (and after any
  underrun) instead of playing from the first byte — the local-speaker
  analogue of the SIP pacer's prebuffer. A priming state machine honors the
  channel's `end_of_response` so short responses are not held back, ignores
  the stale end-of-response that providers fire on barge-in, and drains a
  partial buffer after ~100 ms if the signal never arrives. The new
  `rt_underruns` property counts mid-response starvations (warnings capped at
  the first 5); `rt_prebuffer_ms=0` restores play-on-first-byte.
- **`pacer_prebuffer_ms` / `pacer_jitter_headroom_ms` on `SIPVoiceBackend`**
  (defaults `80` / `60`, unchanged). Forwarded to `OutboundAudioPacer`, which
  already took them — the host could just never set them. Larger headroom
  absorbs longer host-side stalls on PSTN at the cost of barge-in latency.
- **`recent_events_window` on `Channel` and `MemoryProvider`.** Channels
  declare how many recent room events they read per turn (transport channels:
  0; `AIChannel` forwards its memory provider's window;
  `SlidingWindowMemory` reports `max_events`; token-aware providers keep the
  full pool via `DEFAULT_RECENT_EVENTS_WINDOW`).
- **Event-loop hold observability for realtime paths.** Tool-call handler and
  `ON_TOOL_CALL` hook segments log wall-time chronos at DEBUG; a WARNING fires
  when tool-result serialization alone holds the loop past ~50 ms (it runs on
  the full result before truncation) or when the channel falls back to the
  pure-Python sinc resampler (which holds the GIL even inside the resample
  executor). The SIP pacer budget is 60 ms — one fused stretch past it is an
  audible drop-out on a concurrent call.
- **`thinking_coalesce_ms` / `thinking_coalesce_chars` on `AIChannel`
  (defaults `80.0` / `256`).** Reasoning models emit one thinking delta per
  token, and publishing each on the realtime bus costs one ephemeral event +
  fan-out + WS serialise per token — thousands for a long trace, all on the
  shared event loop. Deltas are batched into one `THINKING_DELTA` publish
  per time/size window, cutting bus traffic 10-100x while the reasoning
  stays visibly real-time; clients append deltas, so a coalesced delta
  renders identically. Flushes larger than the per-event preview cap split
  into multiple publishes, so no reasoning text is ever truncated.
  `thinking_coalesce_ms=0` restores one publish per delta. The complete
  trace still arrives at `THINKING_END`, and the inline
  `ThinkingDeltaMarker` stream is unaffected.

### Changed

- **Playback-time AEC reference is fed continuously, silence included.** The
  pipeline AEC reference (wired via `on_audio_played`) skipped silent blocks,
  compressing the reference timeline vs. the actual speaker output; AEC3
  re-estimated its delay at every response start, leaking ~1 s of residual
  echo that the provider's server VAD could mistake for user speech (false
  barge-in → buffer flush → audible cut). Every block now reaches the
  reference, matching how Chrome feeds its AEC3 render stream. The
  transport-level AEC path (`LocalAudioBackend(aec=...)`) keeps its previous
  policy.
- **`RoomContext.recent_events` is sized to what the room's channels read.**
  `_build_context` loaded the full 2000-event ceiling on every call — for a
  persistent voice room that meant deserialising 2000 events several times
  per transcription (~1 s of sync CPU per turn under load). The limit is now
  the largest `recent_events_window` across bound channels, floored at 50 for
  hooks and capped at the ceiling; a transport-only voice room loads 50.
  Text agents with token-aware memory keep the full pool.
- **Tool-call processing yields between segments.** Handler execution, hook
  dispatch, and result submission no longer fuse into one event-loop step, so
  realtime pacing gets a scheduling slot between them.
- **RTP and SIP extras require `aiortp>=0.3.2`, which vectorises every audio
  codec.** G.711 µ-law/A-law run without a per-sample Python loop (encode
  3x, decode 21x), the G.722 wrapper hands the C extension int16 buffers
  instead of boxing every sample (1.4-1.7x including codec time), and L16
  byteswaps in one C-speed pass (12x) — cutting per-frame codec CPU on the
  SIP/RTP voice path. Wideband G.722 negotiation needs the `G722` package
  (`pip install aiortp[g722]`, now `>=1.2.3`).

### Fixed

- **Mid-sentence gaps on local realtime playback.** Any momentary starvation
  (provider burst jitter, loop contention) inserted audible silence
  immediately; underruns now re-prime the buffer, converting scattered gaps
  into one rare, measured re-prime.
- **Outbound resampling no longer blocks the event loop.** A sync resample in
  the provider-audio callback starved RTP pacing under concurrent host load
  (observed: 34.6 ms resample, 186 ms pacer underrun on a live PSTN call).
  Per-session resampling runs in a per-channel single-thread executor that
  also serializes the end-of-response flush and barge-in resets, preserving
  frame order without locks.
- **Realtime DSP held the GIL on hot paths.** `pcm16_to_mulaw` and `rms_db`
  per-sample Python loops are vectorised with NumPy (byte-/value-exact,
  equivalence-tested); the AEC energy diagnostics moved off the lock the
  PortAudio speaker callback contends on. NumPy stays a lazy optional import
  — base installs (no voice extras) are unaffected.
- **Partial transcriptions and speech events skip context builds when no
  hooks are registered.** Partials stream many times per second while the AI
  speaks; each paid a full `RoomContext` build for a no-op hook dispatch.
- **A second realtime session in the same process played no audio.**
  `LocalAudioBackend._rt_closing` persisted across sessions and silently
  dropped every queued chunk; `accept()` re-arms it.
- **FastRTC: sends on a non-open `RTCDataChannel` raised
  `InvalidStateError`.** The peer can close the data channel while provider
  audio or transcriptions are still flowing; sends are now gated on
  `readyState`.
- **The Gemini local example honors its documented `MUTE_MIC` override.**

## [0.8.0] — 2026-06-09

### Added

- **`regenerate_response(room_id)` — re-run the agent on the last inbound
  message.** Finds the most recent transport (human) message and re-broadcasts
  it with intelligence-only visibility, so the agent produces a fresh answer
  without ingesting a new event: the trigger keeps its identity, index, and
  timestamp, and transports never see the user message again (no duplicate
  bubble). The response flows through the existing persistence, streaming, and
  AFTER_BROADCAST machinery like a first-time turn. Removing the prior answer
  is the caller's responsibility. Lives in its own `RegenerateMixin`.
- **`InboundMessage.visibility` — deliver without waking the agent.**
  `process_inbound` previously had no way to post a message that reaches a
  room's transports but not its intelligence channel. The new field (default
  `"all"`) is stamped onto the event, so `visibility="transport"` delivers a
  proactive notification to the human without the agent replying to it.
- **Bounded retry when a tool round ends with no final text.** Small local
  models occasionally run a tool, get the result, then emit nothing instead
  of a final answer. Both tool loops (streaming and non-streaming) now
  re-prompt for the final answer with a corrective nudge, bounded by the new
  `AIChannel(max_empty_retries=...)` parameter (default 1) and guarded by the
  loop deadline and cancellation.
- **`skills_in_prompt` flag on `AIChannel`.** Hosts that render their own
  skills manifest inside `system_prompt` (e.g. positioned above a
  prompt-cache boundary) set `skills_in_prompt=False` to skip the automatic
  preamble + registry XML injection while keeping skill activation tools and
  gating untouched. Default `True` preserves existing behavior.
- **Per-call tool context accessors: `current_tool_room_id()` and
  `current_tool_allowed_names()`.** A channel object is registered once per
  `channel_id` and shared by every room it serves, so room-specific state
  stored on the channel goes stale the moment another room attaches. Both
  accessors (exported from `roomkit.tools`) read the tool loop's per-invocation
  context: the first resolves the originating room from inside a tool handler,
  the second exposes the turn's resolved toolset so handlers validate calls
  against it instead of an attach-time snapshot. Outside a tool loop they
  return `None`.
- **Telegram inline keyboards from `RichContent`.** The Telegram bot provider
  now routes `RichContent` to `sendMessage` with a `reply_markup.inline_keyboard`
  built from `content.buttons` (`{text, callback_data}` or `{text, url}` dicts,
  one button per row), enabling interactive flows such as approve/reject.
- **`ChannelBinding.can_write`.** True iff the binding has write access
  (`READ_WRITE` or `WRITE_ONLY`) and is not muted — the single RFC §7.5 gate
  shared by the inbound pipeline and the event router.

### Fixed

- **Direct injection (`send_event`) traverses the same locked pipeline as
  inbound (RFC §10.5).** It previously persisted and broadcast through a
  separate path, skipping BEFORE_BROADCAST hooks, edit/delete handling, and
  the source write-permission gate. Three more invariants enforced along the
  way: an edit/delete target is mutated only after hooks allow the event, so
  a moderation hook that blocks an edit no longer leaves the target mutated
  (RFC §10.3); a source whose binding cannot write is stored BLOCKED for
  audit instead of injecting a DELIVERED event, with hook side effects still
  collected (RFC §7.5); chain-depth, reentry-blocked, and injected events get
  a unique monotonic index instead of the model default `0` (RFC §8.1/§8.3).
  `tests/test_rfc_conformance.py` encodes the invariants.
- **AFTER_BROADCAST hooks run outside the room lock (RFC §10.1).** They were
  awaited while the room lock was held, so a slow observer hook blocked
  concurrent inbound processing for the same room. The locked pipeline now
  collects the (event, context) pairs and callers run them after releasing
  the lock — still awaited before returning, so observable ordering is
  unchanged.
- **`config_provider` turns reach the tool loop.** `handle_event` gated the
  tool-loop path on attach-time signals only (binding snapshot, constructor
  tools, skills), so a host delivering its toolset via `config_provider` got
  the plain streaming path and the resolved tools were never executable.
- **Streaming tool loops actually inherit the parent context.** The generator
  body runs when the consumer iterates — after `handle_event`'s `finally` has
  reset the contextvar — so participant-role inheritance silently failed and
  the per-round tools re-application (skill gating) was dead code. The parent
  context is now captured at stream creation and passed explicitly.
- **Tool eviction is scoped per room.** The eviction buffer lives on the
  shared channel object; an unscoped store let `read_stored_result` page
  through another conversation's oversized tool output and injected the
  re-read tool into rooms that evicted nothing. The buffer is now keyed by
  `(room, result_id)`.

## [0.7.2] — 2026-06-06

### Added

- **Per-turn config provider for `AIChannel`.** `AIChannel(config_provider=...)`
  resolves an `AIChannelTurnConfig` (system prompt, tools, temperature,
  max_tokens, thinking_budget) fresh at the start of every generation
  turn, so dynamic config — admin edits, per-user gating, feature flags —
  is never served from a stale attach-time snapshot. Explicit
  `binding.metadata` overrides still win for prompt/sampling (per-room
  operator intent); the provider's toolset REPLACES
  `binding.metadata["tools"]`, since that key is itself an attach-time
  snapshot. Without a provider, the static path is unchanged.
  `AIChannelTurnConfig` is exported from `roomkit`. Tests in
  `tests/test_channels/test_turn_config.py`.
- **`AIContext.response_metadata` rides every MESSAGE response event.**
  A `BEFORE_AI_GENERATION` hook can set turn-level attribution (e.g. RAG
  sources, labels) on `ai_context.response_metadata` and it lands in the
  metadata of every MESSAGE event of the turn — non-streaming, streaming,
  and the streaming tool loop — persisted before broadcast, so the stored
  row and the `stream_end` frame carry it from creation with no post-hoc
  store rewrite. `ChannelOutput.response_metadata` carries it on the
  streaming path. Tests in `tests/test_response_metadata.py`.

### Fixed

- **`read_stored_result` pages are size-bounded.** Pagination was
  line-based, but tool results are often single-line JSON: the page
  returned the whole payload, exceeded the eviction threshold, got
  re-stored under a new id, and the agent chased evicted results forever.
  Pages are now char-budgeted under the threshold (lines longer than the
  budget split into chunks) and the response carries an explicit
  `next_offset` cursor.
- **Ollama provider retries stream aborts without an HTTP status.**
  Ollama surfaces chat-template parse failures of the model's own
  tool-call output (e.g. a small model closing `<parameter>` with
  `</function>`) as a `ResponseError` with status `-1`. Those were
  classified non-retryable, killing the turn on a transient sampling
  defect that a regeneration almost always fixes. Statusless aborts now
  join the retryable set; definite HTTP client errors stay fatal.

## [0.7.1] — 2026-05-22

### Added

- **Native Ollama provider** (`OllamaAIProvider`, `OllamaConfig`) built
  on `ollama-python`, including thinking effort levels —
  `OllamaConfig.think` widened from `bool | None` to
  `bool | "low" | "medium" | "high"` per the Ollama 0.7+ API, with
  `ThinkEffort` exported from `roomkit.providers.ollama`.
- **Inline thinking streaming.** New `ThinkingDeltaMarker` in
  `models/streaming.py` delivers thinking in-band with the text stream so
  channels render it in arrival order; `CLIChannel(show_thinking=True)`
  renders it dim-italic inline. `THINKING_DELTA` ephemerals also publish
  over the realtime bus so remote subscribers see reasoning live (the
  buffered `THINKING_END` event still fires for observers joining
  mid-stream).
- **Teams channel owns inbound dispatch + roster lookups end-to-end.**

### Changed

- **`recent_events` ceiling raised from 50 to 2000.** The event-count cap
  predates `BudgetAwareMemory` and silently dropped older turns even when
  the token budget had headroom. A single `_RECENT_EVENTS_LIMIT` constant
  in `core/mixins/helpers.py` now bounds the in-memory footprint while
  token-aware memory does the real trimming.

### Fixed

- **Ollama provider mints unique tool-call ids across turns.** Ollama's
  native `/api/chat` does not return tool-call ids, so the provider
  synthesizes them. The previous format was `call_{name}_{i}` where `i`
  was the index *within a single response message*, so the counter reset
  to `0` on every turn — every same-named tool call in a conversation
  ended up sharing the same id (e.g. `call_scheduled_tasks_0` for 18
  separate calls). Downstream consumers that pair `TOOL_CALL_START` and
  `TOOL_CALL_END` events by `tool_id` then collapsed all N pairs onto a
  single timestamp, bunching the UI's tool pills at one point in the
  chat instead of interleaving them with assistant text. The id now
  carries a `uuid4` suffix (`call_{name}_{hex12}`) so every synthesized
  id is globally unique. New regression test in
  `tests/test_providers/test_ollama.py::test_synthesized_tool_ids_unique_across_turns`.
- **`BEFORE_BROADCAST` block on reentry events now conforms to RFC §9.5.**
  When a sync hook returned `HookResult.block(...)` on an AI-response
  reentry event, the inbound pipeline silently dropped three things: the
  BLOCKED storage of the event, the `event_blocked` framework event, and
  delivery of the hook's `injected_events`. The reentry allow/modify path
  also silently dropped `injected_events` from the hook result. Both
  paths are now symmetric with the main inbound path via a shared
  `_handle_block` helper. Five new tests in
  `tests/test_reentry_block_side_effects.py` lock the behaviour in
  place.

## [0.7.0] — 2026-05-15

First stable release after the `0.7.0a1`–`0.7.0a18` alpha series. The
per-alpha entries below remain as the granular per-PR history; this
section is the upgrade guide from `0.6.x`.

### Highlights

- **Real-time speech-to-speech AI** is the headline feature. The new `RealtimeVoiceChannel` wraps OpenAI Realtime, Gemini Live, xAI, ElevenLabs, Anam, and PersonaPlex behind one Channel ABC, with a 10-mixin architecture (`_realtime_audio`, `_realtime_tools`, `_realtime_speech`, `_realtime_skills`, `_realtime_transcription`, `_realtime_response`, `_realtime_tool_search`, `_realtime_tool_recovery`, `_realtime_context`, `_skill_handlers`) that the channel composes.
- **Tool Search** for tool-heavy realtime sessions — `find_tools(query)` + `list_tools` keep the active tool surface under ~20 (the reliable function-calling threshold for Gemini Live) while exposing thousands of tools dynamically via `provider.reconfigure`.
- **Skill delivery modes** (`on_demand` vs `inline_full`) that handle providers which cannot reconfigure mid-session (Gemini 3.x) by baking skill bodies into `system_instruction` at session start.
- **Carrier-grade SIP**: NAT traversal via `advertised_ip`, BYE routing fixed for inbound calls behind SBCs, RFC 3326 `Reason` header parse + emit, runtime auth resolver (`set_auth_resolver`), runtime invite filter (`set_invite_filter`), PSTN compatibility knobs for outbound dial.
- **Orchestration**: `Supervisor` strategy with `sequential` / `parallel` / `auto_delegate` execution + `async_delivery` for non-blocking pipelines, `HandoffHandler` state machine, `Loop` producer/reviewer pattern, all wired to `kit.status_bus` for observable multi-agent flows.
- **Video / vision**: vision providers (OpenAI, Gemini), avatar providers (MuseTalk lip-sync, WebSocket, Anam cloud), video filters (watermark, YOLO, censor, MediaPipe face-touch detection), screen capture + control tools (`DescribeScreenTool`, `ScreenInputTools`), webcam capture (`DescribeWebcamTool`), PyAV recorder with A/V sync, video bridge.
- **Storage**: `PostgresStore` v2 relational schema with proper indexes (replacing JSONB blobs), `PostgresKnowledgeSource` for full-text retrieval, `SummarizingMemory` + `RetrievalMemory` providers.
- **Delivery backends**: pluggable `InMemoryDeliveryBackend` and `RedisDeliveryBackend` (Streams + consumer groups) so deliveries survive process restarts and scale across workers.
- **Twilio Media Streams** voice backend with stateful soxr resampling and pure-Python G.711 mu-law codec (no `audioop` dependency).
- **Quality**: `ON_AI_RESPONSE` + `ON_FEEDBACK` hooks, `ConversationScorer` ABC, `ScoringHook`, `QualityTracker` reports.

### Migration from 0.6.x

#### Removed APIs (BREAKING)

- `kit.connect_voice` / `kit.disconnect_voice` / `kit.connect_video` / `kit.disconnect_video` / `kit.bind_voice_session` / `kit.connect_realtime_voice` / `kit.disconnect_realtime_voice` → **use `kit.join(...)` and `kit.leave(session)`** (see `0.7.0a1` and `0.7.0a16`).
- `RoomKit(stt=..., tts=..., voice=...)` constructor parameters → **pass providers to `VoiceChannel(stt=..., tts=..., backend=...)` directly**. `kit.stt` / `kit.tts` / `kit.voice` properties now look up from registered channels.
- Top-level `from roomkit import …` exports slimmed from 399 to 66. **Providers, voice/video types, mocks, recording, orchestration, and telemetry must be imported from their subpackages** (e.g. `from roomkit.providers.anthropic.ai import AnthropicAIProvider`).
- `HookTrigger.ON_REALTIME_TOOL_CALL` → **renamed to `HookTrigger.ON_TOOL_CALL`**. The event payload is now a channel-agnostic `ToolCallEvent`. Return results via `HookResult(action="allow", metadata={"result": ...})`.
- Tool handler signature: 3-arg `(session, name, arguments)` → **2-arg `(name, arguments)`**. Use `get_current_voice_session()` contextvar for session access in voice tool handlers.
- `audit_realtime_tool_handler` → **use `audit_tool_handler`** (now channel-agnostic).
- `parse_voicemeup_webhook()` / `configure_voicemeup_mms()` module-level functions → **per-instance `provider.parse_inbound(payload, channel_id)` / `provider.configure_mms(...)`** (enables multi-tenant isolation).
- `GeminiLiveProvider.prime_realtime_input()` → **`provider.start_audio_stream(session)`** (also exposed on `RealtimeVoiceChannel.inject_text(..., start_audio_stream=True)`).

#### Behavior changes

- **Recording is opt-out, not opt-in.** Rooms with recorders now capture every attached channel by default. Disable per-channel with `ChannelRecordingConfig(audio=False, video=False)`. Recording now captures both inbound (mic) and outbound (TTS) audio mixed into a single track.
- **`Tool` protocol is the standard tool registration path.** Pass any object with `.definition: dict` and `.handler(name, args) -> str` via `tools=[my_tool]`. The legacy `tool_handler=` parameter still exists for MCP / audit middleware but `tools=` is the documented surface.
- **`PostgresStore` is now relational (schema v2).** v1 JSONB-blob databases are auto-migrated on first connect; drops old `data` columns and rebuilds the relational schema.
- **`OpenAIRealtimeProvider` honours `input_sample_rate` / `output_sample_rate`.** PCM is only accepted at 24 kHz by the GA API; invalid rates now raise `ValueError` at construction.
- **`audioop` dependency removed.** Replaced with pure-Python G.711 codec + linear interpolation resampler — runs on Python 3.13+ without `audioop-lts`.

### Security

- **HTTP webhook SSRF guard hardened (`HTTPProviderConfig.webhook_url`).** The previous validator only checked literal-string hostnames and the canonical-dotted-quad output of `ipaddress.ip_address`. Five bypasses landed in production: `http://127.1`, `http://2130706433`, `http://0x7f000001`, `http://localhost.` (trailing-dot DNS form), and any hostname whose A record points to RFC 1918 / loopback / link-local. The new validator lives in `roomkit.providers.url_safety.validate_public_url` and (a) normalizes IPv4 numeric forms via `socket.inet_aton`, (b) strips trailing-dot DNS forms, (c) resolves every A/AAAA record at validation time and rejects on any non-public result. Reject reasons now name the resolved address class (loopback, private, link-local, reserved, multicast, unspecified). Note: DNS rebinding between validation and HTTP request is still possible — pin-on-connect is out of scope for a config-time helper; callers that need it must wire a custom `httpx.AsyncHTTPTransport`.
- **`DeepgramSTTProvider` no longer fetches `AudioContent.url` server-side.** The previous code did `httpx.AsyncClient().get(audio.url)` before shipping bytes to Deepgram — an SSRF surface that any inbound webhook could trigger by emitting an `AudioContent` with a non-public URL. The provider now dispatches URL-bearing audio through Deepgram's native `transcribe_url` so the fetch happens from Deepgram's network, not ours. Raw bytes (`AudioChunk` / `AudioFrame`) still go through `transcribe_file` unchanged.
- **`PersonaPlexConfig.ssl_verify` default flipped from `False` to `True`.** The previous default disabled certificate verification (`check_hostname=False`, `verify_mode=CERT_NONE`) on every PersonaPlex connection, justified at the time as a convenience for self-signed dev certs. Secure-by-default is the rule. **Migration**: production deployments are not affected. Local development against self-signed certs must now pass `ssl_verify=False` explicitly. The `PersonaPlexRealtimeProvider(ssl_verify=...)` constructor argument was flipped to match.
- **Telnyx webhook signatures now check timestamp freshness.** `TelnyxSMSProvider.verify_signature` and `TelnyxRCSProvider.verify_signature` previously accepted any correctly-signed timestamp, so a single captured request could be replayed forever. Both now reject signatures whose timestamp is more than 300 seconds away from the current clock; the window is configurable via the new `tolerance_seconds` kwarg. The two byte-identical verifiers were also factored into `roomkit.providers.telnyx._signature.verify_telnyx_signature`. **Migration**: webhook ingest pipelines that buffer requests longer than 5 minutes between Telnyx and the verifier must pass a larger `tolerance_seconds`.
- **`DescribeWebcamTool` no longer exposes `save_path` to the AI.** The previous tool schema let the model pass an arbitrary `save_path: string` that the handler resolved via `Path(p).expanduser().resolve()` and wrote a JPEG to — including auto-creating parent directories. A prompt-injected model could overwrite any file the process could write. The schema field is gone; the constructor now takes an operator-controlled `save_dir` and the handler auto-generates `webcam-<utc-timestamp>-<uuid>.jpg` inside that directory. If `save_dir` is unset, captures are not persisted. The model has no way to influence the destination path. **Migration**: callers passing `save_path=...` to `DescribeWebcamTool.analyze` must instead pass `save_dir=...` at construction time. Any `save_path` field included by the model in tool arguments is now silently ignored.

### Full per-PR detail

See entries `0.7.0a1` through `0.7.0a18` below.

## [0.7.0a18] — 2026-05-13

### Added

- **`RealtimeVoiceProvider.supports_mid_session_reconfigure`** capability flag — providers advertise whether `reconfigure(...)` can safely run mid-session. Defaults to `True` for backwards compatibility; overridden to `False` on the `gemini-3.x` Live family (which rejects `send_client_content` with WS 1007 after the first model turn and has no documented dynamic system_instruction update). Channel code consults the flag before calling `reconfigure` and routes content destined for `system_instruction` through session-start delivery instead.
- **`RealtimeVoiceChannel(skill_delivery_mode=…)`** — explicit selector for how skill bodies reach the model. `"inline_full"` bakes every available skill's full instructions into the initial `system_instruction` at session start under a "binding rules" section; `activate_skill` becomes a declarative ACK and no `provider.reconfigure` is needed. `"on_demand"` keeps the prior behavior. Auto-resolves from `provider.supports_mid_session_reconfigure` when not specified: providers that cannot reconfigure default to `inline_full`, the rest default to `on_demand`. Closes the path for `gemini-3.x` Live, which now has the skill rules in attention from the first token without ever needing a mid-session reconfigure.
- **`SKILLS_INLINE_PREAMBLE`** in `roomkit.channels._skill_constants` — preamble used by `inline_full` mode that tells the model the skill instructions are already loaded as binding rules, so it should follow them and call tools rather than narrate.

### Changed

- **`activate_skill` dispatcher submits the tool result BEFORE reconfiguring.** Pending function calls are bound to the live WebSocket; `reconfigure` tears that connection down and the response would be lost. Previous order (reconfigure → submit) left the model on the original (now-dead) connection waiting forever for a tool response that landed on a fresh `live_session` with no record of the in-flight `call_id`. New order: submit the ACK on the original connection, then (if the provider supports it) reconfigure for the next turn. Same fix applied to the Tool Search dispatcher.
- **Default `GeminiVisionConfig.model` and `GeminiConfig.model` switched to `gemini-3.1-flash-lite`** — Google is GA-ing the model and discontinuing the `gemini-3.1-flash-lite-preview` alias on 2026-05-25. Underlying model architecture is identical per Google; only the identifier changes.

### Fixed

- **Voice agents on `gemini-3.x` Live froze after `activate_skill`.** The activation handler called `provider.reconfigure(system_prompt=…+skill_body, tools=visible)` to push the skill body into `system_instruction`. On Gemini 3.x that reconnect was fatal: every `activate_skill` triggered a WebSocket tear-down and session resumption is fragile with non-trivial system prompts. Combined with the wrong submit/reconfigure order above, the model on the original connection waited forever for a tool response and "forgot the discussion." Now gated on the provider capability flag; on Gemini 3.x the skill body is baked into the initial `system_instruction` instead (via `skill_delivery_mode="inline_full"`) and no mid-session reconfigure is issued.
- **Tool Search silently no-oped on non-reconfigurable providers.** Tool Search's whole mechanic is mid-session `provider.reconfigure(tools=...)` to push newly matched tools onto the live session. When that call is gated off (Gemini 3.x), the `find_tools` tool stayed visible but had no observable effect — confusing the model. `RealtimeVoiceChannel.__init__` now force-disables Tool Search at construction time when the provider can't reconfigure, with a clear INFO log. The full catalogue is exposed verbatim instead.

### Also shipped in this release (work staged on Unreleased before a18)

#### Added

- **Tool Search for `RealtimeVoiceChannel`** — dynamic tool exposure for tool-heavy realtime sessions. Google's Gemini Live recommendation is 10–20 active tools; above that, function-calling reliability degrades sharply (the model narrates instead of invoking). New `tool_search`, `tool_search_pinned`, and `tool_search_threshold` constructor kwargs on `RealtimeVoiceChannel` enable a search-then-invoke pattern: only `find_tools(query)`, `list_tools(category=None)`, and a small pinned set are visible at session start; when the model calls `find_tools`, the catalogue is scored by token overlap (name 3×, description 1×) and the top matches are pushed into the live tool surface via `provider.reconfigure`. Auto-activates when `len(tools) > tool_search_threshold` (default 20) — pass `tool_search=True/False` to force. Per-session exposure window — parallel sessions don't cross-contaminate. Found in `roomkit.channels._realtime_tool_search.RealtimeToolSearchSupport` for direct use.
- **`FIND_TOOLS_SCHEMA`, `LIST_TOOLS_SCHEMA`, `TOOL_SEARCH_PREAMBLE`** in `roomkit.channels._tool_search_constants` — shared definitions for the search infra tools and the system-prompt addendum that tells the model to call `find_tools` before reaching for the rest.
- **Pydantic-style Optional collapsing in `clean_gemini_schema`** — `{"anyOf": [{"type": X}, {"type": "null"}]}` (the shape Pydantic / FastAPI emit for `Optional[X]`) is now folded to `{"type": X, "nullable": true}` *before* the unknown-key strip pass, so MCP / Pydantic-generated tools round-trip cleanly into Gemini Live `FunctionDeclaration`s. `oneOf` / `allOf` are handled the same way for symmetry. Wider unions keep the first non-null branch and mark `nullable` if any branch was null. Without this, `anyOf` was silently dropped and Gemini refused to invoke the affected tools (no error, just silence).
- **`ROOMKIT_GEMINI_DEBUG=1` diagnostic dumps** — `GeminiLiveProvider` now logs the full `LiveConnectConfig` it hands to Gemini Live (system_prompt body, every tool name + param/required count, a warning for any property that emerged typeless after schema cleaning, the first tool's full cleaned schema, voice/temperature/modalities) plus every server event coming the other way (`response_start`, `turn_complete`, `function_call`, `usage` ticks with prompt_tokens > 0, final transcription, `submit_tool_result` previews). Gated on the env var so prod logs stay clean. Single most useful piece of context for diagnosing "the model didn't pick the right tool" / "the model isn't invoking tools at all".
- **`SIPVoiceBackend.set_invite_filter()`** — runtime-installable pre-accept hook. Runs inside ``_handle_invite`` after digest auth has succeeded but before SDP / 200 OK; returns ``None`` to accept or ``(status, reason)`` to reject the INVITE with that 4xx/5xx response. Both sync and async filters are supported. Driving use case: application-layer routing decisions (DID not provisioned, tenant not authorized, outside business hours) that need DB access but should not result in an answered-then-dropped call. Carriers see a clean rejection in CDRs instead of a 200 OK followed by BYE. Filter exceptions are caught and treated as 500 rejection so a buggy callback can't crash the SIP message loop.
- **`InviteFilter` and `InviteFilterDecision` type aliases** in `roomkit.voice.backends.sip_auth`, exported alongside `SIPAuthMixin`.
- **`SIPVoiceBackend.set_auth_resolver()`** — runtime-installable callback for digest-auth credential lookups. The resolver receives the username from the `Authorization` header and returns the matching password (or `None` to deny). Consulted on every authenticated INVITE, so the application owns credential storage — no need to hold every tenant's credentials in process memory or rebuild the backend when one is added/rotated/revoked. Takes precedence over the static `auth_users` dict when both are set; falls through to the dict when the resolver returns `None`. Resolver exceptions are caught and treated as denial so a buggy callback can't crash the SIP message loop. Driving use case: multi-tenant deployments where each SIP trunk has its own credentials and tenants come and go without restarting the backend.
- **`AuthResolver` type alias** in `roomkit.voice.backends.sip_auth` — `Callable[[str], str | None]`, exported alongside `SIPAuthMixin`.
- **`SIPVoiceBackend.has_auth()`** — returns `True` when at least one credential source (the static `auth_users` dict or a resolver) is configured. Used internally by `_handle_invite` to gate the auth challenge; surfaced publicly for apps that need to make their own decisions before an INVITE arrives.
- **RFC 3326 BYE `Reason` exposed on SIP sessions** — `SIPVoiceBackend._handle_bye` now parses the carrier `Reason: Q.850 ;cause=N ;text="…"` header on every BYE and stashes the result on `session.metadata["bye_reason"]` (`{"cause": int, "text": str}`). A canonical Q.850 cause→text map fills in `text` when the carrier omits it. The same dict is attached to the inbound BYE `ProtocolTrace` metadata. Lets dialer orchestrators distinguish "user rejected" from "no circuits" from "normal hangup" without re-parsing the wire — the SIP layer just exposes what it sees; consumers decide what to do with it.
- **`parse_bye_reason()` helper** in `roomkit.voice.backends._sip_types` — accepts `str | bytes | None`, returns the parsed `{"cause", "text"}` dict or `None`.
- **`SIPVoiceBackend.disconnect(session, *, cause, text)`** — new optional kwargs attach an RFC 3326 `Reason: Q.850 ;cause=N ;text="…"` header to outbound BYEs on inbound sessions. Lets applications signal *why* they hung up (e.g. cause=21 "Call rejected" for tenant-routing rejection vs cause=16 "Normal call clearing" for an AI-ended call) so carriers log the right CDR cause and downstream IVR / analytics can branch on intent. Symmetric with the inbound `bye_reason` parsing already in `_handle_bye`. Quote characters and CR/LF in `text` are stripped to preserve header syntax.

#### Changed

- **`activate_skill` returns a small ACK instead of the full skill body.** The skill instructions are now buffered on the channel and pushed into Gemini Live's `system_instruction` (and the OpenAI Realtime equivalent) on the next `provider.reconfigure` call rather than coming back as a multi-KB tool result. Returning long bodies through `submit_tool_result` reliably tipped Gemini Live (and similarly long realtime returns on OpenAI Realtime) into "narrate the script" mode — the model treated the long return as conversational data and stopped emitting function calls for the rest of the session. Routing the body to `system_instruction` keeps it as binding rules and leaves the tool surface intact. New `RealtimeSkillSupport.activated_skills_prompt(session_id)` returns the concatenated active-skill bodies for the channel's reconfigure path.

#### Fixed

- **`GeminiLiveProvider.reconfigure()` wiped tools/voice/temperature on partial updates.** `reconfigure(system_prompt=new)` rebuilt the `LiveConnectConfig` from scratch via `_build_config`, which treats `None` as "absent" — so a prompt-only refresh (e.g. after a skill activation) silently dropped the tools list, leaving the model with no functions to call for the rest of the session. The provider now keeps an effective copy of `system_prompt`, `voice`, `tools`, and `temperature` on the per-session state and folds in the previous value for any field passed as `None`. Passing an explicit empty list / empty string still clears the field — only `None` means "preserve". Tracked on `_GeminiSessionState` so a chain of partial reconfigures composes correctly.
- **"BYE for unknown call_id" warning was indistinguishable from real state desync.** Two cases produced the same log entry: (1) carrier retransmits or counter-BYEs arriving just after our own cleanup — cosmetic noise that fired on every other call — and (2) a BYE for a call_id we never saw, which points to a real desync (dropped INVITE, dialog corruption, hostile probe). `_cleanup_session` now records cleaned-up call_ids in a 60-second TTL set, and `_handle_bye` downgrades the log entry to DEBUG when the call_id is still in that set. Truly-unknown call_ids still WARN. Set is bounded by an opportunistic eviction past a 1024-entry soft cap, so memory stays flat under high call churn.
- **`SIPVoiceBackend.disconnect()` for inbound calls sent BYE through `SipUAC.send_bye` and routed it to the L3 source of the original INVITE — both wrong.** The dialog was created on the UAS side, so the BYE has to use the UAS-side request build path and follow normal SIP routing rules: the dialog's `remote_target` Contact URI determines the L4 destination, not the L3 source. Through any NAT path (Docker bridge, carrier-side SBC) the L3 source is the masqueraded outer address while the Contact is the application-layer endpoint — the two diverge sharply, and BYEs sent to the L3 source leave the private network entirely. `disconnect()` now builds the BYE itself via `dialog.create_request("BYE", …)`, derives the L4 destination from the dialog's `remote_target` (parsed via `parse_uri`), and only falls back to `source_addr` when the dialog has no remote target. The audible symptom: inbound calls rejected from the `on_call` callback would appear connected for tens of seconds until the carrier's own session timer expired.
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

[Unreleased]: https://github.com/roomkit-live/roomkit/compare/v0.19.0...HEAD
[0.19.0]: https://github.com/roomkit-live/roomkit/compare/v0.18.0...v0.19.0
[0.10.0]: https://github.com/roomkit-live/roomkit/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/roomkit-live/roomkit/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/roomkit-live/roomkit/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/roomkit-live/roomkit/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/roomkit-live/roomkit/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/roomkit-live/roomkit/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a18...v0.7.0
[0.7.0a18]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a16...v0.7.0a18
[0.7.0a16]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a15...v0.7.0a16
[0.7.0a15]: https://github.com/roomkit-live/roomkit/compare/v0.7.0a14...v0.7.0a15
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
