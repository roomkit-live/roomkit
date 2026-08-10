# Changelog

All notable changes to RoomKit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`AudioCaptureSource` — capture that outlives a session.** A `VoiceBackend`
  takes the microphone when a session starts and gives it back when the session
  ends. Anything that must listen *before* a session exists — a wake word, a
  level meter — therefore had to hand the device over at the worst possible
  moment, while the person was still talking, and no application buffer repairs
  a closed device. A capture source owns the device instead, and a session
  becomes one subscriber among several.

  ```python
  from roomkit.voice.capture import LocalMicSource
  from roomkit.voice.backends.local import LocalAudioBackend

  mic = LocalMicSource(sample_rate=24000, backlog_seconds=10)
  mic.start()                                        # the device opens once
  detector = mic.subscribe(enqueue, name="wakeword") # no session in sight

  transport = LocalAudioBackend(source=mic)          # a subscriber, not the owner

  mark = mic.mark()                                  # at SPEECH_START
  await channel.start_session(                       # once the trigger matched
      room_id, participant_id, connection=None,
      metadata={"capture_since": mark},
  )
  ```

  The source retains recent audio in a bounded ring; `mark()` names a position
  in it and `subscribe(since=mark)` replays from there before going live.
  Replayed frames travel the ordinary inbound path, so they land in the realtime
  channel's existing pre-connect buffer and flush in order once the provider
  handshake completes — **no new control point on the inbound path**, and the
  discard-on-failed-handshake behaviour already in place applies unchanged.

  Addressing is by mark rather than by duration on purpose: "replay the last N
  seconds" forces the caller to guess, and guessing long replays the tail of the
  previous conversation into the model. A mark that has aged out of the ring
  replays what remains and reports `truncated` — raising at the moment a session
  opens would discard the very utterance the backlog exists to preserve.

  Fan-out is synchronous on the capture thread, which is what keeps AEC's
  capture/reference timing in step. The contract that follows is that a
  subscriber must enqueue the frame and return; the source times each callback
  and warns, naming the subscriber, when one runs long. Lifetime is explicit:
  `start()`/`stop()` alone acquire and release the device, so dropping to zero
  subscribers never stops capture and a detector can safely detach for the
  duration of a call — which it should, since source frames are pre-AEC.

  `LocalMicSource` (sounddevice) and `MockCaptureSource` ship with the ABC.
  `LocalAudioBackend(source=...)` derives its input format from the source and
  rejects a contradicting explicit argument; with no source its behaviour is
  unchanged. RFC Section 12.12, guide `shared-mic-capture.md`, and
  `examples/shared_mic_capture.py`.

- **`archive_room()` — the terminal room status is reachable.** `ARCHIVED` was
  enforced at every write gate but nothing could ever set it, so a status the
  RFC requires (Section 5.1) and a framework event it mandates (`room_archived`,
  Section 8.2) were both unreachable. Archiving refuses new events exactly as
  closing does, keeps history readable, and is idempotent.

- **Nine mandated framework events now fire.** `channel_registered`,
  `channel_unregistered`, `source_connected`, `source_disconnected`,
  `identity_resolved`, `hook_timeout`, `circuit_breaker_opened`,
  `circuit_breaker_closed`, `voice_session_ready` (RFC Section 8.2; the circuit
  breaker pair is also a Section 13.1 MUST). `source_connected` /
  `source_disconnected` are emitted **alongside** the existing
  `source_attached` / `source_detached`, which keep working — the RFC fixed a
  different name after the framework had shipped one. `hook_timeout` is now
  distinct from `hook_error`: a hook that never came back and one that raised
  were previously indistinguishable.

- **`ON_AI_THINKING`, `ON_STATUS_POSTED` and `ON_PLAN_UPDATED` actually fire.**
  All three were declared Implemented, and a hook registered on any of them
  could never run — reasoning and plans surfaced only as ephemeral events, so a
  host with no realtime backend saw nothing at all. New `ThinkingEvent` and
  `PlanUpdatedEvent` payloads.

- **`DTMFRedaction` — configurable masking of DTMF digits (RFC Section 17.6
  MUST).** Set `AudioPipelineConfig.dtmf_redaction` and the digits the framework
  itself exposes are masked: `frame.metadata["dtmf"]`, which travels to
  recorders, debug taps and logs, and the new `DTMFDetectedEvent.redacted_digit`.
  Defaults mask everything — a redaction that leaks the head of a PIN by default
  would be a worse trap than none — with `keep_first` / `keep_last` for the
  RFC's `4111********1111` card shape. `ON_DTMF`'s `digit` stays raw on purpose:
  that hook is how an IVR reads the digits it exists to collect.

- **`RecordingEncryption` — encryption at rest for recordings (RFC Section 17.6
  MUST).** `RecordingConfig.encryption` is honoured by `WavFileRecorder` at
  finalisation: the file is encrypted and the plaintext removed. A file the
  cipher cannot encrypt is discarded rather than handed back in the clear. No
  default cipher ships — a default key is not encryption — so the integrator
  supplies the how, the framework owns the when.

### Fixed

- **The provider's payload was discarded at the door.** RFC Section 5.2 makes
  `raw_payload` a MUST — "the audit trail and the source of truth for
  provider-specific data" — and nothing in the framework ever wrote it. Every
  parser lifted the handful of fields RoomKit models and dropped the rest, so
  delivery annotations, carrier fields and anything a provider added later
  existed nowhere. `InboundMessage` gains `raw_payload` and
  `provider_message_id`; the nine shipped webhook parsers populate them; and
  `TransportChannel`, `WebSocketChannel` and `CLIChannel` carry them onto
  `EventSource`. Both stores already round-tripped the fields — they were
  simply never filled.

  Telegram keeps the whole Update rather than the message object, so
  `update_id` survives; Messenger keeps each messaging entry rather than the
  batch envelope, so a batched webhook leaves every message its own payload.
  The dict is copied, not aliased: "unmodified" has to hold after the caller
  moves on.

- **Room operations could not be scoped to a tenant.** RFC Section 17.2
  requires room operations to be isolated per organization; any caller holding
  a room id operated on it regardless of which organization owned it.
  `get_room()`, `get_timeline()`, `send_event()`, `attach_channel()`,
  `detach_channel()`, `close_room()` and `archive_room()` now take an optional
  `organization_id` and refuse a room belonging to anyone else.

  A room owned by another organization is reported as **not found**, not as a
  distinct "wrong organization" error: a separate error would let a caller
  probe which room ids exist outside its own tenant, which is the thing
  scoping exists to prevent. A room created without an organization belongs to
  no tenant, so a scoped caller does not reach it either.

  The parameter is optional because a library has no caller or auth context of
  its own — the scope has to come from the caller. Left unset, every operation
  behaves exactly as before.

- **An address resolved to one identity across every tenant.**
  `identity_addresses` was keyed on `(channel_type, address)`, so a phone
  number registered by one organization resolved to *that* organization's
  identity for every other one — and a second tenant could not register the
  same number at all, the insert collided. RFC Section 17.2 requires isolation
  per organization. The key is now
  `(channel_type, address, organization_id)`, and `resolve_identity()` /
  `link_address()` take an optional `organization_id`.

  A SQL primary key cannot hold NULL, so "no organization" is stored as the
  empty string — a tenant like any other. The consequence is deliberate: an
  unscoped lookup does **not** find a scoped registration. Were it otherwise,
  omitting the organization would leak exactly as before.

  **Migration.** SQLite moves to schema v3: the primary key cannot be altered
  in place, so the table is rebuilt on first open, with existing rows carrying
  the empty string — they resolve exactly as they did. A v1 file now travels
  v1 → v2 → v3. Postgres adds the column and rebuilds the key with idempotent
  DDL applied at `init()`.

  **For custom stores:** `ConversationStore.resolve_identity` and
  `link_address` gained an optional parameter. An implementation written
  against the previous signature keeps working for its own callers — the
  framework never calls these itself, an integrator's `IdentityResolver` does
  — but it must add the parameter to participate in scoping.

- **A store commit slower than `process_timeout` could be reported blocked.**
  RFC Section 13.6 forbids wrapping the commit in a cancellable unit whose
  expiry leaves a committed event reported as blocked, which is exactly what
  happened: the timeout covered the whole pre-commit coroutine, commit included,
  and with a store that commits on a worker thread the write landed while the
  caller received `blocked, reason="process_timeout"`. The pre-commit phase now
  *decides* and writes nothing; every durable write — the BLOCKED record, the
  edit/delete target mutation, the commit itself — happens after the cancellable
  region is left.

- **A response landing after `close_room()` grew the closed room's timeline.**
  Reentry passes never checked the room status, so an AI answer completing after
  the room closed committed a DELIVERED event into it. The status gate (RFC
  Section 5.1: "at **every** point where the timeline can grow") now also covers
  reentries, lifecycle system events, the identity-challenge injection and the
  post-broadcast blocked-event commits. The one documented exception stays: the
  record *of* the closing transition.

- **A READ_ONLY channel's answer was stored DELIVERED and visible to everyone.**
  RFC Section 7.5 rule 2 requires an event whose source cannot write to be
  stored BLOCKED and never broadcast; the reentry path stored it DELIVERED with
  visibility `all`, so a read-only observer's reply leaked into every channel's
  rebuilt history. It is now stored BLOCKED with `source_read_only`. A muted
  source's suppressed response is likewise recorded BLOCKED with `source_muted`
  instead of being dropped without trace — muting silences the voice, it does
  not erase the record.

- **The CONFIRMED interruption strategy could never interrupt.** Every
  evaluation happened at speech onset, where the sustained duration is 0 by
  construction, so `0 >= min_speech_ms` was always false — and the speech was
  then discarded as echo. The default strategy therefore ignored barge-in
  entirely. A pending evaluation is now re-taken once the speech has had
  `min_speech_ms` to sustain, and the continuous-STT energy path passes its real
  run duration instead of a constant 0.

- **A backend's own barge-in detection overrode the interruption policy.**
  `DISABLED` (and the legacy `enable_barge_in=False`) cancelled TTS anyway on
  any transport with native detection. The transport now reports detected
  speech; the configured strategy still decides. Note the consequence:
  transport barge-in is now subject to `allow_during_first_ms` (200 ms under the
  legacy defaults) like the VAD path already was.

- **`DISABLED` discarded user speech instead of queueing it.** RFC Section 12.6
  says the speech "is queued until the bot finishes"; it was dropped. Segments
  captured during playback are now held (bounded) and replayed once the bot's
  audio has drained.

- **Cancelling a Deepgram stream surfaced as an unhandled websocket error.**
  The sender task closed the stream in a `finally` that ran on the cancellation
  path, where the socket is already gone — so the close raised, and an
  exception raised in a `finally` during cancellation replaces the
  `CancelledError` the caller is suppressing. Stopping a voice session with
  Ctrl+C therefore printed a `ConnectionClosedError` traceback at interpreter
  shutdown. Closing an already-closed stream is now logged at debug and not
  propagated.

- **`RNNoiseDenoiserProvider` could not find a Homebrew-installed library, and
  its error message sent macOS users to the wrong package.** The loader probed
  `~/.local/lib` and `/usr/local/lib` only, so an Apple Silicon install under
  `/opt/homebrew/lib` was invisible; `HOMEBREW_PREFIX` is now honoured too. The
  ImportError no longer recommends `brew install rnnoise` — that package is a
  cask of DAW plugins built from a different project and cannot provide this C
  ABI — and instead names a source build, the pip-installable
  `SherpaOnnxDenoiserProvider` alternative, and the directories it searched.

## [0.46.0] — 2026-08-08

### Added

- **`SQLiteStore` — the embedded persistent conversation store.** One `.db`
  file, stdlib `sqlite3` only: the backend for single-process deployments
  (desktop apps, edge boxes, small bots) where `PostgresStore` is a burden and
  `InMemoryStore` forgets everything on exit. Models are stored as their
  pydantic JSON with the queryable fields extracted into indexed columns, and
  message text is mirrored into an FTS5 table, so history is full-text
  searchable via `SQLiteStore.search_events(query, room_id=…)` — the raw
  material for "what did we talk about last week?" recall. Every call runs on
  one dedicated worker thread (the event loop never blocks on disk I/O) and
  multi-statement operations commit as one `BEGIN IMMEDIATE` transaction, so
  the index-assignment path is serialised by the database file itself. The
  backend is intentionally single-process at the framework level: pairing it
  with `InMemoryLockManager` emits the same warning as any shareable store,
  because SQLite transactions cannot make the full inbound pipeline atomic
  across workers. The whole `InMemoryStore` behavioural suite runs against it.

- **`ConversationStore.is_process_local` — stores declare their own process
  scope.** RoomKit warns when a shareable store is paired with
  `InMemoryLockManager`, because per-process locks cannot serialise the inbound
  pipeline across workers. That check used to test the store against a
  hardcoded list of built-in classes, so a third-party backend could never be
  recognised as process-local and every custom store drew the warning. The
  capability now lives on the contract: the ABC defaults to `False` (persistent
  backends are conservatively assumed shareable) and an implementation that is
  genuinely confined to one process sets `is_process_local = True`, as
  `InMemoryStore` does. Existing stores inherit the safe default and keep their
  current behaviour.

### Fixed

- **Event indexes remain monotonic after deletion in every shipped store.**
  `InMemoryStore` and `SQLiteStore` derived the next index from the current
  event count, while `PostgresStore` used `MAX(index) + 1`; deleting the latest
  event therefore reused its index, violating the RFC timeline invariant.
  All three backends now keep a per-room high-water mark. SQLite migrates v1
  files to schema v2 on open, and enforces unique room indexes and idempotency
  keys at the database boundary; a file written by a newer RoomKit, or a v1 file
  already holding duplicates the v2 constraints would reject, raises the new
  `SQLiteSchemaError` rather than opening in an inconsistent state. Postgres
  gains an `event_sequences` table, created and backfilled from the existing
  room counters by the additive schema batch `init()` already runs on connect —
  no opt-in migration and no data loss for deployments upgrading in place.

- **Gemini Live no longer drops legitimate repeated utterances without VAD.**
  `turn_complete` now acts as the user-utterance boundary when Gemini omits
  `ACTIVITY_START`, and a new assistant response resets final-transcript
  deduplication before its first chunk — including messages that coalesce
  `output_transcription` and `model_turn`.

- **Realtime tool calls could overtake the transcription that triggered
  them.** Transcriptions travel a per-session serialised queue; tool calls run
  in their own task. Even with providers emitting the user final before the
  call, the tool could reach the application first — the tool row closed the
  user's chat entry and the late final re-rendered it (captured in the field:
  tool execution at .801, user final at .836). Tool handling now passes
  through the same per-session FIFO lock as a barrier before executing, and
  releases it during execution so tools never hold transcriptions back.

- **Gemini Live emitted tool calls ahead of the user's final.** Same
  inversion as below through the tool path: a function_call arrives before any
  `model_turn`, so the user transcript stayed buffered across the whole tool
  round and its late final read as new user speech. A tool call is the model
  acting on the utterance — the user buffer now flushes before the call is
  emitted.

- **Gemini Live emitted the reply's transcript ahead of the user's final.**
  Without VAD events, Gemini only finalises the user transcript when the model
  starts replying — and one server message can carry both the reply's first
  transcript chunk and `model_turn`. The handler emitted that chunk before
  flushing the user buffer, so the channel saw the conversation inverted: the
  late user final read downstream as *new* user speech, producing a phantom
  barge-in and a duplicated user entry. The user buffer now flushes before any
  assistant transcription goes out.

- **Gemini Live re-emitted final transcriptions as duplicates.** Gemini
  re-sends a finished utterance after the provider's buffer already flushed it
  at a lifecycle boundary (speech end, model turn); each re-emission reached
  the channel as a second identical final — chat UIs rendered duplicate user
  bubbles and the phantom "user speech" falsely interrupted the assistant's
  streaming reply. The provider now drops consecutive identical finals per
  role, lifting the guard when new speech (`ACTIVITY_START`) or a new model
  turn genuinely begins, so repeating the same words in a later turn still
  comes through.

- **Realtime transcriptions could render out of wire order.** Each
  transcription event runs in its own task and the partial path awaits one hop
  more than the final path, so a final could overtake the partial that
  preceded it on the wire; the late partial then resurrected the
  already-finalised utterance downstream — Gemini ships a short utterance as
  one chunk plus its final in the same server message, and chat UIs rendered
  both as identical duplicate bubbles. A per-session FIFO lock in
  `_process_transcription` keeps hook delivery in arrival order.

## [0.45.0] — 2026-08-07

### Added

- **Deepgram Voice Agent — any TTS vendor in the `speak` stage.** Deepgram
  composes its agent from independently chosen stages, and its `speak` stage
  accepts five provider types (`deepgram`, `eleven_labs`, `cartesia`,
  `open_ai`, `aws_polly`) — but RoomKit hardcoded `type: "deepgram"`, so only
  Aura voices were reachable. `DeepgramAgentConfig.speak_provider` (also a
  per-session `provider_config` key) now carries the full
  `agent.speak.provider` dict verbatim — each vendor has its own field shape
  (`model` vs `model_id`/`voice_id` vs `voice` objects), so RoomKit passes the
  dict through rather than modelling five schemas that would rot.
  `speak_endpoint` rides alongside, carrying the BYO-key endpoint that
  vendors like ElevenLabs require (Deepgram-managed ones need none).
  Mid-session `reconfigure()` swaps vendors wholesale, and a `voice` argument
  naming an Aura model is ignored with a warning when another vendor holds
  the stage.

- **ElevenLabs ConvAI — TTS speed override.** `provider_config["speed"]`
  rides `conversation_config_override.tts`, clamped into the 0.7–1.2 range
  ElevenLabs accepts. The agent must whitelist the speed override in its
  security settings for the value to take effect.

### Fixed

- **Deepgram settings escape hatch — vendor switches no longer blend
  provider fields.** `provider_config["settings"]` is deep-merged into the
  built payload, which used to *merge* a replacement provider into the
  default one — overriding `agent.speak.provider` with an ElevenLabs block
  left the default Aura `model` key inside it, a hybrid Deepgram rejects.
  Deepgram's provider objects are type-discriminated unions, so the merge now
  replaces a dict wholesale whenever the override names a different `type`.

## [0.44.0] — 2026-08-07

### Added

- **Deepgram managed-LLM prompt cap — configurable, and honest about what
  happens past it.** `DeepgramAgentConfig.max_prompt_chars` (default 25,000 —
  Deepgram's documented cap; `None` disables) replaces a hardcoded module
  constant, with a per-session `provider_config["max_prompt_chars"]` override.
  The old warning fired only when silent injections grew the prompt, claimed
  the update would "likely be refused", and fired even for bring-your-own
  `think_endpoint` sessions. What Deepgram actually does past the cap is
  *truncate* the prompt and keep the session (`PROMPT_TOO_LONG`, a non-fatal
  warning) — and it documents no cap at all for BYO endpoints. The check now
  says so, also covers the initial prompt at connect and full replacements
  through `reconfigure()`, and stays silent when a `think_endpoint` is in
  force.

- **Realtime model catalogs — `RealtimeVoiceProvider.available_models()`.**
  The speech-to-speech model ids were the one lineup RoomKit knew only as
  constructor defaults: the chat catalogs exclude them on purpose (the xAI
  catalog's scope note even names `grok-2-audio` as belonging elsewhere — an
  elsewhere that did not exist), so a host building a model picker had
  nothing to enumerate and hardcoded its own list, which then rotted
  independently of the defaults it mirrored.

  The catalog lands as the realtime counterpart of `available_voices()`: a
  classmethod on the provider, answering offline, backed by pure-data
  modules — `providers/{openai,gemini,xai}/realtime_models.py` — that follow
  the image-catalog separation (RFC §25.6): disjoint sets are kept apart
  rather than merged into `AIProvider.available_models`, because no realtime
  id answers a chat completion and no chat id opens a realtime session.
  Entries state what the vendor documents: `supports_vision` follows the
  image-input cut (`gpt-realtime-2.1`+ and every Live model; xAI stays
  `False`, restating 0.43.0's deliberate withholding), `"thinking"` marks the
  `gpt-realtime-2`+ models whose sessions accept `reasoning_effort`, and the
  retired `gpt-4o-*-realtime-preview` pair is flagged `deprecated`. Context
  windows and pricing are omitted deliberately — the realtime channel never
  trims by window, and audio-token rates are a unit `ModelPricing` does not
  model, so restating text rates would price the wrong unit.

  Two providers keep the empty base default on purpose: Deepgram composes
  its agent from stages that have catalogs of their own (`speak` is the
  voice catalog, `think` reads the vendors' *chat* catalogs), and ElevenLabs
  binds a dashboard-configured agent. `scripts/check_models.py` names all
  three new catalogs in `UNMIRRORED_CATALOGS` — the aggregator mirrors chat
  completions, and its `gpt-audio`/`gpt-audio-mini` are that other lineup —
  so every `make check-models` run states who verifies them and when,
  instead of reading silence as coverage. Tests pin the invariant nothing
  else would notice: each provider's constructor default appears in its own
  catalog.

- **`ImageProvider` — an agent can draw, whatever provider holds the
  conversation.** Nothing in RoomKit generated an image: `AIResponse.content` is
  a `str`, and both model catalogs excluded the image lineups on purpose. Images
  travelled inbound only, as `AIImagePart`.

  The capability lands as its own surface (RFC §25), not as a widening of the
  conversational response. That is the whole design: reached *through*
  `AIResponse`, image generation would inherit the provider holding the
  conversation, and would then exist only for the conversations already run by a
  model that draws — acquiring it would mean migrating the conversation. Kept
  apart, an Anthropic agent draws with a Gemini or OpenAI key exactly as it
  transcribes with a Deepgram one. `AIResponse` and every `AIProvider` signature
  are untouched.

  `generate(prompt, size=…, n=…, reference_images=…)` returns `ImageResult`
  objects whose `data` is *always* a `data:` URI — never bare base64, never a
  link that expires — so a result feeds `MediaContent.url` and enters a room with
  no conversion, and `to_image_part()` hands it straight back as the next edit's
  reference. Editing is in the signature from the start rather than bolted on:
  `reference_images` routes to OpenAI's separate edit endpoint or rides Gemini's
  same call, and the caller never sees the difference. `size` is one
  `"WIDTHxHEIGHT"` string everywhere, translated per vendor (Gemini speaks
  aspect ratios and resolution tiers); a size a model cannot produce raises
  rather than silently becoming another one.

  `OpenAIImageProvider` (`/v1/images`) and `GeminiImageProvider` (Interactions
  API) ship with offline catalogs — `ImageProvider.available_models()`, disjoint
  from the conversational one, because no id draws *and* converses and merging
  them would only oblige every consumer of the chat catalog to filter. These
  models are billed **per token**, with the pixels metered apart and an order of
  magnitude above text, so `ModelPricing` gains two optional fields
  (`image_input_per_million`, `image_output_per_million`) and `cost_for()` two
  disjoint counters. Both default to `None`, so every existing catalog and every
  existing call price exactly as before.

  `MockImageProvider` returns a real 1×1 PNG, so a consumer exercises the whole
  path — decode, write, measure — with no key. `examples/image_generation.py`
  runs it end to end: a `generate_image` tool draws, the picture lands in the
  room as `MediaContent` and on disk as a PNG.

- **`GeminiSTTProvider` — batch transcription that returns speaker turns and
  timestamps in the same pass as the words.** Gemini has no speech-to-text
  endpoint; transcription is an instruction to a multimodal model that accepts
  audio. That makes the provider batch by construction — it takes a complete
  recording and answers in seconds — and it is deliberately not a competitor to
  the streaming recognisers already here. `supports_streaming` is `False`, so a
  voice channel transcribes on `SPEECH_END` rather than opening a stream.

  What the batch shape buys is what a streaming recogniser structurally cannot
  give: the model sees the whole recording before it answers, so one call
  returns the transcript, the speaker turns and the timestamps together, with no
  diarization stage and no merge. Until now roomkit had local batch STT
  (sherpa-onnx, Qwen3) and cloud streaming STT (Deepgram, Gradium), and nothing
  for the case the recorders already produce: a finished meeting, a voicemail,
  an imported file.

  `transcribe()` keeps the `STTProvider` contract and returns flat text.
  `transcribe_recording()` returns the `Transcript` — detected language plus one
  `TranscriptSegment` per turn — and accepts a file path, inlining recordings
  under `max_inline_bytes` and uploading larger ones through the Files API
  (deleted after use rather than left to expire). Arbitrary `http(s)` URLs are
  refused rather than fetched, so the provider is not an SSRF vector; pass a
  path, raw audio, or a Files API URI.

  Speaker labels are the model's judgement, not an acoustic decision, and a
  conference records one track per participant — so where the speakers are
  already separated, transcribe each track with `diarize=False` and merge on the
  timestamps. `examples/meeting_transcription.py` runs the whole path: a
  recording becomes speaker turns, enters a room, and an AI channel writes the
  minutes.

- **Offline AEC bench — `scripts/aec_bench.py`.** The upcoming echo-path
  work needs a judge that is not an ear: the bench consumes an
  `aec-dump/1` capture (a JSONL event stream preserving, in arrival
  order, every reference frame the canceller was fed, every capture
  frame with its processed output, and the per-stream activation
  toggles), scores it in 100 ms windows — per-window attenuation over
  echo-active windows, with quiet, too-faint and probable-doubletalk
  windows classified out rather than mis-scored — and replays the same
  events through a fresh `WebRTCAECProvider` under different settings
  (`--delay-ms`, `--ns`) for a side-by-side comparison on the exact audio
  that exposed a problem.  Doubletalk is judged against the dump's own
  echo median: the reference RMS is a digital level and the capture RMS
  an acoustic one, so no fixed ratio between the two scales means
  anything.  The recorder half lives in the host application (RoomKit
  UI's `AECDumpRecorder`) — the format header is the contract.

### Fixed

- **A realtime session's audio-level hooks no longer log tracebacks on
  shutdown.** Levels stream off the audio thread until the very last frame,
  so a few `ON_INPUT_AUDIO_LEVEL`/`ON_OUTPUT_AUDIO_LEVEL` firings are always
  in flight when `close()` seals the framework's resources. Each one then
  failed inside its context build and logged a full `ERROR` traceback —
  every clean shutdown with a level hook registered ended on alarm bells
  that meant nothing. A level during teardown carries nothing: the hook
  path now checks the seal and skips, and a firing that slips past the
  check — sealing races it by construction — is recognised as the framework
  closing and logged at `DEBUG`.

- **Deepgram's agent transcript no longer lands as one entry per
  sentence.** The Voice Agent wire delivers `ConversationText` sentence
  by sentence, final-only, and the provider fired each one as a final
  transcription — every sentence became its own timeline entry (RoomKit
  UI rendered a paragraph gap per sentence: four entries for one
  greeting).  Sentences now accumulate as delta partials and the turn
  closes with one full final — at `AgentAudioDone`, on a barge-in
  (`UserStartedSpeaking` finalizes the truncated transcript before the
  user's turn opens), when a user transcript proves the turn ended even
  if `AgentAudioDone` was lost, or at session teardown.  The user-side
  transcript is untouched: one per utterance, and it still closes the
  user's turn.

- **xAI's streaming input transcription no longer arrives as a pile of
  "finals".** Grok's realtime server re-emits
  `conversation.item.input_audio_transcription.completed` with the
  cumulative text-so-far while the utterance is still in progress — the
  OpenAI protocol this wire mirrors sends exactly one, final, `completed`
  per item, and the shared base read each re-emission accordingly.  Every
  consumer of `on_transcription` therefore got a growing sequence of
  final user transcripts for one sentence (RoomKit UI rendered a chat
  bubble per snapshot: "Quelle aventure ?" / "Quelle aventure ? De quelle
  aventure ?" / the full sentence).  The xAI provider now restores the
  contract itself: snapshots become delta *partials* keyed by item id — a
  non-extending snapshot (server-side STT correction) goes out as one
  more delta, since partials cannot be retracted — and the one true
  final, carrying the full corrected text, fires when the turn provably
  ended: the model starts responding, a new item begins, or the session
  disconnects.  Hosts need no provider-specific handling; the ordinary
  partial/final flow just works.

- **A muted mic desynchronized the pipeline AEC by the mute's full
  duration.** Pausing capture pauses the AEC's world, and the transport-AEC
  feed honoured that — but the pipeline-AEC reference rides
  `on_audio_played`, which kept flowing while `set_input_muted` dropped the
  mic frames.  A 6-second mute measured live left `refs_fed` 736 frames
  ahead of `processed`: on unmute the filter cancelled against audio the
  capture never saw, attenuation collapsed to −12 dB, and the provider's
  server VAD read the leaked echo as user speech — a false barge-in as the
  direct product of pressing mute.  The backend now stamps
  `capture_paused` on every played frame (the broadcast itself continues:
  playback is physically ongoing and level/position listeners must keep
  seeing it), and the pipeline's reference consumer holds its feed while
  the flag is up — both timelines pause and resume together, which the AEC
  bench verifies as the `refs ≈ processed` invariant.

- **`inbound_dsp_threads` silently unplugged every async pipeline callback —
  in a realtime session, that is the microphone.** The pool's workers run
  the stage chain with no event loop, and `_maybe_schedule` answered a
  coroutine created there by logging one line and dropping it. The failure
  wore camouflage: every *sync* callback kept working, so audio flowed
  through the stages, recordings rolled, and what died was precisely what
  rode coroutines — the realtime provider's audio feed and the audio-level
  hooks. A host that enabled the pool got a session that looked alive and
  heard nothing (found by RoomKit UI: mic and VU meter dead with
  `inbound_dsp_threads=2`, fine inline).

  `AudioPipeline` now captures the loop it was built on — the channel
  builds its pipeline inside async context — and `_maybe_schedule` sends an
  off-loop coroutine home with `run_coroutine_threadsafe` instead of
  dropping it, with failures logged through a done-callback exactly like
  the on-loop task path. A pipeline genuinely built outside async context
  still gets the old warn-and-close, there being nowhere to send the work.
  `_pipeline_submit_inbound`'s claim that the callbacks "were already
  thread-tolerant" is corrected to say what is true now, and of what.

## [0.43.0] — 2026-08-07

### Added

- **A `BEFORE_TOOL_USE` hook may rewrite the arguments of the call it is
  gating.** `ON_TOOL_CALL` could already replace a tool's *result* on the way
  out, through `metadata["result"]`. Nothing could touch a call's *arguments*
  on the way in: the hook's callback answered with a bare `bool`, so the only
  thing a host could say about a tool call was yes or no.

  That asymmetry breaks any host that shows the model something other than the
  literal data — redaction being the case that surfaced it. Such a host hands
  the model placeholder text and swaps the real values back into the reply the
  user reads. But when the model passes that text to a tool, the tool is about
  to act on it in the real world, and no seam existed to restore it first. The
  placeholder travels into whatever the tool writes to, and the mechanism meant
  to protect the data quietly corrupts every outbound write instead.

  The callback now answers with a `BeforeToolDecision` carrying the verdict
  and, optionally, the arguments a hook returned under
  `metadata["arguments"]`. Downstream of the gate the AI channel works from
  those arguments, so the tool handler, the `ON_TOOL_CALL` event and the usage
  record all report what actually ran rather than what the model asked for. An
  `ExternalToolHandler` receives the same rewrite as
  `ToolDecision.modified_input`, the field that already existed on that type.
  Rewritten arguments are validated against the tool schema before execution,
  just like the model's original payload.

  The decision is truthy when the call is allowed, so a handler that gates on
  it (`if not decision:`) reads unchanged. A hook that returns a plain `allow`
  leaves the arguments exactly as the model wrote them.

- **An Anthropic turn may carry its caller's own API key.** A shared provider
  still uses its configured credential by default; a host serving users with
  their own Anthropic subscriptions can set `AIContext.metadata["api_key"]`
  in `BEFORE_AI_GENERATION` for that turn alone. Per-key clients are cached in
  a bounded pool, and the credential is represented as a secret so context
  logging and serialization cannot disclose it. Cache eviction leases clients
  to active turns: a burst of distinct credentials may exceed the soft bound
  briefly, but cannot close the HTTP client underneath an in-flight stream.

- **`AIChannel`, `VoiceChannel`, `RealtimeVoiceChannel` and `WebSocketChannel`
  now import from `roomkit.channels`** — where the eleven transport-channel
  factories already lived. The four channel classes were reachable only from
  the top-level package, with nothing marking the difference, so
  `from roomkit.channels import AIChannel` raised `ImportError` while the
  `SMSChannel` on the line next to it worked.

  Purely additive, and free: importing `roomkit.channels` already executes the
  top-level `__init__`, which already imports all four. Both entry points
  return the same object, and a test now asserts that so they cannot drift.

- **Deepgram Voice Agent as a speech-to-speech provider** —
  `DeepgramAgentProvider` in `roomkit.providers.deepgram`, on the new
  `realtime-deepgram` extra (`websockets` only — no SDK).

  Where the other speech-to-speech providers hand you one end-to-end model,
  Deepgram composes an agent from three stages picked independently: `listen`
  (Nova/Flux), `think` (an LLM, managed or pointed at your own OpenAI-compatible
  endpoint) and `speak` (Aura). Swapping the model without touching the voice is
  the reason to reach for it. `mulaw` at 8 kHz is accepted in both directions, so
  a SIP or Twilio leg needs no resampling at all.

  `reconfigure()` is overridden to patch the live session — `UpdateThink` carries
  prompt, model and functions in one message, `UpdateSpeak` swaps the voice — so
  an agent handoff keeps the WebSocket and the conversation context instead of
  reconnecting. Because Deepgram replaces the complete Think/Speak block, each
  update starts from the live session state: omitted per-session models,
  endpoints, temperatures and context settings are preserved, and a
  `provider_config`-only update is applied. The curated catalog is the full
  Aura-2 set across seven languages, plus the twelve Aura-1 voices flagged
  `deprecated`.

  Three constraints are the protocol's, and are documented rather than papered
  over: turn detection is always Deepgram's (`server_vad=False` warns and is
  ignored), `ConversationText` is final-only so there are no interim
  transcriptions, and there is no client-side interrupt message. That last one
  costs nothing in practice — Deepgram signals barge-in the other way, with
  `UserStartedSpeaking`, which the provider surfaces as `on_speech_start`, and
  that is already what makes the channel flush playback and send `clear_audio`.
  `interrupt()` therefore only clears local state. The provider also fires
  `on_speech_end` on the user's own transcript, since Deepgram has no
  speech-stopped event and idle detection would otherwise never fire again.

- **Google Gemini as a TTS provider** — `GeminiTTSProvider` in
  `roomkit.voice.tts.gemini`, on the `gemini` extra roomkit already ships for
  the Gemini AI provider and Gemini Live, so no new dependency.

  Gemini TTS is a generative speech model rather than a voice engine: the
  prompt it receives is an instruction. A `style_prompt` like *"Lis ce texte
  d'une voix calme"* steers delivery and is verifiably not spoken, which no
  other TTS in roomkit can do without markup. 30 prebuilt voices, shared with
  Gemini Live native audio — `available_voices()` reads the one existing
  catalog rather than restating it.

  What it is not is a conversational TTS. Measured against the live API,
  time-to-first-audio is 3–8 seconds depending on the model, so the provider is
  documented for prompts, announcements and generated audio messages, and
  `supports_streaming_input` reports `False` — the API takes a complete prompt,
  and claiming otherwise would route a `VoiceChannel` into a streaming path the
  service cannot honour. The default model is the only one of the three that
  streams as it generates, so playback can start on the first 40 ms frame.

  Output is fixed by the service at 24 kHz mono PCM. The request accepts a
  `sample_rate` field that the service silently ignores, so the config does not
  expose it; use a resampler stage when the transport needs another rate.

- **An image can be shown to a live OpenAI Realtime session.** `inject_image`
  was declared on the provider ABC and implemented only for Gemini Live, and
  `RealtimeVoiceChannel` catches the resulting `NotImplementedError` and logs
  it — so an image sent to an OpenAI session disappeared with a warning and no
  error. `gpt-realtime-2.1` reads images, so the gap was the provider's, not
  the model's.

  The image travels as a data URI in a user message, alongside an optional text
  prompt in the same item. PNG and JPEG are the only formats the API reads, and
  an unsupported MIME type now fails immediately rather than on the wire.
  `provider_config["image_detail"]` (`auto` | `low` | `high`) trades fidelity
  for tokens; left unset the API's default applies, which resolves to high
  detail — worth knowing on a session that injects frames repeatedly.

  This is not the `video.vision` path: that one makes a separate model call and
  returns text. This one puts the picture in the realtime model's own context,
  next to the audio.

  The implementation sits on the OpenAI provider rather than the shared
  WebSocket base, so the xAI provider does not inherit a vision capability its
  models have not been verified to accept.

- **The two pipeline stages you could configure but not run.** `AGCProvider`
  and `DenoiserProvider` shipped ABCs and mocks, but every real implementation
  needed something first — a native library, an ONNX model or a cloud key. A
  quiet microphone or a noisy room meant adding a dependency to fix what the
  audio path could already measure.

  `SimpleAGCProvider` is dependency-free adaptive gain: it measures each frame's
  RMS, walks a stream-local gain toward `AGCConfig.target_level_dbfs` on the
  configured attack/release constants, and limits the peak before converting
  back to PCM. Near-silence is left alone, so an idle mic's noise floor is not
  amplified into apparent speech, and `metadata["gain_applied_db"]` reports what
  was done. `silence_threshold_dbfs` and `min_gain_db` are tunable through
  `AGCConfig.metadata`.

  `WebRTCNoiseSuppressorProvider` runs WebRTC noise suppression on the inbound
  path with no echo canceller attached, on the `webrtc-aec` extra roomkit
  already ships — both bind the same upstream library.

- **Two hooks for provider authors.** `AECProvider.set_stream_active()` toggles
  cancellation for one playback stream rather than globally, and
  `RealtimeVoiceProvider.truncate_audio()` lets a provider drop the unheard tail
  of an interrupted response from its server-side context. Both have defaults
  that preserve existing behaviour, so third-party providers need no change.

### Changed

- **A broken skill now stops discovery instead of disappearing.**
  `SkillRegistry.discover()` logged a warning and skipped any skill whose
  `SKILL.md` failed to parse or validate, and did the same for a directory that
  did not exist. The result was the worst kind of quiet: the skill simply was
  not in the catalogue, the model was never told the capability existed, and
  the agent kept answering as though nothing were missing. A typo in a
  frontmatter name surfaced hours later as an agent that inexplicably could not
  do its job.

  Discovery is now strict by default — an unreadable directory raises
  `SkillDiscoveryError`, an invalid skill re-raises its own `SkillParseError`
  or `SkillValidationError`. It also commits only after every candidate has
  parsed, so a failure leaves the registry exactly as it was rather than half
  filled. Pass `discover(..., strict=False)` for the previous behaviour, which
  remains the right call when skills come from a source you do not control.

  All skill exceptions now derive from a new `SkillError` base, so "something is
  wrong with the skills on disk" is one `except` clause. Existing
  `except SkillParseError` / `except SkillValidationError` are unaffected.

- **The OpenAI Realtime provider defaults to `gpt-realtime-2.1`**, two
  generations up from `gpt-realtime-1.5`. Upstream reports lower p95 latency,
  better alphanumeric recognition — order numbers, phone numbers, confirmation
  codes read back correctly — and more reliable interruption when the caller
  speaks over the model. `gpt-realtime-2.1-mini` remains available as an
  explicit, cheaper choice; a pinned `model=` is unaffected.

  A default that ages is a default that eventually 404s, which is why it is now
  covered by a test. The audio contract is unchanged (PCM at 24 kHz, G.711
  μ-law/A-law at 8 kHz), so telephony paths need nothing.

- **Reasoning effort reaches the OpenAI Realtime session.** Reasoning-capable
  models (`gpt-realtime-2` and later) accept an effort level that trades
  latency for depth, and RoomKit had no way to send it. It is now
  `provider_config={"reasoning_effort": "minimal|low|medium|high|xhigh"}`,
  carried as the session's own `reasoning` field and omitted entirely when
  unset, so models without reasoning are untouched.

- **`AudioPipelineConfig.agc_config` now builds the stage it describes.** It was
  documented as an override and read by nothing: a pipeline that set
  `agc_config` without also constructing an `agc` provider ran no gain control
  at all, and said so nowhere. It now creates a `SimpleAGCProvider` from that
  config when no explicit provider is given. Passing `agc=` is unaffected, and a
  backend declaring `NATIVE_AGC` still skips the stage.

- **BREAKING — `RNNoiseDenoiserProvider` accepts 16, 24 and 48 kHz only.** It
  previously accepted any rate dividing evenly into 48000, which admitted 8 and
  12 kHz into a path RNNoise's exact integer-ratio arithmetic does not exercise.
  Those rates now raise `ValueError` at construction instead of degrading
  quietly. An 8 kHz telephony leg needs a resampler stage ahead of the denoiser.

- **An audio stage bypasses a frame it cannot process instead of mangling it.**
  AGC and every denoiser now check a frame's sample rate, channel count and
  sample width against what they were configured for, pass a mismatch through
  untouched, and warn once per distinct format rather than once per frame.
  Sizes that are not whole native chunks are buffered behind a fixed one-block
  delay rather than dropped or passed through raw, so every input byte has
  exactly one output byte and the timeline stays continuous. The configs
  validate at construction too — `AGCConfig`, `AICousticsDenoiserConfig` (which
  gains an explicit `sample_rate`) and the SherpaOnnx denoiser config raise
  `ValueError` on levels, gains and time constants that cannot describe a stable
  controller.

### Fixed

- **Realtime voice startup is now transactional.** Audio arriving while the
  provider performs its handshake is retained in a bounded buffer and flushed
  in order; a failed transport, provider, negotiated sample rate or client-ready
  notification tears down every partial map, DSP stream, resampler, socket and
  telemetry span. Raw PCM from WebSocket, SIP and FastRTC transports is now
  normalised to `AudioFrame` before entering a configured pipeline, and that
  channel-owned pipeline registers its transport callback only once instead of
  duplicating delivery after every new session.

- **Tool authorization now covers the payload that actually executes.** The
  effective per-room/per-turn catalogue supplies argument schemas (not only
  tools declared in the channel constructor), post-hook rewrites are validated
  in classic, streaming and realtime-voice paths, and a provider cannot invoke
  an undeclared name once a catalogue exists. Provider-executed tools carrying
  `_result` no longer fire a misleading retroactive `BEFORE_TOOL_USE`; they are
  observed through `ON_TOOL_CALL`, because their side effect has already
  happened.

- **Realtime provider teardown no longer strands live resources.** Clean peer
  closes and fatal errors retire OpenAI and Deepgram connections and notify the
  channel; recoverable Deepgram warnings stay active. OpenAI audio truncation is
  reserved atomically under concurrent barge-ins, and failed `session.update`
  sends roll back their connection state. ElevenLabs rejects duplicate in-flight
  tool ids and exposes its fixed 16 kHz input/output contract instead of silently
  accepting a mismatched channel clock.

- **Local audio and AEC state are bounded and paired.** The realtime speaker
  buffer is capped, playback start/stop cannot race a new enqueue, and every AEC
  activation is matched by deactivation on interrupt, disconnect, reset and
  pipeline teardown. A provider failure rolls activity bookkeeping back so a
  later frame can retry rather than leaving cancellation permanently bypassed.

- **The release script verifies the exact dirty file it permits.** A path merely
  ending in `src/roomkit/_version.py` no longer bypasses the clean-tree gate, and
  the real version file must contain exactly one valid assignment so unrelated
  code cannot be swept into the version-only release commit.

- **An ElevenLabs agent can finally call a tool.** `ElevenLabsRealtimeProvider`
  logged that `tools=` was ignored and answered `submit_tool_result` with a
  debug line, so a `tool_handler` on a `RealtimeVoiceChannel` was never
  invoked — while the shipped example and the guide both described tool
  calling as working. Two documents promising a behaviour no code performed.

  The ElevenLabs SDK does not forward JSON schemas the way OpenAI Realtime
  does: it runs **client tools** through a `ClientTools` registry keyed by
  name. The provider now registers a handler per declared tool on that
  registry and parks it on a future, so a call travels the normal RoomKit
  path — `on_tool_call`, the channel's gates and hooks, `submit_tool_result` —
  and the value the handler returns is what the agent receives. The names must
  match the client tools declared on the agent, which is the one half of this
  that lives on ElevenLabs' side rather than in RoomKit. A call nobody answers
  within `tool_timeout_s` (new, default 30 s) is reported to the agent as an
  error rather than hanging its turn.

  The registry is per session and bound to the loop RoomKit runs on. Left to
  itself the SDK starts its own loop in a private thread and would ship the
  tool result from that thread onto a WebSocket owned by ours; and its
  `end_session` stops the registry for good, so a shared instance would break
  on the second connect.

- **ElevenLabs no longer accepts a mid-session reconfigure it cannot honour.**
  `supports_mid_session_reconfigure` was inherited as `True`, so activating a
  skill mid-turn reached the base `reconfigure`, which disconnects and
  reconnects. On ConvAI that ends the conversation server-side and opens a
  different one, dropping the transcript and every pending `tool_call_id`. It
  now reports `False`, which routes callers to the delivery modes meant for
  providers whose session is fixed at connect.

- **A turn on ElevenLabs now ends.** ConvAI sends the agent's text *before* its
  synthesis, and the provider took that text for the end of the response:
  `response_end` fired before the first audio chunk, the chunk then reopened
  the response, and nothing ever closed it — the speaking indicator stayed lit
  and the session never went idle. A response now opens on the first audio
  chunk and closes once the stream has been quiet for `response_idle_ms`
  (new, default 800 ms), with a tool call in flight holding the turn open and
  an interruption closing it.

- **A dead ElevenLabs session no longer looks healthy.** `start_session` only
  spawns the task that opens the WebSocket, so a rejected key, an unknown
  `agent_id` or a dropped connection surfaced nowhere: `connect` returned, the
  session sat in `ACTIVE`, and the failure died inside an unobserved task. The
  provider now supervises the session and reports `connection_failed` /
  `session_ended` through `on_error`, with the session marked `ENDED`.
  `connect` also waits until the SDK has installed its audio-input callback
  before marking the session active, so speech arriving immediately after a
  successful connection is not silently dropped.

### Security

- **A symlink in a skill can no longer read or run a file outside it.**
  `Skill.read_reference()` filtered the *name* — rejecting `..`, `/` and `\` —
  which is no defence against a symlink: `notes.md` contains none of those
  characters and still served `/etc/passwd` when a link of that name sat in
  `references/`. Both ends are now resolved and the result must still be inside
  the skill's directory, so the escape is caught by construction rather than by
  enumerating the spellings of an attack. Full-width look-alikes are normalised
  (NFKC) before inspection, and absolute paths are rejected on POSIX and Windows
  spellings alike.

  The same containment now guards execution. `run_skill_script` resolves the
  name before it calls the executor, so one that escapes the skill is refused
  and no integrator code ever sees it — a check every integrator is expected to
  rediscover is a check some integrator will omit. `Skill.resolve_script()`
  exposes it directly, and `ScriptExecutor` implementations should build their
  command from it rather than joining `scripts/` and a model-supplied name
  themselves. Execution policy — sandboxing, timeouts, allowed interpreters —
  remains entirely the integrator's call; *which file runs* does not.

  The containing directory is checked too: replacing all of `references/` or
  `scripts/` with a symlink is rejected before listing, reading or resolving a
  child. Checking only the child would otherwise make the resolved external
  directory look like the trusted containment root.

  `SkillPathError` subclasses `ValueError`, so integrators already catching the
  `ValueError` this used to raise keep working.

## [0.42.1] — 2026-08-06

### Fixed

- **A channel torn down after being replaced no longer strands the id its
  replacement serves.** A host that rebuilds the channel behind an agent —
  the same agent attached to a second room, an edited configuration — swaps
  the object under the same `channel_id` and closes the old one afterwards,
  once its in-flight turns are done. When both share one `HumanInputHandler`
  (the shape the handler documents as safe), that late close was read as *the
  id is finished*: `AskUserQuestion` requests already armed by the live
  channel were rejected, its `ON_USER_INPUT_REQUIRED` callback was dropped,
  and the next `register_channel` for that id raised
  `RuntimeError: Human input handler is closed for channel <id>` — a crash the
  host could only escape by restarting the process, since nothing ever
  re-opened a closed scope.

  A channel scope now belongs to the channel *object*, not to the id it uses.
  Registering re-opens the scope and returns a token naming that owner;
  `close(channel_id=..., registration=...)` is a no-op once a newer owner
  holds the id, so a departing predecessor closes nothing. `AIChannel` carries
  its token from `register_channel` to `close()` on its own. Handlers used
  directly, and `close()` called without a token, keep closing the scope
  unconditionally, which is what a lone owner wants; the handler-wide
  `close()` still refuses everything after it, because that lifecycle has no
  successor.

## [0.42.0] — 2026-08-06

### Added

- **An ACP host can contribute context to a turn's prompt.** `ACPChannel`
  composed the prompt itself — the room catch-up, then the request — and an
  integrator had no seam to add what only it holds: the member's saved notes, a
  document corpus, an organisation's rules. An `AIChannel` has the
  `MemoryProvider` for that, because it rebuilds its message list every turn;
  an ACP session keeps its history in the agent's own process and offered
  nothing equivalent. Working around it below the channel does not exist: at
  transport level the catch-up is already glued to the text, and the transport
  holds neither the `RoomContext` nor the host's identity.

  `ACPChannel(..., context_contributor=host.blocks_for_turn)` is awaited once
  per solicited turn with the room's context and the triggering event, and the
  blocks it returns open the prompt — ahead of the catch-up, because what the
  agent missed of the conversation belongs nearer the request than background
  does. A contributor that raises costs its blocks and not the turn, like the
  channel's other host-supplied callbacks.

  Three things the contract says out loud. It is turn-scoped: the session
  keeps what it was already told, so a block that never changes is paid for
  again every turn, and what is stable belongs to the agent's own
  configuration — ACP has no instruction channel, and this is not one. Nothing
  is bounded: RoomKit knows neither the blocks' unit nor the agent's tokenizer,
  the model can change mid-session, and a cut would corrupt the host's meaning,
  so both the token budget and the deadline are the host's — with the caveat
  that `on_event` runs inside the broadcast pipeline, where a slow contributor
  delays delivery for the whole room. And RoomKit cannot filter what the blocks
  carry: the catch-up is filtered per reader because it is made of room events
  (RFC §7.5 rule 8), and these are not.

  A channel constructed without one prompts exactly as before.

- **A conference mint may carry attributes, so the identity stops being the
  only field that travels.** `mint_access()` put an identity and a display
  name in the credential and nothing else, while the reception side of the
  same boundary surfaced a provider's whole attribute map and said what it
  vouched for — a channel RoomKit read finely and could not write to. The cost
  landed on `participant_id`: an integrator whose own clients must be told
  *who* is behind a channel identity had only the identity to say it with, so
  the identity became a small format to parse, with as many parsers as
  readers.

  `ConferenceChannel.mint_access(..., attributes={"app.user": "user-42"})`
  now carries string pairs into the credential — `ConferenceBackend` grew the
  optional argument, LiveKit puts it in the token's `attributes` claim, and
  the SFU reports it back on every participant. Three rules keep it honest
  (RFC §12.10.3). It is opt-in per mint: the channel knows each participant's
  `identity_id` and adds none of its own, because minting it unasked would
  publish everyone's platform identity to every peer of a conference that may
  be pseudonymous. What comes back is *unasserted* — it rode a token, which is
  not a thing an SFU established, so it can be read and rendered and cannot
  found an identity. And what may be minted is bounded by what the room would
  persist when the SFU reports it back, with `mint_access()` raising rather
  than truncating: at emission the caller is the integrator, the one party in
  the exchange that can be told.

  Backends written before the argument existed are never handed it, so they
  go on serving every mint that does not ask for attributes.

- **`TelegramBotAPI` — the Bot API as methods, not just the sends a room
  produces.** The provider could send a message and nothing else, so every
  application around it grew a second Telegram client: a second base URL
  holding the same token, a second retry, a second spelling of "Telegram said
  no". What they all reimplemented is the same eleven calls — `get_me`,
  `get_updates`, `set_webhook`, `delete_webhook`, `leave_chat`, `send_message`,
  `send_force_reply`, `send_chat_action`, `answer_callback_query`,
  `edit_message_text`, `edit_message_reply_markup` — alongside `get_file` and
  `download_file`, which already lived here.

  They now live on `TelegramBotAPI`, which `TelegramBotProvider` extends;
  `bot.py` keeps only the translation of a `RoomEvent` into a Telegram message.
  Every call answers with a `ProviderResult`, so success and failure read the
  same way whichever call produced them: `telegram_<code>` when Telegram
  refused, `http_<status>` when the refusal carried no Bot API body, `timeout`
  when nothing came back, and Telegram's own words under
  `metadata["description"]` — the only text precise enough to tell a caller
  that a webhook URL must be HTTPS rather than merely that it was rejected.
  The two reads, `get_me` and `get_updates`, also carry Telegram's `result`
  under `metadata["result"]`.

- **`mentions_bot()` and `entity_text()` — being addressed, and the UTF-16
  arithmetic underneath it.** Telegram marks a message up out of band, and the
  offsets in that markup count UTF-16 code units. Python indexes strings by
  code point, so `text[offset:offset + length]` is right until someone puts an
  emoji in front of the mention, and quietly wrong after. `entity_text` slices
  on the basis Telegram measured in.

  `mentions_bot(msg, bot_username=..., bot_id=...)` answers the question every
  bot in a group has: was this meant for me? True on a reply to the bot, a
  `bot_command` without a target (or qualified with this bot's exact username),
  a `mention` entity, a `text_mention` naming its id, or the handle posted as
  boundary-delimited plain text with no entity at all. A command qualified for
  another bot remains false even when Telegram delivers it here. The helper
  reports the fact and decides no policy — whether a given group answers only
  when addressed is the application's rule, not the kit's.

- **`parse_telegram_update()` — which of its forms an Update took.** One entry
  point for the three a webhook receives: a `message`, an `edited_message`
  (same shape, flagged), and a `callback_query`, whose parts come back as
  `TelegramCallback` — the query id to answer, the `callback_data`, the
  sender, and the message the button hangs off, so an outcome can be appended
  to what was already said. Nothing about who was allowed to press it is
  decided here: `callback_data` is posted by whoever pressed the button, and
  is a claim to check rather than a fact.

- **`TelegramMessageParts` now carries `entities`, `reply_to_message_id` and
  `media_group_id`.** The protocol facts a consumer of the lower layer needed
  and had to go back to the raw update for: a caption's markup (carried under
  its own `caption_entities` key), the message a reply answers — what ties an
  answer to the `force_reply` prompt that asked for it — and the id shared by
  the several messages Telegram splits one album into. They stay off the
  `InboundMessage`: `parse_telegram_webhook`'s metadata is unchanged.

- **`parse_telegram_message()` — reading a Telegram message without deciding
  who sent it.** `parse_telegram_webhook` builds an `InboundMessage`, and in
  doing so settles the sender as `message.from.id`. That rule is not universal:
  under a one-bot-per-user model a direct message belongs to the bot's *owner*,
  not to the account that typed it, and the same process can apply the opposite
  rule in a group. A consumer holding such a model could not reach the new media
  metadata at all, because the only door to it was already attributed.

  The parsing is now two layers. `parse_telegram_message(msg)` returns
  `TelegramMessageParts` — content, metadata, `message_id`, `sender_id` — and
  imposes nothing; `sender_id` is offered, not applied.
  `parse_telegram_webhook` is that function plus the ordinary attribution, and
  its behaviour is unchanged for text, photo and location updates.

- **`TelegramBotProvider.get_file()` and `download_file()`** — the two calls
  that turn an inbound `file_id` into bytes. `get_file(file_id)` resolves the id
  to a Bot API path (valid at least an hour), `download_file(path)` fetches the
  content. They belong here rather than in each application because the bot
  token does: the provider was purely outbound, so a `file_id` reaching a
  consumer had nowhere to go.

  Both return `None` on failure and log a warning that never carries the URL —
  every Bot API URL embeds the token, and httpx names the failing URL in its
  error string. Telegram caps Bot API downloads at 20 MB and refuses larger
  files at the `getFile` step, which surfaces as `None`; an update's
  `metadata["file_size"]` tells a caller before spending the call.

  RoomKit stops at the bytes. Which ASR engine transcribes a voice note is the
  application's decision, not the kit's.

- **`make check-models` — a release can no longer ship a stale catalog
  quietly.** The offline catalogs are the one part of the library a test suite
  structurally cannot validate: a catalog a lineup behind is still internally
  consistent, so everything passes while RoomKit hands out ids the vendor
  retired. `scripts/check_models.py` compares all of them against OpenRouter's
  public `/api/v1/models` — keyless, and it republishes ids, context windows
  and modalities for every major vendor in one request — and reports three
  things: a context window that disagrees, an upstream model newer than
  anything the catalog knows in a family it already tracks, and an id upstream
  no longer lists. `make release` runs it before touching anything and stops on
  a finding; an unreachable mirror only warns, because blocking a release on
  someone else's outage trades one problem for a worse one. It reads a mirror,
  not the vendor, so a finding means "go read the vendor's docs" — and a
  divergence that turns out to be deliberate is recorded in the script with its
  reason, next to the four already there. `SKIP_MODEL_CHECK=1` for a release
  that must go out first.

- **`AIProvider.catalog_entry()`** returns the offline `ModelInfo` for the
  active model, or `None` for an id the catalog does not carry. It is the one
  place a provider should read its own model's metadata from; `context_window`
  and both fixed `supports_vision` implementations now go through it.

- **The model catalog carries the vendor's price, so a new model cannot bill
  zero.** RoomKit declared the models and every consumer kept its own rate
  sheet, which is a list that drifts by construction: `gemini-3.6-flash` shipped
  in the catalog while a downstream sheet stopped at 3.5, and a whole
  conversation was recorded at `cost=N/A` — tokens counted, nothing billed, no
  warning anywhere. `ModelInfo.pricing` is now a `ModelPricing` sitting beside
  `context_window`, filled from the same vendor page on the same date as the
  rest of the entry.

  Four rates, not two: `input_per_million`, `output_per_million`,
  `cache_read_per_million`, `cache_write_per_million` — one per counter RoomKit
  reports in `usage`. A cached prefix costs a tenth of fresh input at
  Anthropic's rates, so a two-rate sheet is not a rounding error on a long
  conversation. An unset rate is a claim, not a gap: OpenAI bills nothing to
  write a cache, Google bills cache storage by the hour, and neither is a
  per-token write. `ModelPricing.cost_for(usage)` prices a response in one
  call. A cache counter whose rate is unset is omitted because the catalog does
  not represent a separate per-token charge for it. `currency` and `verified`
  travel with the rates. Rates must be finite and non-negative and multipliers
  finite and positive; `cost_for()` rejects non-integer or negative token
  counters rather than emitting a negative bill. `ModelPricing` is exported
  from both `roomkit.providers.ai` and the package root. A price changes without
  the model changing, and Claude Sonnet 5's introductory
  $2/$10 expiring on 2026-08-31 is exactly why a consumer needs the date.

  Priced: Anthropic, OpenAI, Gemini, Mistral, xAI, and OpenRouter (its own rate
  card — it is the seller there, and it resells `gpt-5.6-terra` at half OpenAI's
  own price). Unpriced by nature: Ollama's locally pulled weights, PolarGrid's
  private edges, and Azure/vLLM, which have no offline catalog at all. Two
  guards: a test asserts every model in a priced catalog has a rate — it fails
  on the commit that adds one without, which is when this bug was introduced —
  and `make check-models` now reports a rate that disagrees with the upstream
  mirror as `PRICE`, alongside the existing context-window drift.

### Changed

- **The offline catalogs say what they are for, which is not discovery.** Each
  `providers/*/models.py` presented itself as "the catalog of what this
  provider offers" — which is what `list_models()` does, against the provider,
  and precisely the framing that guarantees the file rots. The list cannot go
  away: `context_window` is a sync property, so history trimming needs a number
  before any request exists and cannot await one. So the contract narrowed
  instead — offline metadata for the models RoomKit can describe without a key.
  Nothing changes at runtime, but a model missing from a catalog is now
  documented as an ordinary outcome (`context_window is None`, degrade) rather
  than a gap to be raced against every vendor announcement.

- **BREAKING — `model` is now required on `AnthropicConfig` and
  `OpenAIConfig`.** Their former defaults (`claude-sonnet-4-20250514` and
  `gpt-4o`) could age into a 404, while silently advancing either one would
  change a caller's cost, latency and behavior on a library upgrade. RoomKit
  therefore makes neither product choice: callers select a model explicitly.
  Migration: pass the id at construction —
  `AnthropicConfig(api_key=..., model="claude-opus-5")`. A caller that already
  passes `model=` is unaffected; every example now shows the choice it makes.

### Fixed

- **GPT-5.6 function tools no longer inherit an incompatible reasoning
  default.** The OpenAI provider uses Chat Completions, where GPT-5.6 function
  tools require effective reasoning `none`; omitting the parameter selected the
  family's `medium` default and made a configured GPT-5.6 model fail as soon as
  an `AIChannel` exposed a tool. Official GPT-5.6 tool turns now send
  `reasoning_effort="none"` explicitly. Custom OpenAI-compatible endpoints are
  not profiled. The same vendor check corrected Luna's offline metadata from
  the Sol/Terra 1.05M window to 400k and removed the Sol/Terra-only long-context
  price multipliers.

- **A release interrupted between its version commit and tag can resume.** The
  release script recognizes a clean, version-only HEAD without its tag, checks
  CI on the code-bearing parent, and finishes the tag and publication. It also
  accepts only the PEP 440 final/prerelease spellings its artifact names can
  represent, rejecting a hyphenated SemVer prerelease before mutation.

- **AI tool execution and planning are isolated per room.** An `AIChannel` is
  shared by every room that attaches it, but tool hooks, human-input context,
  usage memory and plan events read a mutable channel-wide room id. Two
  interleaved streams could therefore authorize or record room A's tool under
  room B, while the planner injected its last plan into every room even without
  concurrency. All of these paths now resolve the invocation-scoped room from
  the tool-loop context, and plans are stored and rendered by room id in the
  same bounded 100-room working set as the other channel memories. Model-authored
  plans are validated and copied before that state changes, limited to 100 tasks
  with 500-character titles, and stripped to their declared fields, so malformed
  or oversized nested arguments cannot poison the room's later context.

- **An empty per-turn toolset now fails closed.** `current_tool_allowed_names()`
  returned `None` both before context construction and after resolving a room to
  zero tools. A host following its fallback contract could therefore replace a
  deny-all decision with a broader static allowlist. A resolved empty toolset now
  returns `set()`, and `AIChannel` rejects a provider-authored tool name outside
  the resolved current-turn toolset before it reaches the shared user handler.

- **Human-input notifications stop with their owning channel.** The tracked
  callback tasks introduced in this release had no shutdown path, so a slow or
  hung hook could outlive `AIChannel.close()`. Closing now rejects and retires
  that channel's pending requests, cancels and awaits its notification tasks,
  and leaves work belonging to another channel intact when a handler is shared.
  It marks the scope closed before taking that snapshot, so a concurrent create
  cannot slip past and wait until timeout. Framework callbacks are routed per
  channel rather than the last registered owner replacing every earlier one.

- **Telegram treats malformed success responses as provider failures.** A 2xx
  response containing valid JSON of the wrong shape escaped as
  `AttributeError`, despite the API's `ProviderResult` contract. Bot API
  envelopes, `ok`, `result` and `getFile` payloads are now shape-checked against
  the operation that produced them — bot, update list, literal `true`, or
  Message id. A 200 refusal remains a Telegram error and an incomplete or
  mismatched response becomes `invalid_response` (or `None` for `get_file`).

- **Release compatibility and security review.** Official OpenAI and Anthropic
  configurations now profile the explicitly selected model with request
  parameters it accepts, while explicit flags, older model ids and compatible
  custom endpoints retain their prior behavior. The release script pushes only
  its own tag instead of every local tag. Token pricing now represents
  long-context tiers and explicit GPT-5.6 cache writes; usage normalization
  separates those writes from ordinary input; and the live catalog check flags
  a positive upstream rate that RoomKit accidentally leaves unset.

- **ACP reconnects no longer lose room history.** A dead transport now clears
  the catch-up marks together with the dead sessions, and the lazy prompt
  computes its catch-up only after recovery has completed. The replacement
  agent therefore receives all visible history its empty session missed.

- **Telegram input boundaries are hardened.** Network errors no longer return
  an exception string containing the token-bearing Bot API URL, malformed
  nested webhook objects are rejected without escaping parser exceptions, and
  inbound idempotency keys are scoped by both chat and message id.

- **The release path is reproducible again.** Integration CI installs the
  `httpx2` peer imported by the FastRTC/Starlette stack, `make docs` targets the
  sibling documentation repository explicitly, and `.pypirc` credentials are
  passed to `uv publish` outside the process command line. A resumed release
  additionally validates that its tag contains the requested version, points
  to a version-only commit that is HEAD or the direct parent of the next-dev
  HEAD, and derives the CI commit from that validated topology; a stale local
  tag can no longer combine one GitHub source tree with different PyPI artifacts.

- **An activated skill was re-loaded, whole, on every turn.** `activate_skill`
  returned the skill's full body every time it was called, and nothing in the
  text channel remembered an activation past the turn that made it. Nothing
  could: the rebuilt context carries message events, not tool calls, so the
  body genuinely vanished between turns and a model still working on the task
  obeyed the preamble and fetched it again. In one production onboarding room a
  9 KB skill cost 28 KB of reloading over three exchanges — more than the skill
  itself, and enough to undo the trimming that had just been done to it. The
  channel now records which skills a room activated and renders their bodies
  into each turn's system prompt, so `activate_skill` answers later calls with
  a short ack instead of the body, and the tools a skill gates stay revealed
  across turns instead of re-hiding. This is the lifecycle the realtime channel
  already ran per session, now specified in RFC §24.4. Losing the record
  (restart, a channel object replaced) loses the prompt block with it, so the
  next activation delivers the body again — the mechanism degrades to
  reloading, it never leaves the model holding an ack with no rules. The record
  is hydrated from the room's persisted tool-call history, keeps four skills
  per room by recency, and is injected regardless of `skills_in_prompt`: that
  flag governs the static catalogue, while active bodies are runtime state a
  host cannot know.

- **A large skill body could reach the model as a preview and a pointer.** Tool
  results over `evict_threshold_tokens` are stored aside and replaced with a
  head/tail preview plus a `read_stored_result` id — sound for data, wrong for
  instructions: a skill past the threshold (~20 KB by default) had its binding
  rules truncated into a summary the model was left to act on. `activate_skill`
  results are now exempt. `read_skill_reference` still evicts, since a
  reference is data and paginating data is what eviction is for.

- **An OpenAI-compatible provider counted its cached prefix twice.**
  `prompt_tokens` includes the tokens read from cache, and RoomKit reported it
  as `input_tokens` while also reporting `cache_read_input_tokens` beside it —
  so anything pricing a response charged the cached prefix at the full input
  rate *and* again at the cached one. `input_tokens` now counts what was billed
  at the input rate, matching how Anthropic reports natively and how the Gemini
  provider already normalized. Affects `openai`, `openrouter`, `azure` and
  `vllm`, which share the mapping; a caller summing the two keys to recover the
  provider's raw `prompt_tokens` still gets it.

- **A streamed turn dropped its cache counters before the response hook.** The
  streaming loop summed `input_tokens` and `output_tokens` across rounds and
  forwarded only those, so `AIResponseEvent.usage` — the payload a consumer
  prices a turn from — could not tell a cached prefix from fresh input. Every
  integer counter a round reports is now summed and forwarded; the two canonical
  keys are still always present.

- **A parallel Gemini tool call replayed unsigned when the signature landed on
  a later call.** Gemini puts a `thought_signature` on one function call of a
  parallel round, and its validator then demands one on *every* function call in
  the history — so the provider lends the round's signature to the calls that
  came back bare. It resolved that signature by carrying it forward as it walked
  the message, which works only while the signed call comes first: a call
  *preceding* it replayed with nothing, and Gemini 3 rejected the whole next turn
  with `400 "Function call is missing a thought_signature"`. The round is now
  scanned up front, so the borrowed signature reaches every call whatever the
  order, and a call carrying its own still replays with that one. The warning
  that went with this moves to where the fact is known: it fired per unsigned
  call during streaming, announcing a rejection that the replay then prevented
  (13 times in one observed conversation, none of them rejected), and now fires
  once per round, only when *no* call in it carries a signature — the case with
  nothing to lend. The two `INFO` diagnostics that instrumented this are `DEBUG`.

- **A history ending on a model turn reached the conversation as an opaque
  provider rejection.** Gemini answers a user turn; handed a history whose last
  content is a model one it replies `400 "Requests ending with a model turn are
  not supported."`, and the raw status is what the room read. The provider now
  refuses the request before it leaves, with a `ProviderError` naming the
  condition and how many trailing model contents it found. It does not append a
  continuation turn: the cause is upstream — a turn generated with no new input
  to answer, most often concurrent turns on one room each rebuilding a history
  that ends on another's reply — and answering a prompt the application never
  wrote would hide the race that produced it. Tool results are unaffected; they
  are sent as user contents.

- **A human-input request only started listening once the notification came
  back.** `HumanInputHandler.create()` awaited the `ON_USER_INPUT_REQUIRED`
  callback before returning, and the tool reached `wait()` only after that — so
  a slow broadcast, or a hook burning its full 30-second budget, left the
  request armed but unattended for the whole window. Answers arriving in it were
  still recorded, but a caller that kept its own bookkeeping and dropped the
  entry on `resolve()` turned an answer that had arrived into
  `ValueError: No pending request`, handed to the model as a tool failure; one
  session re-asked the same question six times. `create()` now returns as soon
  as the request is answerable and runs the callback in a tracked background
  task. The hooks keep their sync semantics — priority order, and a
  `HookResult.block()` still rejects the request, reported by `wait()` wherever
  it has got to.

- **A recorded outcome was indistinguishable from an id that never existed.**
  `wait()` raised `ValueError` for both, so waiting twice — or waiting after a
  host that keeps its own bookkeeping dropped the request on `resolve()` —
  reported a failure instead of what happened. An outcome now goes into a
  bounded retention the moment it settles (the last 128,
  `HumanInputHandler(retention=…)`) and `wait()` replays it: the same answer,
  the same rejection, the same timeout. Recorded at settle time and not at read
  time, because the host that drops the request is the one that recorded the
  answer a moment earlier. `ValueError` is left to mean what it says.
  `create_detached()` and `release()` name the other half of the problem: the
  external-runtime path calls `create()` and never `wait()`, which left the
  cleanup ownership for each caller to guess at.

- **A vLLM server described itself with OpenAI's model list.**
  `create_vllm_provider()` returned a plain `OpenAIAIProvider`, so
  `available_models()` answered with OpenAI's hosted catalog — someone else's
  models, for a server running whatever weights you loaded onto it. Worse, an
  id that happened to collide (a proxy named `gpt-4o`, a fine-tune) borrowed
  that model's context window and trimmed history against it. It also inherited
  OpenAI's vision prefixes, which no local model id matches, so every vLLM
  deployment reported text-only and dropped images before the wire. The factory
  now returns an `OpenAIAIProvider` subclass — identical on the wire — whose
  catalog is empty (`context_window is None`, the honest answer for a model
  RoomKit has never heard of) and which passes images through, letting a
  multimodal server work and a text-only one answer with an error. Same bargain
  Ollama and Mistral already make. `list_models()` is unchanged and still
  queries the server's `/v1/models`.

- **RoomKit refused seven PolarGrid edges the SDK can route.** The offline
  region list carried nine, so `lax-01`, `sea-01`, `chi-01`, `phx-01`,
  `was-01`, `mia-01` and `sfo-03` were rejected by `resolve_region_id()` as if
  they were typos — including two US East edges, which matters where the
  Canada/US split is the data-residency signal. The list now mirrors the SDK's
  own `POLARGRID_REGIONS` (all 16), and cites that shipped table rather than a
  doc page, because the table is what actually routes. The `polargrid` extra's
  floor moves to `polargrid-sdk>=0.9.2` accordingly: `chi`/`lax`/`phx`/`sea`/
  `was` landed in 0.9.0, `mia-01` in 0.9.1, `sfo-03` in 0.9.2. A test asserts
  the mirror equals the SDK's table whenever the optional SDK is installed, so
  the two cannot drift again silently. The alias table is deliberately *not*
  extended — the new edges have no alias upstream, and inventing one would make
  it a table RoomKit maintains rather than a mirror it tracks.

- **Images stopped reaching the wire on every current OpenAI and Anthropic
  model.** Both providers answered `supports_vision` from a hardcoded tuple of
  model-name prefixes kept alongside the catalog that already states the same
  fact per model. Nothing updated the tuples with the lineup, and the two
  drifted exactly as a duplicated fact does: Anthropic's stopped at
  `claude-opus-4`, so Opus 5, Sonnet 5, Haiku 4.5, Fable 5 and Mythos 5 all
  reported text-only, and OpenAI's predated GPT-5 entirely, so every GPT-5.x
  model did too. A room routing an image to one of them dropped it silently —
  no error, just a reply that never mentions the picture, on models that can
  all see. Both now read `ModelInfo.supports_vision` from the catalog via the
  new `AIProvider.catalog_entry()`, with a family prefix as fallback for ids
  the catalog does not carry (a snapshot newer than the release, an
  OpenAI-compatible server naming its own model). No test caught this, because
  a catalog and a prefix table that disagree are each internally consistent;
  the regression guard now walks every vision model in both catalogs.

- **The model catalogs had fallen a lineup behind, in the places it costs
  most.** OpenAI's whole current frontier (`gpt-5.6-sol`, `-terra`, `-luna`)
  was absent, `gpt-5-codex` was listed as merely deprecated three weeks after
  it was shut down, and six more models with announced shutdown dates were
  unflagged; Anthropic was missing `claude-opus-5`; Gemini was missing
  `gemini-3.6-flash` and `gemini-3.5-flash-lite`; OpenRouter carried
  `mistralai/mistral-medium-3.5`, a slug that has never existed (the real one
  spells the version with a hyphen); and Mistral gave Ministral 3 3B a 256k
  window when it has 128k. All eight catalogs re-verified against the
  vendors' own docs on 2026-08-05.

- **An agent that owns its own turn is no longer invisible when it answers.**
  `ON_AI_RESPONSE` is the one signal RoomKit gives a host that a turn of
  intelligence just finished, and it was wired on `isinstance(channel,
  AIChannel)`. An ACP coding agent is not an `AIChannel` — it is a channel of
  category `INTELLIGENCE` that runs its tool loop in another process — so a
  conversation it held produced events, persisted them, broadcast them, and
  fired nothing afterwards. Every post-processing task an integrator hangs off
  that trigger (memory extraction, conversation summary, chat title, metrics)
  simply did not run for those rooms, and would not have run for any future
  agent channel either.

  The wiring now follows the **category**, which is the capability the trigger
  is about: any channel declaring `ChannelCategory.INTELLIGENCE` gets the
  report, whatever class it descends from. The four other callbacks in that
  block stay on `AIChannel` — they presume a tool loop running in this
  process, which an ACP agent does not have.

  `ACPChannel` now fills the report at the end of a turn: the text it produced,
  how many tools it called, and how long it took. A turn only reports when it
  reaches its terminal item without an error — a stream closed from the
  outside cancels the agent and delivers nothing, so it is not a response.

  `usage` carries the token counters off the `PromptResponse` the agent
  returns, relayed unaltered, beside the context occupancy and running cost
  its usage notifications announce. The ACP schema annotates those counters as
  running session figures while the reference agent fills them per prompt —
  measured against it, `cached_read_tokens` is the whole prefix re-read on that
  turn, not a sum over turns. One reading cannot tell the two apart, and
  reinterpreting either way corrupts the number where nothing downstream can
  notice, so RoomKit does no arithmetic on them. Read `total_tokens`: a coding
  agent's context arrives almost entirely as cache reads, which makes
  `input_tokens` alone a large understatement.

- **A participant reached on a second channel is no longer adopted in
  silence.** `ensure_participant(room, channel, participant_id)` looks a
  participant up by `(room, id)` and returns whatever it finds. When that
  record belonged to another channel — a conference asking for a participant
  under the id a WebSocket channel had already used — the caller got it with no
  error and no log, believing it held a participant on the channel it had
  named. Downstream, a conference lifecycle then drove a team-channel
  membership: leaving the call wrote `LEFT` over it, and joining one revived a
  membership deliberately dropped. It took weeks to find because nothing said a
  word.

  The lookup is unchanged, and deliberately so: a participant is **one record
  per (room, id)**, and the same person reached by SMS and then by email being
  one participant is the point of a cross-channel identity, not a defect. What
  changes is that the reuse is now said and kept. `connected_via` — declared in
  the model and in RFC §5.5 since the beginning, written by nothing until now —
  carries every channel the room has reached a participant through, primary
  first, and is persisted. A lookup naming a channel that is not the record's
  primary one logs a warning naming both. `add_member` still moves the primary
  channel on a deliberate join through another channel, but keeps the one it
  replaced on the list and logs the move.

  `ensure_participant` now runs under the room lock, like `add_member` and
  `remove_member`, since recording a channel is a read-modify-write on the same
  record. Recording a channel is bookkeeping, not presentation: no
  `PARTICIPANT_UPDATED` event, no `ON_PARTICIPANT_UPDATED` hook.

  Postgres gains `participants.connected_via TEXT[]` through the additive,
  idempotent DDL `init()` already runs — no migration step, and an existing row
  reads as an empty list, which is what it was stored under. RFC §5.5 states the
  rule ("One record, several channels"); the library follows it. (RMK-108)

- **A voice note sent to a Telegram bot no longer vanishes.**
  `parse_telegram_webhook` knew `text`, `photo` and `location`; every `voice`,
  `audio`, `video_note`, `video` and `document` update fell through to
  `content is None` and returned an empty list. No log, no error — the message
  simply disappeared.

  All six media kinds now parse the way `photo` already did: the caption
  becomes the body and the file reference goes to metadata as `file_id` and
  `media_type`, along with whichever of `duration`, `mime_type`, `file_name`
  and `file_size` Telegram supplied. A voice note has no caption, so its body
  is empty — that is the right answer, the `file_id` is what carries the
  message.

  Compatibility: a media update used to yield nothing and now yields one
  `InboundMessage` whose body may be empty. That is not a new shape — a photo
  without a caption already produced exactly it — but a consumer that filters
  on a non-empty body will see one more message.

### Security

- **The `smart-turn` extra no longer resolves a `transformers` with a known
  RCE.** Its floor was `transformers>=4.57`, and every version below 5.5.0
  carries GHSA-fgcw-684q-jj6r — arbitrary code execution during model
  initialization. RoomKit's own use is narrow: `SmartTurnDetector` constructs
  `WhisperFeatureExtractor(chunk_length=...)` directly and never calls
  `from_pretrained`, so the library does not walk the advisory's path itself.
  Shipping an extra that resolves to a vulnerable version is still a fact an
  auditing consumer has to answer for, and an application sharing the
  environment may well load a model. The floor is now `transformers>=5.5`.

  That has a packaging consequence worth stating plainly. `qwen-tts`,
  `qwen-asr` and `neutts` pin `transformers` exactly, at 4.57.3, 4.57.6 and
  5.1.0 — all below the new floor — and the `all` extra includes `smart-turn`.
  So `roomkit[all,qwen-tts]` and its siblings can no longer resolve, and
  `pyproject.toml` now declares those six conflicts explicitly: a resolver
  error naming the two extras, instead of a silent backtrack onto a vulnerable
  `transformers`. Install those three engines in an environment without
  `smart-turn` or `all`.

## [0.41.4] — 2026-08-05

### Fixed

- **A Gemini Live session no longer dies on a tool whose parameter is written
  `{"type": ["string", "null"]}`.** That is JSON Schema's own spelling of an
  optional string, and what a generator that is not Pydantic emits — a
  TypeScript MCP server going through `zod-to-json-schema`, for one.
  `clean_gemini_schema` already folded the Pydantic shape
  (`anyOf: [{type: X}, {type: null}]`) down to `{type: X, nullable: true}`, but
  `type` is a key Gemini accepts, so a type *list* travelled through the
  cleaning untouched and failed inside `FunctionDeclaration`, whose `type` is a
  single-valued enum. The failure is not a degraded tool: `_build_config` raises
  while assembling the declarations, so `provider.connect` dies and the whole
  voice session never opens — one such tool anywhere in the tenant's set takes
  realtime voice down for that room. Both spellings now collapse the same way, a
  wider list keeping its first non-null member exactly as a wider `anyOf` does.
  Reported from a production deployment.

## [0.41.3] — 2026-08-04

### Added

- **`kit.send_event(addressed_to=[...])` — direct injection has a sender too.**
  An address is set on an event *by its sender* (RFC §19.3), and every entry
  point has one: `InboundMessage` could name who it asked since 0.40, direct
  injection could not, though it traverses the same pipeline (RFC §10.5) and
  lands the same event in the same room. The gap showed as soon as an
  application stored a message and triggered the answer itself — the shape
  behind every "the AI replies when it is mentioned" surface. With no way to
  say *this stored message asks nobody*, the only lever left was muting the
  intelligence channel around the write and unmuting it for the turn: a state
  mutation standing in for a per-event decision, which races the write it
  guards, has to be undone under the same lock, and leaves the room's agent
  silenced or shouting if anything in between fails. `addressed_to=[]` says it
  on the event, where there is nothing to race.

## [0.41.2] — 2026-08-04

### Added

- **`kit.set_agent_response_policy(room_id, policy)` — the room that turned
  multi-agent can say so.** `AgentResponsePolicy` was selectable at creation
  only, and a room rarely knows then how many agents it will end up holding: a
  chat that opens with one assistant and gains a second when the human asks for
  it becomes a room of independent agents at that moment, and under
  `AGENT_CHAIN` the first answer solicits the newcomer, whose answer comes back,
  down to `max_chain_depth`. The only way to switch a live room was to mutate
  the `Room` and call `store.update_room()`, which is the framework's own state
  written from outside the framework. Setting the policy a room already holds is
  a no-op, so the call sits safely on an attach path; the change applies to
  events processed after it, never retroactively to a turn already routed
  (RFC §19.3.1).

### Changed

- **An intelligence channel that is not solicited now costs nothing.** The
  registry lookup, the transcode and the `max_length` pass all ran before
  `_solicits` was consulted, so a binding the event never addressed still paid
  for them — and an unregistered one logged `Channel … not found in registry` on
  every single broadcast. That is the normal state of a room whose roster is
  rehydrated lazily: after a restart the bindings are all in the store and none
  of the channels are live, and only the agent actually being talked to has to
  be rebuilt. Solicitation is now decided first, from the untranscoded event
  (transcoding rewrites content, never the address). The warning still fires for
  a channel that *was* asked to act, which is the case worth knowing about.

- `examples/acp_multi_agent.py` sets `ADDRESSED_ONLY` on its room rather than on
  the kit. A kit-wide default would also silence the single-agent rooms a server
  hosts alongside it, where chaining is exactly what they want.

## [0.41.1] — 2026-08-04

### Fixed

- **An ACP turn that dies no longer leaves its tool calls spinning forever.**
  A turn interrupted mid-tool — the agent's process restarted, its host gone —
  emitted a `TOOL_CALL_START` that nothing ever closed. That row is persisted,
  so the tool card read as *running* on every reload of the conversation,
  indefinitely.

  `ACPChannel` now closes what a turn leaves open, whichever way it ends:
  every tool started without a terminal `tool_call_update` gets a
  `ToolCallEndMarker` with `status="failed"` and an error saying the turn ended
  before the tool reported a result. The distinction is for whoever reads the
  thread — a tool that never returned because the turn died is not a tool that
  failed on its own. Cancellation takes the same path: a stop the user asks for
  returns through the ordinary end of a prompt, and takes the open tool with it.
  A turn whose tools all reported emits nothing extra.

  The closing markers are emitted *into the stream*, before the turn's terminal
  return or error, because the stored `TOOL_CALL_END` is persisted from the
  marker — the channel's `finally` runs when nothing can be yielded any more.
  A stream closed from the outside (its consumer cancelled, a muted binding) is
  past that point: it still publishes the ephemeral `TOOL_CALL_END` for live
  surfaces, but its stored row stays pending.

## [0.41.0] — 2026-08-04

### Added

- **`ConferenceTranscription` says where the utterance sat in time.** A new
  `timing: UtteranceTiming` field carries `started_at` / `ended_at`, both read
  from `time.monotonic()` at the VAD's own speech boundaries — not when the
  transcription came back, which is a recogniser round trip later. A transcript
  writer could previously only stamp arrival, which drifts by the recogniser's
  latency and gives every utterance one instant for both its ends.

  One clock deliberately: the VAD also counts the audio it kept, in frame
  durations, and a start read from the wall clock against a duration counted in
  audio agree only for as long as frames arrive in real time. Every lane in a
  conference shares this timeline, which is what lets a transcript order two
  speakers against each other. `duration_ms` is derived from the two ends, and
  the span includes the trailing silence the VAD needs before it will call an
  utterance over.

  `UtteranceTiming` is exported from `roomkit` and
  `roomkit.channels.conference`.

### Changed

- **`UtteranceCallback` receives the utterance's timing.** Lane callbacks are
  internal to `ConferenceChannel`; integrators reach this through
  `ON_TRANSCRIPTION`, where the new field is additive.

## [0.40.0] — 2026-08-04

### Changed

- **The room lock ends at broadcast planning — external delivery moved to
  per-room delivery lanes** (RFC §10.1 steps 12-14, §10.2, §13.5; the
  roomkit-specs amendment `e0aabcc`). Measured on the scale bench: with
  `channel.on_event`/`deliver` (provider round trips, AI generation) inside
  the room's critical section, adding workers made rooms *slower* — 74
  backends parked on advisory locks while 3-4 worked. Now the lock covers
  the pre-commit gates, the atomic commit and the *planning* of the
  delivery set; execution runs off the lock, ordered per room by the
  `Room.delivered_index` cursor (strict CAS) under a delivery claim — a
  derived `__delivery__:{room_id}` key on the existing lock manager. One
  lane per room and per process executes only the plans its process
  enqueued; the shared cursor forces the global order. Observable
  semantics preserved: `process_inbound`/`send_event` still return after
  the full delivery cascade (trigger set + every reentry pass it spawned),
  so "the AI response is committed when the call returns" holds.
  Post-delivery failures — including the side-effect persistence write — are
  recorded on that cascade and reach the caller instead of being log-only;
  cancelling or closing a lane releases even the delivery already dequeued,
  so a waiter cannot be stranded mid-round. Response events re-enter as
  their own commit passes (fresh room lock) instead of being drained inside
  the trigger's lock tenure — a concurrent inbound
  MAY now commit between a trigger and its response (the RFC's explicit
  relaxation: index monotonicity and parent linkage, never adjacency).
  AFTER_BROADCAST keeps its contract (fires after the event's delivery
  set completes); its relative order across trigger/reentries follows
  execution order. A worker that commits and crashes before delivering
  leaves a cursor hole: the waiting lane skips it after
  `delivery_gap_timeout` with a `delivery_skipped` framework event —
  the same bounded loss as the previous crash window, now observable and
  without wedging the room. The Postgres store backfills
  `delivered_index = latest_index` once, when the column is first created
  (pre-lane deployments delivered under the lock, so everything stored is
  delivered). Known limit: the cursor advances when a delivery set executes,
  and the post-delivery pass that follows it (AFTER_BROADCAST hooks and the
  side-effect persistence write) runs after that advance. A crash in that
  window leaves the cursor past an event whose post-effects never ran, and
  unlike an undelivered event this leaves no hole to gap-skip, so it is not
  retried. The durable outbox that turns this into recovery is a separate
  step (RFC §13.6); until then a post-effect side effect is best-effort
  across a process crash.

  The lane is the room's *only* delivery primitive: every committed event
  with recipients goes through it, because a commit publishes its index on
  the cursor and that is exactly what releases the lane to execute the next
  one — broadcasting inline right after committing would let index N+1 be
  delivered before N. Two paths visibly change as a result. A greeting now
  fires its AFTER_BROADCAST hooks like any other delivered event. And a
  streamed answer's segments reach the non-streaming channels **as they are
  produced** rather than in one batch after the stream, each firing its own
  AFTER_BROADCAST once its delivery set completes (RFC §10.1 step 16) — so
  an SMS participant follows the answer at the pace the web one does. A
  stream that fails part-way now sends only the text produced *after* the
  failure to the streaming channel, instead of re-delivering every segment
  it had already rendered.

- **An inbound message deserialises the room's history once, not twice**
  (RMK-105). Every message built two complete `RoomContext`: one before the
  room lock, for the channel and the identity resolver (RFC §10.1 steps 3-5),
  and one under the lock whose first statement overwrote the parameter it had
  just been handed — the signature promised a reuse that never happened, across
  two call levels. A context deserialises up to 50 stored events into pydantic
  models; on the measured bench (1 worker, 1 room, sequential, 2000 messages)
  that history was 1574 µs of the 3852 µs of worker CPU a message costs, half
  of it for a copy nobody read. The locked pass still re-reads room, bindings
  and participants — the lock exists so the status gate and the delivery plan
  read fresh state (RFC §10.1 steps 6 and 12) — but it now carries the history
  it was handed, and only when the room's `event_count` proves nothing
  committed in between: the committed timeline is append-only (§8.1), so an
  unmoved counter means no event entered it. An EDIT or DELETE event is itself
  a commit, so it moves the counter and sends the pass back to a full read.
  Direct host calls to `update_event()` / hard `delete_event()` are the
  documented exception: they mutate stored snapshots without moving the room
  tally, so a concurrent call may keep the pre-lock snapshot for that pass
  (the same snapshot boundary RFC §14.4 gives every event read). Nothing a hook
  or an AI channel sees changes on the ordinary commit path. `send_event`
  stops building a context under the lock for the locked pass to rebuild a
  line later, and passes none.

- **`Room.event_count` is a maintained commit tally, not an implicit recount**
  (RMK-97, RFC §10.1 step 12). Both shipped stores now apply the specified
  `event_count += 1` when committing instead of deriving the value from the
  events still present. PostgreSQL therefore drops the only O(room history)
  query left on the inbound hot path — one `COUNT(*)` and one SQL statement
  fewer per message. The distinction matters after a hard deletion: the room
  tally remains monotonic and may exceed the number of stored events;
  `ConversationStore.get_event_count()` remains the exact on-demand count for
  callers that need it. `InMemoryStore`, `PostgresStore` and the ABC default
  now share that contract.

### Added

- **Distributed ephemeral backends: Redis realtime + Redis status bus.**
  The two surfaces that stayed process-local after the scale-out work now
  cross process boundaries. `RedisRealtimeBackend` distributes ephemeral
  events (typing, presence, reactions, thinking deltas, tool-call markers)
  over Redis pub/sub — single shared reader per process, per-subscription
  bounded queues so a slow callback never stalls the rest (same isolation
  as `InMemoryRealtime`), and `subscribe()` only returns once the server
  confirmed the subscription. `RedisStatusBackend` gives the multi-agent
  `StatusBus` a shared capped history (`LPUSH`/`LTRIM`) plus cross-process
  notifications (pub/sub) — every worker observes every agent's entries and
  `recent()` reads the same log everywhere. Both follow the
  `RedisDeliveryBackend` conventions (URL or injected client, lazy import,
  `roomkit[redis]` extra) and are re-exported from `roomkit.realtime` /
  `roomkit.orchestration`. New example `examples/realtime_redis.py`
  (two-terminal cross-process demo). The `redis` extra floor moves to
  `>=5.0.1` — the delivery backend already called `Redis.aclose()`, which
  only exists from that release.

- **`RoomKit.get_timeline(newest_first=True)`.** The store-level read has
  offered it since the tool-usage hydration work; the framework method did
  not, so the one thing a reconnecting client actually wants — the last N
  events, ascending — was only reachable through cursor gymnastics. Default
  stays `False` (page 1 of a log reads from the beginning).

- **An ACP agent catches up on the room it was not asked about** (RMK-102, RFC
  §19.3.2). §19.3.2 was left open on purpose — build the addressing machinery,
  run a room of real agents, decide after — and the room answered: Claude Code
  and Codex under `ADDRESSED_ONLY`, and "@codex do you see the previous
  message?" gets a confident "yes" about Codex's own session preamble. An
  unsolicited channel is skipped entirely, which costs an `AIChannel` nothing
  (its context is rebuilt from the store each turn) and costs an ACP session
  everything (its history lives in the agent's process). The RFC now says so:
  not asked, and not told, with the counterpart made normative — the timeline
  MUST be available to a channel at the moment it *is* solicited. `ACPChannel`
  therefore declares `recent_events_window` (new `room_history`, default 20,
  `0` opts out) and prefixes what it missed to the next prompt it sends: only
  the gap since its last prompt, never its own words, filtered per reader by
  `visible_events` so catching up is not a second door into the room (RFC §7.5
  rule 8), and honest about its bound — *"the 20 most recent of 47 messages you
  did not receive"* — because an agent that knows it was truncated can ask for
  the rest and one that believes it holds the whole room cannot. The mark
  advances only when a prompt is actually dispatched, so a muted turn leaves
  the catch-up for the next one, and a closed session starts over. No change to
  the event router.

- **`ConversationStore.connection()` — a store call no longer has to pay for
  its own connection** (RMK-97). Every call into `PostgresStore` took a
  connection from the pool and gave it back, and asyncpg resets a connection on
  release (`pg_advisory_unlock_all(); CLOSE ALL; UNLISTEN *; RESET ALL;`) — a
  full extra round trip per call, ~36% of the SQL statements an inbound message
  issued on the scale bench. `connection()` is an async context manager on the
  ABC: the calls inside the block are served from one connection instead of one
  each. The default binds nothing and yields, so `InMemoryStore` and any
  third-party backend behave exactly as before; `PostgresStore` overrides it,
  binding through the single `_acquire()` choke point every one of its queries
  already goes through. It is deliberately **not** a transaction — no
  atomicity, no isolation, no rollback, no snapshot; it bounds connection
  *tenure* and nothing else, and the block must hold store calls only, awaited
  sequentially (a hook, a provider call or a lock inside it would park a pooled
  connection behind foreign code — the failure mode the delivery lanes were
  built to remove). Reentrant. Applied to the two stretches of the inbound path
  that are pure store reads: building a room's context, and asking whether a
  room and a binding exist during routing. Measured against a real PostgreSQL
  on the simplest room: **11 → 8 pooled connections per inbound message**, i.e.
  three fewer round trips, held there by a new deterministic guard
  (`tests/test_inbound_connection_budget.py`).

- **`ACPChannel(transport=...)` — the agent connection is injectable, so the
  agent no longer has to be a subprocess of this process.** `ACPChannel` spoke
  ACP only over the stdio of an agent it spawned itself, which rules out an
  agent running anywhere else — on a user's own machine behind a relay, in
  another container. The protocol never needed the process: the SDK builds its
  connection from a plain reader/writer pair (`acp.connect_to_agent`), and the
  update-to-RoomKit-event mapping never referenced the process at all. That
  seam is now public. `ACPTransport` is the ABC — `open()` returns a live
  connection, `close()` undoes it, `is_alive()` says whether it still stands —
  and `StdioACPTransport` is the spawn, still built for you when you pass
  `command=`. Exactly one of `command` / `transport` is required, and
  `env`/`inherit_env` are refused alongside a transport rather than silently
  ignored (they configure the spawn). Everything the *protocol* does —
  `initialize`, version negotiation, `authenticate`, per-room sessions,
  prompts, permissions, config options — stays on the channel, so a new
  transport inherits it. `info["transport"]` reports the transport's name
  instead of a hardcoded `"stdio"`. The channel now asks the transport whether
  the connection is alive, which generalises "the subprocess died, respawn and
  drop its sessions" to any transport that can tell. Nothing changes for
  existing callers: `command` stays positional.

- **`RoomKit(agent_response_policy=...)` / `create_room(...)` — a room says
  whether its agents answer each other (RFC §19.3.1).** `AGENT_CHAIN` (the
  default, unchanged) lets an agent's output solicit every eligible agent —
  the chaining Appendix B.4 describes, bounded by `max_chain_depth`.
  `ADDRESSED_ONLY` lets it solicit only the channels it addressed, which is
  what a room of independent agents needs: without it, two agents answer each
  other until the depth limit stops them. It lives on the `Room` and is
  persisted, because a policy consulted on every broadcast must reach the
  same verdict in whichever worker owns the delivery lane — `update_room()`
  can change it later, and a room created before this reads `agent_chain`
  from the column default, the behaviour it was already running under. An
  explicit address wins under either policy.

- **`CLIChannel.run(visibility=...)` — a submission can scope itself.** The
  twin of `addressed_to`: addressing says who is *asked*, visibility says who
  may *see*, and the two are not interchangeable — "everyone sees, one agent
  answers" cannot be written with visibility alone, whose comma form matches
  channel ids only (`"transport,codex"` looks like a category plus an agent
  and is neither). The hook returns a keyword, a sequence of channel ids, or
  `None`; ids are joined for you, which is the form that avoids the trap. It
  sets **both** the message's `visibility` and its `response_visibility` —
  scoping the question while publishing the answer would be worse than not
  scoping at all. `InboundMessage` gains `response_visibility` for that,
  stamped centrally beside `addressed_to`. Default unchanged: no hook, no
  restriction.

- **The console transcript names people, not the wire.** A speaker with a
  participant behind them shows as `@marie · sms` — their display name and
  the *kind* of channel they reach you through. Two colleagues texting into
  one room used to share a single handle, and neither had a name. No new API:
  `RoomContext` already carried the participants and `deliver()` already
  received it. `source.participant_id` holds a `Participant.id` or an
  `Identity.id` depending on the path, so both are looked up. Events with no
  participant — every agent — keep the channel-derived label.

- **`examples/sms_and_agents.py` — a colleague on SMS in your agent's room.**
  Her message does not wake the agent (the SMS binding is `"transport"`), the
  agent's answers do not reach her phone (its binding is your console), and
  `/dm` scopes a line to her alone. Runs with no credentials: the mock SMS
  provider prints what it would have sent, and `/sms` fakes one coming back.

- **Esc interrupts the turn in flight, and keeps what it already said.** In
  console mode, Escape cancels the running `process_inbound` — not Ctrl-C,
  which still ends the session: interrupting a long answer and leaving are
  different intentions. The queue keeps draining, so a message typed behind
  the interrupted turn still runs. What the agent had streamed is persisted
  rather than lost: it is on the user's screen, so the timeline must hold it
  too, or the room disagrees with what the human read and the agent's next
  context is missing what it already said. The segment carries
  `metadata["cancelled"] = True` so a reader can tell a finished answer from
  a stopped one. Cancellation is **not** an error — nobody failed — so
  ON_ERROR stays silent and the `CancelledError` propagates untouched.
  Previously `CancelledError`, a `BaseException`, slipped past the streaming
  path's `except Exception` and took the partial text with it.

- **`CLIChannel.run(status_extra=...)` — the application's own status-bar
  segment.** Asked fresh on every render, it sits between the model and the
  live status: the multi-agent example shows `→ @codex`, so the bar answers
  *where does my next line go?* before you type it. The shell cannot know
  that — the address comes from the application — so the segment is a hook
  rather than a built-in. A segment that raises costs its own line, never the
  bar. Console mode on a real terminal only; the classic loop has no bar.

- **`CLIChannel.run(addressed_to=...)` — a submission can name who it asks.**
  Given the submitted line, return channel ids (or `None` to leave the
  message unaddressed). Evaluated after `content_factory`, so a line that
  switched which agent you are talking to is already reflected. The
  multi-agent example uses it with `@codex review hello.py` and a `/agent`
  picker — and lost its `ConversationRouter`, its two routing rules and its
  routing predicate in the process, 69 lines out for 37 in.

- **Event addressing — `RoomEvent.addressed_to` / `InboundMessage.addressed_to`
  (RFC §19.3).** Names the intelligence channels *asked to act* on an event.
  It is **not** visibility: `visibility` is configured on a binding and says
  who may *see*; an address is set on an event by its sender and says who is
  *solicited*. The two are independent — addressing one agent hides the event
  from nobody, and transport delivery is never narrowed, so the humans in the
  room still get the message. `None` means unaddressed (every eligible agent
  acts, or the router decides, exactly as before); an empty list addresses
  nobody, which is a decision rather than an absence. An address outranks a
  router's stamp: a router cannot override what the sender asked for. The
  address is stored with the event, so a transcript can show who was asked
  and a replay reproduces the same solicitation. RoomKit takes the decision,
  never the syntax — `@mentions`, a `/agent` command or a picker all live in
  the application. PostgresStore gains a nullable `addressed_to TEXT[]`,
  added additively: an existing row is unaddressed, which is what it was
  stored under, so nothing is backfilled.

- **`ExternalToolHandler.channel_id` — a handler can say who it serves.** The
  framework already wired the channel id in privately; it is now readable, so
  a permission prompt can answer the question a human must be asked when
  several agents share one terminal: *who wants to run this?* Empty until the
  channel is registered (handlers are built first), so read it when a tool
  call arrives. Wiring the same handler instance to a second channel now logs
  a warning — the injected hook callbacks are per-channel, so the second
  wiring silently re-attributed the first channel's tool events.

- **`CLIChannel.run(commands={...})` — local commands the loop awaits.** A
  line whose first word matches a key never reaches `content_factory` or the
  room; its async handler runs with the rest of the line as its argument,
  **in submission order**. That ordering is the feature: a handler may prompt
  (`terminal_input`, `terminal_select`) without racing the loop for stdin,
  and a command typed behind a message runs after that message's turn rather
  than inside it. Under the pinned bar commands ride the submission queue (so
  the bar is up and can be suspended for a prompt); in the classic loop they
  are awaited between reads. Previously an example had to spawn such work as
  a detached task, which raced the classic loop's own `input()` — both ACP
  examples now use `commands=`. The prefix is yours: `"/model"`, `":q"`,
  anything.

- **`roomkit.console.terminal_select()` — an inline keyboard menu, the twin
  of `terminal_input()`.** Ask the user to pick one option mid-session:
  ↑/↓ to move, Enter to choose, Esc to cancel. Under the pinned-bar shell it
  renders where the bar sits (no alternate screen, transcript untouched) and
  erases itself once answered, borrowing the shell's own input and output so
  nothing competes for the keys. Without a shell it falls back to a numbered
  list read from stdin, so piped and CI runs still work. Options are
  `(value, label)` pairs or bare strings; the return is the chosen value, or
  `None` on cancel. The new multi-agent ACP example uses it for `/agent`.

- **`examples/acp_multi_agent.py` — Claude Code and Codex in one Room.**
  Two ACP agents, one console, addressed by mention (`@codex review
  hello.py`): `CLIChannel.run(addressed_to=...)` sends each message to the
  selected agent alone, and the room's `ADDRESSED_ONLY` policy keeps agent
  output from triggering the other one. An ACP session that was skipped
  catches up from the visible room timeline before its next prompt.

- **Console transcript: a quiet handle, a marked answer, and what the turn
  cost.** A turn now opens with `@claude code` (dim italic — it names the
  speaker without shouting), the prose leads with `●` on its own first line
  with continuations aligned under the text, and the turn closes with
  `⎿ took 2m 30s · 3 tools`. The marker leads each stretch of prose, so an
  answer resuming after a tool round starts a fresh one. Replaces the
  `● Label` header above an unmarked body.

- **The console status bar says who is working, and for how long.** While a
  turn is in flight the pinned bar shows a spinner, the agent's own label,
  the elapsed wait and what it is doing right now — `⠹ Claude Code working
  32s · Edit · 12.3k ctx`. Activity is tracked per source channel, not as
  one global flag, so a room running several intelligence channels reads
  `⠹ 2 agents working 32s · Planner, Coder` (the oldest turn owns the clock
  — that is the wait being lived). The spinner runs only while work is in
  flight; an idle console never repaints. The bar also carries the model
  each agent reports for itself, replacing the banner's startup guess as
  soon as a session exists.

- **`ACPChannel.session_config()` / `.config_options()` /
  `.set_config_option()` — the agent's session tunables.** ACP agents
  advertise `model`, `mode`, `effort` and vendor switches as session config
  options; RoomKit now records them when a session opens, follows the
  agent's `config_option_update` notifications, and can set one:
  `await channel.set_config_option(room_id, "model", "sonnet")` (opening the
  session if the first prompt has not). Each change publishes an ephemeral
  `acp_config_options` CUSTOM event carrying values, human labels and the
  full descriptors — which is how the console bar tracks the live model.
  **Caveat, verified against `claude-agent-acp` 0.61.0:** a model switched
  from *inside* the agent with its own `/model` slash command is handled
  locally by the Claude SDK and announced to nobody, so the client's view
  goes stale; drive switches through `set_config_option()` when the value
  must stay observable. The example wires exactly that as its own `/model`
  command (and `--model` pins the startup model through `ANTHROPIC_MODEL`).

- **`ACPChannel(inherit_env=[...])` — forward named parent-env variables to
  the agent.** The ACP SDK strips the agent's environment down to
  `HOME/LOGNAME/PATH/SHELL/TERM/USER` (MCP practice), which silently breaks
  tooling a coding agent depends on — without `SSH_AUTH_SOCK`, every
  git-over-SSH operation prompts for key passphrases on the controlling
  terminal (and steals keystrokes from the CLI input loop). Named variables
  are read at each spawn, unset names are skipped, explicit `env=` entries
  win, and nothing is forwarded by default. The Claude Code example forwards
  `SSH_AUTH_SOCK`.

- **`CLIChannel(console=True)` — branded console mode with an inline input
  zone.** The classic REPL stays the default; console mode renders inline
  (normal scrollback, no alt-screen, unlike the voice `RoomKitConsole`
  dashboard) with a startup banner — RoomKit logo, version, AI model(s)
  discovered from the room's intelligence bindings, room id, attached
  channels — plus brand-palette styling, progressive Markdown, and
  Claude-Code-style tool activity lines (`⏺ tool(args)` / `⎿ ✓ 42 ms`)
  **with result previews**: command output and colored ± diffs (ACP diff
  blocks, MCP-style payloads) render under the completion line, capped
  with a `… +N lines` marker; a start renders without parentheses when
  arguments are not yet known (ACP enriches titles mid-run).
  On a real terminal the input zone opens directly below the banner, then
  settles at the bottom as the transcript grows; its status toolbar shows the
  room, model, idle/working state and queued count. The user keeps typing while
  the agent streams, and submissions queue and process strictly one at a time.
  Above the zone the stream flushes append-only per completed Markdown block
  (fence-aware — code blocks are never split);
  non-TTY sessions (pipes, CI) fall back to the sequential loop with the
  phase-agnostic inline renderer. Subsumes `markdown=True`; requires the
  `console` extra, which now ships `prompt-toolkit>=3.0.36` alongside
  rich. Examples opt in via `CONSOLE=1` (`shared.console_enabled()`),
  mirroring the voice-console convention. New public API:
  **`AIChannel.provider`** (read-only, mirrors
  `RealtimeVoiceChannel.provider`; used by the banner) and
  **`roomkit.console.terminal_input()`** — a terminal read that suspends
  the pinned bar for the duration (used by the ACP example's tool
  permission prompt; plain `input()` when no shell is active).

- **`RoomKit(delivery_gap_timeout=30.0)`** — how long a lane waits on a
  cursor hole owned by an absent worker before skipping it, and
  **`RoomKit(delivery_claim_lock_manager=...)`** — a dedicated lock manager
  (own pool) for delivery claims, so claim tenures (which span provider
  round trips) cannot starve the room-lock pool commits depend on. New
  framework event: **`delivery_skipped`** (`{from_index, to_index}`).

- **First-class Buzz agents: owner commands + `BuzzAgent` lifecycle runner.**
  A RoomKit process can now honor the full lifecycle contract Buzz expects
  of *every* agent, however launched (the platform's remote-agents spec,
  layer L1):
  - **Owner control commands.** `BuzzRelaySource` intercepts the platform's
    `!shutdown` / `!cancel` / `!rotate` — kind-9, exact content, mentioning
    the agent — when authored by the **proven** owner: the NIP-OA auth tag's
    attester (Schnorr-verified by buzzkit against the agent's own pubkey),
    else the new `BuzzConfig.owner_pubkey`. Commands are consumed before the
    pipeline, so the AI can no longer answer its own stop command;
    `!shutdown` stops the source gracefully (or defers to the new
    `on_owner_command` callback). Fail-closed: no provable owner → commands
    stay regular messages, as does any command from a non-owner. Replay-safe
    (live-tested): the relay replays recent history on every subscribe, so a
    command issued *before the source started* is stale — consumed without
    action, never obeyed, never forwarded — while one issued during a
    disconnection is still honored when the reconnect replays it. Inbound
    metadata gains `nostr_created_at` so apps can likewise tell live traffic
    from replay (see the example's echo guard). Governed by
    `BuzzConfig.obey_owner_commands` (default on — a bot with an auth tag now
    obeys its owner; set it to `False` to keep the old answer-everything
    behavior).
  - **`BuzzAgent`** (`roomkit.providers.buzz`): the runner that owns waiting
    and dying — attaches the sources, installs SIGTERM/SIGINT handlers, arms
    an opt-in `exit_after_inactivity` bound (default off; reaper on its own
    timer), and exits every cause through the same graceful path
    (`kit.close()` → presence `offline` → sockets closed), returning a
    `BuzzAgentStopCause`. A startup that fails — an unreachable relay, two
    sources sharing a `channel_id` — takes that same exit before the
    exception reaches the caller, so a half-started agent leaves no reaper
    task, no installed signal handler and no open kit behind. Intentional
    stops are final: the source supervisor only restarts sources that
    *raise*, never a clean stop.
  - **`BuzzConfig.from_env()`** reads the reserved identity triplet
    (`BUZZ_PRIVATE_KEY`/`NOSTR_PRIVATE_KEY`, `BUZZ_RELAY_URL`,
    `BUZZ_AUTH_TAG`), fail-closed — a RoomKit agent is launchable by the
    same script/unit/entrypoint as any other Buzz agent. New example:
    `examples/buzz_agent.py`.
  - The `buzz` extra floor moves to `buzzkit>=0.3.0` (adds
    `parse_owner_command` and the Schnorr-verified
    `BuzzClient.verified_owner_hex` these features are built on).

### Fixed

- **A configured `supervisor_id` receives an addressed message again** (RFC
  §19.4 step 4). Addressing (`addressed_to`) decides who acts, and the router
  correctly does not override it — but the supervisor is the one addition
  step 4 makes on top of the address: it ALWAYS receives the event, "in
  addition to the addressed channels". The router's addressed branch returned
  without stamping the supervisor, and solicitation short-circuited on the
  address before consulting the always-process list, so any message that named
  a recipient silently dropped the supervisor — the single channel an address
  is not allowed to silence. The router now stamps only the supervisor (never a
  routing target) on an addressed event, and solicitation honours it; a sender
  who already named the supervisor gets no redundant stamp.

- **A coding agent that answers in raw JSON now renders as output in the
  console, and a shell command behind an ACP `terminal` block renders at
  all.** ACP fixes the envelope — `content` blocks and a free-form
  `raw_output` — and leaves the payload inside it to each agent, so Claude
  Code's text and diff blocks rendered as a transcript while Codex's
  `{"formatted_output": …, "exit_code": N}` reached the terminal verbatim,
  escaped newlines and all, cut mid-string at 200 characters. Three shapes,
  each captured from a live `@agentclientprotocol/codex-acp` 1.1.9 session:
  a file read arrived as that JSON dump; a shell command arrived with a
  `terminal` content block — a handle on live output, carrying no text —
  which won over the raw result and so displayed **nothing**; a failed
  command put the same JSON in `error`, printed as one unwrapped line.
  Preview extraction moves to `roomkit.console._tool_preview`, which reads
  the envelope instead of printing it: payload wrappers (`formatted_output`,
  `output`, `text`, `content`, MCP's `result`/`error`) are unwrapped down to
  their text, an envelope that arrived as a JSON *string* is parsed, a
  display payload that yields nothing falls back to the raw result,
  `image`/`audio`/`resource` blocks are named rather than dumping their
  base64, and every preview line is clipped at 200 characters. A command
  that fails without printing anything now reports `exit code N` instead of
  a bare `✗ tool failed`. Output that merely happens to be JSON
  (`cat package.json`) keeps its own lines, and a payload shape nobody
  recognises still degrades to the compact JSON it rendered before.

- **An event visibility hid from a channel no longer comes back to it as
  context on the next turn** (RFC §7.5 rule 8, new in `roomkit-specs`).
  Broadcast kept `visibility`'s promise; the turn after broke it. A channel the
  visibility hid an event from was correctly not called at broadcast, but the
  event stayed in `RoomContext.recent_events`, which the memory provider re-read
  verbatim and handed to the model. Measured on all eight combinations — scope
  `transport` / `none` / `internal` / a specific channel id, set on the event
  *or* on the source binding: every one leaked. The `internal` case is the
  starkest: those are the events the framework produces itself (delegation,
  handoff, system) and documents as living only in stored room history — they
  were arriving in a model's prompt one turn later.

  The filter runs per reader, in `AIChannel`, on the way *into* the memory
  provider: one `RoomContext` serves every channel of a broadcast, so filtering
  it once at load would give the same wrong answer to everyone, and filtering
  the provider's *output* would still let a summarizing provider re-emit hidden
  content as a summary. No `MemoryProvider` signature changed — every provider,
  shipped or third-party, inherits the rule. Two deliberate exemptions: a
  channel always keeps its **own** events (dropping them would erase an
  assistant's own turns from its own prompt whenever it is bound with a narrow
  visibility — the RFC §7.4 assistant pattern), and hooks keep the whole
  timeline (host code, in the integrator's process, holding the store anyway).

  Two consequences to know about. **This narrows what an AI sees**: an
  application that set `visibility="transport"` believing it only delayed
  delivery will find its model's context smaller. And because the source
  binding's scope is resolved at read time — the router stamps it onto a copy
  made *after* the commit, so storage keeps `"all"` — **visibility is a live
  policy**: widening a binding widens its past too. A source whose channel has
  since been detached leaves the event's own `visibility` as the whole answer,
  so ordinary history survives a detach while framework-internal events, which
  carry their scope on the event, stay hidden. New:
  `roomkit.core.visibility.visible_events(context, channel_id)` and
  `effective_visibility(event, source_binding)`, the latter now also backing
  `EventRouter._check_visibility` so delivery and re-read cannot drift.

- **`RoomContext.recent_events` holds the room's most recent messages, not its
  first ones.** `ConversationStore.get_conversation()` delegated to
  `list_events()` without `newest_first`, so a call with no cursor returned the
  *oldest* `limit` messages — the head of the room, frozen. That is the read
  behind `_build_context()`, which means every hook and every AI channel on a
  room whose history outgrew the channel's window saw the opening of the
  conversation and never what had just been said; a sliding-window memory
  provider slicing `recent_events[-N:]` only took the last of the oldest. Both
  shipped stores were affected (the in-memory slice and the Postgres
  `ORDER BY created_at LIMIT n OFFSET 0`). Without a cursor `get_conversation`
  now returns the newest `limit` messages in ascending order — the last element
  is the newest. With `after_index` it stays a forward keyset cursor ("what
  arrived since I last looked"), unchanged. The delegated-agent room context
  (`_child_execution`) was reading its head the same way and is fixed too.
  This is an observable semantic change for a caller that used
  `get_conversation()` without a cursor to read a room's *opening* messages —
  ask `list_events()` / `get_timeline()` for those. A third-party store that
  implements `list_events` while ignoring the `newest_first` argument (already
  part of the ABC) keeps the old behaviour silently.

- **`PostgresStore.init()` serializes its DDL across processes.** Idempotent
  DDL is not concurrent DDL: PostgreSQL resolves `IF NOT EXISTS` *before*
  taking the lock the statement needs, so a fleet restarting together had
  every worker see the same object missing and the losers raise
  (`duplicate_object` on a table or index, `duplicate_column` on an additive
  column migration) — a failed boot on exactly the deploy that touches the
  schema. `init()` now runs as one transaction behind the same advisory lock
  `migrate()` takes, so the two also exclude each other. The additive column
  guards are additionally scoped to `current_schema()`: `information_schema`
  spans every schema the role can see, and a namesake `rooms` table in
  another one would have skipped a migration our tables needed.

- **`ACPChannel.close_session()` frees everything the session owned.** It
  dropped the session and its room mapping but kept the agent's reported
  config options and the room's turn lock until the channel itself closed —
  so a long-lived channel cycling sessions (one per conversation, one per
  reconnect) accumulated one dead entry per cycle. Both go with the session
  now; a coroutine queued on a retired turn lock re-checks and retries on
  the current one, so retiring it cannot hand two callers the same critical
  section.

- **`ACPChannel.close()` is bounded and always tears the process down.** The
  graceful half (cancel the turns, close the sessions) now shares a 5-second
  budget, and the subprocess teardown runs in a `finally` — so an agent that
  has stopped answering, or a second Ctrl-C landing on `close_session`, can
  no longer hang `RoomKit.close()` nor leave the agent process orphaned. The
  Claude Code example also exits quietly on Ctrl-C instead of printing a
  cancellation traceback.

- **Buzz presence heartbeat no longer dies on a transient failure.**
  `BuzzRelaySource._presence_loop` returned (silently, DEBUG-only) on the
  first failed kind-20001 publish, leaving a live agent showing offline
  until the next reconnect. Presence is the only liveness signal a Buzz
  agent has (Buzz's remote-agents spec makes it the status, with a
  relay-side TTL), so the loop now logs a WARNING and retries at the next
  beat — a dead socket still ends the loop via the subscribe failure and
  reconnect.

- **Buzz agents now flip to offline on a deliberate stop.**
  `BuzzRelaySource.stop()` publishes presence `"offline"` (best-effort,
  gated on `announce_presence`) before leaving/closing, so a stopped agent's
  dot turns grey immediately instead of lingering "online" until the
  relay-side presence TTL lapses — the avoidable half of the staleness
  window Buzz's remote-agents spec (I3) bounds.

## [0.39.0] — 2026-08-02

### Added

- **Tool Search: conversation-scoped tool memory.** Three legs, one goal —
  an agent's working knowledge of its tools now spans the conversation, not
  the turn or the process:
  - `find_tools` reveals persist across turns via `ToolUsageMemory.record_revealed`
    (the tool's description already promised "the rest of the session"; the
    code now honours it — a tool found in turn N is often only called in
    turn N+1).
  - `ToolUsageMemory` hydrates from the persisted event store on first use
    per room: the framework injects a loader (`register_channel`) that reads
    the room's recent `TOOL_CALL_END` events, so a process restart or channel
    cache expiry no longer wipes the digest + re-reveal set mid-conversation.
  - `find_tools` results list unmatched same-family tool names
    (`related_tools_same_source`, name-prefix family, both text and realtime
    paths) — peripheral vision so a model that found `get-menu` knows the
    same source also does carts instead of refusing the next in-domain ask.

- **`read_stored_result`: explicit partial-page warning.** Small models skip
  the bare `has_more`/`next_offset` fields and conclude absence off one page;
  a partial page now carries a prose `warning` ("PARTIAL CONTENT … never
  conclude something is absent until you have read EVERY page"), and the page
  envelope allowance grew to keep the warning from re-evicting the page.

- **MCP: `structuredContent` survives result flattening and eviction.**
  `MCPToolProvider.call_tool` publishes a successful call's
  `CallToolResult.structuredContent` (dict, ≤512KB serialized) on the active
  `ToolCallContext`, and the AI channel carries it through
  `AIToolResultPart` → `ToolCallEndMarker` → `ToolCallContent.structured_content`
  on both the streaming and non-streaming paths. The LLM-facing string is
  unchanged (large results still evict to a placeholder); UI surfaces that
  render from structured tool output (MCP Apps widgets) read the field off
  the persisted `TOOL_CALL_END` event instead of re-parsing — or losing —
  the text form.

- **Buzz: threaded replies (NIP-10).** The inbound parser reads a message's
  NIP-10 `e`-tags and sets `InboundMessage.thread_id` to the thread root
  (plus `metadata["nostr_reply_to"]` for the immediate parent), and
  `BuzzProvider.send` threads outbound messages under
  `channel_data.thread_id` via `send_message(..., reply_to=...)` — the same
  provider-native threading contract Discord and Teams use.

- **Buzz: reactions.** `BuzzRelaySource(on_event=...)` surfaces NIP-25
  reactions (kind 7 → `action: "add"`) and their retractions (kind 5 →
  `action: "remove"`) as normalised dicts outside the message pipeline —
  matching the Discord/WhatsApp-personal reaction contract — and widens the
  default subscription to kinds 9 + 7 + 5. Outbound,
  `BuzzRelayProvider.send_reaction(target_event_id, emoji)` /
  `remove_reaction(reaction_event_id)` publish through the shared client
  (`MockBuzzProvider` records them). New pure helper `parse_buzz_reaction`
  and kind constants `KIND_STREAM_MESSAGE` / `KIND_REACTION` /
  `KIND_DELETION` in `roomkit.sources.buzz`.

- **Buzz: `BuzzConfig.leave_on_stop`.** Opt-in NIP-29 leave (kind 9022) when
  the source stops. Off by default: on a private channel the membership was
  granted by an admin and self-join cannot get it back, so leaving on every
  shutdown would lock the agent out.

- **Stress lane.** `make stress` (pytest marker `stress`) runs load/contention
  tests excluded from the default run: 100 rooms × 5 concurrent turns, 50
  turns racing into one room, flaky delivery under load, and a 1000-turn
  conversation asserting the framework's transient state stays flat. The
  `wallclock` marker is registered for timing-sensitive tests so loaded CI
  runners can deselect them.

### Fixed

- **G.711 mu-law encode is bit-exact with the reference.** The encode table
  was indexed by ``magnitude >> 1`` but built without re-doubling, so every
  outbound sample was encoded as if it had half its amplitude (~6 dB low,
  with shifted segment boundaries) on every G.711 path — Twilio media
  streams and RTP/SIP calls. The codec now transcribes Sun/CCITT
  ``g711.c`` exactly, verified against ``audioop`` over the full 16-bit
  sweep; decode was already exact and is now vectorised too (22.5 → 1.5 µs
  per 20 ms frame).

- **A revoked channel stays revoked across restarts and workers (RFC §7.5-7).**
  `detach_channel()` now records the revocation in room metadata
  (`roomkit:detached_channels`, written atomically via
  `patch_room_metadata`) instead of a per-process in-memory set. Previously a
  process restart — or a sibling worker sharing the store — lost the record,
  and the next inbound message silently re-attached the detached channel at
  default permissions. An explicit `attach_channel()` still clears it, and
  the metadata key is removed when its last tombstone goes.

- **Dishonest concurrency tests now assert their invariant.** The lock
  manager's serialization test measures critical-section overlap (a no-op
  lock now fails it); the framework's concurrent mute/access test asserts
  both writes land (lost-update proof) instead of `muted in (True, False)`.

### Added

- **Inbound DSP off the event loop: `AudioPipelineConfig.inbound_dsp_threads`.**
  With a pool size set, each session's stage chain (resampler → AEC →
  denoiser → VAD → …) runs on a small thread pool instead of the event
  loop: frames of one session stay strictly FIFO, sessions spread across
  workers, and the native stages release the GIL — so the concurrent-call
  ceiling scales with cores instead of one, and a slow stage no longer
  delays every other session and all message traffic. Backpressure is per
  stream and bounded (drop-oldest, counted). Applies to `VoiceChannel` and
  `RealtimeVoiceChannel`; unset (default) keeps inline processing.
  Measured 3.9× on 4 workers in the stress bench (`make stress`).

### Changed

- **The messaging path stops paying O(rooms) and O(events) store costs**
  (measured at 5 000 rooms / 50-event contexts, Apple Silicon):
  - `find_latest_room` — called once per inbound message — resolves
    through a participant→rooms candidate index instead of scanning every
    room and binding: 1 467 → 23 µs (64×). The index only narrows
    candidates; the full match predicate re-runs per candidate, so
    behaviour is unchanged.
  - `InMemoryStore` event reads share the stored snapshot instead of
    deep-copying every event on every read: `get_conversation(50)`
    1 757 → 6.6 µs (266×), a full `RoomContext` build 1 607 → 80 µs
    (20×). The copy moved to the write side — the store deep-copies once
    per `add_event`/`commit_event`/`update_event`, so a caller's later
    mutation of a written object still cannot reach the log. Committed
    events are immutable (RFC §4); the new RFC §14.4 makes the
    returned-object ownership explicit: treat read events as frozen,
    rely on neither aliasing nor isolation. Rooms, bindings and
    participants keep their caller-owned copies.
  - FastRTC resolves websocket→session through a dict maintained at
    registration instead of scanning all sessions per audio frame.
  - The stress lane pins both budgets (`test_stress_messaging.py`).

- **The WAV recorder is off the frame path.** Its taps now only enqueue:
  all disk I/O — file opens, `writeframes`, spooling, mixing — runs on a
  dedicated writer thread behind a bounded queue (a full queue drops the
  frame with a counted warning; recording observes the call, it never
  brakes it). MIXED/STEREO/ALL directions spool to raw files instead of
  accumulating ~1.9 MB/min/session in RAM, and the stop-time mix — a
  per-sample Python loop on the event loop, ~9.6 M iterations for a
  10-minute call — is vectorised and runs on the writer thread. `stop()`
  queues behind the session's remaining frames, so the files it reports
  are complete when it returns.

- **The voice frame path sheds its residual per-sample Python** (measured
  per 20 ms frame on Apple Silicon): Speex AEC energy diagnostics are
  DEBUG-gated and computed outside the stream lock (−28 µs/frame in
  production, and the playback path no longer waits on them); the
  sherpa-onnx VAD feeds ``accept_waveform`` a NumPy array instead of a
  per-element Python list (14.5 → 1.7 µs) and its RMS helper is vectorised
  (→ 2 µs); the pipeline logs a warning when the pure-Python resampler
  fallback is selected (~13× slower than the NumPy one it stands in for).

- **`register_channel()` refuses a duplicate `channel_id`** with the new
  `ChannelAlreadyRegisteredError` instead of silently replacing the live
  channel (which left existing bindings routing to an orphan). Call
  `unregister_channel()` first to swap an implementation deliberately.

- **`RoomKit.close()` is idempotent.** A second call returns immediately
  instead of re-running the teardown against already-released resources.

- **Buzz: graceful relay restarts reconnect quietly.** The source checks
  `BuzzClient.close_code` — a 1012 close (relay restart) reconnects at the
  initial backoff with an INFO log instead of counting as an error; replayed
  events were already deduped by id.

- **Buzz: presence heartbeat every 30 s (was 55 s).** Relays at buzz >= 0.5.x
  hold presence for a 180 s TTL and expect a beat every 60 s; older relays
  used 90 s / 30 s. 30 s is the cadence buzzkit documents as safe on both.

- The `buzz` extra now requires `buzzkit>=0.2.1` (reply threading, reactions,
  `close_code`, kind-9022 leave; 0.2.1 surfaces the HTTP bridge's error body
  on rejected sends).

## [0.38.0] — 2026-07-31

### Added

- **Skills: unavailable skills stay visible with a reason.**
  `SkillRegistry.mark_unavailable(name, reason)` records a skill that exists
  but cannot be used in the current context (e.g. a `requires` gate whose
  tools are not granted); `unavailable_skills` / `get_unavailable_reason()`
  expose the mapping, `register()` clears a stale mark. `to_prompt_xml()`
  emits an `<unavailable_skills>` block with per-skill `<reason>` (rendered
  even when no skill is available), and the `activate_skill` /
  `read_skill_reference` / `run_skill_script` handlers — AIChannel and
  realtime voice alike — answer `Skill 'X' is unavailable in this context:
  <reason>` instead of a misleading "not found". The AIChannel tools-hint
  fallback ("X is not a skill, but these TOOLS match") no longer fires for a
  known-but-unavailable skill.

- **Conference: speech-to-speech composition (RFC §12.10.12).** A realtime
  provider (Gemini Live, OpenAI Realtime, …) can now be the conference's
  intelligence: `ConferenceChannel(realtime=ConferenceRealtimeConfig(...))`
  mixes every subscribed audio track N→1 (additive, `1/√k` headroom, 20 ms
  windows, silence-only windows never forwarded), feeds one provider session
  per conference, and publishes the provider's voice on the bot track under
  the ordinary utterance contract — floor, terminal `is_final`, barge-in
  included. Attribution ends at the provider boundary: its user-side
  transcriptions are discarded (configure `stt=` beside it for the attributed
  transcript — the lanes run in parallel with the mix), while its assistant
  finals become room events attributed to the channel. The per-lane VAD stays
  the interruption sensor: `ConferenceInterruptionConfig` scope is enforced
  on it, and a landed barge-in also cancels the provider's response
  (best-effort — documented no-op on Gemini Live). `tts=` and `realtime=`
  are mutually exclusive (one bot track, one voice); inbound text events are
  injected into the provider's context rather than synthesized. The slot
  hot-plugs like the others — `plug_realtime()` / `unplug_realtime()`, first
  need, occupied-slot refusal, last-need retirement — and the lanes it
  shares with a recognizer survive whichever of the two unplugs first.
  `info()` gains `realtime_configured` / `realtime_provider` and per-room
  `realtime_active` / `realtime_dropped_windows`. New example:
  `examples/conference_realtime_ai.py`.

- **Conference: runtime ownership of explicit bot grants (RFC §12.10.4).**
  The plugs never rewrite an explicit `bot_grants` — which makes the set
  owned, not immutable — and `ConferenceChannel.set_bot_grants()` is now
  the owner speaking at runtime: pass a new grant set to replace the
  explicit one (the caller keeps coverage of the configured needs on
  themselves, exactly as at construction), or `None` to return the channel
  to derivation. Unlike a plug's alignment, the change is an instruction
  applied to every live session in full — in place where the backend
  declares `BOT_GRANT_UPDATE`, by the announced re-join where it cannot or
  where the update fails. Visibility moves asymmetrically, verified live
  against LiveKit: removing `hidden` in place makes the SFU announce the
  bot to the clients already connected — the observer that reveals itself
  when the host starts the notetaker — while no SFU interface can un-tell
  them, so a visible→hidden change always replaces the session, the
  announced leave being the one retraction every backend delivers. Each
  effective change on a connected session emits the new
  `conference_bot_grants_changed` framework event; `info()` gains
  `bot_grant_update_in_place` (the price of a change, answered before the
  call) and a per-room `bot_hidden` (the status in force on the session,
  §17.7).

- **Conference: hot-plugging intelligence (RFC §12.10.4).** The configuration
  first need is read from is no longer fixed at construction:
  `ConferenceChannel` gains `plug_stt()` / `unplug_stt()`, `plug_tts()` /
  `unplug_tts()` and `plug_recording()` / `unplug_recording()`. Plugging a
  need is a first need — the attach's occupancy probe is re-run, an occupied
  conference is joined at once, and the tracks already published are
  subscribed retroactively, so a meeting is transcribed from the plug
  forward. Unplugging the last need takes the bot out (`conference_ended`
  announced): the channel returns to pure transport, same channel, same
  room. A plug refuses exactly what construction refuses (E2EE × stt,
  E2EE × recording, an already-filled slot); unplugging an empty slot is a
  no-op; an unplugged provider is closed under the existing
  `close_providers` rule. The bot's derived grants now follow the
  configuration in force at each join, and a change that widens what a live
  session must do is applied in place through the new optional backend
  surface `ConferenceBackend.update_bot_grants()` (capability
  `ConferenceCapability.BOT_GRANT_UPDATE` — declared unconditionally by the
  LiveKit backend, which implements it over `UpdateParticipant`), falling
  back to an announced re-join on backends that cannot re-permission a
  connected session. `info()` answers §17.7 with the configuration in
  force, not the constructor's. The notetaker-on-demand flow is the use
  case: see `examples/conference_notetaker_on_demand.py`.

- **xAI (Grok) chat provider.** `XAIAIProvider` + `XAIConfig`
  (`pip install roomkit[xai]`) put Grok's text models on the same footing as
  every other AI provider. xAI serves the OpenAI Chat Completions API verbatim,
  so the provider subclasses `OpenAIAIProvider` and inherits message building,
  tool handling, streaming, `/v1/models` discovery and client construction
  unchanged; `XAIConfig` subclasses `OpenAIConfig` so no request field can drift
  between the two. Three things are genuinely xAI's own:

  - **The catalog** (`available_models()`): the six current Grok text models
    with their real context windows (`grok-4.5` 500k, `grok-4.3` and the 4.20
    variants 1M, `grok-build-0.1` 256k).
  - **Vision.** The inherited implementation prefix-matches *OpenAI's* vision
    model names, so it reports every `grok-*` id as text-only and silently drops
    images. `supports_vision` now reads the catalog instead, and an id the
    catalog does not know (an alias like `grok-latest`, or a model newer than
    the snapshot) defaults to capable — the whole Grok text line is multimodal.
  - **Reasoning.** Depth rides the top-level `reasoning_effort` string, as on
    OpenAI (the nested `reasoning: {effort}` object is `/v1/responses`, not Chat
    Completions). Unlike the parent it is sent on **tool turns too**: Grok
    reasons unconditionally, so effort is the only lever over the cost of an
    agentic turn, and dropping it on exactly the turns that spend the most would
    defeat the setting. It is withheld only from `grok-4.20-0309-non-reasoning`,
    which the catalog marks as refusing it.

  `XAIConfig` also flips two parent defaults to match the API: the output cap
  goes out as `max_completion_tokens` (xAI deprecated `max_tokens`) and
  `stream_options.include_usage` is on so streamed turns account their tokens.
  Runnable example in `examples/xai_ai.py`. The pre-existing
  `XAIRealtimeProvider` (Grok speech-to-speech) is untouched — same vendor,
  different protocol.

- **Multi-party conference support.** RoomKit can now join a meeting it does
  not host. `ConferenceChannel` attaches a room to a conference whose media
  plane an external SFU owns: it brings a bot into it, transcribes its
  participants, speaks the AI's answers into it, records it, and can be asked
  what it is doing there. RFC §12.10 is the normative reference (conformance
  Level 3); the pieces, briefly:

  - **Transport.** `ConferenceBackend` is the ABC — join/leave, track
    subscription, `publish_audio()` on a single bot track, `mint_access()`,
    room lifecycle — and `MockConferenceBackend` implements the whole of it
    for tests, with fault injection to make the failure paths reachable:
    `fail(method, ...)`, `delay(operation, ...)`, per-track audio formats
    (`MockTrackFormat`), and the bot's output grouped by utterance. A room
    holds at most one conference: attaching a second conference channel is
    refused with `ConferenceAlreadyAttachedError` (RFC §12.10.4), and the
    reservation outlives the binding — a room whose previous conference
    channel still has a session in the meeting or a teardown running keeps
    refusing, because a detach removes the binding at its start and takes
    the bot out at its end. The refusal is retryable, never a wait: the
    attach may come from inside the very announcement the teardown is
    deferred behind. The reservation's authority is one RoomKit instance —
    its bindings plus the books of its registered channels; making it hold
    across workers sharing one store is a contract decision tracked
    separately. A backend that observes the SFU ending the bot's
    session without a `leave()` — a dropped connection, an eviction —
    reports it through `on_bot_session_ended`; the channel takes the
    session off its books, finalizes its recordings, announces
    `conference_ended`, and re-joins on a bounded, backed-off supervisor
    while the room stays attached and collecting — the dead session was
    what received the frames, so no backend event could produce the lazy
    join's "next need". Past the attempts, the lazy join remains the
    fallback. A detach's own `leave()` runs on the same budget-and-grace
    discipline as the close's: a wedged SFU costs the teardown one budget,
    and the session goes back on the books where the close retries it.
  - **A real SFU.** `LiveKitConferenceBackend` (`pip install roomkit[livekit]`)
    implements the whole of that ABC against LiveKit, media plane only —
    `livekit-agents` stays out, because RoomKit already owns VAD, recognition,
    synthesis and interruption, and a transport that segmented speech would
    break the separation the ABC exists to draw. It joins with
    auto-subscription off so the framework's subscription set stays the
    authoritative one, announces the participants that were already there when
    the bot arrived, and hands the lane 48 kHz frames that *declare* their
    format rather than resampling them. `capabilities` reports what is wired
    rather than what LiveKit sells: screen share, active speaker and connection
    quality always; remote unmute and SIP dial-in only where the deployment
    says the server was configured for them; not E2EE, whose key exchange
    `ConferenceBackend` has no contract for, and not bot video, which has no
    source until an avatar gives it one. Identity is founded only on
    attributes LiveKit itself asserts — the `sip.` attributes of a participant
    the *server* marked as a dial-in — so a client writing its own
    `sip.phoneNumber` cannot reach someone else's Identity. A disconnect the
    SDK refuses propagates — the session stays registered until it is
    genuinely out, and `close()` raises for the sessions it could not take
    out instead of logging them — and its control-plane event bridge is
    bounded with the loss made explicit, never silent: active-speaker and
    quality events coalesce to their latest value (a flapping participant
    costs bounded memory), and a lifecycle event that would not fit ends
    the session as a *reported discontinuity* through `bot_session_ended`
    rather than being dropped where nothing would say so. The end is
    reported only once the old connection is confirmed disconnected — a
    disconnect that will not go through keeps the session on the books,
    refusing a replacement, for a later `leave()` to retry — the
    disconnect itself is single-flight (a `leave()` arriving while the
    unhealthy end's call is on the wire joins it rather than issuing a
    second one, and a requested leave owns the books: nothing spontaneous
    is reported over it), and the reason counts the events discarded
    undelivered. The supervisor's
    re-join then announces the conference's *current* state; what happened
    entirely inside the outage window is genuinely lost, and the reason
    string is the signal for §17.7 implementations to treat that window
    as unaccounted rather than observed-and-empty.
  - **Transcription.** Each subscribed AUDIO track runs through the shared
    `AudioPipeline` in a lane of its own, under the track's stream identity:
    one utterance becomes one transcription event attributed to its speaker,
    and one participant's recognizer latency never delays another's frames.
    Backpressure is bounded and counted (`max_queued_frames`, oldest frame
    dropped). A lane requires a VAD, and `InterruptionStrategy.SEMANTIC` is
    refused — a backchannel can only be classified once the utterance has
    ended, too late to interrupt anything.
  - **Identity.** An arrival the framework did not name is resolved when it
    arrives, from the attributes the SFU itself vouches for
    (`ConferenceParticipant.asserted_metadata`); a participant's own claims
    reach a resolver only on a channel told to trust them. Provider
    attributes land under `participant.metadata["conference"]`, bounded and
    provenance-kept, never over the integrator's own keys. Utterances skip
    re-resolution: the roster already carries the answer.
  - **Recording.** With `recorder=` and `recording=ConferenceRecordingConfig()`,
    every subscribed track is recorded separately and attributed to its
    publisher — the bot's own audio included — through the room-level
    `MediaRecorder` contract now specified in RFC §12.11. Writes stay off
    the frame-delivery path, the per-track backlog is bounded and counted,
    and `ON_RECORDING_STARTED` / `ON_RECORDING_STOPPED` report where each
    recording was written. No audio reaches the recorder before
    `ON_RECORDING_STARTED` has been heard (RFC §17.6): the audio buffered
    during the announcement flows to the recorder once the hook returns, and
    a handler that refuses — detaching the channel is the ordinary way —
    captures nothing, the buffered frames dropped and counted. The hook is a
    consent point, not a notification of capture under way; the full consent
    and encryption-at-rest mechanism is tracked separately.
    `ConferenceRecordingConfig.metadata` reaches the recorder verbatim on
    `MediaRecordingConfig.metadata`, one copy per recording.
  - **Speaking and interruption.** AI responses are synthesized once and
    published on the bot track, one utterance at a time, every utterance
    closed; who may interrupt the bot is policy
    (`ConferenceInterruptionConfig`), and `ON_BARGE_IN` names the
    interrupting participant. A barge-in that lands reaches the SFU as
    `ConferenceBackend.stop_playback()` — the audio the transport had queued
    is discarded instead of playing to the end of its buffer, so the
    interruption is as immediate as the transport allows rather than bounded
    by its queue. Non-AI text is spoken only with `speak_text_events=True` —
    a meeting is not a place to read unrelated channel traffic aloud.
  - **Admission.** `mint_access()` issues `ConferenceAccess` under
    `default_grants`, validates before it mints, and refuses a banned
    participant and a room the channel is leaving; a mint still in flight
    when a detach lands is taken back rather than left valid against a
    meeting the framework has left. Bans stick — no SFU event lifts one.
    A credential that goes out also starts the lazy bot join, in the
    background: presence is observable only through a connection, so no
    backend callback can make the *first* join happen — the mint is the
    framework's own advance notice that a human is about to connect (RFC
    §12.10.3/.4), and it is what lets a meeting where humans speak first
    be joined and transcribed without the framework having to speak. The
    join never delays the mint's answer, and its failure never fails the
    mint. An attach is the other trigger that owes nothing to the
    backend's callbacks: it may be landing over a conference already
    underway — a channel restarted mid-meeting re-attaches above
    participants an earlier life admitted, with no mint left to wait
    for — so it probes the conference's occupancy with
    `list_participants()`, off its own path, and anyone in there who is
    not the channel's own bot starts the same lazy join. An empty
    conference stays unjoined, and the probe's failure is never the
    attach's. Both triggers answer to a need: the join exists for the
    intelligence, so a channel configured with no stt, no tts and no
    recording — pure transport — never joins on a mint or an arrival and
    skips the probe entirely. RoomKit stays the meeting's admission gate
    and roster with no participant of its own in it, at the stated price
    that the bot's connection was the event bridge: no real-time
    participant, track, speaker or quality callbacks from a backend that
    observes presence only through a connection (RFC §12.10.4).
  - **Observability.** `conference_started` / `conference_ended` name and
    measure the bot session, and `info()` answers RFC §17.7's disclosure
    questions per room — bot present, collection permitted, STT and
    recording *active* as distinct from configured — keeping a session on
    its way out visible until it has actually left.
  - **Management surface.** What an interface reflecting the meeting reads,
    live. The lanes announce the VAD's utterance boundaries on
    `ON_SPEECH_START`/`ON_SPEECH_END`, named per participant and track —
    the real-time "who is speaking right now" the SFU's dominant-speaker
    signal (relayed on `ON_ACTIVE_SPEAKER_CHANGED`) cannot give, having no
    way to say that nobody is. The SFU's view of each participant's
    connection is relayed on `ON_CONNECTION_QUALITY_CHANGED`. A publisher
    muting or unmuting a track is relayed on `ON_CONFERENCE_TRACK_MUTED` /
    `ON_CONFERENCE_TRACK_UNMUTED`, naming the track's kind — a muted VIDEO
    track is how most clients say "camera off", so microphone and camera
    indicators both read from this pair; screen share keeps its own
    `ON_SCREEN_SHARE_STARTED`/`STOPPED`. And the name
    a room gave a participant rides the minted credential: LiveKit renders
    it in its own clients, reports it back on `list_participants()` and the
    catch-up, and a roster record without a name takes the reported one —
    never overwriting one the integrator set — which is how a roster
    rebuilt after a restart gets its names back.
  - **Shutdown.** There is one logical shutdown per channel: concurrent
    `close()` calls join the same shielded task, a caller cancelled mid-wait
    abandons only its own wait, and once the shutdown reaches its terminal
    result later calls replay it — an immediate return after a success, the
    same `ConferenceCloseError` after a failure — instead of re-running the
    steps. Departures are exact-once: every path a session leaves through —
    a detach, an abandoned join, the close's sweep — funnels into one
    `leave()` per session, and a path that finds one in flight joins it.
    The media plane outranks the bookkeeping: a detach and a close take the
    bot out of the meeting on bounded budgets, and the media calls are no
    exception — `leave()`, the backend's close and a lane's recogniser are
    cancelled past their budget, given a bounded grace, then abandoned and
    reported rather than waited for again. Every backend and provider call
    the channel admits holds a *lease* on the resources it uses — the
    backend under a publish or a late join, the pipeline and recognizer
    under a lane, the synthesizer under a stream — and a resource is closed
    only once no lease on it remains: one still in use past the budget is
    retained, closes in the background when its operations truly end, and
    fails the current close explicitly. The pipeline's own close runs off
    the event loop and closes every provider whatever became of the ones
    before it; the recorder reports finalizations it could not finish and a
    provider it had to keep alive instead of leaving them in the log. A
    session the channel could not remove, a resource it had to retain, or a
    backend/provider close that failed is a *failed* close: `close()` raises
    `ConferenceCloseError` carrying the structured report (component,
    operation, status per issue) at the very end, once every other step has
    run — `info()` goes on reporting retained sessions. The waits that
    cannot be bounded belong to `RoomKit.close()`: every operation the
    channel starts on the store or the lock manager — the reads included,
    and the room lock from the moment its acquisition begins to the moment
    it is let go — runs under a framework *resource lease*, and the
    framework finishes every lease after all the channels have closed and
    before it releases either resource. Nothing integrator-owned ever runs
    under a lease, and once the final wait has concluded the registry is
    sealed: work that resumes later — a callback parked in a backend past
    every closing budget — is refused with a clear error rather than run
    against a released resource (RFC §12.10.4).

  See `examples/conference_quickstart.py` (the whole arrangement on the mock
  backend, deterministic), `examples/conference_livekit.py` (the same against
  a real LiveKit SFU — with `ANTHROPIC_API_KEY`, a live AIChannel answers the
  meeting out loud), `examples/conference_ai_meeting.py` (the STT → LLM → TTS
  loop deterministic on the mock: the AI answers a spoken question, BEFORE_TTS
  holds an answer back, and both anti-loop protections are measured),
  `examples/conference_notetaker_on_demand.py`,
  `examples/conference_fault_injection.py`,
  `examples/conference_identity_provenance.py`,
  `examples/conference_recording_result.py`, and RFC §12.10 for the
  contracts.

- **`rename_member()`.** Change what a member is called — never who they
  are (RFC §5.5). `add_member()` on an ACTIVE member is deliberately a
  no-op, so a display name set at join stayed put with no first-class way
  to change it and no event for an interface to react to. The new verb
  updates `display_name` in place, emits the new `PARTICIPANT_UPDATED`
  event and fires `ON_PARTICIPANT_UPDATED`; a rename to the name already
  held is a no-op — no write, no event. `id` and `identity_id` are what
  attribution and correlation stand on, and no rename touches them.

- **`AudioPipeline.process_inbound_stream(stream, frame)`** — a stream-keyed
  inbound entry point that returns what the stages produced instead of fanning
  out to callbacks typed on a `VoiceSession`, plus `release_stream(stream)` for
  the cleanup a lane owes when its track goes away. `process_inbound(session,
  frame)` is unchanged and now shares the same stage chain.

### Changed

- **`RoomKit.close()` finishes the shutdown before it raises.** A channel
  whose `close()` raised used to abort the framework's close on the spot:
  every channel after it kept its media — for a conference channel, a bot
  left sitting in a meeting — and the store, the lock manager and the rest
  were never released. The failure is now collected, every remaining step of
  the shutdown runs, and what was collected is re-raised at the very end as
  an `ExceptionGroup` naming each channel that failed. Raised, not swallowed:
  the channel that failed may still be holding its media, and a close that
  returns cleanly over that turns a logged error into an operational and
  disclosure risk (RFC §12.10.4).

- **The recorder contract bounds its close, and names when a recorder may not
  be released.** RFC §12.11 bounded the writes and said nothing about
  `on_track_removed()` / `on_recording_stop()`, which block the same way and
  which a conference teardown waits for before its bot leaves; it now requires
  those to be bounded too, and a recording the implementation stopped waiting
  for to be reported rather than guessed at. It also states the rule that
  follows from every one of those bounds: `close()` MUST NOT be called while a
  call the implementation gave up on is still running, because freeing the
  context a call is inside is not an error a recorder can return. An
  implementation that cannot settle them leaves the recorder unreleased and
  says so.

- **`FrameworkAwareChannel` is how a channel asks for the framework.**
  `register_channel()` hands session-based channels the `RoomKit` instance they
  were registered with; it used to pick them out with a hardcoded list of three
  concrete classes, briefly by `hasattr(channel, "set_framework")`, and now by
  inheritance from the exported `FrameworkAwareChannel`. The list meant a new
  session-based channel could not be written without editing the framework; the
  attribute check meant any channel owning a method of that name — a wrapper
  around another framework, say — was called with an argument it never asked
  for. A `runtime_checkable` `Protocol` would not have helped: its `isinstance`
  tests the name and nothing else. Inheriting is a declaration, and it puts the
  override's signature under the type checker. `VoiceChannel`,
  `RealtimeVoiceChannel`, `VideoChannel` and `ConferenceChannel` declare it —
  `AudioVideoChannel` and `RealtimeAudioVideoChannel` through their parent.

### Changed — BREAKING

- **Audio pipeline stages now take a stream identity.** `process()` gains a
  required `stream` argument and `reset()` is keyed by it on all seven stage
  interfaces — `VADProvider`, `DenoiserProvider`, `AECProvider`, `AGCProvider`,
  `DTMFDetector`, `DiarizationProvider`, `AudioPostProcessor`.
  `AECProvider.feed_reference()` takes the key too, since each stream owns its
  echo canceller and an unkeyed reference cannot reach the right one.

  ```python
  # before
  def process(self, frame: AudioFrame) -> VADEvent | None: ...
  def reset(self) -> None: ...

  # after
  def process(self, frame: AudioFrame, stream: str) -> VADEvent | None: ...
  def reset(self, stream: str) -> None: ...
  ```

  A `VoiceChannel` holds one `AudioPipeline` for every session on the channel —
  up to ten with `AudioBridge` — so the stages were sharing VAD hangover,
  denoiser history and AGC gain between speakers: one speaker's silence closed
  another's utterance. A conference lane will be another stream through the same
  stages, so the defect was about to become structural rather than incidental.

  `stream` deliberately has **no default**. A default would let a provider
  accept the argument, ignore it, keep compiling, and mix streams silently —
  the exact failure this change removes. Only code that *implements* a stage is
  affected; callers go through `AudioPipeline`, which passes the session id
  itself. `tests/voice/pipeline/stream_conformance.py` ships a check third-party
  implementers can run against their own stage.

- **The resampler takes the stream identity too.** `ResamplerProvider.resample()`
  and `flush()` gain a required `stream` argument, and `reset()` accepts one —
  `reset()` with no argument still clears every stream, which is what a blanket
  pipeline reset asks for.

  ```python
  # before
  def resample(self, frame, target_rate, target_channels, target_width): ...
  def flush(self, target_rate, target_channels, target_width): ...
  def reset(self) -> None: ...

  # after
  def resample(self, frame, target_rate, target_channels, target_width, stream): ...
  def flush(self, target_rate, target_channels, target_width, stream): ...
  def reset(self, stream: str | None = None) -> None: ...
  ```

  The resampler is stage 1 of the inbound pipeline and was the one stage the
  key stopped short of. `SincResamplerProvider` holds a one-frame delay line for
  look-ahead context and keyed it on audio format alone, so in a conference the
  frame buffered for one participant was emitted as the *next* participant's
  output — their voice, and the transcript drawn from it, attributed to someone
  else. `LinearResamplerProvider` and `NumpyResamplerProvider` are stateless and
  were never affected; the argument is required of them for the same reason it
  is required of the stages, so a provider cannot quietly ignore it.

  Resamplers are now covered by
  `tests/voice/pipeline/stream_conformance.py`, which had enumerated the six
  stage directories and left this one out — reporting full coverage over the
  gap that hid the defect.

  `AudioBridge` keys its conversions on the `source -> target` pair rather than
  the destination alone. Both bundled resamplers are stateless so nothing leaked
  today, but a target mixes a frame from every other participant, and those are
  separate continuous signals: keying on the destination would have been the
  same defect one layer down.

### Security

- **`symmetric_rtp` is reachable from the SIP backend.** aiortp has had RTP
  latching all along, but `aiosipua`'s bridge forwarded eleven RTP options to
  it and not that one, so the setting could not be turned on from RoomKit at
  all. `aiosipua` 0.7.1 forwards it; `SIPVoiceBackend(symmetric_rtp=True)` now
  reaches every session it creates — inbound INVITE, outbound `dial()`, and
  the A/V backend's audio and video sessions alike. The `sip` extra floors at
  `aiosipua[rtp]>=0.7.1`.

  Default stays `False`, matching aiortp and aiosipua: latching changes how
  media is addressed mid-call, so it is opted into. What it buys is the
  ordinary NAT fix, plus media redirection stops being followed the moment
  the caller sends anything of its own. What it does **not** buy — and
  `SECURITY.md` said otherwise before this, wrongly — is protection from a
  caller that stays silent: latching only fires on an inbound packet, so an
  offer that advertises a third party and then sends nothing keeps the stream
  aimed there. `rtp_establishment_timeout` bounds that one; authentication
  prevents it.

- **A SIP session is released when its 2xx is never acknowledged.** aiosipua
  gives up on the ACK after 64×T1, drops its own call state and calls
  `on_ack_timeout` — a hook RoomKit never set, so only aiosipua let go while
  our session, its RTP port, its socket and its RTCP task stayed for the life
  of the process. Answering an INVITE and never acknowledging is the cheapest
  way to leak one: no BYE arrives, and the inactivity watchdog has no packet
  to measure against. The handler now reuses the BYE teardown.

- **Webhook signature verification is discoverable.** The helpers were correct
  and invisible: not one of the 164 examples called `verify_signature`, no
  docs page mentioned it, and the parser docstrings said nothing —
  `process_webhook`, described in its own docstring as "the simplest
  integration method", showed an endpoint with no check at all. An opt-in
  control nothing points at is off in practice. The parser docstrings now name
  the header and the method, `process_webhook` states that it does not
  authenticate, `SECURITY.md` gains a per-provider table plus the two mistakes
  that break Twilio verification (the URL must be the public one; the body
  must be the raw bytes), and `examples/webhook_signature_verification.py`
  demonstrates acceptance, a tampered body, a replay to another URL and a
  missing header — with no credentials and no network.

- **`TwilioRCSProvider` can verify its webhooks.** It defined no
  `verify_signature`, so it inherited the base's `NotImplementedError` — an
  RCS endpoint had no way to establish that Twilio sent the request, although
  Twilio signs RCS exactly as it signs SMS and the config already held the
  credentials. The HMAC moves to a shared `providers/twilio/_signature.py`
  (the shape Telnyx already uses for its own pair) and both providers call it.

- **Config secrets stay out of `repr()`** (RFC §17.7). Two pydantic configs
  and nine dataclasses carried a credential that renders in `repr()`, against
  the repo's own convention on both sides. Latent rather than active — nothing
  logs a config object today — but a traceback renders every local it passes.
  `SecretStr` on the two, `field(repr=False)` on the nine, and a test that
  fails if a new config forgets.

- **The AI channel no longer prints its provider's error to the room.** A
  provider failure in the non-streaming tool loop returned the partial answer
  plus the SDK's own error string — status code, request id, model and
  organisation names — as the assistant's message, committed and broadcast.
  The partial answer stays; the reason goes to the log.

- **Dependency floors raised, and the extras are audited.** The resolved
  versions were fine; the declared minimums let a low resolution land on known
  advisories. `mcp>=1.23.0`, `Pillow>=10.3`, `botbuilder-core>=4.17`, and
  `onnxruntime`/`transformers` — previously the only two dependencies with no
  constraint at all — floored at `>=1.20` / `>=4.57`. CI audited the core
  alone, so nothing in any extra was ever looked at; it now audits them too,
  non-blocking, and `make audit` runs the same passes locally.

- **A source's URL no longer carries its token into logs and events**
  (CWE-532). `WebSocketSource.name` and `SSESource.name` returned the URL
  verbatim, and `name` is documented as being for logging and framework
  events — it reached both, plus whatever observability exporter consumes
  them, and the connection log line wrote it at INFO. Authenticating one of
  these endpoints with a token in the query string is ordinary; several
  providers document no other way. The new `telemetry.redaction.safe_url()`
  keeps the scheme, host and path — what makes a log line diagnosable — and
  drops the query string and any `user:pass@`, marking a query that existed so
  a reader can tell "no parameters" from "parameters removed". Unlike
  `redact()` it is not gated on `ROOMKIT_LOG_CONTENT`: a credential is not
  content, and no debugging session justifies logging it. The connection
  itself still dials the real URL.

- **Outbound audio queues are bounded, and the binary inbound path is capped.**
  The realtime channel's per-session send queue and the Twilio backend's write
  queue were both unbounded `asyncio.Queue()`. Neither producer can be
  back-pressured — the realtime one is a synchronous provider callback that
  returns immediately by contract — so a client that stops reading its socket
  makes the queue grow for as long as the provider keeps talking, at roughly
  48 KB/s for 24 kHz PCM16, while the provider goes on billing for audio
  nobody will hear. The existing drop paths (barge-in, mute, teardown) all
  assume an interruption or an ending; a client that simply goes quiet and
  stops reading triggers none of them.

  Both are capped at ~10 s of speech, dropping the *newest* chunk. That is the
  opposite of the conference backlog's policy, deliberately: control items —
  the end-of-response marker transports use to settle playback state, and the
  teardown sentinel — share the realtime queue, so evicting from the head
  could swallow one; and truncating the tail of an utterance is kinder than
  punching a gap in its middle.

  On the inbound side, `MAX_INBOUND_AUDIO_FRAME_BYTES` was applied to the
  base64 path and not to the raw-binary path ten lines above it in the same
  function. Under Starlette/FastAPI `receive_bytes()` has no size limit of its
  own, so that branch accepted whatever an untrusted client sent. It is capped
  now too.

- **One unresponsive WebSocket client no longer freezes its room.** Delivery
  fanned out sequentially with no timeout on the send. A socket that is closed
  raises and gets evicted after a few failures, but one that is merely gone —
  a dropped connection the kernel has not noticed, a client that stopped
  reading — never returns, and the existing eviction counter only ever
  incremented on exceptions, so it was blind to exactly this case. The wait
  was not merely slow for the other clients: broadcast runs under the room
  lock and, unlike the pre-commit phase, is unbounded by design, so the room
  stopped accepting anything at all.

  Sends now run concurrently and each is bounded by `send_timeout` (5 s by
  default, constructor argument), with a timeout counting toward eviction like
  any other failure. Five slow clients now cost what one costs, and a client
  that recovers keeps its place.

- **A WebSocket connection receives its rooms, and only its rooms.**
  `WebSocketChannel` held a flat `{connection_id: send_fn}` registry with no
  room dimension anywhere in its API — not on `register_connection`, not on
  `connect_websocket`. `deliver()` was handed the binding naming the room and
  had nothing to filter against, so it sent every room's events to every
  socket the channel held. A channel shared across conversations leaked them
  into each other, and the leak was durable: the client saw them.

  There was no smaller fix available. Filtering needs data the API did not
  carry; the only alternative was for each integrator's `send_fn` to check
  `event.room_id` itself, which is the undocumented status quo that caused
  this. So the dimension is now explicit.

  **BREAKING:** `room_id` is a required keyword on
  `WebSocketChannel.register_connection()` and `RoomKit.connect_websocket()`.
  Every call site fails loudly and takes one argument to fix. A socket that
  follows several conversations calls `subscribe()` / `unsubscribe()`
  (`kit.subscribe_websocket()` / `kit.unsubscribe_websocket()`) instead of
  opening one socket per room. `deliver()` and `deliver_stream()` both scope
  to `binding.room_id`, so a connection the channel cannot place receives
  nothing. `Channel.supports_streaming_delivery_for(room_id)` joins the ABC —
  defaulting to the channel-wide property, overridden by `WebSocketChannel` —
  so a room whose clients cannot stream no longer takes the streaming path
  only to fall back at the end.

- **The inbound router no longer guesses which room a message belongs to.** It
  tried the channel binding first and returned the first match, so a channel
  bound to several active rooms sent the message to whichever the store
  happened to hand back — the oldest binding in the in-memory store, and
  whatever the planner chose in Postgres, where the query had no `ORDER BY` at
  all. Two deployments of the same code could route differently. This is a
  durable cross-room disclosure, not a transient one: the message is stored in
  the wrong room, broadcast to that room's channels, and read back as context
  by that room's agent. It is also not exotic — `delegate(share_channels=...)`
  creates exactly this shape, as does a channel re-attached after its room
  closed.

  The order now follows RFC §10.4: the sender's own latest room first (a
  binding identifies the pipe, a participant identifies the conversation),
  then a channel bound to *exactly one* active room. More than one, and the
  router returns null with a warning naming the channel — a new room is
  recoverable, a message in someone else's conversation is not. Both stores
  order their candidates the same way, so the answer no longer depends on the
  backend. `ConversationStore` gains `find_room_ids_by_channel()`, non-abstract
  with a fallback, so existing third-party stores keep working.

- **A closed room refuses new events.** `RoomStatus.CLOSED` and `ARCHIVED` were
  enforced nowhere: the inbound router skipped non-ACTIVE rooms, which made
  implicit routing look safe, but every path that *names* the room went
  straight through — `process_inbound(room_id=...)`, `send_event()`, and the
  framework's own re-injection on the delegation path. A closed room went on
  storing events, broadcasting them and letting its agent reply. This was not
  even a spec violation to point at: RFC §5.1 said "no new events accepted" in
  a table cell with no RFC 2119 keyword, and no step of the normative §10.1
  pipeline enforced it. The spec was fixed first, then this.

  The check sits at the one point every entry converges on, under the room
  lock — `close_room()` takes the same lock, so an earlier answer could be
  stale by commit time. Nothing is written for a refused event, not even a
  `BLOCKED` record, since an audit entry appended to a closed room is exactly
  what the status forbids. The inbound path returns
  `InboundResult(blocked=True, reason="room_closed")`; `send_event()` raises
  the new `RoomClosedError`, because its contract is to return the committed
  event and handing back one marked `DELIVERED` for a write that never
  happened is worse than raising — the same reason it already raises for a
  room that does not exist. PAUSED still accepts events, closing a room stays
  observable through `ON_ROOM_CLOSED`, and history stays readable.

- **A binding is no longer widened by accident.** Two paths handed out more
  access than the integrator had granted, both by letting a default fill a gap.
  Sharing a channel into a delegated room (`delegate(share_channels=...)`)
  copied the parent binding's category and metadata but not its `access`,
  `visibility` or `muted` — so a read-only observer, or a muted one, became a
  full participant in the child room. And the inbound pipeline's convenience
  auto-attach did not distinguish a channel that had never been bound from one
  the integrator had deliberately `detach_channel()`ed: the next message naming
  that room re-attached it at `READ_WRITE`, undoing the revocation. Both now
  follow RFC §7.5-6 and §7.5-7. Re-granting access remains available, as an
  explicit `attach_channel()` — which is the point.

- **SIP auth and trace hygiene.** Three small things in the same area. The
  digest comparison used `!=`, the only credential comparison in the codebase
  that was not constant-time; it is `hmac.compare_digest` now. (No timing
  oracle was reachable — the nonce is single-use, so each attempt is measured
  against a different expected value — but that is a poor reason to be the
  exception.) The nonce table was rebuilt on *every* challenge, making the
  sweep quadratic in the challenge rate, which is precisely what an
  unauthenticated INVITE flood drives for free; it now sweeps once the table
  is large enough to be worth walking. And `ProtocolTrace` carried the raw
  INVITE including `Authorization: Digest … response="<md5>"` — not replayable,
  the nonce having been consumed, but an offline dictionary attack on the
  password when read beside the username, realm and nonce in the same header.
  `response` is masked; everything else in the header stays, because the trace
  exists to debug authentication.

- **A SIP offer can no longer point the media stream anywhere it likes.** The
  RTP destination was taken from the offer's `c=`/`m=` lines and applied with
  no validation at all — `0.0.0.0`, port 0, loopback and multicast included —
  and since symmetric RTP is off, nothing downstream ever corrected it by
  observing where packets actually arrived from. `is_usable_rtp_address()` now
  gates every point where an offer moves the destination, in the audio backend
  and the A/V one. A hold offer no longer redirects the stream; it leaves it
  where it was.

  Note what this does *not* claim: an address that is merely wrong rather than
  impossible — a third party's, which turns the call into an RTP reflector
  aimed at them — is not detectable here, because a caller behind NAT
  legitimately advertises an address its packets do not come from. Symmetric
  RTP in the transport is the defence for that, and it belongs to aiortp.

  The re-INVITE shortcut is narrowed to the case it was written for. It exists
  because re-INVITEs on *outbound* calls reach `on_invite` rather than
  `on_reinvite`; it was matching on Call-ID alone, so an out-of-dialog INVITE
  that reused a known Call-ID had its SDP applied to the existing session
  without ever being authenticated. Inbound sessions now fall through to the
  normal path, where the UAS has already declined to treat the request as
  in-dialog and the session-id claim refuses it.

- **A SIP call that is answered and then says nothing no longer holds its port
  forever.** The RTP watchdog only judged sessions that had received at least
  one packet — it measures time since the last one, and there had been none —
  so taking the 200 OK and staying silent kept an RTP port, a UDP socket and a
  periodic RTCP task until the process exited. `SIPSessionState` had no
  creation timestamp, so no establishment timer was even expressible. It now
  records `created_at`, and `rtp_establishment_timeout` (default 60 s) reaps a
  session that never received RTP.

  Two more bounds close the same drain. `max_sessions` (default 0, off) answers
  `503` past the cap instead of allocating into an exhausted pool of 5000
  ports. And the INVITE task now carries a done-callback: the `RuntimeError`
  raised when the pool *is* exhausted used to surface only as asyncio's "Task
  exception was never retrieved" at collection time — no call id, nothing in
  the SIP log, and no final response to the caller, who waited out its own
  timer.

- **A SIP caller can no longer seize a live session by naming its
  `X-Session-ID`.** The session id came from the caller's `X-Session-ID`
  header and the backend stored it with `self._session_states[session.id] =
  state`, overwriting whatever was there. Since `send_audio`, `send_dtmf`,
  `disconnect` and the voice channel's own room binding all resolve on that
  id alone, a second INVITE naming a live session took over the first call's
  audio path: the agent's synthesised speech went to the new caller's RTP
  address and the new caller's audio arrived in the victim's room. The
  displaced session became unreachable — its RTP port, socket and periodic
  RTCP task were never freed, because cleanup pops by id and the id now
  pointed elsewhere — and its `_call_to_session` entry survived, so a BYE on
  the old dialog tore down the *current* call.

  A colliding id is now refused with `486 Busy Here` before a port is
  allocated or a 200 OK is sent, and the claim is held across the awaits in
  setup so two INVITEs racing on one id cannot both pass. A legitimate PBX
  does not reuse a live id, and nothing in an INVITE distinguishes a confused
  one from a hostile one. `SIPVideoBackend` takes the same claim on its A/V
  path. Ending a call still frees its id.

- **An `m=video` line no longer buys a SIP caller past authentication.**
  `SIPVideoBackend` overrides `_handle_invite` to dispatch offers carrying
  video to its own A/V path — and that path ran no digest challenge and no
  invite filter. An operator who had configured `auth_users` believed the
  port was authenticated; adding a video section to the SDP skipped the check
  entirely, along with whatever tenant routing the invite filter enforced.
  The gate is now a single `_authorize_invite()` that both dispatch branches
  call before any port is allocated.

- **Edit and delete authorization no longer trusts the payload that requests
  it.** The author check only ran when `edit_source` was `None`/`"sender"` or
  `delete_type` was `SENDER`, so any other value skipped it: on a channel
  whose remote party controls the content — WebSocket, most transports — a
  participant could rewrite or delete anyone's messages, the AI's included,
  by sending `edit_source="admin"`. `delete_type=ADMIN` and `SYSTEM` were
  likewise accepted with no authority check at all, contrary to RFC §10.3.

  Authorization is now fail-closed: anything outside the RFC's
  `"sender" | "system"` vocabulary is unprivileged and still requires the
  sender to be the original author, `ADMIN` requires a verified `OWNER` role
  on the room roster, and `SYSTEM` requires the event to originate from a
  system channel. `EditContent.edit_source` stays a `str`, so callers using
  the documented values are unaffected; moderation that legitimately outranks
  the roster belongs on `update_event`/`delete_event`, where the host owns
  authorization.

- **`WebTransportBackend` can be authenticated, and refuses to be anonymous by
  accident.** The backend accepted any client that reached its UDP port: the
  CONNECT handshake was checked for its path and nothing else, and the HTTP
  headers — the only place a WebTransport client can put a credential — were
  read and thrown away before the application ever saw them. With the default
  `0.0.0.0` bind, that is an open voice endpoint whose sessions bill STT and
  TTS to the operator.

  `authenticate` now receives a `WebTransportConnectRequest` carrying the
  path and the full header block, and returns metadata to accept or `None` to
  reject with 403; the metadata reaches the session factory through
  `auth_context`, as it already did for WebSocket and WebRTC. `start()` raises
  unless either `authenticate` or `allow_anonymous=True` is given, mirroring
  the guard the WebRTC offer endpoint has had since 0.28.0 — existing
  deployments that meant to be open say so in one argument.

- **The Teams bot example validates its webhook JWT.** `examples/teams_bot.py`
  is the only example that stands up a real HTTP endpoint — on `0.0.0.0:3978`
  — and it read the request body without ever looking at the `Authorization`
  header, so anyone able to reach it could impersonate any Teams user. It now
  routes the activity through `provider.process_inbound(payload, auth_header,
  on_turn)` and answers 401 on `PermissionError`. Worth copying rather than
  the shape it had: the validation helpers existed all along, no example
  showed them.

### Fixed

- **Sync hooks fail closed, and can rewrite every payload they are given.**
  `HookResult.event` was typed as a `RoomEvent`, but only one of the nine sync
  triggers passes one — the rest carry a string, a media frame or their own
  event type, so a hook returning `action="modify"` on `BEFORE_TTS` or
  `ON_TRANSCRIPTION` raised inside the engine, which logged the error and
  carried on with the *original* payload: a redaction hook published exactly
  what it existed to suppress. The field now accepts the trigger's own
  payload, `HookResult.modify()` takes it too, and each reader substitutes a
  rewritten value of the type it expects.

  Failing is no longer a way through either. On the triggers whose payload a
  hook may exist to withhold, every way a hook can fail to produce a usable
  result — raising, exceeding its timeout, returning something that is not a
  `HookResult`, returning a rewrite of a type the consumer cannot use — now
  blocks the payload instead of letting it pass unmodified, and a rewrite to
  an empty string reads as a rewrite rather than as no modification.
  Everywhere else a raising hook stays non-fatal, so a broken hook still
  cannot take a room down.

- **A refused attach no longer destroys the attachment it failed to replace.**
  `attach_channel()` writes the binding before asking the channel to establish
  it, and rolled back by *deleting* that binding. Over a live attachment that
  is the wrong inverse: the second attach replaced the first's binding, and a
  channel that refuses the new one has said nothing about the old — it is still
  attached, its conference still running, its bot still in the meeting. The
  delete took the room's only handle on that away, so `detach_channel()` found
  nothing to remove, returned false, and the attachment ran on with nothing able
  to reach it. The previous binding is now restored rather than removed; a first
  attach, which had nothing to restore, still leaves nothing behind.

- **A detach is announced even when the channel raises on its way out.** By the
  time `on_room_detached()` runs, the binding is gone and `CHANNEL_DETACHED` is
  indexed — the detach has happened as far as the room is concerned — but an
  exception from the channel skipped `ON_CHANNEL_DETACHED` and
  `room_channel_detached` entirely, leaving every observer believing the channel
  was still attached. Both now fire before the error is re-raised at the caller.
  An announcement that fails as well is logged rather than allowed to displace
  the channel's own failure, which is the one the caller is owed.

- **A sender the room has already named is not resolved again.** Every
  conference utterance re-entered the inbound pipeline as a message whose
  `sender_id` was the speaker's backend identity, and identity resolution ran on
  it — per sentence, per participant, for the whole meeting. No resolver can
  match a framework identifier, so the answer was `UNKNOWN` every time. That is
  a lookup per sentence where a resolver reads a CRM, and worse than noise: the
  standard `ON_IDENTITY_UNKNOWN` hook that refuses unknown senders then blocked
  every transcript of a participant the framework had identified when they
  dialled in — silently, since a blocked event leaves nothing in the room.

  Two senders now skip resolution (RFC §11.6). One the room has already marked
  `IDENTIFIED`, read off the participants the pipeline had already loaded — the
  answer is on the roster, and `identity_id` carries it. And one arriving on a
  channel that declares `sender_is_participant`, meaning its `sender_id` is a
  room `Participant.id` rather than an address: `ConferenceChannel` sets it,
  because a conference resolves when a participant *arrives* and the address its
  provider attached is there to resolve (§12.10.2), and speaking again asks
  nothing new. This also covers the participant the framework minted access for,
  whose id no resolver knows any better.

  `PENDING` and `AMBIGUOUS` participants are deliberately still resolved: a
  participant the room *has* is not one it has *identified*, a resolver may
  still be what settles it, and a hook may still want to challenge or refuse.
  Nothing changes for a genuinely unknown sender on a text channel.

- **What is typed at a terminal is not an address, and is no longer resolved as
  one.** `CLIChannel.run()` names the human at the keyboard — its `sender_id`
  defaults to `"user"` and its own documentation calls it a Participant ID — and
  that value went straight into the inbound pipeline as something to look up. No
  resolver matches it, so every line came back `UNKNOWN`, `ON_IDENTITY_UNKNOWN`
  fired per line, and the standard hook that refuses unknown senders discarded
  everything typed, silently: a blocked event leaves nothing in the room.
  `ensure_participant()` did not help, since the record it creates is `PENDING`
  and a `PENDING` sender on a text channel stays deliberately resolvable.

  `CLIChannel` now declares `sender_is_participant`, as `ConferenceChannel`
  does: its `sender_id` is a room `Participant.id` rather than an address, so
  resolution is skipped (RFC §11.6, case 1). Nothing is lost — the resolution
  removed is one that could never answer.

  The declaration belongs to a channel whose `sender_id` the framework itself
  chooses. `WebSocketChannel` and `VoiceChannel` deliberately keep the default:
  what reaches them comes from the integrator or from the backend — a SIP
  session id for one call, a caller number for the next — and excluding them is
  an integrator's call, made with `identity_channel_types` (RFC §11.4).

- **`read_stored_result` exists in the very turn that evicted.** A tool result
  crossing the eviction threshold mid-loop replaces itself with a preview whose
  instruction is to page the full output back with `read_stored_result` — but
  the definition was only injected when the *next* inbound event rebuilt the
  context, and every round of both tool loops re-filters from the turn's frozen
  tool snapshot. So the tool did not exist exactly where its preview recommended
  it: the model burned rounds hunting for it through its discovery tools, and a
  one-shot automation run (webhook, schedule) has no next event at all — its
  evicted content was unreachable for the whole run, and a model was observed
  guessing at it instead, wrongly. The definition is now injected per round as
  soon as the store holds anything, deduped against a context that already
  carries it; a turn with nothing evicted is untouched.

## [0.37.1] — 2026-07-24

### Fixed

- **`buzz` extra now requires buzzkit 0.1.4.** `BuzzHuddleBackend` drives the
  huddle client with `paced=False` so RoomKit's `OutboundAudioPacer` owns the
  outbound clock — an argument that only exists in buzzkit 0.1.4, so the
  0.37.0 floor (`buzzkit>=0.1.3`) allowed installs whose huddle sessions fail
  with a `TypeError` on connect. buzzkit 0.1.4 also runs all WebSocket I/O on
  a dedicated thread and drops late frames instead of bursting them, curing
  choppy huddle audio under event-loop load.

## [0.37.0] — 2026-07-24

### Added

- **ACP intelligence channel for external coding agents.** The new
  `ACPChannel` makes RoomKit an ACP client over stdio (stable ACP v1): it starts
  one agent subprocess lazily, maps each Room to an isolated ACP session,
  serializes prompts within a Room, and lets different Rooms run concurrently.
  Agent text and reasoning stream as they arrive; tool lifecycle, plan, usage,
  and progress updates are exposed through RoomKit stream/realtime events.
  Permission requests pass through `ExternalToolHandler` — and therefore the
  existing `BEFORE_TOOL_USE` / `ON_TOOL_USE` hooks — with a deny-by-default
  policy when no handler is configured. Sessions can be inspected, cancelled,
  or closed explicitly, and subprocess/session cleanup is handled by
  `RoomKit.close()`. Install the optional `roomkit[acp]` extra.
- **Claude Code over ACP, through the existing CLI channel.** The new
  `examples/acp_claude_code.py` wires `CLIChannel` to `ACPChannel`, runs the
  official Claude Agent ACP adapter, streams visible reasoning and tool
  activity, asks for each tool permission in the terminal, and scopes the
  coding agent to a selected workspace.
- **Progressive Markdown and tool activity in `CLIChannel`.** Set
  `markdown=True` to render both complete and streaming agent responses with
  Rich via the new `roomkit[console]` extra. The live document refreshes on
  every real text delta instead of waiting for turn completion, while
  `show_thinking=True` renders reasoning deltas and tool start/end events remain
  visible inline. Plain terminal output also now shows tool names, arguments,
  completion status, and duration.
- **Buzz transport channel over Nostr.** `ChannelType.BUZZ`, `BuzzChannel`,
  `BuzzConfig`, `BuzzProvider`, `MockBuzzProvider`, and `BuzzRelaySource`
  provide bidirectional Buzz messaging through one shared `buzzkit` client.
  The source authenticates with NIP-42, converts relay events into idempotent
  RoomKit messages, filters the agent's own events by default, reconnects with
  backoff, self-joins its NIP-29 channel as a bot, and publishes an online
  presence heartbeat. `BuzzConfig.auth_tag` carries an optional NIP-OA owner
  attestation, while custom `kinds` and parsers allow non-chat subscriptions.
  Install the isolated `roomkit[buzz]` extra.
- **Buzz Huddles realtime voice transport.** `BuzzHuddleBackend` bridges
  `buzzkit.HuddleClient` Opus sessions to RoomKit's realtime voice pipeline,
  including streaming 48 kHz resampling, outbound pacing, barge-in, silence
  fill for server-side VAD, roster metadata, and deterministic disconnect
  reasons. With `end_when_alone=True` (the default), the transport leaves when
  the last remote peer is gone instead of keeping the huddle alive by itself.
  `BuzzHuddleWatcher` owns the full announcement-to-call lifecycle: it watches
  kind-48100 announcements through an auto-restarting `BuzzRelaySource`, dials
  one huddle at a time, rejoins after connection loss, and waits for the next
  announcement after a normal end. `RealtimeVoiceChannel.transport` exposes
  the backend for this and other transport-level orchestration.
- **Replay-safe programmatic publishing.** `RoomKit.send_event()` accepts an
  optional `idempotency_key`. Replaying the same key in one Room is blocked by
  the existing locked idempotency pipeline and unique store index, preventing a
  second persistence and re-broadcast while preserving the previous behaviour
  when the key is omitted.

### Changed

- **Fixed-rate voice backends share a stateful streaming resampler.** Buzz
  Huddles and Twilio use the same low-latency soxr QQ implementation, with the
  existing pure-Python linear fallback when soxr is unavailable.
- **`fastrtc` extra caps NumPy below 2.5.** Numba (pulled in through librosa)
  does not support NumPy 2.5 yet, so the extra now declares
  `numpy>=1.26,<2.5`.

### Fixed

- **Idle silence no longer splices into bursty voice responses.**
  `OutboundAudioPacer(fill_with_silence_when_idle=True)` previously inserted a
  silence frame after a short provider lull even while its jitter buffer was
  still ahead of wall clock, permanently displacing subsequent speech and
  producing chopped audio. Silence is now emitted only after the pacer has
  actually fallen behind.

## [0.36.0] — 2026-07-20

### Fixed

- **Muting an intelligence channel silences its streaming voice, not just its
  events.** A muted channel's non-streaming `response_events` were suppressed in
  the router, but a streaming response was captured and returned *before* the
  mute check — so a muted streaming provider still replied. Once 0.35.0 made
  `send_event` deliver those streams, a directly-injected message (e.g. a REST
  team-channel post carrying no `@`-mention) woke the muted agent. The router
  now drops a muted channel's stream without iterating it — no provider
  round-trip, the reply is never generated — matching the `response_events`
  suppression and the RFC contract "muting silences the voice, not the brain".

## [0.35.0] — 2026-07-20

### Fixed

- **`send_event` delivers a streaming AI response instead of dropping it.** A
  directly-injected event that woke a streaming intelligence channel had its
  response generated and then silently discarded — `send_event` ran the locked
  pipeline but omitted the post-lock streaming-response drain the inbound path
  performs. It now consumes `pending_streams` like `process_inbound`, so the
  reply is persisted and delivered. Non-streaming providers were unaffected;
  injections that don't wake an agent are a no-op.
- **`regenerate_response` fires ON_ERROR on a non-streaming failure.** A failed
  regeneration with a non-streaming provider surfaced on `InboundResult.error`
  but rendered no error card; it now fires `ON_ERROR` too (parity with the
  inbound path). The streaming path already fires its own, so the two never
  double up.
- **Voice failures surface as events, not just log lines.** A TTS provider
  without `synthesize_stream` now emits `tts_error` (not only an ERROR log); a
  continuous-mode STT routing failure emits `stt_error` like the VAD path; and
  the Deepgram stream raises on an SDK `on_error` so the consumer marks the
  stream failed and reconnects instead of seeing a clean, empty end.

## [0.34.0] — 2026-07-20

### Changed

- **A headless turn failure logs once, at the host's layer.** When there is no
  streaming target — a one-shot programmatic caller that reads
  `InboundResult.error` and logs it itself — a `ProviderError` now logs at DEBUG
  in the framework instead of WARNING, so the framework line no longer
  duplicates the caller's WARNING for the same incident. With a streaming target
  (interactive) the framework WARNING is unchanged; unexpected errors still keep
  their traceback.

## [0.33.0] — 2026-07-20

### Added

- **Turn failures reach the caller via `InboundResult.error`.** When an
  intelligence channel's response fails — while consuming a streaming response,
  or raised by a non-streaming provider (`generate`) — `process_inbound` now
  returns the exception on the new `InboundResult.error` field (cause chain
  intact), in addition to firing `ON_ERROR`. A headless caller with no streaming
  target — which previously saw the failure fire `ON_ERROR` and then vanish,
  leaving `process_inbound` to return an empty result — can now observe and
  classify it. Both the streaming and non-streaming paths surface identically
  (`BroadcastResult.errors_exc` carries the live exception per channel, not just
  `str(exc)`). Interactive callers ignore the field; the `ON_ERROR` error-card
  behaviour is unchanged.
- **`regenerate_response` surfaces the same error.** A failure while
  regenerating a turn is returned on the result's `error` instead of a
  success-looking `InboundResult` — it used to discard the stream error and
  never read the broadcast error.

### Changed

- **Turn-failure logging at the right verbosity.** A `ProviderError` (backend
  unreachable, 5xx, timeout, context overflow) — an expected transient now
  returned to the caller and delivered to `ON_ERROR` — is logged as a single
  WARNING line without a traceback, instead of a full `logger.exception` ERROR.
  Applies to both the streaming consumption path (`_handle_streaming_response`)
  and the broadcast path (`event_router`, non-streaming generation). Any other
  exception is unexpected and keeps its traceback.

## [0.32.0] — 2026-07-19

### Added

- **Atomic room-metadata patch API.** New
  `ConversationStore.patch_room_metadata(room_id, patch, *, unset=())` merges
  keys into a room's metadata (optionally removing `unset` keys first) without
  rewriting the whole row. `update_room` is a full-row read-modify-write: a
  caller holding a stale `Room` silently clobbers concurrent metadata patches
  and regresses the `event_count` / `latest_index` / `timers` counters
  maintained by `commit_event`. The base implementation is a documented
  non-atomic fallback (sufficient for `InMemoryStore`); the Postgres store
  overrides it with a single `(metadata - unset) || patch` JSONB update.
  Returns the updated `Room`, or `None` when the room does not exist.

## [0.31.0] — 2026-07-19

### Added

- **Direct event-mutation APIs with hooks.** New `RoomKit.update_event()` and
  `RoomKit.delete_event()` (EventOpsMixin) let a host application mutate a
  persisted event — replace content/source/metadata, or hard-delete a thread
  root with its replies (`ConversationStore.delete_event`, implemented for
  memory and Postgres) — under the room lock, with authorization owned by the
  caller. Both fire the new hook triggers `ON_EVENT_UPDATED` /
  `ON_EVENT_DELETED` after the lock is released.
- **RFC §10.3 inbound edits/deletes fire the mutation triggers.** The
  `_apply_edit_delete_state` path (channel-originated EDIT/DELETE events) now
  fires `ON_EVENT_UPDATED` / `ON_EVENT_DELETED` with the mutated target, so
  observers (e.g. denormalized-projection maintainers) see every stored-state
  change regardless of origin. Firings are deferred until the room lock is
  released, like AFTER_BROADCAST.

### Fixed

- **Postgres `update_event` now persists the `source_*` columns.** The UPDATE
  omitted `source_channel_id/type`, `source_participant_id`, `source_provider`
  and `source_extra`, silently dropping sender reclassification on Postgres
  (the in-memory store, which replaces the whole object, honored it).

## [0.30.0] — 2026-07-16

### Changed

- **Thread-reply pagination reads from a composite index.** The query filters
  `events` by `parent_event_id` then reads forward `ORDER BY index`; the
  single-column `idx_events_parent` forced a sort of the whole thread on every
  page. The index now carries `(parent_event_id, index)` — the leading column
  still serves plain `parent_event_id` lookups. It ships under a new name,
  `idx_events_parent_index`, so `init()`'s additive `CREATE INDEX IF NOT EXISTS`
  actually creates it on databases that predate the composite (reusing the old
  name would have no-op'd against the existing single-column index).

### Added

- `PostgresStore.drop_legacy_parent_index(dry_run=True)` — opt-in migration that
  removes the now-redundant single-column `idx_events_parent` on databases that
  predate the composite. `init()` is additive and never drops, so this is the
  explicit path to reclaim the old index. Idempotent; dry-run by default.

## [0.29.0] — 2026-07-13

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

[Unreleased]: https://github.com/roomkit-live/roomkit/compare/v0.46.0...HEAD
[0.46.0]: https://github.com/roomkit-live/roomkit/compare/v0.45.0...v0.46.0
[0.45.0]: https://github.com/roomkit-live/roomkit/compare/v0.44.0...v0.45.0
[0.44.0]: https://github.com/roomkit-live/roomkit/compare/v0.43.0...v0.44.0
[0.43.0]: https://github.com/roomkit-live/roomkit/compare/v0.42.1...v0.43.0
[0.42.1]: https://github.com/roomkit-live/roomkit/compare/v0.42.0...v0.42.1
[0.42.0]: https://github.com/roomkit-live/roomkit/compare/v0.41.4...v0.42.0
[0.41.4]: https://github.com/roomkit-live/roomkit/compare/v0.41.3...v0.41.4
[0.41.3]: https://github.com/roomkit-live/roomkit/compare/v0.41.2...v0.41.3
[0.41.2]: https://github.com/roomkit-live/roomkit/compare/v0.41.1...v0.41.2
[0.41.1]: https://github.com/roomkit-live/roomkit/compare/v0.41.0...v0.41.1
[0.41.0]: https://github.com/roomkit-live/roomkit/compare/v0.40.0...v0.41.0
[0.40.0]: https://github.com/roomkit-live/roomkit/compare/v0.39.0...v0.40.0
[0.39.0]: https://github.com/roomkit-live/roomkit/compare/v0.38.0...v0.39.0
[0.38.0]: https://github.com/roomkit-live/roomkit/compare/v0.37.1...v0.38.0
[0.37.1]: https://github.com/roomkit-live/roomkit/compare/v0.37.0...v0.37.1
[0.37.0]: https://github.com/roomkit-live/roomkit/compare/v0.36.0...v0.37.0
[0.36.0]: https://github.com/roomkit-live/roomkit/compare/v0.35.0...v0.36.0
[0.35.0]: https://github.com/roomkit-live/roomkit/compare/v0.34.0...v0.35.0
[0.34.0]: https://github.com/roomkit-live/roomkit/compare/v0.33.0...v0.34.0
[0.33.0]: https://github.com/roomkit-live/roomkit/compare/v0.32.0...v0.33.0
[0.32.0]: https://github.com/roomkit-live/roomkit/compare/v0.31.0...v0.32.0
[0.31.0]: https://github.com/roomkit-live/roomkit/compare/v0.30.0...v0.31.0
[0.30.0]: https://github.com/roomkit-live/roomkit/compare/v0.29.0...v0.30.0
[0.29.0]: https://github.com/roomkit-live/roomkit/compare/v0.28.0...v0.29.0
[0.28.0]: https://github.com/roomkit-live/roomkit/compare/v0.27.0...v0.28.0
[0.27.0]: https://github.com/roomkit-live/roomkit/compare/v0.26.0...v0.27.0
[0.26.0]: https://github.com/roomkit-live/roomkit/compare/v0.25.0...v0.26.0
[0.25.0]: https://github.com/roomkit-live/roomkit/compare/v0.24.0...v0.25.0
[0.24.0]: https://github.com/roomkit-live/roomkit/compare/v0.23.0...v0.24.0
[0.23.0]: https://github.com/roomkit-live/roomkit/compare/v0.22.0...v0.23.0
[0.22.0]: https://github.com/roomkit-live/roomkit/compare/v0.20.0...v0.22.0
[0.20.0]: https://github.com/roomkit-live/roomkit/compare/v0.19.0...v0.20.0
[0.19.0]: https://github.com/roomkit-live/roomkit/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/roomkit-live/roomkit/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/roomkit-live/roomkit/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/roomkit-live/roomkit/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/roomkit-live/roomkit/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/roomkit-live/roomkit/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/roomkit-live/roomkit/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/roomkit-live/roomkit/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/roomkit-live/roomkit/compare/v0.10.0...v0.11.0
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
