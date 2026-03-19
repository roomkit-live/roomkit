# xAI Grok Realtime Provider

## Plan

xAI's Realtime API uses the same wire protocol as OpenAI Realtime (WebSocket, same event names) but with a simpler/flatter session config format (no nested `audio.input`/`audio.output`). Key xAI-specific features: `web_search`/`x_search` native tools, `grok-2-audio` transcription model, voices `eve`/`ara`/`rex`/`sal`/`leo`.

## Tasks

- [x] Create `src/roomkit/providers/xai/__init__.py` — exports
- [x] Create `src/roomkit/providers/xai/config.py` — `XAIRealtimeConfig` (Pydantic)
- [x] Create `src/roomkit/providers/xai/realtime.py` — `XAIRealtimeProvider` (RealtimeVoiceProvider)
- [x] Add lazy loaders in `src/roomkit/voice/__init__.py`
- [x] Export `XAIRealtimeConfig` + `XAIRealtimeProvider` from `src/roomkit/__init__.py`
- [x] Create `tests/test_realtime_xai.py` — 38 unit tests, all passing
- [x] Create `examples/realtime_voice_local_xai.py` — runnable example with local mic/speakers

## Review

All 38 tests pass. Lint clean. Type check clean. Top-level import verified.
