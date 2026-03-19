# API Redesign — Implementation Plan — COMPLETED

## Goal

Simplify the voice API surface to eliminate boilerplate and fix bugs. After this
work, a local-audio voice app is 6 lines of setup, SIP apps use explicit
`join/leave`, and recording "just works" when a room has recorders.

**Status: ALL SECTIONS COMPLETE** ✅

---

## Target API

### Pattern A — Local backend (auto-start)

```python
kit = RoomKit()

voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend, pipeline=pipeline_config)
ai = AIChannel("ai", provider=ai_provider, system_prompt=system_prompt)

kit.register_channel(voice)
kit.register_channel(ai)

room = await kit.create_room(
    room_id="voice-demo",
    recorders=[
        RoomRecorderBinding(
            recorder=PyAVMediaRecorder(),
            config=MediaRecordingConfig(storage="./recordings"),
        ),
    ],
)

await kit.attach_channel("voice-demo", "ai", category=ChannelCategory.INTELLIGENCE)
# Hooks registered here...
await kit.attach_channel("voice-demo", "voice")  # auto-starts session

# That's it. Voice is live, recording captures everything.
await stop.wait()
await kit.close()
```

### Pattern B — SIP / push backend (explicit join/leave)

```python
# Setup (once)
room = await kit.create_room(room_id="support")
await kit.attach_channel("support", "voice")
await kit.attach_channel("support", "ai", category=ChannelCategory.INTELLIGENCE)

# Push — SIP caller joins the room
@backend.on_call
async def handle_call(session):
    await kit.join("support", "voice", session=session)

# Pull — local mic joins the room
session = await kit.join("voice-demo", "voice")

# Cross-transport bridge
await kit.join("bridge", "voice", session=sip_session, backend=sip_backend)

# Leave
await kit.leave(session)
```

---

## Implementation Summary

### 1. ~~TTS Bug Fix~~ — NOT A BUG

Investigated in depth. Both `connect_voice()` and `bind_voice_session()`
correctly deliver TTS through the store-binding path. Existing integration
tests cover this. No library changes needed.

### 2. Auto-start on `attach_channel` — DONE ✅

Commit: `c3d3cf2`

- Added `VoiceBackend.auto_connect` property (default `False`)
- `LocalAudioBackend` overrides → `True`
- `attach_channel` calls `_post_attach` AFTER lock release
- `_post_attach` calls `join()` for auto_connect backends
- 4 new tests in `tests/test_auto_connect.py`

### 3. `kit.join()` / `kit.leave()` — DONE ✅

Commit: `7a30c5d`, `8f3b359`

- `join(room_id, channel_id)` — pull model (creates session)
- `join(room_id, channel_id, session=session)` — push model (binds existing)
- `join(..., backend=other_backend)` — cross-transport bridging
- `leave(session)` — stops, unbinds, disconnects, cleans recording
- Dispatches by channel type (VoiceChannel, VideoChannel)
- Old methods are deprecation wrappers with `DeprecationWarning`
- pytest filterwarnings added for existing tests

### 4. Recording simplification — DONE ✅

Commit: `fd563a8`

- Opt-out recording: rooms with recorders auto-record all channels
- `ChannelRecordingConfig` only needed to disable (e.g. `audio=False`)
- Outbound TTS recording: ring buffer mixing into single track
- Thread-safe with 5s cap, sample-by-sample clamping
- Fixed TOCTOU race, added opt-out guard to legacy backend path
- 4 new tests, updated 3 doc pages

### 5. SIP example cleanup — DONE ✅

Commits: `6bb5aa3`, `45e085f`, `346b541`

- All SIP examples use `kit.join(room_id, channel_id, session=session)`
- Disconnect handlers use `kit.leave(session)`
- Rooms created once at startup, not per-call

### 6. Example + docs updates — DONE ✅

Commits: `6bb5aa3`, `45e085f`, `8f3b359`, `346b541`

- **35 examples migrated** — zero old patterns remain
- 12 LocalAudio examples → auto-start via `attach_channel`
- 7 SIP examples → `join`/`leave`
- 2 bridge examples → `join(backend=)` for cross-transport
- 14 other examples → `join`/`leave`
- **18 doc pages updated** in roomkit-docs
- CHANGELOG.md updated with all 5 new features
- Net removal of ~350 lines of boilerplate

---

## Commits (chronological)

| Commit | Description |
|--------|-------------|
| `fd563a8` | feat(recording): opt-out + outbound TTS capture |
| `8cf5cb4` | docs: recording changelog |
| `7a30c5d` | feat: kit.join() / kit.leave() |
| `c3d3cf2` | feat: auto-start on attach_channel |
| `6bb5aa3` | refactor: migrate 25 examples |
| `3581e4a` | docs: join/leave changelog |
| `45e085f` | refactor: migrate 10 more examples |
| `8f3b359` | feat: backend= param for cross-transport bridging |
| `08c4bf5` | docs: backend= in changelog |
| `346b541` | refactor: migrate last 5 examples |

---

## Remaining / out of scope

- `RealtimeVoiceChannel` not supported by `join()` — uses
  `connect_realtime_voice` / `disconnect_realtime_voice` directly
  (different session model with WebSocket connection param)
- `websocket_streaming.py` uses `ChannelBinding` for WebSocket channel
  (not voice) — intentionally unchanged
- Two bridge examples keep `bind_session(backend=)` because `join(backend=)`
  was added to support this — now migrated
