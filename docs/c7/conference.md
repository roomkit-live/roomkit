# Conference (Multi-Party SFU)

ConferenceChannel bridges an external SFU conference into a RoomKit room. The SFU carries all human-to-human media; RoomKit joins as a single bot participant for transcription, AI voice, recording, and speech-to-speech — it never proxies human media (RFC §12.10).

## Constructor

```python
ConferenceChannel(
    channel_id,
    *,
    backend,                 # ConferenceBackend (required)
    stt=None,                # STTProvider — per-track transcription
    tts=None,                # TTSProvider — the bot's voice
    realtime=None,           # ConferenceRealtimeConfig — speech-to-speech (excludes tts)
    pipeline=None,           # default: 16 kHz mono contract + EnergyVADProvider
    interruption=None,       # ConferenceInterruptionConfig
    recording=None, recorder=None,   # ConferenceRecordingConfig + MediaRecorder, both or neither
    bot_identity="roomkit",
    bot_grants=None,         # explicit ConferenceGrants; default derived via for_bot()
    default_grants=None,     # what mint_access() grants humans; default ConferenceGrants()
    e2ee=False, close_room_on_detach=False,
    speak_text_events=False, # off: only AI-channel text events are spoken
    close_providers=True,
    max_queued_frames=100,   # per-track backpressure bound (lane + recording)
    identity_address_keys=None, identity_trusts_unasserted_metadata=False,
)
```

Refused at construction (and identically at plug time): `e2ee=True` with stt/recording/realtime (bot receives ciphertext); `ConferenceRecordingMode.EGRESS` (only `FRAMEWORK` is implemented); `tts` + `realtime` together (one bot track, one voice); `realtime.tools` without `tool_handler`; a `pipeline` without a VAD when stt/realtime is set.

Public surface: `mint_access()`, `plug_*/unplug_*()`, `set_bot_grants()`, `may_interrupt(participant_id)`, `active_lanes` (dict `track_id -> ConferenceLane`; `drain()`, `dropped_frames`), `info()`, `close()`.

## How the Bridge Works

The bot joins lazily, only when the channel has a need (stt/tts/recording/realtime) and someone is coming: a `mint_access()`, an arrival, an occupancy probe. With no need configured the channel is pure transport — admission gate and roster, no bot in the meeting.

- **Inbound**: each subscribed AUDIO track gets its own `ConferenceLane` (queue + task + VAD state). The VAD segments utterances; one utterance → one STT call → one `RoomEvent` attributed to the track's `participant_id` (track identity replaces diarization). A full lane drops oldest frames.
- **Outbound**: `deliver()` speaks AI-channel TextContent on the single bot track via TTS (all text events if `speak_text_events=True`); with `realtime`, text is injected into the provider's context instead.
- **Video**: never subscribed — `capabilities()` announces AUDIO only.

```python
from roomkit import ConferenceTranscription, HookResult, HookTrigger

@kit.hook(HookTrigger.ON_TRANSCRIPTION)   # sync, runs BEFORE the room; block/modify = redaction
async def on_text(payload: ConferenceTranscription, ctx) -> HookResult:
    # payload: track_id, participant_id, room_id, text
    return HookResult.allow()
```

## Conference Hooks

SYSTEM lifecycle events; payload in `event.content.data` (always includes `channel_id`).

| Trigger | Fired when | Data |
|---------|-----------|------|
| `ON_CONFERENCE_PARTICIPANT_JOINED` / `_LEFT` | SFU reports arrival/departure | `participant_id` |
| `ON_CONFERENCE_TRACK_PUBLISHED` / `_UNPUBLISHED` | Track publish/unpublish | `track_id`, `participant_id`, `kind` |
| `ON_CONFERENCE_TRACK_MUTED` / `_UNMUTED` | Publisher (un)mutes; muted VIDEO = camera off | `track_id`, `participant_id`, `kind` |
| `ON_SCREEN_SHARE_STARTED` / `_STOPPED` | SCREEN_SHARE track published/unpublished | `track_id`, `participant_id` |
| `ON_ACTIVE_SPEAKER_CHANGED` | Dominant speaker (`ACTIVE_SPEAKER` capability) | `participant_id` |
| `ON_CONNECTION_QUALITY_CHANGED` | Quality report (backend label, not normalized) | `participant_id`, `quality` |

`ON_SPEECH_START`/`ON_SPEECH_END` fire per lane at VAD edges. `ON_BARGE_IN` carries `ConferenceBargeIn(room_id, track_id, participant_id, interrupted_text, audio_position_ms)`.

## ConferenceBackend ABC (`roomkit.conference`)

- Control plane: `ensure_room(room_id, metadata=None, e2ee=False)`, `close_room()`, `mint_access(room_id, participant_id, grants, *, display_name=None) -> ConferenceAccess`, `list_participants()`, `remove_participant()`, `mute_track()`, `unmute_track()` (needs `REMOTE_UNMUTE`).
- Bot session: `join_as_bot(room_id, identity, grants) -> BotSession`, `leave(bot)`, `update_bot_grants(bot, grants)` (needs `BOT_GRANT_UPDATE`, else `ConferenceCapabilityError`), `subscribe_track(bot, track_id)`, `unsubscribe_track()`, `publish_audio(bot, chunk)` (PCM `AudioChunk`; `is_final` ends the utterance), `stop_playback(bot)` (barge-in: discard queued unplayed audio; utterance still ends on `is_final`; no-op for a gone session), `publish_video()` (needs `VIDEO_PUBLISH`), `close()`.
- Callbacks (what drives the channel): `on_participant_joined/left`, `on_track_published/unpublished/muted/unmuted`, `on_track_audio`, `on_track_video`, `on_active_speaker_changed`, `on_connection_quality`, `on_bot_session_ended` (SFU dropped the bot without a `leave()`).
- `ConferenceCapability` flags: `SCREEN_SHARE`, `EGRESS_RECORDING`, `SIP_GATEWAY`, `ACTIVE_SPEAKER`, `CONNECTION_QUALITY`, `VIDEO_PUBLISH`, `REMOTE_UNMUTE`, `BOT_GRANT_UPDATE`, `E2EE`.

## Backends

| Backend | Class | Config | Extra |
|---------|-------|--------|-------|
| LiveKit | `LiveKitConferenceBackend` | `LiveKitConfig` | `roomkit[livekit]` |
| Mock | `MockConferenceBackend` | `capabilities=` kwarg | built-in |

```python
from roomkit.conference.livekit import LiveKitConfig, LiveKitConferenceBackend

backend = LiveKitConferenceBackend(LiveKitConfig(url="wss://my-project.livekit.cloud"))
```

`LiveKitConfig`: `url`/`api_key`/`api_secret` fall back to `LIVEKIT_URL`/`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`; `access_ttl=timedelta(minutes=15)`, `audio_sample_rate=48_000`, `audio_channels=1`, `publish_queue_ms=300`, `remote_unmute=False`, `sip_gateway=False`, `room_metadata_key="roomkit"`. Declares `SCREEN_SHARE | ACTIVE_SPEAKER | CONNECTION_QUALITY | BOT_GRANT_UPDATE` (+`REMOTE_UNMUTE`/`SIP_GATEWAY` when configured).

`MockConferenceBackend` scripts SFU events: `simulate_participant_joined/left`, `simulate_track_published(room_id, participant_id, kind=TrackKind.AUDIO)`, `simulate_track_unpublished`, `simulate_audio(track, frame)`, `simulate_track_muted/unmuted`, `simulate_active_speaker`, `simulate_connection_quality`, `simulate_bot_disconnected(bot, reason)`; fault injection `fail(method, exc)` / `delay(method, seconds)`; assertion state `calls`, `published_audio`, `utterances`, `subscriptions`, `playback_stops`.

## Models & Grants

- `ConferenceGrants(publish_audio=True, publish_video=True, publish_screen_share=True, subscribe=True, moderate=False, hidden=False)` — human defaults. `ConferenceGrants.for_bot(speaks=False, listens=True)` = least privilege; `ConferenceGrants.observer()` = subscribe-only + hidden.
- `ConferenceAccess(url, token, expires_at, provider_data)` — opaque client credential; `token` excluded from repr.
- `BotSession(id, room_id, identity, joined_at, metadata)`; `ConferenceTrack(id, room_id, participant_id, kind, muted, metadata)`; `ConferenceParticipant(participant_id, display_name, connected_at, tracks, metadata, asserted_metadata)`; `TrackKind.AUDIO/VIDEO/SCREEN_SHARE`.
- `ConferenceInterruptionConfig(strategy=InterruptionStrategy.IMMEDIATE, scope=ConferenceInterruptionScope.ANY, allowlist=[])` — scope `ANY`/`NONE`/`ALLOWLIST` decides who may barge in.
- `ConferenceRealtimeConfig(provider, system_prompt=None, voice=None, tools=None, tool_handler=None, temperature=None, input_sample_rate=24000, output_sample_rate=24000, server_vad=True, provider_config=None)` — lanes mixed N→1 into one speech-to-speech session per room; the provider speaks on the bot track.

## Hot-Plugging

Effects are in force when the plug returns: occupied conferences joined, published tracks subscribed, bot grants realigned (in place with `BOT_GRANT_UPDATE`, announced re-join otherwise). An occupied slot is refused — swap = unplug, then plug. Unplugging the last need makes the bot leave every conference.

```python
await channel.plug_stt(stt)          # optional pipeline=; joins an existing pipeline
await channel.plug_tts(tts)
await channel.plug_recording(ConferenceRecordingConfig(), recorder=recorder)
await channel.plug_realtime(config)
await channel.unplug_stt(); await channel.unplug_tts()
await channel.unplug_recording(); await channel.unplug_realtime()
await channel.set_bot_grants(ConferenceGrants.observer())  # None returns to derivation
```

## Shutdown Semantics (RFC §12.10.4)

`close()` runs exactly one shutdown per channel (`ConferenceShutdownCoordinator`): concurrent callers join the same shielded task and the terminal result is replayed on later calls. Order: close admission → stop playbacks → wait teardowns/joins → disconnect realtime sessions → bots leave every conference → close lanes → finalize recordings → close backend → close providers → settle roster writes. Every backend/provider call holds an operation lease; a resource still leased is **retained**, not closed — it closes in the background when its last lease returns, and the close **fails**: `ConferenceCloseError` aggregates structured `CloseIssue`s (`component`, `operation`, `status` in `FAILED`/`TIMED_OUT`/`ABANDONED`/`RETAINED`, `step`, `detail`) plus any bot session that could not be removed (still reported by `info()`). All waits are bounded — a slow store never holds a bot in a meeting.

`info()` answers RFC §17.7 disclosure per room: `bot_present`, `bot_hidden`, `stt_active`, `realtime_active`, `recording_active`, `recording_dropped_frames`, `active_lanes`, `collecting`, `leave_failed`.

## Minimal Example

```python
import asyncio
from roomkit import MockConferenceBackend, RoomKit
from roomkit.channels.conference import ConferenceChannel
from roomkit.voice.stt.mock import MockSTTProvider

async def main() -> None:
    backend = MockConferenceBackend()
    channel = ConferenceChannel("conf", backend=backend, stt=MockSTTProvider(transcripts=["Hi."]))
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room("standup")
    await kit.attach_channel("standup", "conf")

    # Mint SFU credentials for a human client; the mint also starts the lazy bot join
    await kit.ensure_participant("standup", "conf", "alice", display_name="Alice")
    access = await channel.mint_access("standup", "alice")  # hand access.url/token to the client

    # With a real backend, SFU events drive these
    await backend.simulate_participant_joined("standup", "alice", display_name="Alice")
    mic = await backend.simulate_track_published("standup", "alice")
    # feed AudioFrames: simulate_audio(mic, frame) — speech, then silence — then
    # await channel.active_lanes[mic.id].drain() and read kit.store.list_events("standup")
    await kit.close()

asyncio.run(main())
```

Runnable references: `examples/conference_quickstart.py`, `conference_ai_meeting.py`, `conference_realtime_ai.py`, `conference_livekit.py`.
