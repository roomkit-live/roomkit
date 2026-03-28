# RoomKit Examples

Runnable examples demonstrating RoomKit features. Each file is self-contained
with a docstring explaining prerequisites and run command.

```bash
# Run any example:
uv run python examples/<example>.py

# Enable console dashboard (voice examples):
CONSOLE=1 uv run python examples/<example>.py
```

## Voice

| Example | Feature | Description |
|---------|---------|-------------|
| `voice_local_audio.py` | Local audio | Mock STT/TTS with local mic and speakers — no external deps |
| `voice_cloud.py` | Cloud pipeline | Deepgram STT + Claude + ElevenLabs TTS with full audio pipeline |
| `voice_gradium.py` | Cloud pipeline | Gradium STT/TTS + Claude with local mic |
| `voice_deepgram_grok.py` | Cloud pipeline | Deepgram STT + Claude Haiku + Grok TTS |
| `voice_expressive.py` | TTS | ElevenLabs expressive voice with emotion and accent tags |
| `voice_neutts.py` | TTS | NeuTTS with zero-shot voice cloning from reference audio |
| `voice_local_onnx_vllm.py` | Local AI | Fully local voice assistant with sherpa-onnx STT/TTS + vLLM |
| `voice_parakeet_tdt.py` | Local STT | NVIDIA Parakeet TDT ASR for offline speech recognition |
| `voice_qwen3_asr.py` | Local STT | Qwen3-ASR speech recognition pipeline |
| `voice_qwen3_tts.py` | Local TTS | Qwen3 text-to-speech with voice cloning |
| `voice_pipeline.py` | Audio pipeline | Voice channel with audio processing pipeline (AEC, VAD, diarization, DTMF) |
| `voice_pipeline_advanced.py` | Audio pipeline | Advanced pipeline: AGC, denoiser, debug taps, multiple recorders |
| `voice_sherpa_onnx_vad.py` | VAD | Neural VAD + denoiser with sherpa-onnx models |
| `voice_smart_turn.py` | Turn detection | Audio-native turn detection with smart-turn ONNX model |
| `voice_greeting.py` | Voice UX | Patterns for greeting callers on session start |
| `voice_say_play.py` | Voice UX | Proactive voice output via `say()` and `play()` methods |
| `voice_console_demo.py` | Console | RoomKitConsole dashboard demo with audio meters and colored logs |
| `audio_level_vu_meter.py` | Audio levels | VU meter / audio level monitoring with LocalAudioBackend |
| `wav_recorder.py` | Recording | Debug audio recording with WavFileRecorder pipeline stage |
| `grok_tts.py` | TTS | xAI Grok text-to-speech provider demo |

## Voice -- SIP / RTP / Transport

| Example | Feature | Description |
|---------|---------|-------------|
| `voice_sip.py` | SIP | Accept SIP calls from a PBX with full hook pipeline |
| `voice_sip_bridge.py` | SIP bridge | Bridge two SIP calls with live Deepgram transcription |
| `voice_sip_bridge_summary.py` | SIP bridge | Bridge SIP calls and summarize transcript with Claude |
| `voice_sip_dtmf.py` | SIP + DTMF | SIP with DTMF tone detection as AI tool calls for IVR |
| `voice_sip_local_agent.py` | SIP + local AI | SIP to fully local AI agent — no cloud dependencies |
| `voice_sip_auth.py` | SIP auth | SIP with authentication and registrar support |
| `voice_sip_dial.py` | SIP outbound | Outbound SIP dialing with Gemini Live AI |
| `voice_rtp.py` | RTP | RTP voice backend with mock providers |
| `voice_fastrtc.py` | WebRTC | Voice assistant over FastRTC WebSocket transport |
| `voice_webtransport.py` | WebTransport | Voice echo via QUIC datagrams (WebTransport) |
| `voice_bridge_with_ai.py` | Multi-party | Multi-party bridge with AI moderator that can interject |
| `voice_bridge_summary.py` | Multi-party | Bridge multi-party call and summarize with AI |
| `voice_bridge_live_analyst.py` | Multi-party | Live AI analyst on bridged call with sentiment analysis |
| `voice_multibackend_bridge.py` | Multi-transport | SIP + WebRTC + WebSocket all bridged together |
| `rtp_gradium_stt.py` | RTP + STT | RTP receiver with Gradium STT transcription |
| `rtp_video_call.py` | RTP + video | RTP audio + video direct transport |

## Voice -- Realtime (Speech-to-Speech)

| Example | Feature | Description |
|---------|---------|-------------|
| `realtime_voice_gemini.py` | Gemini Live | Gemini Live API minimal example over WebSocket |
| `realtime_voice_fastrtc.py` | Gemini + WebRTC | Gemini Live speech-to-speech over FastRTC WebRTC |
| `realtime_voice_local_gemini.py` | Gemini + local | Gemini Live with local mic and speakers |
| `realtime_voice_local_openai.py` | OpenAI Realtime | OpenAI Realtime with local mic and speakers |
| `realtime_voice_local_elevenlabs.py` | ElevenLabs | ElevenLabs speech-to-speech with local mic |
| `realtime_voice_local_xai.py` | xAI Realtime | xAI Grok Realtime with local mic and speakers |
| `realtime_voice_local_personaplex.py` | PersonaPlex | PersonaPlex speech-to-speech with local mic |
| `realtime_voice_sip_gemini.py` | SIP + Gemini | SIP calls routed to Gemini Live |
| `realtime_voice_sip_elevenlabs.py` | SIP + ElevenLabs | SIP calls routed to ElevenLabs Conversational AI |
| `realtime_elevenlabs_tools.py` | ElevenLabs tools | ElevenLabs Conversational AI with client-side tool calling |
| `realtime_av_anam.py` | Avatar | Realtime audio-video with Anam AI avatar |

## Video

| Example | Feature | Description |
|---------|---------|-------------|
| `webrtc_video.py` | WebRTC | WebRTC audio + video with AI vision analysis |
| `websocket_video.py` | WebSocket | WebSocket video streaming with AI vision |
| `webcam_assistant.py` | Webcam | Chat with Claude about what it sees on your webcam |
| `webcam_vision.py` | Webcam | Webcam vision analysis with periodic AI snapshots |
| `webcam_recording.py` | Recording | Webcam recording to MP4 file |
| `webcam_censor.py` | Vision filter | Webcam content censoring with recording |
| `screen_describe.py` | Screen capture | Screen description with AI analysis |
| `screen_assistant_ia.py` | Screen + voice | AI screen assistant with speech-to-speech voice |
| `screen_agent_orchestrated.py` | Screen + agents | Orchestrated screen agent with OmniView vision |
| `video_live_subtitles.py` | Subtitles | Real-time translated subtitles on webcam video |
| `avatar_call.py` | Avatar + SIP | SIP call with lip-synced avatar video |
| `sip_anam_avatar.py` | Avatar + SIP | SIP-to-Anam AI Avatar bridge |
| `sip_openai_anam_avatar.py` | Avatar + OpenAI | SIP + OpenAI Realtime + Anam Avatar |
| `sip_video_call.py` | SIP video | SIP audio + video call handler |
| `sip_video_bridge.py` | SIP video | Bridge two SIP audio + video calls |
| `sip_send_video.py` | SIP video | SIP send video test pattern |
| `pyav_video_recorder.py` | Recording | PyAV H.264 webcam recorder to MP4 |
| `room_media_recorder.py` | Recording | Room-level media recording: mic + webcam to MP4 |

## Text / Messaging

| Example | Feature | Description |
|---------|---------|-------------|
| `quickstart.py` | WebSocket | Two WebSocket users chatting with an AI assistant |
| `anthropic_ai.py` | Anthropic | AI-powered assistant using Anthropic Claude |
| `openai_ai.py` | OpenAI | AI-powered assistant using OpenAI GPT |
| `mistral_ai.py` | Mistral | AI-powered assistant using Mistral AI |
| `ai_azure.py` | Azure | Azure AI Foundry with OpenAI-compatible API |
| `multichannel_bridge.py` | Multi-channel | WebSocket + SMS + Email + HTTP + AI bridged together |
| `teams_bot.py` | Teams | Microsoft Teams bot via Bot Framework |
| `teams_echo.py` | Teams | Microsoft Teams echo bot with MockTeamsProvider |
| `telegram_bot.py` | Telegram | Telegram bot with webhook integration |
| `whatsapp_personal.py` | WhatsApp | WhatsApp Personal via neonize protocol |
| `facebook_messenger.py` | Messenger | Facebook Messenger chat integration |
| `http_webhook.py` | HTTP | Generic HTTP webhook channel |
| `voicemeup_sms.py` | SMS | VoiceMeUp SMS provider with webhooks |
| `elasticemail.py` | Email | Elastic Email provider integration |
| `async_sms_notification.py` | SMS + voice | Cross-channel SMS notification during a voice call |
| `websocket_streaming.py` | WebSocket | WebSocket streaming text delivery protocol |
| `delivery_status_webhook.py` | Webhooks | Delivery status tracking with webhook callbacks |

## AI Features

| Example | Feature | Description |
|---------|---------|-------------|
| `ai_tools_function_calling.py` | Tools | Custom tools, function calling, and per-room tool config |
| `ai_thinking.py` | Reasoning | Extended thinking / chain-of-thought support |
| `ai_planning.py` | Planning | Structured task tracking with `_plan_tasks` tool |
| `ai_eviction.py` | Token mgmt | Auto-pagination of large tool results to avoid overflow |
| `ai_memory.py` | Memory | SummarizingMemory for long conversations with compression |
| `ai_knowledge_scoring.py` | Retrieval | Knowledge retrieval and response scoring with RetrievalMemory |
| `memory_provider.py` | Memory | Custom MemoryProvider for AI context construction |
| `mcp_tool_provider.py` | MCP | Model Context Protocol tool provider integration |
| `agent_skills.py` | Skills | Agent skills discovery and registration |
| `streaming_tools.py` | Streaming | Streaming text delivery with interleaved tool calls |
| `tool_call_events.py` | Tool events | Tool call ephemeral event tracking |
| `guardrails.py` | Safety | Multi-layer safety: PII redaction, jailbreak detection, output guards |

## Orchestration (Multi-Agent)

| Example | Feature | Description |
|---------|---------|-------------|
| `orchestration_routing.py` | Routing | Rule-based routing with ConversationRouter |
| `orchestration_loop.py` | Loop | Coder/reviewer loop with `can_return_to` |
| `orchestration_loop_cli.py` | Loop CLI | Interactive loop with multiple parallel reviewers |
| `orchestration_swarm.py` | Swarm | Swarm where any agent can hand off to any other |
| `orchestration_swarm_cli.py` | Swarm CLI | Interactive swarm orchestration with specialist agents |
| `orchestration_pipeline.py` | Pipeline | Multi-agent pipeline: triage -> handler -> resolver |
| `orchestration_pipeline_cli.py` | Pipeline CLI | Interactive pipeline triage |
| `orchestration_supervisor.py` | Supervisor | Manual delegation mode supervisor |
| `orchestration_supervisor_parallel_tasks.py` | Supervisor | Parallel analysis workflow with supervisor |
| `orchestration_supervisor_sequential_content_workflow.py` | Supervisor | Sequential content workflow with supervisor |
| `orchestration_supervisor_voice_parallel.py` | Supervisor + voice | Supervisor with parallel voice analysis |
| `orchestration_approval_loop.py` | Approval | Produce/review approval loop strategy |
| `orchestration_voice_mediator.py` | Voice routing | Channel-aware content adaptation with visibility routing |
| `orchestration_voice_triage.py` | Voice triage | SIP voice orchestration with triage and delegation |
| `orchestration_realtime_triage.py` | Realtime triage | SIP + Gemini Live speech-to-speech triage |

## Infrastructure

| Example | Feature | Description |
|---------|---------|-------------|
| `hook_logging_analytics.py` | Hooks | AFTER_BROADCAST async hooks for logging and analytics |
| `hook_moderation.py` | Hooks | Content moderation with BEFORE_BROADCAST hooks |
| `hook_inject_welcome.py` | Hooks | Auto-inject welcome messages with hooks |
| `channel_mute_unmute.py` | Channel mgmt | Dynamic channel muting/unmuting and lifecycle hooks |
| `presence_tracking.py` | Presence | Online/away/offline presence tracking |
| `typing_indicators.py` | Typing | Typing start/stop indicator events |
| `reactions.py` | Reactions | Emoji reactions on messages via ephemeral events |
| `read_receipts.py` | Read receipts | Read receipt and delivery tracking |
| `edit_delete_messages.py` | Message ops | Message editing and deletion with EditContent/DeleteContent |
| `response_visibility.py` | Visibility | Control where AI responses are delivered per channel |
| `rich_content_buttons.py` | Rich content | Rich formatted messages with buttons, cards, quick replies |
| `composite_messages.py` | Content types | Multi-part messages with text, media, and location |
| `location_sharing.py` | Location | Geographic location sharing between channels |
| `room_lifecycle.py` | Room mgmt | Room creation, pausing, closing with timers |
| `background_agent_task.py` | Delegation | Background task delegation with `kit.delegate()` |
| `custom_identity_resolver.py` | Identity | Custom IdentityResolver for inbound message senders |
| `rate_limiting.py` | Rate limiting | TokenBucketRateLimiter for message throttling |
| `circuit_breaker_retry.py` | Resilience | Circuit breaker and retry patterns |
| `postgres_store.py` | Storage | PostgreSQL storage backend for production |
| `delivery_backend.py` | Delivery | Persistent delivery with InMemoryDeliveryBackend |
| `delivery_redis.py` | Delivery | Redis-backed persistent distributed delivery |
| `telemetry_console.py` | Telemetry | Console telemetry provider for span timing |
| `telemetry_otel.py` | Telemetry | OpenTelemetry integration |
| `telemetry_pyroscope.py` | Profiling | Continuous CPU profiling with Pyroscope |
| `aicoustics_denoiser.py` | Denoising | AICoustics Quail denoising with RMS stats |
| `test_rnnoise_live.py` | Denoising | RNNoise noise reduction live testing |
| `trace_audio.py` | Debugging | Audio path tracer for RoomKit debugging |

## Shared Helpers

The `shared/` directory provides reusable utilities for examples:

| Helper | Description |
|--------|-------------|
| `setup_logging(name)` | Standard logging configuration |
| `require_env(*names)` | Validate required environment variables |
| `run_until_stopped(kit, cleanup=fn)` | Graceful shutdown with signal handling |
| `setup_console(kit)` | Optional console dashboard (`CONSOLE=1`) |
| `build_pipeline(...)` | Audio pipeline assembly from env vars |
| `build_aec(rate, block_ms)` | AEC provider factory (`AEC=webrtc\|speex\|0`) |
| `build_vad(rate)` | VAD provider factory (`VAD=energy\|silero\|ten\|0`) |
| `build_denoiser(rate)` | Denoiser factory (`DENOISE=rnnoise\|sherpa\|0`) |
| `build_turn_detector()` | Turn detector factory (`TURN_DETECTOR=smart-turn\|0`) |
| `build_debug_taps()` | Debug audio taps (`DEBUG_AUDIO=1`) |
| `auto_select_provider(env, label)` | Interactive provider selection (OpenAI/Gemini) |
| `os_info()` | OS-specific info for system prompts |
| `log_tool_call(event)` | Format and log tool calls in hooks |
