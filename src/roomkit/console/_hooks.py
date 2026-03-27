"""Hook registration for the RoomKit console dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.console._state import ConsoleState, ConversationTurn, VoiceEvent
from roomkit.core.hooks import HookRegistration
from roomkit.models.enums import HookExecution, HookTrigger

_ACCENT = "rgb(6,182,212)"

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.core.hooks import HookEngine

# All console hook names share this prefix for clean removal.
HOOK_PREFIX = "_console_"

# Triggers to register with their state-update callbacks.
_HOOK_DEFS: list[tuple[HookTrigger, str]] = [
    (HookTrigger.ON_INPUT_AUDIO_LEVEL, "input_level"),
    (HookTrigger.ON_OUTPUT_AUDIO_LEVEL, "output_level"),
    (HookTrigger.ON_VAD_AUDIO_LEVEL, "vad_level"),
    (HookTrigger.ON_SESSION_STARTED, "session_started"),
    (HookTrigger.ON_SPEECH_START, "speech_start"),
    (HookTrigger.ON_SPEECH_END, "speech_end"),
    (HookTrigger.ON_TRANSCRIPTION, "transcription"),
    (HookTrigger.ON_PARTIAL_TRANSCRIPTION, "partial_transcription"),
    (HookTrigger.BEFORE_TTS, "before_tts"),
    (HookTrigger.AFTER_TTS, "after_tts"),
    (HookTrigger.ON_BARGE_IN, "barge_in"),
    (HookTrigger.ON_TOOL_CALL, "tool_call"),
]


def _make_handler(tag: str, state: ConsoleState, kit: RoomKit | None = None) -> Any:
    """Build a hook handler that updates *state* for the given *tag*."""

    async def on_input_level(event: Any, ctx: Any) -> None:
        state.push_input_level(event.level_db)

    async def on_output_level(event: Any, ctx: Any) -> None:
        state.push_output_level(event.level_db)

    async def on_vad_level(event: Any, ctx: Any) -> None:
        state.push_input_level(event.level_db)
        state.is_speech = event.is_speech

    async def on_session_started(event: Any, ctx: Any) -> None:
        # event.session is VoiceSession (may be None for text channels).
        session = getattr(event, "session", None)
        state.session_id = session.id if session else None
        state.participant_id = getattr(event, "participant_id", None)
        state.room_id = event.room_id
        state.channel_id = event.channel_id
        state.session_started_at = event.timestamp
        state.voice_state = "idle"

        # Detect barge-in and skills from channel.
        if kit is not None:
            channel = kit.channels.get(event.channel_id)
            if channel is not None:
                transport = getattr(channel, "_transport", None)
                if transport is not None:
                    mute = getattr(transport, "_mute_mic_during_playback", None)
                    state.barge_in_enabled = not mute if mute is not None else None
                skills = getattr(channel, "_skills", None)
                if skills is not None:
                    state.skill_names = list(getattr(skills, "skill_names", []))
                    state.skills_count = len(state.skill_names)

        state.voice_events.append(VoiceEvent("SESSION STARTED", f"bold {_ACCENT}"))

    async def on_speech_start(event: Any, ctx: Any) -> None:
        state.voice_state = "listening"
        state.partial_text = ""
        state.partial_assistant_text = ""
        state.voice_events.append(VoiceEvent("SPEECH START", "bold green"))

    async def on_speech_end(event: Any, ctx: Any) -> None:
        state.voice_state = "processing"
        state.voice_events.append(VoiceEvent("SPEECH END", "yellow"))

    async def on_transcription(event: Any, ctx: Any) -> None:
        # RealtimeTranscriptionEvent has .role;
        # TranscriptionEvent (VoiceChannel) has no .role — always user.
        role = getattr(event, "role", "user")

        if role == "assistant":
            state.last_tts_text = event.text
            state.partial_assistant_text = ""
            state.tts_count += 1
            state.conversation.append(ConversationTurn(role="assistant", text=event.text))
            state.voice_state = "idle"
            state.voice_events.append(VoiceEvent("AI DONE", f"{_ACCENT}"))
        else:
            state.last_final_text = event.text
            state.partial_text = ""
            state.transcription_count += 1
            state.conversation.append(ConversationTurn(role="user", text=event.text))
            state.voice_state = "processing"
            state.voice_events.append(VoiceEvent("STT FINAL", "cyan"))

    async def on_partial_transcription(event: Any, ctx: Any) -> None:
        # PartialTranscriptionEvent.role defaults to "user".
        # Both VoiceChannel and RealtimeVoiceChannel fire this hook.
        role = getattr(event, "role", "user")
        if role == "assistant":
            state.partial_assistant_text += event.text
            state.voice_state = "speaking"
        else:
            state.partial_text += event.text
            state.voice_state = "listening"

    async def on_before_tts(text: Any, ctx: Any) -> None:
        state.voice_state = "speaking"
        tts_text = text if isinstance(text, str) else str(text)
        state.last_tts_text = tts_text
        state.tts_count += 1
        state.conversation.append(ConversationTurn(role="assistant", text=tts_text))
        state.voice_events.append(VoiceEvent("TTS START", f"bold {_ACCENT}"))

    async def on_after_tts(text: Any, ctx: Any) -> None:
        state.voice_state = "idle"
        state.voice_events.append(VoiceEvent("TTS END", _ACCENT))

    async def on_barge_in(event: Any, ctx: Any) -> None:
        state.barge_in_count += 1
        state.voice_state = "listening"
        state.voice_events.append(VoiceEvent("BARGE-IN", "bold red"))

    async def on_tool_call(event: Any, ctx: Any) -> None:
        state.tool_call_count += 1
        name = getattr(event, "name", "?")
        state.voice_events.append(VoiceEvent(f"TOOL: {name}", "yellow"))

    handlers: dict[str, Any] = {
        "input_level": on_input_level,
        "output_level": on_output_level,
        "vad_level": on_vad_level,
        "session_started": on_session_started,
        "speech_start": on_speech_start,
        "speech_end": on_speech_end,
        "transcription": on_transcription,
        "partial_transcription": on_partial_transcription,
        "before_tts": on_before_tts,
        "after_tts": on_after_tts,
        "barge_in": on_barge_in,
        "tool_call": on_tool_call,
    }
    return handlers[tag]


def register_console_hooks(
    engine: HookEngine, state: ConsoleState, kit: RoomKit | None = None
) -> list[str]:
    """Register all console observer hooks on *engine*.

    Returns the list of hook names for later removal.
    """
    names: list[str] = []
    for trigger, tag in _HOOK_DEFS:
        name = f"{HOOK_PREFIX}{tag}"
        engine.register(
            HookRegistration(
                trigger=trigger,
                execution=HookExecution.ASYNC,
                fn=_make_handler(tag, state, kit),
                priority=999,
                name=name,
            )
        )
        names.append(name)
    return names


def unregister_console_hooks(engine: HookEngine, names: list[str]) -> None:
    """Remove previously registered console hooks from *engine*."""
    for name in names:
        engine.remove_global_hook(name)
