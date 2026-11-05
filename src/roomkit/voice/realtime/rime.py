"""Rime Prism speech-to-speech provider (``roomkit[rime]``).

Prism owns turn detection. It accepts continuous mono PCM16 at 16 kHz and
returns 24 kHz audio. This adapter keeps silence flowing between input
frames, maps the Prism-specific text/tool protocol, and reports playback
truncation through RoomKit's existing realtime contract.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

from roomkit.core.task_utils import cancel_and_wait
from roomkit.providers.ai.base import ModelInfo
from roomkit.providers.ai.tool_declaration import declared_parameters
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime._rime_session import PrismSession
from roomkit.voice.realtime.injection import VoiceInjectionResult, say_line_instruction
from roomkit.voice.realtime.provider import RealtimeVoiceProvider
from roomkit.voice.voices import VoiceInfo

_VOICES = ("astra", "aurora", "brando", "wawona")
_CONFIG_KEYS = {"interrupt_response", "eagerness", "end_of_turn_timeout_ms", "truncation"}


@dataclass
class RimeRealtimeConfig:
    """Prism connection and turn settings.

    ``timeout_s`` bounds connection setup and acknowledged commands.
    ``eagerness`` and ``end_of_turn_timeout_ms`` are fixed for a session.
    Context preservation is not advertised: Prism can evict old turns.
    """

    api_key: str = field(repr=False)
    voice_id: str = "astra"
    websocket_url: str = "wss://prism.api.rime.ai/v1/realtime"
    timeout_s: float = 10.0
    interrupt_response: bool = True
    eagerness: Literal["auto", "low", "medium", "high"] = "auto"
    end_of_turn_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if (
            not self.api_key
            or not self.api_key.isascii()
            or any(c.isspace() for c in self.api_key)
        ):
            raise ValueError("RimeRealtimeConfig.api_key must be a nonempty ASCII token")
        _voice(self.voice_id)
        if self.timeout_s <= 0:
            raise ValueError("Prism timeout_s must be positive")
        _turn_settings(
            {"eagerness": self.eagerness, "end_of_turn_timeout_ms": self.end_of_turn_timeout_ms}
        )


class RimeRealtimeProvider(RealtimeVoiceProvider):
    """English Prism conversations with transcripts, function tools and barge-in.

    Use 16 kHz input and 24 kHz output on the channel, without a pipeline VAD.
    The channel then selects ``server_vad=True`` when connecting the provider.
    ``provider_config`` accepts ``interrupt_response``, ``eagerness``,
    ``end_of_turn_timeout_ms`` and ``truncation``. Reconfiguration retains
    the live conversation; changing the two fixed turn settings is refused.
    A dropped connection is reported, never silently replaced with a new
    conversation. ``full_duplex`` stays false: Prism exposes turn boundaries.
    """

    def __init__(self, config: RimeRealtimeConfig) -> None:
        super().__init__()
        self._config = config
        self._sessions: dict[str, PrismSession] = {}
        self._connecting: dict[str, asyncio.Task[Any]] = {}

    @property
    def name(self) -> str:
        return "RimeRealtime"

    @property
    def model_name(self) -> str:
        return "prism"

    @classmethod
    def available_voices(cls) -> list[VoiceInfo]:
        return [VoiceInfo(id=voice, name=voice.title(), language="en") for voice in _VOICES]

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        return [ModelInfo(id="prism", display_name="Rime Prism", context_window=65536)]

    async def connect(
        self,
        session: VoiceSession,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        server_vad: bool = True,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        if session.state == VoiceSessionState.ENDED:
            raise ValueError("Cannot connect an ended voice session")
        if session.id in self._sessions or session.id in self._connecting:
            raise ValueError("This voice session is already connected or connecting")
        if (input_sample_rate, output_sample_rate) != (16000, 24000):
            raise ValueError("Prism requires 16000 Hz input and 24000 Hz output")
        if not server_vad:
            raise ValueError("Prism requires server_vad=True; client endpointing is unavailable")
        settings = self._settings(system_prompt, voice, tools, temperature, provider_config)
        current = asyncio.current_task()
        assert current is not None
        self._connecting[session.id] = current
        try:
            await self._open(session, settings)
        except BaseException:
            session.state = VoiceSessionState.ENDED
            raise
        finally:
            self._connecting.pop(session.id, None)

    async def _open(self, session: VoiceSession, settings: dict[str, Any]) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError("Install Rime dependencies with: pip install roomkit[rime]") from exc
        await self._abandon_previous_connection(session)
        ws = await connect(
            self._config.websocket_url,
            additional_headers={"Authorization": f"Bearer {self._config.api_key}"},
            open_timeout=self._config.timeout_s,
            close_timeout=2,
        )
        state = PrismSession(self, session, ws, self._config.timeout_s)
        self._sessions[session.id] = state
        try:
            await state.initialize(settings)
        except BaseException:
            await state.close()
            raise

    def _settings(
        self,
        prompt: str | None,
        voice: str | None,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        provider_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        _check_config(temperature, provider_config)
        pc = provider_config or {}
        settings: dict[str, Any] = {
            "voice": _voice(voice if voice is not None else self._config.voice_id),
            "turn_detection": _turn_settings(
                {
                    "interrupt_response": self._config.interrupt_response,
                    "eagerness": self._config.eagerness,
                    "end_of_turn_timeout_ms": self._config.end_of_turn_timeout_ms,
                    **pc,
                }
            ),
        }
        _add_settings(settings, prompt, tools, pc)
        return settings

    async def send_audio(self, session: VoiceSession, audio: bytes) -> None:
        """Queue PCM16 for continuous 20 ms frames, bounded to five seconds."""
        self._state(session).append_audio(audio)

    async def inject_text(
        self,
        session: VoiceSession,
        text: str,
        *,
        role: str = "user",
        silent: bool = False,
    ) -> VoiceInjectionResult:
        """Add history silently, or request a typed turn / application instruction."""
        state = self._sessions.get(session.id)
        if state is None or state.closed:
            return VoiceInjectionResult(status="not_sent", reason="disconnected")
        if role not in {"user", "system", "assistant"}:
            raise ValueError("Prism injection role must be user, system or assistant")
        if not text.strip() or len(text) > 4000:
            raise ValueError("Prism injected text must contain 1 to 4000 characters")
        if silent:
            event = {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": role, "text": text},
            }
            return await state.inject(event, "conversation.item.created", wait_ready=False)
        if role == "assistant":
            text = say_line_instruction(text)
            if len(text) > 4000:
                raise ValueError("Prism spoken instruction exceeds 4000 characters")
        response = {
            "metadata": {"prism_cause": "user_text" if role == "user" else "proactive"},
            "prism_input_text" if role == "user" else "prism_instruction": text,
        }
        return await state.inject(
            {"type": "response.create", "response": response}, "response.created", wait_ready=True
        )

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        state = self._sessions.get(session.id)
        if state is None or state.closed:
            self._log_dropped_result(session, call_id)
            return
        await state.tools.result(call_id, result)

    async def interrupt(self, session: VoiceSession) -> None:
        state = self._state(session)
        if state.response_id:
            await state.send({"type": "response.cancel", "response_id": state.response_id})

    async def truncate_audio(self, session: VoiceSession, audio_end_ms: int) -> None:
        state = self._state(session)
        if state.output_item:
            await state.request(
                {
                    "type": "conversation.item.truncate",
                    "item_id": state.output_item,
                    "content_index": 0,
                    "audio_end_ms": max(0, min(audio_end_ms, state.output_bytes // 48)),
                },
                "conversation.item.truncated",
            )

    async def reconfigure(
        self,
        session: VoiceSession,
        *,
        system_prompt: str | None = None,
        voice: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> None:
        """Apply a per-session patch after the server acknowledges it."""
        _check_config(temperature, provider_config)
        state = self._state(session)
        pc = provider_config or {}
        patch: dict[str, Any] = {}
        if voice is not None:
            patch["voice"] = _voice(voice)
        turn = _turn_settings(pc)
        for key in ("eagerness", "prism_end_of_turn_timeout_ms"):
            if key in turn and turn[key] != state.settings.get("turn_detection", {}).get(key):
                raise ValueError(f"Prism {key} is fixed for the session")
        if turn:
            patch["turn_detection"] = turn
        _add_settings(patch, system_prompt, tools, pc)
        async with state.update_lock:
            await state.request({"type": "session.update", "session": patch}, "session.updated")

    async def send_event(self, session: VoiceSession, event: dict[str, Any]) -> None:
        """Send a raw Prism event. Its errors still reach on_error."""
        await self._state(session).send(event)

    def is_responding(self, session_id: str) -> bool:
        state = self._sessions.get(session_id)
        return state is not None and bool(state.response_id or state.response_requested)

    async def disconnect(self, session: VoiceSession) -> None:
        connecting = self._connecting.get(session.id)
        if connecting is not None and connecting is not asyncio.current_task():
            await cancel_and_wait(connecting)
        state = self._sessions.get(session.id)
        if state is not None:
            await state.close()

    async def close(self) -> None:
        for task in tuple(self._connecting.values()):
            if task is not asyncio.current_task():
                await cancel_and_wait(task)
        await asyncio.gather(*(state.close() for state in tuple(self._sessions.values())))
        await super().close()

    def _state(self, session: VoiceSession) -> PrismSession:
        state = self._sessions.get(session.id)
        if state is None or state.closed:
            raise RuntimeError("Prism session is not connected")
        return state


def _voice(value: str) -> str:
    if value not in _VOICES:
        raise ValueError(f"Prism voice must be one of {', '.join(_VOICES)}")
    return value


def _check_config(temperature: float | None, config: dict[str, Any] | None) -> None:
    if temperature is not None:
        raise ValueError("Prism does not support temperature")
    unknown = (config or {}).keys() - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unsupported Prism provider_config keys: {', '.join(sorted(unknown))}")


def _turn_settings(config: dict[str, Any]) -> dict[str, Any]:
    turn = {key: config[key] for key in ("interrupt_response", "eagerness") if key in config}
    if "interrupt_response" in turn and not isinstance(turn["interrupt_response"], bool):
        raise ValueError("interrupt_response must be a boolean")
    if "eagerness" in turn and turn["eagerness"] not in {"auto", "low", "medium", "high"}:
        raise ValueError("Prism eagerness must be auto, low, medium or high")
    if "end_of_turn_timeout_ms" in config:
        timeout = config["end_of_turn_timeout_ms"]
        if not isinstance(timeout, int) or not 1000 <= timeout <= 10000:
            raise ValueError("end_of_turn_timeout_ms must be an integer between 1000 and 10000")
        turn["prism_end_of_turn_timeout_ms"] = timeout
    return turn


def _add_settings(
    settings: dict[str, Any],
    prompt: str | None,
    tools: list[dict[str, Any]] | None,
    config: dict[str, Any],
) -> None:
    if prompt is not None:
        settings["instructions"] = prompt
    if tools is not None:
        settings["tools"] = [_tool(tool) for tool in tools]
    if "truncation" in config:
        settings["truncation"] = config["truncation"]


def _tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type", "function") != "function":
        raise ValueError("Prism supports function tools only")
    spec = tool.get("function", tool)
    if not spec.get("name"):
        raise ValueError("Prism function tools require a name")
    result = {key: spec[key] for key in ("name", "description", "prism_filler") if key in spec}
    result.update(type="function", parameters=declared_parameters(spec.get("parameters")))
    return result
