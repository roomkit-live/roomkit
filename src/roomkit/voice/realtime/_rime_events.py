"""Map Prism server events onto RoomKit's realtime callbacks."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.voice.realtime._rime_session import PrismSession


class PrismEvents:
    def __init__(self, state: PrismSession) -> None:
        self.state = state
        self._handlers = {
            "session.updated": self._updated,
            "prism.typed_input.ready": self._ready,
            "input_audio_buffer.speech_started": self._speech_start,
            "input_audio_buffer.speech_stopped": self._speech_end,
            "conversation.item.input_audio_transcription.delta": self._user_delta,
            "conversation.item.input_audio_transcription.completed": self._user_final,
            "conversation.item.input_audio_transcription.failed": self._transcription_failed,
            "response.created": self._response_start,
            "response.output_item.added": self._output_item,
            "response.text.delta": self._text_delta,
            "response.text.done": self._text_done,
            "response.audio.delta": self._audio,
            "response.done": self._response_done,
            "response.function_call_arguments.done": state.tools.call,
            "conversation.item.created": self._item_created,
            "error": self._error,
            "session.idle_timeout": self._idle_timeout,
        }

    async def handle(self, event: dict[str, Any]) -> None:
        handler = self._handlers.get(event.get("type", ""))
        if handler:
            await handler(event)

    async def _updated(self, event: dict[str, Any]) -> None:
        self.state.settings = event["session"]

    async def _ready(self, event: dict[str, Any]) -> None:
        if self.state.response_id or self.state.response_requested or self.state.speaking:
            return
        self.state.ready.set()
        await self.state.tools.continue_ready()

    async def _speech_start(self, event: dict[str, Any]) -> None:
        self.state.speaking = True
        self.state.ready.clear()
        await self.state.fire("speech_start")

    async def _speech_end(self, event: dict[str, Any]) -> None:
        self.state.speaking = False
        await self.state.fire("speech_end")

    async def _user_delta(self, event: dict[str, Any]) -> None:
        await self.state.fire("transcription", event.get("delta", ""), "user", False)

    async def _user_final(self, event: dict[str, Any]) -> None:
        # This may arrive after response.done. Never attach it to the active reply.
        await self.state.fire("transcription", event.get("transcript", ""), "user", True)

    async def _transcription_failed(self, event: dict[str, Any]) -> None:
        error = event.get("error", {})
        await self.state.fire(
            "error",
            error.get("code", "transcription_failed"),
            error.get("message", "Prism transcription failed"),
        )

    async def _response_start(self, event: dict[str, Any]) -> None:
        self.state.response_id = event["response"]["id"]
        self.state.response_requested = None
        self.state.ready.clear()
        await self.state.fire("response_start")

    async def _output_item(self, event: dict[str, Any]) -> None:
        item = event.get("item", {})
        if item.get("type") == "message" and item.get("role") == "assistant":
            self.state.output_item = item.get("id")
            self.state.output_bytes = 0

    async def _text_delta(self, event: dict[str, Any]) -> None:
        await self.state.fire("transcription", event.get("delta", ""), "assistant", False)

    async def _text_done(self, event: dict[str, Any]) -> None:
        await self.state.fire("transcription", event.get("text", ""), "assistant", True)

    async def _audio(self, event: dict[str, Any]) -> None:
        audio = base64.b64decode(event.get("delta", ""), validate=True)
        if len(audio) % 2:
            raise ValueError("Prism returned an incomplete PCM16 sample")
        self.state.output_bytes += len(audio)
        if audio:
            await self.state.fire("audio", audio)

    async def _response_done(self, event: dict[str, Any]) -> None:
        state = self.state
        response = event["response"]
        if usage := response.get("usage"):
            state.owner._record_usage(
                state.session,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                details=usage,
            )
        if state.response_id == response.get("id"):
            state.response_id = None
            await state.fire("response_end")
        await state.tools.finished(response)

    async def _item_created(self, event: dict[str, Any]) -> None:
        await self.state.tools.acknowledged(event.get("item", {}))

    async def _error(self, event: dict[str, Any]) -> None:
        state = self.state
        error = event.get("error", {})
        if error.get("event_id") == state.response_requested:
            state.response_requested = None
        if error.get("scope") == "tool_roundtrip":
            owner = error.get("owner", {})
            await state.tools.abandon(owner.get("response_id", ""), owner.get("call_ids"))
        code, message = error.get("code", "unknown"), error.get("message", "Prism failed")
        if error.get("scope") == "session":
            await state.fail(code, message)
        else:
            await state.fire("error", code, message)

    async def _idle_timeout(self, event: dict[str, Any]) -> None:
        await self.state.fail("idle_timeout", "Prism closed an idle conversation")
