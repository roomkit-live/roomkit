"""Scripted recognition for the voice scenarios (RFC §12.2, §12.2.3, §12.3.12).

A recognizer whose streams follow the scenario's ``stt`` script, a turn
detector whose verdicts may wait at a gate, and the calls that choose a
session's STT language and flush a batch-mode buffer.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

from roomkit import VoiceChannel
from roomkit.voice.base import AudioChunk, SpeakerSegment, TranscriptionResult
from roomkit.voice.pipeline.turn.base import TurnContext, TurnDecision, TurnDetector
from roomkit.voice.stt.base import STTProvider
from tests.channel_tasks import settled

if TYPE_CHECKING:
    from roomkit import RoomKit
    from roomkit.models.event import AudioContent
    from roomkit.voice.audio_frame import AudioFrame
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates


_CONTINUOUS_LOOP = "run_continuous"
"""The coroutine of a continuous-mode session's STT loop (roomkit's own name)."""


class ScriptedStreamError(RuntimeError):
    """A stream the script fails, as a provider's error ends one."""


def _result(spec: Mapping[str, Any]) -> TranscriptionResult:
    segments = [
        SpeakerSegment(speaker=item["speaker"], text=item["text"])
        for item in spec.get("segments", [])
    ]
    return TranscriptionResult(
        text=spec["text"],
        is_final=spec.get("final", True),
        language=spec.get("language"),
        segments=segments,
        is_speech_start=spec.get("speech_start", False),
    )


def _milliseconds(data: bytes, sample_rate: int, channels: int) -> float:
    return len(data) * 1000 / (2 * max(channels, 1) * sample_rate)


class ScriptedSTT(STTProvider):
    """A streaming recognizer whose every stream follows the next ``streams`` entry.

    A stream gives its next ``results`` item on each chunk with sound it
    receives. Once they are all given, a stream with ``then`` holds at its
    ``gate`` and ends or fails; one without reads its audio to the end, holds
    at its ``gate``, then gives its ``at_end`` results. Each stream records the
    language it was opened with and how much sound and silence it heard.
    """

    def __init__(self, spec: Mapping[str, Any], gates: Gates) -> None:
        self._spec = spec
        self._gates = gates
        self.streams: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "ScriptedSTT"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_diarization(self) -> bool:
        return bool(self._spec.get("supports_diarization", False))

    @property
    def keeps_stream_across_turns(self) -> bool:
        return bool(self._spec.get("keeps_stream_across_turns", False))

    @property
    def supports_language_override(self) -> bool:
        return bool(self._spec.get("supports_language_override", False))

    def _open(self, language: str | None) -> tuple[dict[str, Any], Mapping[str, Any]]:
        record: dict[str, Any] = {"language": language, "sound_ms": 0.0, "silence_ms": 0.0}
        scripts = self._spec.get("streams", [])
        script = scripts[len(self.streams)] if len(self.streams) < len(scripts) else {}
        self.streams.append(record)
        return record, script

    async def _finish(self, script: Mapping[str, Any]) -> None:
        if "gate" in script:
            await self._gates.hold(script["gate"])
        if script.get("then") == "fail":
            raise ScriptedStreamError("the scripted stream fails")

    async def transcribe(
        self, audio: AudioContent | AudioChunk | AudioFrame, *, language: str | None = None
    ) -> TranscriptionResult:
        """A batch call is a stream of one chunk: its first final answers it."""
        record, script = self._open(language)
        data = getattr(audio, "data", b"")
        rate = getattr(audio, "sample_rate", 16000)
        key = "sound_ms" if any(data) else "silence_ms"
        record[key] += _milliseconds(data, rate, getattr(audio, "channels", 1))
        await self._finish(script)
        for spec in [*script.get("results", []), *script.get("at_end", [])]:
            if spec.get("final", True):
                return _result(spec)
        return TranscriptionResult(text="")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        record, script = self._open(language)
        results = list(script.get("results", []))
        then = script.get("then")
        if then and not results:
            await self._finish(script)
            return
        async for chunk in audio_stream:
            sound = any(chunk.data)
            key = "sound_ms" if sound else "silence_ms"
            record[key] += _milliseconds(chunk.data, chunk.sample_rate, chunk.channels)
            if sound and results:
                yield _result(results.pop(0))
            if then and not results:
                await self._finish(script)
                return
        await self._finish(script)
        for spec in script.get("at_end", []):
            yield _result(spec)

    def read(self) -> list[dict[str, Any]]:
        return [
            {
                "language": s["language"],
                "sound_ms": round(s["sound_ms"]),
                "silence_ms": round(s["silence_ms"]),
            }
            for s in self.streams
        ]


class ScriptedTurnDetector(TurnDetector):
    """One ``turn_decisions`` verdict per evaluation; a ``gate`` holds it first.

    The channel runs a detector off its event loop, so the verdict waits for
    its gate on that loop.
    """

    def __init__(
        self, decisions: list[Mapping[str, Any]], gates: Gates, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._decisions = list(decisions)
        self._gates = gates
        self._loop = loop

    @property
    def name(self) -> str:
        return "ScriptedTurnDetector"

    def evaluate(self, context: TurnContext) -> TurnDecision:
        spec: Mapping[str, Any] = (
            self._decisions.pop(0) if self._decisions else {"is_complete": True}
        )
        if "gate" in spec:
            held = asyncio.run_coroutine_threadsafe(self._gates.hold(spec["gate"]), self._loop)
            held.result(timeout=10)
        return TurnDecision(
            is_complete=spec["is_complete"],
            confidence=spec.get("confidence", 1.0),
            suggested_wait_ms=spec.get("suggested_wait_ms"),
        )


def recognizer(spec: Mapping[str, Any], gates: Gates) -> ScriptedSTT | None:
    return ScriptedSTT(spec["stt"], gates) if "stt" in spec else None


def turn_detector(spec: Mapping[str, Any], gates: Gates) -> ScriptedTurnDetector | None:
    if "turn_decisions" not in spec:
        return None
    return ScriptedTurnDetector(spec["turn_decisions"], gates, asyncio.get_running_loop())


def _quiet(channel: VoiceChannel) -> bool:
    """No audio waits for a stream, and only session-long loops still run."""
    return all(state.queue.empty() for state in channel._stt_streams.values()) and not any(
        not task.done() and not _lasts(task) for task in channel._scheduled_tasks
    )


def _lasts(task: asyncio.Task[Any]) -> bool:
    return getattr(task.get_coro(), "__name__", "") == _CONTINUOUS_LOOP


async def settle_voice(channel: VoiceChannel) -> None:
    """Until the voice work the steps started has run (README, ``voice.settle``).

    A continuous STT loop lasts as long as its session: what it was fed is
    outstanding until its queue is empty and the work it scheduled has run.
    """
    async with asyncio.timeout(5):
        while True:
            await settled(channel, lasting=(_CONTINUOUS_LOOP,))
            if _quiet(channel):
                await asyncio.sleep(0)
                if _quiet(channel):
                    return
            await asyncio.sleep(0.01)


async def _set_stt_language(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    fixture.channel.set_stt_language(fixture.sessions[spec["participant_id"]], spec["language"])


async def _flush_stt(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    session = fixture.sessions[spec["participant_id"]]
    result = await fixture.channel.flush_stt(session, route=spec.get("route", False))
    return {"text": result.text}


STT_CALLS = {"voice.set_stt_language": _set_stt_language, "voice.flush_stt": _flush_stt}
STT_ARGUMENTS = {
    "voice.set_stt_language": {"channel_id", "participant_id", "language"},
    "voice.flush_stt": {"channel_id", "participant_id", "route"},
}
OPEN_ARGUMENTS = {
    "stt",
    "batch_mode",
    "pipeline_speakers",
    "turn_decisions",
    "turn_incomplete_wait_ms",
}
