"""What the scripted transport sends out, for the voice scenarios (RFC §12.2, §12.3.6).

``voice.send_dtmf`` sends a digit through the channel; the transport sends it
only when it declares ``dtmf_signaling``, and otherwise keeps the base
interface's refusal. ``playback_fails`` names the participants whose playback
the transport fails. ``voice.read`` returns the digits sent and how many
times each participant's playback was cancelled.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

from roomkit.voice import AudioChunk, VoiceBackend, VoiceCapability, VoiceSession
from tests.conformance._voice_synthesis import UnsupportedError
from tests.conformance._voice_transport import PipelineTransport

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness


class ScriptedPlaybackError(RuntimeError):
    """A playback the script fails, as a transport's error ends one."""


class OutboundTransport(PipelineTransport):
    """The pipeline transport, sending DTMF as its capabilities say, failing
    the playback of the participants ``playback_fails`` names."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        super().__init__(spec)
        self._failing: set[str] = set(spec.get("playback_fails", []))

    def send_dtmf(self, session: VoiceSession, digit: str, duration_ms: int = 160) -> None:
        if VoiceCapability.DTMF_SIGNALING not in self.capabilities:
            # A transport that cannot signal keeps the interface's refusal.
            VoiceBackend.send_dtmf(self, session, digit, duration_ms)
        super().send_dtmf(session, digit, duration_ms)

    async def send_audio(
        self, session: VoiceSession, audio: bytes | AsyncIterator[AudioChunk]
    ) -> None:
        if session.participant_id in self._failing:
            raise ScriptedPlaybackError(f"the playback to {session.participant_id} fails")
        await super().send_audio(session, audio)


def read_outbound(
    transports: list[PipelineTransport], sessions: Mapping[str, VoiceSession]
) -> dict[str, Any]:
    """``voice.read``'s view of what the transports sent out: the digits each
    participant's transport sent and how many times its playback was cancelled."""
    sent = [entry for transport in transports for entry in transport.sent_dtmf]
    cancels = [
        call.args["session_id"]
        for transport in transports
        for call in transport.calls
        if call.method == "cancel_audio"
    ]
    return {
        "dtmf_sent": {
            name: [
                {"digit": digit, "duration_ms": duration}
                for sid, digit, duration in sent
                if sid == session.id
            ]
            for name, session in sessions.items()
        },
        "cancel_audio": {name: cancels.count(session.id) for name, session in sessions.items()},
    }


async def _send_dtmf(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    session = fixture.sessions[spec["participant_id"]]
    arguments = {"duration_ms": spec["duration_ms"]} if "duration_ms" in spec else {}
    try:
        fixture.channel.send_dtmf(session, spec["digit"], **arguments)
    except NotImplementedError as error:
        raise UnsupportedError(str(error)) from error


OUTBOUND_CALLS = {"voice.send_dtmf": _send_dtmf}
OUTBOUND_ARGUMENTS = {
    "voice.send_dtmf": {"channel_id", "participant_id", "digit", "duration_ms"},
}
OUTBOUND_OPEN_ARGUMENTS = {"playback_fails"}
