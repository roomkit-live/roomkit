"""A realtime channel's configuration and its sessions', what its provider
and its transport were asked, its delegations' outputs, and the usage its
provider reports (roomkit-specs ``conformance/README.md``, "Scripted
realtime sessions"; RFC §12.4, §12.4.1, §12.4.2).

``realtime.reconfigure`` reconfigures one of the channel's sessions, another
room's among them; ``realtime.usage`` has the provider report what the
channel's first session consumed, which the ``on_usage`` callback the
channel's spec scripts records. ``realtime.read`` reads what the provider and
the transport were asked, and the usage both surfaces carry.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit.voice.realtime.mock import MockCall, MockRealtimeProvider, MockRealtimeTransport

if TYPE_CHECKING:
    from roomkit import RoomKit
    from roomkit.voice.base import VoiceSession
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates

# The members of a channel's ``realtime`` spec this module reads, and of its
# ``on_usage``; the session's own configuration ``realtime.start`` and
# ``realtime.configure`` take.
SESSION_MEMBERS = {"system_prompt", "voice", "on_usage"}
ON_USAGE_MEMBERS = {"gate", "fails"}
CONFIGURATION = ("system_prompt", "voice")

# The calls ``provider_calls`` and ``transport_calls`` list, by method name.
_PROVIDER_CALLS = {
    "interrupt",
    "truncate_audio",
    "send_activity_start",
    "send_activity_end",
    "disconnect",
}
_TRANSPORT_CALLS = {"interrupt", "disconnect"}


class ScriptedUsageCallbackError(RuntimeError):
    """The failure an ``on_usage`` callback the spec scripts raises."""


class RecordingTransport(MockRealtimeTransport):
    """The mock transport, which also records each ``interrupt`` it is asked:
    the channel's flush of its playback (RFC §12.4.1)."""

    def interrupt(self, session: VoiceSession) -> None:
        self.calls.append(MockCall(method="interrupt", args={"session_id": session.id}))


def prompt_and_voice(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The prompt and voice *spec* names: the channel's, from its ``realtime``
    spec, or the session's own, from a ``realtime.start``."""
    return {key: spec[key] for key in CONFIGURATION if key in spec}


def session_names(session: VoiceSession) -> dict[str, str]:
    """A session as the reads name it: its room and its participant."""
    return {"room_id": session.room_id, "participant_id": session.participant_id}


def configuration(session: VoiceSession, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """A connect's or a reconfiguration's entry: the session's room, and the
    prompt and voice the provider was asked."""
    return {
        "room_id": session.room_id,
        "system_prompt": kwargs.get("system_prompt"),
        "voice": kwargs.get("voice"),
    }


def on_usage(
    provider: MockRealtimeProvider, spec: Mapping[str, Any], gates: Gates
) -> list[dict[str, Any]]:
    """Register the ``on_usage`` callback the spec scripts, if any; the list
    its calls are recorded in, before it holds at its gate or fails."""
    reports: list[dict[str, Any]] = []
    script = spec.get("on_usage")
    if script is None:
        return reports

    async def callback(session: VoiceSession, usage: dict[str, Any]) -> None:
        reports.append({"session": session_names(session), "usage": dict(usage)})
        if "gate" in script:
            await gates.hold(script["gate"])
        if "fails" in script:
            raise ScriptedUsageCallbackError(script["fails"])

    provider.on_usage(callback)
    return reports


def read(
    provider: MockRealtimeProvider,
    transport: MockRealtimeTransport,
    session: VoiceSession | None,
    reconfigures: list[dict[str, Any]],
    usage_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """What ``realtime.read`` adds: the provider's and the transport's calls,
    the provider's reconfigurations and delegation outputs, the first
    session's ``last_usage`` (null while empty) and the ``on_usage`` calls."""
    last_usage = session.last_usage if session is not None else {}
    return {
        "reconfigures": list(reconfigures),
        "provider_calls": [
            _provider_call(call) for call in provider.calls if call.method in _PROVIDER_CALLS
        ],
        "transport_calls": [
            {"call": call.method} for call in transport.calls if call.method in _TRANSPORT_CALLS
        ],
        "delegation_outputs": [
            {"delegation_id": delegation_id, "text": text, "spoken": spoken}
            for _, delegation_id, text, spoken in provider.delegation_outputs
        ],
        "last_usage": last_usage or None,
        "usage_reports": list(usage_reports),
    }


def _provider_call(call: MockCall) -> dict[str, Any]:
    entry: dict[str, Any] = {"call": call.method}
    if call.method == "truncate_audio":
        entry["audio_end_ms"] = call.args["audio_end_ms"]
    return entry


READS_ARGUMENTS = {
    "realtime.reconfigure": {"channel_id", "room_id", "system_prompt", "voice", "tools"},
    "realtime.usage": {"channel_id", "input_tokens", "output_tokens", "details"},
}


async def reads_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    fixture = harness.realtime_sessions[spec["channel_id"]]
    match spec["call"]:
        case "realtime.reconfigure":
            await fixture.reconfigure(spec)
        case "realtime.usage":
            await fixture.provider.simulate_usage(
                fixture.session,
                spec["input_tokens"],
                spec["output_tokens"],
                details=spec.get("details"),
            )
        case _:
            raise ValueError(f"Unknown realtime call {spec['call']}")
    return None
