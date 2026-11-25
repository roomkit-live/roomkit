"""The audio bridge's steps for RFC §12.7.

``voice.open`` configures the bridge (``bridge``, an AudioBridgeConfig) and may
build the channel with no TTS provider; ``voice.add_backend`` adds a scripted
transport to the channel; ``voice.connect`` has a transport take a call on its
own, which ``voice.join`` then joins (the push model);
``voice.client_disconnected`` is a transport reporting its client gone; and
``voice.read`` reports what each transport was asked to do (``transports``).
A bridge hook's ``modify`` replaces the frame it forwards.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from roomkit.voice import AudioBridgeConfig, BridgeAudioEvent
from tests.conformance._voice_transport import PRIMARY, transport_of

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness

# What a transport is asked to do for one session, or for itself (``close``),
# as ``voice.read`` reports it (RFC §12.2 VoiceBackend, §12.7.3).
_TRANSPORT_CALLS = (
    "connect",
    "start_listening",
    "send_audio",
    "send_audio_sync",
    "send_transcription",
    "cancel_audio",
    "send_dtmf",
    "disconnect",
    "close",
)


def bridge_config(spec: Mapping[str, Any]) -> bool | AudioBridgeConfig:
    """``voice.open``'s ``bridge``: true for the defaults, or an AudioBridgeConfig."""
    bridge = spec.get("bridge", False)
    if isinstance(bridge, Mapping):
        return AudioBridgeConfig(**bridge)
    return bool(bridge)


async def _add_backend(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    """The channel's ``add_backend``: the named transport, built on its first add."""
    fixture = harness.voice_fixtures[spec["channel_id"]]
    name = spec["backend"]
    transport = fixture.transports.get(name)
    if transport is None:
        assert fixture.new_transport is not None
        transport = fixture.new_transport(spec)
        fixture.transports[name] = transport
    fixture.channel.add_backend(transport)


async def _connect(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    """The transport takes the participant's call on its own: the session it
    makes is the participant's, joined to no room until ``voice.join`` pushes it."""
    fixture = harness.voice_fixtures[spec["channel_id"]]
    transport = fixture.transports[spec.get("backend", PRIMARY)]
    participant = spec["participant_id"]
    fixture.sessions[participant] = await transport.connect(
        spec.get("room_id", fixture.room_id), participant, fixture.channel.channel_id
    )


async def _client_disconnected(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    """The participant's transport reports its client gone (``on_client_disconnected``)."""
    fixture = harness.voice_fixtures[spec["channel_id"]]
    participant = spec["participant_id"]
    session = fixture.sessions[participant]
    await transport_of(fixture, participant).simulate_client_disconnected(session)


def read_transports(fixture: Any) -> dict[str, list[dict[str, Any]]]:
    """Each transport's calls, each naming its session's participant: every
    distinct call once, in the order it was first made (how many times a
    transport is asked for a chunk or a transcription is the implementation's)."""
    owners = {sid: p for t in fixture.transports.values() for sid, p in t.owners.items()}
    read: dict[str, list[dict[str, Any]]] = {}
    for name, transport in fixture.transports.items():
        calls: list[dict[str, Any]] = []
        for call in transport.calls:
            if call.method not in _TRANSPORT_CALLS:
                continue
            record: dict[str, Any] = {"call": call.method}
            session_id = call.args.get("session_id")
            if call.method == "connect":
                record["participant_id"] = call.args["participant_id"]
            elif session_id is not None:
                record["participant_id"] = owners.get(session_id, session_id)
            if record not in calls:
                calls.append(record)
        read[name] = calls
    return read


def patched_bridge_audio(payload: BridgeAudioEvent, patch: Mapping[str, Any]) -> BridgeAudioEvent:
    """A bridge hook's ``modify``: ``event.frame.data`` (base64) replaces the bytes."""
    if set(patch) != {"frame"} or set(patch["frame"]) != {"data"}:
        raise ValueError("a bridge hook's modify replaces event.frame.data alone")
    data = base64.b64decode(patch["frame"]["data"], validate=True)
    return replace(payload, frame=replace(payload.frame, data=data))


BRIDGE_CALLS = {
    "voice.add_backend": _add_backend,
    "voice.connect": _connect,
    "voice.client_disconnected": _client_disconnected,
}
BRIDGE_ARGUMENTS = {
    "voice.add_backend": {
        "channel_id",
        "backend",
        "capabilities",
        "session_ids",
        "sample_rates",
        "playback_gate",
    },
    "voice.connect": {"channel_id", "participant_id", "backend", "room_id"},
    "voice.client_disconnected": {"channel_id", "participant_id"},
}
BRIDGE_OPEN_ARGUMENTS = {"tts"}
