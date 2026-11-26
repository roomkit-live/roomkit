"""The settings a conference channel is built with, and what its model's tools
do (roomkit-specs ``conformance/README.md``, "Scripted conference options";
RFC §12.10.2, §12.10.4, §12.10.5, §12.10.12).

``conference.open``'s settings build the ConferenceChannel's own keywords: its
synthesizer, its interruption policy, E2EE and the scripted SFU's
capabilities, the bot's identity and explicit grants, the providers'
ownership, the identity keys, and the realtime model's tool handlers and
bounds. ``conference.cancel`` makes the model abandon calls, and
``conference.read`` reads what the synthesizer was asked, what a lane
discarded, which providers were closed and how each tool handler ended.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit import (
    ConferenceCapability,
    ConferenceGrants,
    ConferenceInterruptionConfig,
    ConferenceInterruptionScope,
    ConferenceRealtimeConfig,
    ToolPolicy,
)
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice import InterruptionStrategy
from roomkit.voice.pipeline import AudioPipelineConfig
from tests.conformance._calls import UnreadableError
from tests.conformance._voice_pipeline import add_pipeline_stages
from tests.conformance._voice_stt import ScriptedSTT
from tests.conformance._voice_synthesis import ScriptedTTS, _context_record, scripted_tts

if TYPE_CHECKING:
    from roomkit import RoomKit
    from roomkit.voice.realtime.mock import MockRealtimeProvider
    from tests.conformance._conference import ConferenceFixture
    from tests.conformance._conference_media import ConferenceMedia
    from tests.conformance._conference_sfu import ScriptedSFU
    from tests.conformance._harness import Harness

# The conference.open members this module reads.
OPTIONS_OPEN = {
    "tts",
    "speak_text_events",
    "interruption",
    "e2ee",
    "capabilities",
    "occupants",
    "bot_identity",
    "bot_grants",
    "close_room_on_detach",
    "close_providers",
    "identity_address_keys",
    "identity_trusts_unasserted_metadata",
    "recorder",
    "tool_handler",
    "tool_policy",
    "tool_timeout_seconds",
    "tool_timeouts",
}
# ConferenceChannel's own keywords, each passed as the scenario writes it.
_PLAIN = (
    "e2ee",
    "bot_identity",
    "close_room_on_detach",
    "close_providers",
    "identity_address_keys",
    "identity_trusts_unasserted_metadata",
)


@dataclass
class ConferenceOptions:
    """What the scenario scripted of a conference's own settings, kept to be read."""

    tts: ScriptedTTS | None = None
    closed: list[str] = field(default_factory=list)
    handlers: list[dict[str, str]] = field(default_factory=list)


def capabilities(spec: Mapping[str, Any]) -> ConferenceCapability:
    """The scripted SFU's capabilities ``conference.open`` names."""
    held = ConferenceCapability.NONE
    for name in spec.get("capabilities", []):
        held |= ConferenceCapability[name.upper()]
    return held


def _closing[T: (ScriptedSTT, ScriptedTTS)](provider: T, kind: str, closed: list[str]) -> T:
    """The provider, noting its kind in ``closed`` when the channel closes it."""
    close = provider.close

    async def closing() -> None:
        closed.append(kind)
        await close()

    provider.close = closing  # ty: ignore[invalid-assignment]
    return provider


def _interruption(spec: Mapping[str, Any]) -> ConferenceInterruptionConfig:
    members: dict[str, Any] = {}
    if "strategy" in spec:
        members["strategy"] = InterruptionStrategy(spec["strategy"])
    if "scope" in spec:
        members["scope"] = ConferenceInterruptionScope(spec["scope"])
    if "allowlist" in spec:
        members["allowlist"] = set(spec["allowlist"])
    return ConferenceInterruptionConfig(**members)


def media_spec(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """``conference.open`` as its media inputs read it: ``vad: null``, lanes
    with no VAD, is this module's to build."""
    if "vad" in spec and spec["vad"] is None:
        return {name: value for name, value in spec.items() if name != "vad"}
    return spec


def channel_options(
    spec: Mapping[str, Any], harness: Harness, options: dict[str, Any], media: ConferenceMedia
) -> ConferenceOptions:
    """Adds ``conference.open``'s settings to the channel's keywords *options*,
    the media inputs' already among them."""
    kept = ConferenceOptions()
    if isinstance(options.get("stt"), ScriptedSTT):
        options["stt"] = _closing(options["stt"], "stt", kept.closed)
    if "tts" in spec:
        kept.tts = _closing(scripted_tts(spec["tts"], harness.gates), "tts", kept.closed)
        options["tts"] = kept.tts
    options["speak_text_events"] = spec.get("speak_text_events", True)
    if "interruption" in spec:
        options["interruption"] = _interruption(spec["interruption"])
    if "bot_grants" in spec:
        options["bot_grants"] = ConferenceGrants(**spec["bot_grants"])
    options.update({name: spec[name] for name in _PLAIN if name in spec})
    if "vad" in spec and spec["vad"] is None:
        # Lanes with every stage the scenario names but a VAD.
        pipeline = AudioPipelineConfig()
        add_pipeline_stages(
            pipeline, {k: spec[k] for k in ("agc", "denoiser") if k in spec}, media.trace
        )
        options["pipeline"] = pipeline
    if "recorder" in spec:
        _recorder(spec, harness, options)
    return kept


def _recorder(spec: Mapping[str, Any], harness: Harness, options: dict[str, Any]) -> None:
    """``recorder``: none beside a recording, or a recorder with no recording."""
    name = spec["recorder"]
    if "recording" in spec and name is None:
        options["recorder"] = None
    elif "recording" not in spec and name is not None:
        recorder = MockMediaRecorder()
        harness.media_recorders[name] = recorder
        options["recorder"] = recorder
    else:
        raise UnreadableError("recorder is null beside recording, or a name without it")


def realtime_config(
    spec: Mapping[str, Any], harness: Harness, model: MockRealtimeProvider, kept: ConferenceOptions
) -> ConferenceRealtimeConfig:
    """The model ``conference.open``'s ``tools`` declares, each tool served by
    its scripted handler, with the handler, policy and bounds it names."""
    tools: Mapping[str, Any] = spec["tools"]

    async def answer(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        return await _handle(name, tools[name], harness, kept.handlers)

    members: dict[str, Any] = {}
    if "tool_policy" in spec:
        members["tool_policy"] = ToolPolicy.model_validate(spec["tool_policy"])
    if "tool_timeout_seconds" in spec:
        members["tool_timeout_seconds"] = spec["tool_timeout_seconds"]
    if "tool_timeouts" in spec:
        members["tool_timeouts"] = dict(spec["tool_timeouts"])
    return ConferenceRealtimeConfig(
        provider=model,
        tools=[
            {"name": name, "description": name, "parameters": {"type": "object"}} for name in tools
        ],
        tool_handler=None if spec.get("tool_handler") is False else answer,
        **members,
    )


async def _handle(
    name: str, tool: str | Mapping[str, Any], harness: Harness, handlers: list[dict[str, str]]
) -> str:
    """One run of a tool's scripted handler: a result string, or a script that
    waits at its gate, then returns or raises (README, Scripted conference options)."""
    if isinstance(tool, str):
        handlers.append({"name": name, "outcome": "returned"})
        return tool
    try:
        if "gate" in tool:
            await harness.gates.hold(tool["gate"])
    except asyncio.CancelledError:
        handlers.append({"name": name, "outcome": "interrupted"})
        raise
    if "fails" in tool:
        handlers.append({"name": name, "outcome": "raised"})
        raise RuntimeError(tool["fails"])
    handlers.append({"name": name, "outcome": "returned"})
    return tool["result"]


async def seat_occupants(backend: ScriptedSFU, spec: Mapping[str, Any]) -> None:
    """The participants the scripted SFU's conference holds before any channel
    listens to it: no callback announces them, only ``list_participants``."""
    for participant_id in spec.get("occupants", []):
        await backend.simulate_participant_joined(spec["room_id"], participant_id)


async def _cancel(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.conferences[spec["channel_id"]]
    sessions = fixture.channel.get_room_sessions(fixture.room)
    if fixture.model is None or not sessions:
        raise ValueError("the conference model must be connected first")
    await fixture.model.simulate_tool_call_cancellation(sessions[0], list(spec["call_ids"]))


def _lane_dropped(fixture: ConferenceFixture) -> dict[str, int]:
    """Per track whose lane runs, the frames it discarded, as the channel's
    lanes expose them."""
    return {
        track_id: lane.dropped_frames
        for track_id, lane in sorted(fixture.channel.active_lanes.items())
    }


def read_options(fixture: ConferenceFixture) -> dict[str, Any]:
    """``conference.read``'s members of the conference's own settings."""
    kept = fixture.options or ConferenceOptions()
    tts = kept.tts
    return {
        "synthesized": [call["text"] for call in tts.calls] if tts is not None else [],
        "tts_contexts": [_context_record(c) for c in tts.contexts] if tts is not None else [],
        "lane_dropped": _lane_dropped(fixture),
        "closed": list(kept.closed),
        "handlers": list(kept.handlers),
    }


OPTIONS_CALLS = {"conference.cancel": _cancel}
OPTIONS_ARGUMENTS = {"conference.cancel": {"channel_id", "call_ids"}}
