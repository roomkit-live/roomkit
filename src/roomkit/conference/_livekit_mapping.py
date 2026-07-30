"""Translation between RoomKit's conference vocabulary and LiveKit's.

Everything here is a pure function over primitives — no ``livekit`` import, no
SDK object. That is deliberate: the extra is optional, so a contract that lived
inside the backend class would only be checkable on a machine that installed
LiveKit, and the translations are exactly the part worth checking. Callers pass
protobuf enum *names* (``rtc.TrackKind.Name(...)``) rather than their integer
values, because a name is readable in an assertion and a number is not.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from roomkit.conference.models import (
    ConferenceCapability,
    ConferenceGrants,
    ConferenceParticipant,
    ConferenceTrack,
    TrackKind,
)
from roomkit.voice.base import AudioChunk

# LiveKit's own names for the sources a token may grant. Passed as strings
# because that is what VideoGrants.can_publish_sources carries.
MICROPHONE = "microphone"
CAMERA = "camera"
SCREEN_SHARE = "screen_share"

SIP_ATTRIBUTE_PREFIX = "sip."
"""Prefix LiveKit's SIP service uses for the attributes it sets itself."""

SIP_PARTICIPANT_KIND = "PARTICIPANT_KIND_SIP"
"""``ParticipantKind`` name LiveKit gives a participant that dialled in."""

_QUALITY_LABELS = {
    "QUALITY_EXCELLENT": "excellent",
    "QUALITY_GOOD": "good",
    "QUALITY_POOR": "poor",
    "QUALITY_LOST": "lost",
}

_INBOUND_VIDEO_CODECS = {
    "I420": "raw_yuv420p",
    "NV12": "raw_nv12",
    "RGB24": "raw_rgb24",
}


def capabilities_for(*, remote_unmute: bool, sip_gateway: bool) -> ConferenceCapability:
    """What the LiveKit backend declares, given what the deployment allows.

    Three are unconditional because this backend translates the events behind
    them: screen-share tracks arrive on their own source, dominant-speaker
    changes and quality reports both have handlers. Two are conditional because
    they turn on server configuration this backend cannot read — remote unmute
    needs ``room.enable_remote_unmute``, and a phone cannot reach a room without
    a SIP trunk and a dispatch rule.

    Everything else is left out on purpose; see
    :attr:`LiveKitConferenceBackend.capabilities`.
    """
    capabilities = (
        ConferenceCapability.SCREEN_SHARE
        | ConferenceCapability.ACTIVE_SPEAKER
        | ConferenceCapability.CONNECTION_QUALITY
    )
    if remote_unmute:
        capabilities |= ConferenceCapability.REMOTE_UNMUTE
    if sip_gateway:
        capabilities |= ConferenceCapability.SIP_GATEWAY
    return capabilities


def publish_source_names(grants: ConferenceGrants) -> list[str]:
    """Which LiveKit track sources ``grants`` allows publishing.

    Screen-share audio is not listed: RoomKit's grant covers a screen share,
    and a participant sharing a tab with sound would need it separately. Adding
    it here would hand out a publish right no RoomKit grant asked for.
    """
    sources = []
    if grants.publish_audio:
        sources.append(MICROPHONE)
    if grants.publish_video:
        sources.append(CAMERA)
    if grants.publish_screen_share:
        sources.append(SCREEN_SHARE)
    return sources


def video_grant_kwargs(
    room_id: str,
    grants: ConferenceGrants,
    *,
    publish_data: bool,
) -> dict[str, Any]:
    """Translate ConferenceGrants into ``api.VideoGrants`` keyword arguments.

    ``can_publish_sources`` supersedes ``can_publish`` on the LiveKit side, so
    the two are set together rather than independently: a list is sent only
    when something may in fact be published, and a participant granted nothing
    gets ``can_publish=False`` with no list at all. An empty list would be a
    third state whose meaning is the server's to decide, and this boundary does
    not guess at it.

    ``publish_data`` has no ConferenceGrants equivalent, so the caller decides.
    The bot passes ``False`` — the framework configured it and knows it publishes
    no data — and a human keeps LiveKit's permissive default, because the
    framework does not know what an integrator's client application needs. That
    is the reasoning of :meth:`ConferenceGrants.for_bot`, applied one layer down.

    ``can_update_own_metadata`` is deliberately left unset, which is LiveKit's
    deny. It is what keeps a participant from writing its own attributes after
    joining, and therefore part of what makes :func:`asserted_attributes`
    truthful.
    """
    sources = publish_source_names(grants)
    kwargs: dict[str, Any] = {
        "room": room_id,
        "room_join": True,
        "can_publish": bool(sources),
        "can_subscribe": grants.subscribe,
        "can_publish_data": publish_data,
    }
    if sources:
        kwargs["can_publish_sources"] = sources
    if grants.moderate:
        kwargs["room_admin"] = True
    if grants.hidden:
        kwargs["hidden"] = True
    return kwargs


# LiveKit speaks two dialects of the same enums: the realtime SDK's FFI protocol
# names a track kind "KIND_AUDIO", while the server API calls the same thing
# "AUDIO". The realtime names are the ones this backend works in, because that
# is where the media arrives; these tables bring the control plane's answers over
# so that one participant looks the same whichever call reported it.
_RTC_TRACK_KIND_NAMES = {"AUDIO": "KIND_AUDIO", "VIDEO": "KIND_VIDEO", "DATA": "KIND_DATA"}
_RTC_TRACK_SOURCE_NAMES = {
    "UNKNOWN": "SOURCE_UNKNOWN",
    "CAMERA": "SOURCE_CAMERA",
    "MICROPHONE": "SOURCE_MICROPHONE",
    "SCREEN_SHARE": "SOURCE_SCREENSHARE",
    "SCREEN_SHARE_AUDIO": "SOURCE_SCREENSHARE_AUDIO",
}
_RTC_PARTICIPANT_KIND_NAMES = {
    "STANDARD": "PARTICIPANT_KIND_STANDARD",
    "INGRESS": "PARTICIPANT_KIND_INGRESS",
    "EGRESS": "PARTICIPANT_KIND_EGRESS",
    "SIP": SIP_PARTICIPANT_KIND,
    "AGENT": "PARTICIPANT_KIND_AGENT",
    "CONNECTOR": "PARTICIPANT_KIND_CONNECTOR",
    "BRIDGE": "PARTICIPANT_KIND_BRIDGE",
}


def rtc_track_kind_name(api_name: str) -> str:
    """Bring a server-API track type over to the realtime dialect.

    An unrecognised name is passed through rather than guessed at, so that
    :func:`track_kind_for` refuses it downstream instead of this table quietly
    inventing a kind for a LiveKit version that grew one.
    """
    return _RTC_TRACK_KIND_NAMES.get(api_name, api_name)


def rtc_track_source_name(api_name: str) -> str:
    """Bring a server-API track source over to the realtime dialect."""
    return _RTC_TRACK_SOURCE_NAMES.get(api_name, api_name)


def rtc_participant_kind_name(api_name: str) -> str:
    """Bring a server-API participant kind over to the realtime dialect.

    This one carries weight: :func:`asserted_attributes` decides provenance on
    the realtime name, so a control-plane answer that kept the API spelling
    would make a dial-in's caller number unasserted and leave the participant
    unresolvable.
    """
    return _RTC_PARTICIPANT_KIND_NAMES.get(api_name, api_name)


def track_kind_for(kind_name: str, source_name: str) -> TrackKind:
    """Map a LiveKit track kind and source onto RoomKit's TrackKind.

    Only a *video* track sourced from a screen share is ``SCREEN_SHARE``. The
    audio of a shared tab arrives on its own track and is audio like any other:
    it is speech to transcribe and sound to record, and calling it a screen
    share would route it away from both.

    Raises:
        ValueError: for a kind LiveKit could not classify. RoomKit has no
            unknown TrackKind, and inventing one would attach a lane to a track
            whose contents nothing has established.
    """
    if kind_name == "KIND_AUDIO":
        return TrackKind.AUDIO
    if kind_name == "KIND_VIDEO":
        if source_name == "SOURCE_SCREENSHARE":
            return TrackKind.SCREEN_SHARE
        return TrackKind.VIDEO
    raise ValueError(
        f"LiveKit reported a track of kind {kind_name!r}, which is neither audio nor video. "
        "RoomKit has no TrackKind for it, and guessing one would open a lane on a track "
        "whose contents nothing has established."
    )


def asserted_attributes(
    participant_kind_name: str, attributes: Mapping[str, str]
) -> dict[str, str]:
    """The subset of a LiveKit participant's attributes that LiveKit asserts.

    LiveKit carries one flat attribute map, and two very different things live
    in it: what the server established, and what a participant's own client
    supplied. Nothing in the map's shape tells them apart, and an address is
    only as good as its provenance (RFC section 12.10.2) — a caller that writes
    its own ``phone_number`` and is resolved on it reaches someone else's
    Identity.

    So the rule is narrow and rests on something a client cannot forge: the
    participant's *kind*. LiveKit's SIP service sets the ``sip.`` attributes and
    the server sets the kind, so on a participant whose kind is SIP those
    attributes are the server's own statement — that is where a dial-in's caller
    number comes from, and it is what makes a phone participant resolvable.
    Everything else is surfaced but unvouched, so a browser participant gets
    ``{}``: this backend distinguishes, and for that participant it asserts
    nothing.

    Not ``None``, which would say the backend cannot distinguish at all and
    would leave even a dial-in unresolvable. And deliberately not "every
    attribute on a minted participant", which would be defensible only as long
    as no token in the deployment grants ``can_update_own_metadata`` — a
    condition this backend cannot see and therefore must not rely on.
    """
    if participant_kind_name != SIP_PARTICIPANT_KIND:
        return {}
    return {
        key: value for key, value in attributes.items() if key.startswith(SIP_ATTRIBUTE_PREFIX)
    }


def participant_record(
    *,
    identity: str,
    sid: str,
    kind_name: str,
    name: str,
    metadata: str,
    attributes: Mapping[str, str],
    connected_at: datetime | None,
) -> ConferenceParticipant:
    """Build a ConferenceParticipant from what LiveKit reports about one.

    ``participant_id`` is LiveKit's identity, which is the value the framework
    passed to ``mint_access()`` and got echoed back — the correlation rule 2 of
    RFC section 12.10.2 depends on, and the reason no mapping table is needed
    here.

    Two provider fields sit beside the attributes under a ``livekit.`` prefix
    rather than merged flat, so that no LiveKit field can shadow an attribute
    key an integrator reads. Of the four, ``sid`` and ``kind`` are the server's
    own; ``name`` and ``metadata`` come from a token this backend did not
    necessarily mint, so they are surfaced without being asserted.
    """
    surfaced: dict[str, Any] = dict(attributes)
    surfaced["livekit.sid"] = sid
    surfaced["livekit.kind"] = kind_name
    if name:
        surfaced["livekit.name"] = name
    if metadata:
        surfaced["livekit.metadata"] = metadata
    asserted: dict[str, Any] = dict(asserted_attributes(kind_name, attributes))
    asserted["livekit.sid"] = sid
    asserted["livekit.kind"] = kind_name
    record = ConferenceParticipant(
        participant_id=identity,
        metadata=surfaced,
        asserted_metadata=asserted,
    )
    if connected_at is not None:
        record.connected_at = connected_at
    return record


def track_record(
    *,
    sid: str,
    room_id: str,
    participant_id: str,
    kind_name: str,
    source_name: str,
    muted: bool,
    name: str = "",
    mime_type: str = "",
) -> ConferenceTrack:
    """Build a ConferenceTrack from a LiveKit publication.

    The publication's ``sid`` becomes the track id the framework subscribes,
    mutes and records on, so it is the one identifier both sides agree about.
    """
    metadata: dict[str, Any] = {"sid": sid, "source": source_name}
    if name:
        metadata["name"] = name
    if mime_type:
        metadata["mime_type"] = mime_type
    return ConferenceTrack(
        id=sid,
        room_id=room_id,
        participant_id=participant_id,
        kind=track_kind_for(kind_name, source_name),
        muted=muted,
        metadata=metadata,
    )


def quality_label(quality_name: str) -> str | None:
    """Map a LiveKit ``ConnectionQuality`` name onto a report label.

    ``None`` for a quality LiveKit itself calls unknown: the callback exists to
    report a level, and forwarding "unknown" as one would put a word in the
    integrator's dashboard that means less than no report at all.
    """
    return _QUALITY_LABELS.get(quality_name)


def codec_for_buffer_type(buffer_type_name: str) -> str:
    """Map a LiveKit ``VideoBufferType`` name onto a RoomKit raw video codec.

    Raises:
        ValueError: for a buffer layout RoomKit's VideoFrame has no codec for.
            Converting it here would be media-plane work in the wrong place.
    """
    codec = _INBOUND_VIDEO_CODECS.get(buffer_type_name)
    if codec is None:
        raise ValueError(
            f"LiveKit delivered video as {buffer_type_name!r}, which RoomKit's VideoFrame "
            f"has no codec for (known: {sorted(_INBOUND_VIDEO_CODECS)})"
        )
    return codec


SAMPLE_WIDTH = 2
"""Bytes per sample on both sides: LiveKit's audio frames are 16-bit signed."""

PUBLISHABLE_FORMATS = frozenset({"pcm", "pcm_s16le"})
"""Chunk formats ``rtc.AudioFrame`` can carry. It is 16-bit signed, always."""


def require_publishable_pcm(chunk: AudioChunk) -> None:
    """Reject a chunk this backend cannot put on the bot's track.

    Two refusals, for two different reasons. An encoded chunk is refused because
    encoding belongs to the backend (RFC section 12.10.3): a caller choosing the
    wire format would defeat the boundary this interface exists to draw. Another
    PCM width is refused because ``rtc.AudioFrame`` is 16-bit signed and nothing
    else — handing it 32-bit float would not fail, it would publish noise, and
    noise that reaches a conference is worse than a chunk that was refused.
    """
    if not chunk.format.startswith("pcm"):
        raise ValueError(
            f"publish_audio expects decoded PCM, got format {chunk.format!r}. "
            "Encoding belongs to the backend: a caller choosing the wire format "
            "defeats the abstraction boundary."
        )
    if chunk.format not in PUBLISHABLE_FORMATS:
        raise ValueError(
            f"LiveKit publishes 16-bit signed PCM, and this chunk is {chunk.format!r}. "
            f"Reinterpreting it would publish noise rather than fail. Accepted: "
            f"{sorted(PUBLISHABLE_FORMATS)}."
        )
