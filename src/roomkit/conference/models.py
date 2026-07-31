"""Data models for SFU conference orchestration.

RoomKit does not own the conference media plane: an external SFU routes media
between human participants, and RoomKit joins as a bot participant to provide
transcription, vision, AI voice and cross-channel integration. These models are
the vocabulary of that boundary — see RFC section 12.10.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Flag, StrEnum, auto, unique
from typing import Any

from roomkit.voice.interruption import InterruptionStrategy


@unique
class TrackKind(StrEnum):
    """Kind of media carried by a conference track."""

    AUDIO = "audio"
    """Microphone audio."""

    VIDEO = "video"
    """Camera video."""

    SCREEN_SHARE = "screen_share"
    """Screen share video."""


class ConferenceCapability(Flag):
    """Capabilities a ConferenceBackend can support.

    Backends declare these via their ``capabilities`` property so the framework
    can refuse configurations the backend cannot honour, rather than failing at
    runtime.

    Example::

        class MyBackend(ConferenceBackend):
            @property
            def capabilities(self) -> ConferenceCapability:
                return (
                    ConferenceCapability.SCREEN_SHARE
                    | ConferenceCapability.ACTIVE_SPEAKER
                )
    """

    NONE = 0
    """No optional capabilities (default)."""

    SCREEN_SHARE = auto()
    """Separate screen-share tracks."""

    EGRESS_RECORDING = auto()
    """Server-side (SFU) recording and export."""

    SIP_GATEWAY = auto()
    """PSTN/SIP participants can dial into the conference."""

    ACTIVE_SPEAKER = auto()
    """Dominant-speaker change events."""

    CONNECTION_QUALITY = auto()
    """Per-participant connection quality reports."""

    VIDEO_PUBLISH = auto()
    """The bot can publish video tracks (avatar embodiment)."""

    REMOTE_UNMUTE = auto()
    """A moderator can unmute another participant's track.

    Separate from muting because unmuting someone else's microphone is a
    privacy decision, not a technical one: SFUs commonly refuse it by default
    and require an explicit server-side opt-in. Muting is always available.
    """

    BOT_GRANT_UPDATE = auto()
    """A connected bot session's grants can be changed in place.

    A server-side participant update — the SFU changes what the session may
    do without reconnecting it. A capability because many SFUs can only set
    permissions at admission; against those, the one way to change a live
    bot's grants is to replace the session, and hot-plugging falls back to
    exactly that re-join (RFC 12.10.4). What this buys is continuity: a
    re-permission with the session, its subscriptions and the event bridge
    intact.
    """

    E2EE = auto()
    """End-to-end encryption between clients.

    Constrains rather than extends what the framework can do: with E2EE active
    the bot receives ciphertext, so STT, vision and recording are unavailable
    unless the bot is admitted as a key holder.
    """


@dataclass
class ConferenceTrack:
    """A single media stream published by a conference participant."""

    id: str
    """Backend-scoped stable identifier."""

    room_id: str
    """Owning conference room.

    Carried on the track because the frame callbacks receive only a track: a
    single backend instance serves many rooms, so without this the frames
    would not be routable.
    """

    participant_id: str
    """Publishing participant.

    Track identity is what attributes speech to a speaker, which is why a
    conference needs no diarization.
    """

    kind: TrackKind
    """What the track carries."""

    muted: bool = False
    """Whether the publisher has muted this track."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Provider-specific fields (sid, source, ...)."""


@dataclass
class ConferenceParticipant:
    """A participant's media presence in a conference."""

    participant_id: str
    """Identity of the participant.

    For a participant the framework admitted, this is the RoomKit
    ``Participant.id`` passed to ``mint_access()`` and echoed back by the
    backend. For one it did not admit — a PSTN dial-in, or an out-of-band
    admission — it is the backend's own stable identity.
    """

    display_name: str | None = None
    """Human-readable name, when the SFU carries one.

    Presentation, never identity: attribution rides ``participant_id`` alone
    (RFC 12.10.2), and this is what the SFU's own clients render. It usually
    rode in on the credential ``mint_access()`` issued, which is what lets a
    roster rebuilt from the join's catch-up get its names back after a
    restart — the credential outlives the process that minted it.
    """

    connected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the participant joined the media session."""

    tracks: list[ConferenceTrack] = field(default_factory=list)
    """Tracks this participant currently publishes."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Provider-supplied participant attributes.

    Not decoration: for a participant the framework did not name, this is where
    the resolvable address lives — a PSTN dial-in carries its caller number
    here, and that number is what identity resolution consumes.

    What it does *not* say is who put each attribute there, which is what
    :attr:`asserted_metadata` is for.
    """

    asserted_metadata: dict[str, Any] | None = None
    """The subset of :attr:`metadata` the SFU itself asserts.

    One attribute map on most SFUs carries two very different things: facts the
    server established — the number a SIP trunk reported, a claim in a token it
    authenticated, an attribute set through a server-side API — and values a
    participant's own client supplied when it joined. Only the first kind can
    found an identity, and nothing in the map's shape tells them apart, so the
    backend says which is which here.

    Three states, all meaningful (RFC §12.10.2):

    - a mapping: these attributes the SFU asserts, and identity may be resolved
      on an address among them;
    - ``{}``: this backend distinguishes, and the SFU asserts nothing here;
    - ``None``: this backend cannot distinguish. A statement, not an omission —
      the channel resolves nothing from it unless the integrator says otherwise.

    A backend that fills this with everything it has is asserting a guess, and
    a guess is indistinguishable from a fact to whoever acts on it.
    """


@dataclass
class ConferenceGrants:
    """Permissions encoded into a participant's conference access.

    The defaults are deliberately permissive so the common case works
    unconfigured. Narrowing them is the integrator's call and is recommended
    wherever a role does not need to publish. This is a SHOULD, not a MUST: do
    not flip these defaults to deny-by-default without changing the
    specification first.
    """

    publish_audio: bool = True
    """May publish a microphone track."""

    publish_video: bool = True
    """May publish a camera track."""

    publish_screen_share: bool = True
    """May publish a screen-share track."""

    subscribe: bool = True
    """May receive other participants' tracks."""

    moderate: bool = False
    """May mute or remove other participants."""

    hidden: bool = False
    """Invisible to other participants (bots, monitors)."""

    @classmethod
    def for_bot(cls, *, speaks: bool = False, listens: bool = True) -> ConferenceGrants:
        """Least privilege for the framework's own bot.

        The permissive defaults above are for humans, whose needs the framework
        cannot know: an attendee may unmute, turn a camera on or share a screen
        at any point, and refusing that by default would break the common case.
        The bot is the opposite — the framework configured it, so it knows
        exactly what it will do, and asking the SFU for more than that is
        privilege nobody will use.

        So: ``publish_audio`` only when a synthesizer is configured — without
        one there is nothing to publish. ``subscribe`` only when something
        consumes the tracks it would receive; a channel that only speaks
        subscribes to none, and the grant would be permission to receive every
        participant's media for nobody to read. ``publish_screen_share`` never:
        the bot has no screen. ``publish_video`` stays off until the bot is
        given something to show; an avatar would be what turns it on, and none
        is configurable yet.

        ``listens`` defaults to true because that is what a conference bot is
        usually for, and because it is what makes :meth:`observer` mean what it
        says.

        ``hidden`` is deliberately not decided here. It is a disclosure choice,
        not a privilege, and Section 17.7 leaves it to the integrator.
        """
        return cls(
            publish_audio=speaks,
            publish_video=False,
            publish_screen_share=False,
            subscribe=listens,
        )

    @classmethod
    def observer(cls) -> ConferenceGrants:
        """Subscribe-only and hidden — the Observer participation pattern.

        A silent bot that is also invisible, which is why it is expressed as
        one: whether a silent transcribing bot may stay invisible to
        participants is a legal question rather than a framework one, and the
        framework exposes the bot's ``hidden`` status so integrators can meet
        the disclosure rules that apply to them.
        """
        return replace(cls.for_bot(), hidden=True)


@dataclass
class ConferenceAccess:
    """Credentials a client uses to join the conference directly.

    Treated as opaque: the backend mints it, the integrator hands it to its
    client application, and the provider's client SDK consumes it. Framework
    code must not depend on its internal structure beyond these fields.
    """

    url: str
    """Endpoint the client connects to."""

    token: str = field(repr=False)
    """Provider-specific credential.

    Excluded from ``repr()`` so that logging an access object, or letting one
    surface in a traceback, cannot leak the credential.
    """

    expires_at: datetime | None = None
    """When the credential stops being valid, if it expires.

    Short-lived credentials are recommended.
    """

    provider_data: dict[str, Any] = field(default_factory=dict)
    """Additional provider-specific fields.

    Must not carry credentials: unlike ``token`` it appears in ``repr()``, and
    the framework has no way to know which provider keys are sensitive.
    """


@dataclass
class BotSession:
    """The framework's own connection to a conference.

    One per conference. Every frame the framework receives and every frame it
    publishes passes through this single connection — it is the only crossing
    of the media-plane boundary.
    """

    id: str
    """Backend-scoped session identifier."""

    room_id: str
    """Conference the bot is connected to."""

    identity: str
    """The bot's identity in the conference.

    Used to recognise the bot's own participant and tracks when a backend
    reports them back, so they can be excluded from participant records,
    processing lanes and subscriptions.
    """

    joined_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the bot connected.

    Defaulted to construction time, which is when a backend builds the session
    it is returning from ``join_as_bot()``. A backend with a more accurate
    figure — one the SFU reports — sets it instead. What reads it is
    ``conference_ended``'s ``duration_ms`` (RFC 8.2).
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """Provider-specific fields."""


@unique
class ConferenceInterruptionScope(StrEnum):
    """Who may interrupt the bot while it is speaking."""

    ANY = "any"
    """Speech on any audio track interrupts playback."""

    NONE = "none"
    """The bot always finishes speaking (presentation or IVR style)."""

    ALLOWLIST = "allowlist"
    """Only listed participants may interrupt (moderator pattern)."""


@dataclass
class ConferenceInterruptionConfig:
    """Multi-party interruption policy.

    In a 1:1 voice session any user speech may interrupt playback. In a
    conference, *who* may interrupt is policy rather than mechanics.
    """

    strategy: InterruptionStrategy = InterruptionStrategy.IMMEDIATE
    """How an interruption is confirmed once it is allowed."""

    scope: ConferenceInterruptionScope = ConferenceInterruptionScope.ANY
    """Which participants are allowed to interrupt at all."""

    allowlist: list[str] = field(default_factory=list)
    """Participant identities allowed to interrupt when scope is ALLOWLIST."""


@unique
class ConferenceRecordingMode(StrEnum):
    """Where a conference recording is produced."""

    FRAMEWORK = "framework"
    """Recorded by RoomKit from the tracks the bot subscribes to.

    The path that always works: no backend capability, functions against the
    mock backend, and the file lands wherever the implementation writes it.
    Audio tracks are already subscribed for transcription, so recording them
    adds a file write and no additional media subscription.
    """

    EGRESS = "egress"
    """Delegated to the SFU. Requires ``EGRESS_RECORDING``.

    Exists for one reason: a *composed* video recording — grid or
    active-speaker layout — cannot be produced by the framework without
    subscribing every video track, decoding all of them, compositing and
    re-encoding, which is the media-plane work RoomKit does not do. Carries no
    unified result contract: the SFU announces completion out of band, and the
    integrator collects the output through the provider's own mechanism.
    """


@dataclass
class ConferenceRecordingConfig:
    """Configuration for recording a conference."""

    mode: ConferenceRecordingMode = ConferenceRecordingMode.FRAMEWORK
    """Who produces the recording."""

    storage: str = "local"
    """Integrator-defined storage identifier, resolved at runtime."""

    format: str = "wav"
    """Output format. Composed video egress typically uses ``mp4``."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Recording metadata (room_id, participant_id, ...)."""
