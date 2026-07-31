"""ConferenceChannel — orchestrates an SFU conference for a room.

The channel owns a ConferenceBackend, demultiplexes published tracks into
per-track processing lanes, and represents the AI's voice in the conference. It
never carries media between human participants: the SFU does that, and the
channel's only media connection is the bot session it joins with.

See RFC section 12.10.4. Vision and bot video are out of scope here; this is
the audio core those build on. Framework-side recording is in scope and runs
on the same collection gate as transcription (RFC section 12.10.8), and so is
speech-to-speech composition (RFC section 12.10.12): a realtime provider
plugged in as the conference's intelligence hears the lanes mixed N→1 and
speaks on the bot track — see ``_conference_realtime`` and
``_conference_mixer``.

Which is why the channel announces AUDIO alone. Vision is a SHOULD in RFC
section 12.10.11, so an audio-only conference conforms; announcing a media type
no code carries does not. Nothing here registers ``on_track_video`` and
selective subscription never subscribes a video track, so a binding claiming
VIDEO would promise frames that never arrive.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from roomkit.channels import _conference_activity
from roomkit.channels._conference_access import ConferenceAccessMixin
from roomkit.channels._conference_activity import RoomActivity
from roomkit.channels._conference_attachment import ConferenceAttachmentMixin
from roomkit.channels._conference_identity import (
    CONFERENCE_ADDRESS_KEYS,
    CONFERENCE_UNASSERTED_METADATA_KEY,
    ConferenceIdentity,
)
from roomkit.channels._conference_lane import (
    ConferenceBargeIn,
    ConferenceLane,
    ConferenceTranscription,
)
from roomkit.channels._conference_lanes import ConferenceLanesMixin
from roomkit.channels._conference_metadata import CONFERENCE_METADATA_KEY
from roomkit.channels._conference_operations import ConferenceOperations, ConferenceResource
from roomkit.channels._conference_plug import ConferencePlugMixin
from roomkit.channels._conference_realtime import ConferenceRealtime
from roomkit.channels._conference_recording import ConferenceRecording
from roomkit.channels._conference_recording_events import (
    ConferenceRecordingEvents,
    ConferenceRecordingStarted,
    ConferenceRecordingStopped,
)
from roomkit.channels._conference_room_state import ConferenceRoomState
from roomkit.channels._conference_roster import ConferenceRoster
from roomkit.channels._conference_session import ConferenceSessionMixin
from roomkit.channels._conference_shutdown import CloseStatus, ConferenceShutdownCoordinator
from roomkit.channels._conference_subscription import ConferenceSubscriptionMixin
from roomkit.channels._conference_voice import ConferenceVoice
from roomkit.channels.base import Channel, FrameworkAwareChannel
from roomkit.conference.base import ConferenceBackend
from roomkit.conference.models import (
    ConferenceAccess,
    ConferenceCapability,
    ConferenceGrants,
    ConferenceInterruptionConfig,
    ConferenceParticipant,
    ConferenceRealtimeConfig,
    ConferenceRecordingConfig,
    ConferenceRecordingMode,
    TrackKind,
)
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelMediaType,
    ChannelType,
    EventType,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.voice.pipeline.config import AudioPipelineConfig, AudioPipelineContract
from roomkit.voice.pipeline.engine import AudioPipeline
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager, AbstractContextManager

    from roomkit.core.framework import RoomKit
    from roomkit.models.identity import IdentityResult
    from roomkit.recorder.base import MediaRecorder
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

logger = logging.getLogger("roomkit.channels.conference")


__all__ = [
    "CONFERENCE_ADDRESS_KEYS",
    "CONFERENCE_METADATA_KEY",
    "CONFERENCE_UNASSERTED_METADATA_KEY",
    "ConferenceBargeIn",
    "ConferenceChannel",
    "ConferenceLane",
    "ConferenceRecordingStarted",
    "ConferenceRecordingStopped",
    "ConferenceTranscription",
]


class ConferenceChannel(
    ConferenceAttachmentMixin,
    ConferenceSessionMixin,
    ConferenceAccessMixin,
    ConferenceSubscriptionMixin,
    ConferencePlugMixin,
    ConferenceLanesMixin,
    FrameworkAwareChannel,
    Channel,
):
    """Multi-party conference channel backed by an external SFU.

    Example::

        channel = ConferenceChannel("conf", backend=backend, stt=stt, tts=tts)
        kit.register_channel(channel)
        await kit.attach_channel("room-1", "conf")

        access = await channel.mint_access("room-1", "p-alice")
    """

    sender_is_participant = True
    """A conference utterance is attributed to a participant, not to an address.

    What a lane puts on ``sender_id`` is the identity the track was published
    under, which is the Room ``Participant.id`` this channel keeps its roster on
    (RFC §12.10.2 rule 2). Resolving it would be resolving on the backend's
    opaque identity, which rule 3 rules out for exactly the reason it fails
    here: no resolver can match it, so every utterance comes back UNKNOWN, and a
    hook written to refuse unknown senders makes the transcripts of a
    participant the framework identified on arrival disappear.

    A conference resolves once, when a participant arrives and its provider's
    address is still there to resolve (:mod:`roomkit.channels._conference_identity`).
    Speaking again asks nothing new.
    """

    def __init__(
        self,
        channel_id: str,
        *,
        backend: ConferenceBackend,
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
        realtime: ConferenceRealtimeConfig | None = None,
        pipeline: AudioPipelineConfig | None = None,
        interruption: ConferenceInterruptionConfig | None = None,
        recording: ConferenceRecordingConfig | None = None,
        recorder: MediaRecorder | None = None,
        bot_identity: str = "roomkit",
        bot_grants: ConferenceGrants | None = None,
        default_grants: ConferenceGrants | None = None,
        e2ee: bool = False,
        close_room_on_detach: bool = False,
        speak_text_events: bool = False,
        close_providers: bool = True,
        max_queued_frames: int = 100,
        identity_address_keys: Sequence[str] | None = None,
        identity_trusts_unasserted_metadata: bool = False,
    ) -> None:
        super().__init__(channel_id)
        if max_queued_frames < 1:
            raise ValueError(
                f"max_queued_frames must be at least 1, got {max_queued_frames}. "
                "A non-positive value makes a track's queues unbounded, so a lane "
                "or a recording that falls behind grows without limit instead of "
                "dropping its oldest frames — the backpressure guarantee the bound "
                "exists for."
            )
        if e2ee and stt is not None:
            # RFC 12.10.2: an implementation offering E2EE must either admit the
            # bot as a key holder in the conference's key exchange, or refuse
            # media intelligence on an encrypted conference. There is no
            # key-holder contract in ConferenceBackend, so the bot receives
            # ciphertext it cannot decode; transcribing it would produce
            # nonsense while the configuration read as if it worked.
            raise ValueError(
                "STT cannot run on an end-to-end encrypted conference: the bot "
                "receives ciphertext it has no key for, so the lanes would "
                "transcribe noise. Admitting the bot to the key exchange is a "
                "backend capability RoomKit does not yet model, so pass e2ee=False "
                "to transcribe, or drop stt= to keep the conference encrypted."
            )
        if realtime is not None:
            self._validate_realtime(realtime, tts=tts, e2ee=e2ee)
        self._backend = backend
        self._stt = stt
        self._recording = recording
        self._recorder = self._resolve_recorder(
            recording, recorder, e2ee=e2ee, max_queued_frames=max_queued_frames
        )
        # Separate from the recorder because what a recorder is asked never
        # awaits — it is synchronous throughout, which is why it runs on a
        # worker thread — and announcing does.
        self._recording_events = ConferenceRecordingEvents(channel_id)
        self._pipeline_config = self._resolve_pipeline(pipeline, stt, realtime=realtime)
        self._pipeline = (
            AudioPipeline(self._pipeline_config) if self._pipeline_config is not None else None
        )
        # Shared with ConferenceVoice: a publication in flight is work a detach
        # must not overtake, exactly like an announcement is.
        self._activity = RoomActivity()
        # The one ledger of which operations are using which of the channel's
        # resources, and the one shutdown those operations are closed by. Every
        # backend and provider call the channel admits takes a lease here, and
        # a resource is closed only once no lease on it remains — see
        # _conference_operations and _conference_shutdown.
        self._operations = ConferenceOperations()
        self._shutdown = ConferenceShutdownCoordinator(channel_id, self._operations)
        self._voice = ConferenceVoice(
            backend=backend,
            tts=tts,
            interruption=interruption or ConferenceInterruptionConfig(),
            ensure_bot=self._ensure_bot,
            activity=self._activity,
            operations=self._operations,
            on_published=None if self._recorder is None else self._record_bot_audio,
        )
        # The speech-to-speech composition (RFC 12.10.12), built inert and
        # activated by configuration: its mixer taps every lane either way,
        # which is what lets a provider plugged mid-meeting hear the tracks
        # already open.
        self._realtime = ConferenceRealtime(
            channel_id=channel_id,
            bot_identity=bot_identity,
            voice=self._voice,
            operations=self._operations,
            ensure_bot=self._ensure_bot,
        )
        if realtime is not None:
            self._realtime.configure(realtime)
        self._roster = ConferenceRoster(channel_id)
        # `identity_trusts_unasserted_metadata` widens what an arrival may be
        # identified on, from what the SFU asserts to every attribute it
        # surfaced. The safe reading is the one that holds unconfigured (RFC
        # §12.10.2), so this is the integrator saying their deployment has a
        # reason the framework cannot know — a closed client fleet, or a backend
        # whose provenance they establish elsewhere.
        self._identity = ConferenceIdentity(
            channel_id,
            identity_address_keys,
            trust_unasserted=identity_trusts_unasserted_metadata,
        )
        self._max_queued_frames = max_queued_frames
        self._bot_identity = bot_identity
        # Kept apart from the derivation rather than merged into it: an
        # explicit `bot_grants` is the caller saying their deployment knows
        # something the configuration does not, and hot-plugging never
        # rewrites it (RFC 12.10.4) — the caller who set it took coverage of
        # the configured needs on themselves. Owned, not immutable: the
        # caller replaces it, or returns to derivation, through
        # set_bot_grants().
        self._explicit_bot_grants = bot_grants
        # Serialises plug_stt/unplug_tts/... against each other: everything a
        # change triggers — the grants derived, the subscriptions re-evaluated,
        # the join or leave decided — is read from the configuration as a
        # whole, so two changes interleaving would act on a configuration
        # neither of them describes (RFC 12.10.4).
        self._plug_lock = asyncio.Lock()
        self._default_grants = default_grants or ConferenceGrants()
        self._e2ee = e2ee
        self._close_room_on_detach = close_room_on_detach
        self._speak_text_events = speak_text_events
        self._close_providers = close_providers

        self._framework: RoomKit | None = None
        # Everything the channel knows about a room, held together because the
        # guarantees are written across it — see ConferenceRoomState.
        self._rooms: dict[str, ConferenceRoomState] = {}
        self._teardowns: set[asyncio.Task[None]] = set()
        # Providers an unplug retired while an operation that outlived its own
        # grace was still inside them: each closes in the background once its
        # last lease is back, off both the unplug's clock and the shutdown's
        # (RFC 12.10.4). See ConferencePlugMixin._close_retired.
        self._deferred_closes: set[asyncio.Task[None]] = set()
        # Participant callbacks in flight. Not room activity — that is what a
        # teardown drains, and these hold the lock a teardown holds across that
        # drain — so `close()` has its own barrier. See `_participant_callback`.
        self._roster_writes: set[asyncio.Event] = set()
        # Set once that barrier is closed. Past it nothing touches the store or
        # the room lock, because `RoomKit.close()` releases both right after the
        # channels — so a budget that runs out stops the work rather than
        # letting it run on into what is being released.
        self._roster_closed = False
        self._lanes: dict[str, ConferenceLane] = {}
        # Credentials a teardown took back. Kept off the room's record because
        # `_mint` reads it after the record has stopped listing the request.
        self._abandoned_mints: set[asyncio.Task[ConferenceAccess]] = set()
        # Set the moment the backend is closed, so nothing that outlived
        # `close()`'s budget calls into it afterwards.
        self._backend_closed = False

        backend.on_participant_joined(self._on_participant_joined)
        backend.on_participant_left(self._on_participant_left)
        backend.on_track_published(self._on_track_published)
        backend.on_track_unpublished(self._on_track_unpublished)
        backend.on_track_muted(self._on_track_muted)
        backend.on_track_unmuted(self._on_track_unmuted)
        backend.on_track_audio(self._on_track_audio)
        backend.on_active_speaker_changed(self._on_active_speaker_changed)
        backend.on_connection_quality(self._on_connection_quality)
        backend.on_bot_session_ended(self._on_bot_session_ended)

    def _resolve_recorder(
        self,
        recording: ConferenceRecordingConfig | None,
        recorder: MediaRecorder | None,
        *,
        e2ee: bool,
        max_queued_frames: int,
    ) -> ConferenceRecording | None:
        """Settle what will record this conference, or refuse what cannot record it.

        Recording is off unless a configuration asks for it, and a configuration
        that asks for it names what does the recording. The two halves are
        separate arguments because they answer different questions — whether to
        record, and what to record with — and each of them alone is a
        configuration that reads as if it worked and does nothing.

        Egress is specified (RFC 12.10.8) and unimplemented: there is no egress
        surface on ConferenceBackend to delegate to, and the RFC gives the mode
        no unified result contract, so a channel accepting it would subscribe
        nothing, produce nothing, and report nothing. Refused by name rather
        than ignored, since the difference between "recorded by the SFU" and
        "not recorded" is not one an integrator should have to discover.
        """
        if recorder is not None and recording is None:
            raise ValueError(
                "A conference recorder was passed with no recording configuration, so "
                "nothing would ever be fed to it. Pass recording=ConferenceRecordingConfig() "
                "to record, or drop recorder=."
            )
        if recording is None:
            return None
        if e2ee:
            # RFC 12.10.2, the same key-holder gap that refuses STT: the bot
            # receives ciphertext, so what a recorder wrote would be noise
            # while the configuration read as a compliant recording.
            raise ValueError(
                "A conference cannot be recorded end-to-end encrypted: the bot receives "
                "ciphertext it has no key for, so the recording would hold noise while "
                "reading as evidence. Pass e2ee=False to record, or drop recording= to "
                "keep the conference encrypted."
            )
        if recording.mode is not ConferenceRecordingMode.FRAMEWORK:
            raise ValueError(
                f"Conference recording mode {recording.mode.value!r} is not implemented: "
                "ConferenceBackend has no egress surface to delegate to, and egress carries "
                "no result contract the framework could report on. Use "
                f"mode={ConferenceRecordingMode.FRAMEWORK.value!r}, or drive the SFU's own "
                "egress API outside RoomKit."
            )
        if recorder is None:
            raise ValueError(
                "Framework-mode conference recording needs a recorder to write with: the "
                "tracks would be subscribed and dropped. Pass recorder= a MediaRecorder "
                "(roomkit.recorder), which receives one recording per track, attributed to "
                "its participant."
            )
        return ConferenceRecording(
            recorder=recorder,
            config=recording,
            channel_id=self.channel_id,
            # The same bound as the lanes, and deliberately one knob rather
            # than two: it answers "how much of a track may this channel hold
            # before it starts dropping", and a caller who tuned it for a slow
            # recognizer meant it for a slow disk as well.
            max_queued_frames=max_queued_frames,
            # Announced from the writer's task, which is where a recording is
            # opened and therefore the first moment there is one to announce.
            on_opened=self._announce_recording_started,
            # Read once the announcement has been heard, before the audio
            # buffered during it reaches the recorder: an ON_RECORDING_STARTED
            # handler that refuses — detaching the channel is the ordinary
            # way — closes admission synchronously, so this check turning
            # false is what keeps the pre-consent audio out of the file
            # (RFC 17.6).
            may_capture=self._may_record,
        )

    def _may_record(self, room_id: str) -> bool:
        """Whether the channel is still allowed to record a room's audio."""
        room = self._attached_room(room_id)
        return room is not None and room.may_collect()

    def _validate_realtime(
        self, realtime: ConferenceRealtimeConfig, *, tts: TTSProvider | None, e2ee: bool
    ) -> None:
        """Refuse a speech-to-speech configuration that cannot work.

        One voice per bot (RFC 12.10.12): a synthesizer and a realtime
        provider both publish on the one bot track, and no floor discipline
        turns two intelligences into one voice. The E2EE refusal is the same
        key-holder gap as STT's — the mix would be ciphertext. And tools
        without a handler are a conversation that wedges: the provider's
        turn waits on a result nothing will ever submit.
        """
        if tts is not None:
            raise ValueError(
                "tts= and realtime= are mutually exclusive: both publish on the one "
                "bot track, and two components answering the same room answer over "
                "each other (RFC 12.10.12). Configure one, or trade them at runtime "
                "through unplug_tts()/plug_realtime()."
            )
        if e2ee:
            raise ValueError(
                "A speech-to-speech provider cannot run on an end-to-end encrypted "
                "conference: the bot receives ciphertext it has no key for, so the "
                "mix it hears would be noise. Pass e2ee=False, or drop realtime= to "
                "keep the conference encrypted."
            )
        if realtime.tools and realtime.tool_handler is None:
            raise ValueError(
                "realtime.tools were configured with no tool_handler: the provider's "
                "turn waits on a result nothing will ever submit. Pass "
                "tool_handler=, or drop tools=."
            )

    @property
    def _realtime_config(self) -> ConferenceRealtimeConfig | None:
        """The speech-to-speech configuration in force, if one is plugged."""
        return self._realtime.config

    @property
    def _transport_only(self) -> bool:
        """Pure transport: nothing plugged in consumes a track or can speak.

        A bot session would be a participant with no function, so the mint,
        arrival and occupancy-probe triggers of the lazy join stand down on
        this (RFC 12.10.4 step 1); the channel stays the room's admission gate
        and roster. A property rather than a flag because the configuration it
        reads is not fixed at construction — plug_stt() and its family change
        the answer while the channel runs. Read off the configuration
        and not off the bot's grants, because an explicit grant is what the
        SFU would allow, not what the channel was configured to do.
        """
        return self._voice.tts is None and not any(self._consumes(kind) for kind in TrackKind)

    @property
    def _bot_grants(self) -> ConferenceGrants:
        """What the bot joins with, as the configuration stands right now.

        Derived rather than defaulted: the framework knows what it configured
        the bot to do, so asking the SFU for more is privilege nobody will
        use. `listens` is asked in the same terms `_consumes` answers in, so
        the grant and the subscriptions cannot drift apart. An explicit
        `bot_grants` still wins — the caller may know something about its
        deployment that the configuration does not say — and is never
        rewritten by a plug or an unplug (RFC 12.10.4); ``set_bot_grants()``
        is how its owner replaces it, or hands the channel back to this
        derivation.
        """
        if self._explicit_bot_grants is not None:
            return self._explicit_bot_grants
        return ConferenceGrants.for_bot(
            speaks=self._voice.tts is not None or self._realtime_config is not None,
            listens=any(self._consumes(kind) for kind in TrackKind),
        )

    def _holds_conference(self, room_id: str) -> bool:
        """Whether this channel still holds the room's one conference slot.

        Read by ``attach_channel`` when another conference channel asks for
        the room (RFC 12.10.4): the reservation outlives the binding, because
        a detach removes the binding at its start and takes the bot out at
        its end — possibly on a deferred teardown. The channel holds the room
        for as long as it is attached, has a session in the meeting or on its
        books, or has that teardown still running; a second conference
        admitted inside that window is two bots in one meeting.
        """
        room = self._rooms.get(room_id)
        if room is None:
            return False
        return (
            room.attached
            or room.bot is not None
            or bool(room.leaving)
            or (room.pending_teardown is not None and not room.pending_teardown.done())
        )

    def _resolve_pipeline(
        self,
        config: AudioPipelineConfig | None,
        stt: STTProvider | None,
        realtime: ConferenceRealtimeConfig | None = None,
    ) -> AudioPipelineConfig | None:
        """Settle what the lanes will run, or refuse a configuration that cannot work.

        ``stt`` and ``realtime`` are parameters rather than reads of ``self``
        because the two callers stand on opposite sides of the assignment: the
        constructor resolves what it is about to install, and the plugs must
        refuse a bad configuration *before* touching the channel's state.

        There is no pipeline without a consumer that needs one: recognition
        and speech-to-speech are the two, and without either, building one
        would load a VAD for frames the channel never subscribes to.

        With either, a VAD is not optional. For recognition (RFC 12.10.4):
        without segmentation the lane calls the recognizer once per 20 ms
        frame and produces a transcript cut at frame boundaries. For
        speech-to-speech (RFC 12.10.12): the per-lane VAD is the interruption
        policy's one sensor — the provider's own detection hears the mix and
        can name no interrupting participant — so without it the bot cannot
        be barged in on at all. So an unconfigured channel gets a working
        default, and a configuration that names its stages but omits the VAD
        is refused rather than degraded into that.

        Format normalisation is not optional either, and unlike the VAD it has
        an obvious default, so a configuration without a contract gets one
        instead of an error. Participants negotiate their own formats with the
        SFU, so two tracks in one conference need not agree, while every stage
        downstream assumes they do.

        AEC and Diarization are neither required nor forbidden: a conference
        has no server-side echo path and gets attribution from track identity,
        so neither is needed — but the specification says MUST NOT be
        required, not MUST NOT be configured.
        """
        if stt is None and realtime is None:
            return None
        # The default contract's internal format is 16 kHz mono 16-bit, which
        # is what every track is resampled to before the stages run.
        if config is None:
            return AudioPipelineConfig(vad=EnergyVADProvider(), contract=AudioPipelineContract())
        if config.vad is None:
            raise ValueError(
                "A conference lane requires a VAD when STT or a realtime provider is "
                "configured: without segmentation the lane transcribes every frame "
                "instead of every utterance, and without speech detection the "
                "interruption policy has no sensor (RFC 12.10.12). Pass "
                "AudioPipelineConfig(vad=...), or omit the pipeline argument to get "
                "the default one."
            )
        if config.contract is None:
            # Copied rather than mutated: the caller may share this config.
            return replace(config, contract=AudioPipelineContract())
        return config

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.CONFERENCE

    @property
    def active_lanes(self) -> dict[str, ConferenceLane]:
        """Processing lanes currently running, keyed by track."""
        return dict(self._lanes)

    def capabilities(self) -> ChannelCapabilities:
        """Audio, and only audio — see the module docstring for why.

        The binding copies this at attachment and it is what routing and
        transcoding read, so it is the one place an integrator can find out
        what a conference carries without waiting for frames that never come.
        """
        return ChannelCapabilities(
            media_types=[ChannelMediaType.AUDIO],
        )

    def info(self) -> dict[str, Any]:
        """What the bot is, where it is, and what it is doing with the media.

        Disclosure rules for a transcribing bot differ by jurisdiction, so the
        RFC mandates no announcement and instead requires that an integrator be
        able to ask (§17.7): the bot's identity and hidden status, and whether
        speech recognition, vision or recording is running — at any time, not
        only when the channel was configured.

        Which is why the answer is per conference, under ``rooms``. A channel
        serving three rooms is configured once and behaves differently in each:
        the bot may be in one and not another, and a binding closed to
        ``Access.NONE`` stops collection in that room alone. "Is this meeting
        being transcribed" is the question a disclosure obligation asks, and a
        channel-wide flag cannot answer it — ``stt_configured`` says only what
        the channel was built with.
        """
        return {
            "backend": self._backend.name,
            "bot_identity": self._bot_identity,
            "bot_hidden": self._bot_grants.hidden,
            # Whether set_bot_grants() reaches a live session without a
            # re-join. Answered here, before the call, because the fallback
            # costs the event bridge a cut — the caller weighs that against
            # the change, and can only weigh what it can read (RFC 12.10.4).
            "bot_grant_update_in_place": (
                ConferenceCapability.BOT_GRANT_UPDATE in self._backend.capabilities
            ),
            "stt_configured": self._stt is not None,
            "stt_provider": self._stt.name if self._stt is not None else None,
            "realtime_configured": self._realtime_config is not None,
            "realtime_provider": (
                self._realtime_config.provider.name if self._realtime_config is not None else None
            ),
            # Constant because there is nothing to configure: the channel takes
            # no VisionProvider and announces no VIDEO. The key stays so the
            # disclosure surface answers the question rather than omitting it —
            # "no" is an answer, a missing key is not.
            "vision_configured": False,
            "recording_configured": self._recording is not None,
            "e2ee": self._e2ee,
            # A record exists for every room the channel has ever served; the
            # ones worth reporting are those it is in or on its way out of.
            "rooms": {
                room_id: self._room_info(room_id)
                for room_id, room in sorted(self._rooms.items())
                if room.attached or room.leaving
            },
        }

    def _room_info(self, room_id: str) -> dict[str, Any]:
        """What the bot is doing in one conference.

        ``stt_active`` is the whole point: a recognizer is only running on this
        room if there is a bot in it, the binding still permits collection, and
        a lane is actually carrying a track. Any one of those missing and
        nothing is being transcribed here, whatever the channel was configured
        with.

        ``recording_active`` answers the same way and is read off the
        subscriptions rather than off the open recordings. A track's recording
        opens on its first frame, so a meeting where nobody has spoken yet has
        none — and answering "not being recorded" about a conference whose next
        frame lands in a file is the wrong side to err on for a question asked
        out of a disclosure obligation.

        ``recording_dropped_frames`` is what the room's recordings never wrote,
        and it is here because RFC sections 12.10.4 and 12.10.8 ask for the loss
        to be exposed and not only logged: a recording with a hole in it that
        nothing reports reads as a defective recorder rather than as storage
        that could not keep up. It counts the whole attachment, closed tracks
        included, since a participant who left took the gap in their file with
        them.

        A room being torn down is still reported, with ``detaching`` set. The
        bot is out of the channel's books from the first moment of the detach
        but only leaves the conference at the end of it, and reporting it absent
        in between would tell an integrator the meeting is unattended while the
        bot is still sitting in it.

        Which is separate from *which* session is reported, and the two came
        apart once a detach could be deferred past a re-attach: for as long as
        that teardown runs, the room holds a session on its way out and a live
        one the new attachment brought in.

        So the two are reported separately and neither stands in for the other.
        ``bot_session_id`` is the bot in the meeting now — the one an integrator
        would act on — and is absent when there is none, even while a session it
        replaced is still leaving. ``leaving_session_ids`` lists every session
        still on its way out, which is more than one when a room is leaving
        twice over. ``bot_present`` is the disclosure answer and covers both: a
        session that has not left is a bot in the room, whichever list it is on.
        ``detaching`` is then what it says — this attachment is being torn down,
        which a room that has since been re-attached is not.

        ``leave_failed`` is the case where none of that resolves on its own: a
        ``leave()`` the backend refused leaves the bot in the meeting, and the
        session stays on ``leaving`` — reported present, because it is —
        carrying what went wrong. Empty is the normal answer. Anything in it is
        a bot an operator may have to go and remove by hand, and the reason it
        is here rather than only in the log is that a disclosure obligation is
        answered by asking, not by reading logs.
        """
        room = self._room(room_id)
        bot = room.bot
        leaving = room.leaving
        lanes = self._lane_ids(room_id)
        collecting = room.may_collect()
        live = bot is not None and collecting
        return {
            "bot_present": bot is not None or bool(leaving),
            "bot_session_id": bot.id if bot is not None else None,
            # The hidden status in force on this session — what the SFU
            # holds, which is what a disclosure obligation asks about (RFC
            # 17.7) and can differ transiently from the channel-level answer:
            # grants change while a session runs (set_bot_grants), and an
            # update that failed leaves the session on what it joined with.
            # None when there is no live session to report on.
            "bot_hidden": (
                room.bot_grants.hidden if bot is not None and room.bot_grants is not None else None
            ),
            "detaching": bool(leaving) and not room.attached,
            "leaving_session_ids": sorted(leaving),
            "leave_failed": room.leave_failures(),
            "collecting": collecting,
            "active_lanes": len(lanes),
            "stt_active": self._stt is not None and live and bool(lanes),
            # Active means connected: a session is established on the first
            # mixed window or injection, so a conference where nobody has
            # spoken yet answers no — nothing has reached the provider.
            "realtime_active": (
                self._realtime_config is not None
                and live
                and self._realtime.session_for(room_id) is not None
            ),
            # Audio the mix discarded to stay near-live, in whole windows
            # (RFC 12.10.4 asks for loss to be exposed, not only logged).
            "realtime_dropped_windows": self._realtime.mixer.dropped_windows(room_id),
            # No video track is ever subscribed here, so no room can be an
            # exception to `vision_configured` above.
            "vision_active": False,
            "recording_active": self._recorder is not None and live and bool(room.subscribed),
            "recording_dropped_frames": (
                0 if self._recorder is None else self._recorder.dropped_frames(room_id)
            ),
            # Why the bot has stopped speaking in this room, when it has. An
            # utterance nothing could close makes the track unusable — anything
            # published after it is heard as its continuation — and an
            # integrator whose AI has gone quiet needs somewhere to read that
            # rather than a silence to interpret.
            "bot_track_unterminated": self._voice.unterminated(room_id),
        }

    def set_framework(self, framework: RoomKit) -> None:
        """Wire the channel to the framework.

        Attaching does not come through here: ``on_room_attached`` and
        ``on_room_detached`` are the ``Channel`` contract and the framework
        awaits them itself — see :mod:`roomkit.channels._conference_attachment`.
        """
        self._framework = framework
        self._voice.set_framework(framework)
        self._realtime.set_framework(framework)
        self._roster.set_store(
            framework.store, framework.lock_manager, lease=framework._resource_lease
        )
        self._identity.set_framework(framework)
        self._recording_events.set_framework(framework)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _participant_callback(self) -> AsyncIterator[bool]:
        """Hold a closing channel open for one participant callback.

        The whole callback, from entry to exit, and not merely the write at the
        end of it. Everything in between reaches for something ``RoomKit.close()``
        releases straight after the channels — an identity resolver, the store,
        the framework's room lock — so a barrier that only covered the write
        photographed an empty registry while a resolver was still suspended, and
        the callback came back to a lock manager that had been closed.

        Yields whether the callback may run at all. Past the barrier the answer
        is no: a closing channel admits no new work, because there is nothing
        left for new work to safely touch. That is what makes this a barrier
        rather than a drain — closing admission comes first, and only then does
        anything get waited for.
        """
        if self._roster_closed:
            yield False
            return
        done = asyncio.Event()
        self._roster_writes.add(done)
        try:
            yield True
        finally:
            self._roster_writes.discard(done)
            done.set()

    @contextlib.asynccontextmanager
    async def _roster_write(self, room_id: str) -> AsyncIterator[bool]:
        """Take the room lock for a roster write, if there is still any point.

        The lock orders the write against ``remove_member()`` and against a
        detach, and it has to be taken outside the teardown's drain — a detach
        holds it across that drain, so work the drain waits for cannot be
        waiting for the lock.

        Yields whether the write may go ahead. The barrier is read again here,
        immediately before the lock, because a callback the barrier gave up
        waiting for is one whose remaining steps must do nothing: acquiring a
        lock and writing a record are both uses of something the framework is
        about to release.

        A lock acquisition that fails once the channel has closed is caught
        rather than propagated into the backend's callback fan-out, where it
        would read as a defect in the conference rather than as a channel that
        has closed. Caught around the *acquisition alone*: the write itself
        runs inside the block, so a single ``try`` around the whole thing
        turned any failure the store raised into a message about waiting for a
        room lock — which is both untrue and one severity too quiet.

        The framework's resource lease is taken *before* the acquisition
        begins, not after it succeeds. An acquisition is already an operation
        the lock manager is running — on an advisory-lock backend it is a call
        holding a pool connection — so a lease that started at success left
        the framework free to release the pool underneath a caller still
        queued on it. Everything under the lease is this channel's own code
        and the resource calls themselves — the acquisition, the generation
        checks, one store write, the lock's release — never integrator code,
        which is what makes it safe for ``RoomKit.close()`` to wait for the
        lease with no deadline. And the wait terminates: every holder of the
        room lock is itself under a lease and finishes, so a queued
        acquisition gets the lock, finds the barrier down, writes nothing and
        lets go. The channel's own ``close()`` does not wait for any of it —
        a write the store already has is not the channel's to wait for, and
        holding ``close()`` open on it held every channel behind this one in
        its conference (RFC 12.10.4).
        """
        if self._roster_closed:
            yield False
            return
        with self._resource_lease():
            room = contextlib.AsyncExitStack()
            try:
                await room.enter_async_context(self._locked_room(room_id))
            except Exception:
                if not self._roster_closed:
                    raise
                logger.info(
                    "Conference channel %r abandoned a roster write for room %s: the channel "
                    "closed while it was waiting for the room",
                    self.channel_id,
                    room_id,
                )
                yield False
                return
            async with room:
                yield not self._roster_closed

    def _close_the_roster(self) -> None:
        """Stop admitting participant callbacks. Nothing waits here.

        Closing admission comes first and on its own, which is the whole of what
        a barrier is and the whole of what belongs at the *start* of a close: a
        callback arriving later is refused, so nothing new accumulates while the
        conference is being taken apart. Waiting for what was already admitted
        is :meth:`_settle_the_roster`, and it happens at the end — after the bot
        has left.
        """
        self._roster_closed = True

    async def _settle_the_roster(self) -> None:
        """Wait for the participant callbacks the barrier admitted before it closed.

        Last, deliberately. An earlier draft waited here before taking the bot
        out of the conference, which made a slow store into a bot left in a
        meeting — the one failure this module ranks above every other, and the
        reason every wait in it is bounded. The media plane is released first:
        the bot leaves, the backend closes, and only then does this wait for
        the bookkeeping.

        Bounded, like everything else in this ``close()``. A callback is
        *abandonable*: it may be suspended in an identity resolver, which is
        integrator code, so it is waited for on the usual budget — one budget
        for the whole of this, not one per step — and past it the barrier
        stands: a callback that has not reached the store yet finds the channel
        closed and writes nothing.

        A write the store already has is a different thing, and deliberately
        not waited for here. It cannot be taken back, but the store it is
        inside of is not the channel's — it is the framework's, released by
        ``RoomKit.close()`` only after every channel has closed. Each such
        write sits under the framework's resource lease (see
        :meth:`_roster_write`), and the framework waits for every lease, with
        no deadline, before releasing the store and the lock manager. Waiting
        for it *here*, without a deadline, is what a review of the multichannel
        shutdown found wanting: the framework closes channels in sequence, so
        one channel's slow store held every channel behind it in its
        conference — the exact failure the media-first ordering above exists
        to prevent (RFC 12.10.4).
        """
        await self._settle(
            self._roster_writes,
            _conference_activity.DRAIN_TIMEOUT_S,
            "Closing conference channel %r with %d participant callback(s) still running "
            "after %.1fs. The barrier stands, so a callback that has not reached the store "
            "writes nothing; a write the store already has finishes under the framework's "
            "resource lease before the store is released",
        )

    async def _settle(
        self, registry: set[asyncio.Event], budget: float, unfinished_note: str
    ) -> None:
        """Wait for a registry of in-flight work to empty, on one shared budget.

        ``budget`` is for the whole call and not for each item: a channel with
        five callbacks in flight would otherwise pay five times over, and the
        module's convention is that a deadline covers a step rather than each of
        the things inside it.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget
        while registry:
            pending = [asyncio.ensure_future(done.wait()) for done in list(registry)]
            remaining = deadline - loop.time()
            if remaining <= 0:
                for waiter in pending:
                    waiter.cancel()
                logger.error(unfinished_note, self.channel_id, len(pending), budget)
                return
            try:
                await asyncio.wait(pending, timeout=remaining)
            finally:
                for waiter in pending:
                    waiter.cancel()

    def _locked_room(self, room_id: str) -> AbstractAsyncContextManager[None]:
        """The framework's per-room lock, which every writer of a roster holds.

        ``add_member()`` and ``remove_member()`` take it, and so does every
        detach — so a step taken under it is one no membership change and no
        detach can interleave with. What must never be done under it is anything
        the teardown's drain waits for: the detach holds this lock across that
        drain, and the two would clear each other only when it timed out.
        """
        if self._framework is None:
            return contextlib.nullcontext()
        return self._framework.lock_manager.locked(room_id)

    def _resource_lease(self) -> AbstractContextManager[None]:
        """The framework's lease on the store and the room lock, or nothing.

        Taken around every operation this channel starts on either resource —
        a roster write from the moment the lock acquisition begins to the
        moment the lock is let go, and each store read on its own (the roster
        takes those itself). ``RoomKit.close()`` waits for every lease — after
        all the channels have closed and their media is released — before it
        releases the store and the lock manager, which is what lets this
        channel's own ``close()`` stay bounded without anything being released
        under work it is still running (RFC 12.10.4). Nothing integrator-owned
        ever runs under one: a resolver or a hook can suspend forever, and a
        lease is a promise the framework waits on without a deadline. Without
        a framework there is no store and no lock manager, so there is nothing
        to hold open.
        """
        if self._framework is None:
            return contextlib.nullcontext()
        return self._framework._resource_lease()

    def _room(self, room_id: str) -> ConferenceRoomState:
        """The channel's record for a room, created on first mention.

        A record is never removed: the join lock and the generation counter must
        outlive the attachment they belong to. Its existence says nothing — a
        room the channel has never attached to has one, unattached and empty.
        """
        state = self._rooms.get(room_id)
        if state is None:
            state = self._rooms[room_id] = ConferenceRoomState()
        return state

    def _attached_room(self, room_id: str) -> ConferenceRoomState | None:
        """The record for a room this channel is in, or ``None``.

        What every backend callback opens with, and it does not create: the
        callbacks are fed by the far side, and a backend reporting a room this
        channel never attached to must not leave a record behind for it. Same
        for `mint_access`, where the room id comes from the caller.
        """
        room = self._rooms.get(room_id)
        return room if room is not None and room.attached else None

    # -------------------------------------------------------------------------
    # Participants
    # -------------------------------------------------------------------------

    async def _on_participant_joined(
        self, room_id: str, participant: ConferenceParticipant
    ) -> None:
        room = self._attached_room(room_id)
        if room is None:
            return
        if self._is_own_bot(room_id, participant.participant_id):
            return
        generation = room.generation
        # Held for the whole callback, the identity resolution included: it is an
        # await into integrator code, and a channel that closed while it was
        # suspended would have released the store and the room lock that the
        # rest of this needs.
        async with self._participant_callback() as admitted:
            if not admitted:
                return
            await self._record_arrival(room_id, room, participant, generation)

    async def _record_arrival(
        self,
        room_id: str,
        room: ConferenceRoomState,
        participant: ConferenceParticipant,
        generation: int,
    ) -> None:
        """Put one arrival on the roster and announce it. See the caller."""
        # Not a precondition: the arrival is recorded whether or not the bot got
        # in, and only a channel that has left the conference stops it. See
        # `_ensure_bot_for_arrival`.
        if not await self._ensure_bot_for_arrival(room_id):
            return
        # Re-read the world before the first thing that touches the store. The
        # join above can suspend in the backend past the closing budget: the
        # shutdown gives up on it, releases the store and the lock manager,
        # and the join resumes here — its failure swallowed, because an
        # arrival is recorded even without a bot. A closed channel's arrival
        # is not, and neither is a detached room's: the identity lookup below
        # is a store read, and past this point every step is work for an
        # attachment that no longer exists.
        if self._roster_closed or room.generation != generation or not room.attached:
            return
        identity = await self._identify(room_id, participant)
        # The roster write takes the framework's room lock, so it stays outside
        # the drain: a detach holds that lock for the whole of itself, and work
        # the drain is waiting for cannot be waiting for it back.
        #
        # Which is also what lets the check and the write be one step. Under the
        # lock a detach is not running — it takes the same one — so the room is
        # still attached when the record is written, or it is not and nothing is
        # written. Announcing it is the part that stays inside the drain.
        async with self._roster_write(room_id) as writable:
            if not writable or room.generation != generation or not room.attached:
                return
            await self._roster.record(room_id, participant, identity)
        # Announcing it is two awaits — the hook and the event — and a detach
        # landing between them would announce a participant joining a
        # conference observers have already been told ended. So they are one
        # transaction the detach drains, with the generation check inside it: it
        # runs whole, or it does not start.
        #
        # Bringing the bot in is deliberately outside, and so is identifying the
        # arrival. Both run integrator code — ON_SESSION_STARTED, the resolver —
        # and holding a detach behind it is the head-of-line block the join lock
        # is shaped to avoid. Neither writes anything a discarded generation
        # would leave behind.
        async with self._activity.track(room_id):
            if room.generation != generation or not room.attached:
                return
            await self._fire(
                room_id,
                HookTrigger.ON_CONFERENCE_PARTICIPANT_JOINED,
                "conference_participant_joined",
                f"Participant {participant.participant_id} joined the conference",
                {"participant_id": participant.participant_id},
            )
            await self._emit_framework_event(
                "conference_participant_joined",
                room_id,
                participant_id=participant.participant_id,
            )

    async def _identify(
        self, room_id: str, participant: ConferenceParticipant
    ) -> IdentityResult | None:
        """Resolve an arrival the framework did not name, from its address.

        Only that case. A participant the framework minted access for already
        has a Room participant the integrator created and named, and running a
        resolver over the attributes an SFU happens to attach would overwrite
        that with a guess. Being unknown to the roster is what says the
        framework did not name this one (RFC 12.10.2, rule 3).

        Before the record is written, not after: the answer goes into the record
        the same moment it is created, so ON_CONFERENCE_PARTICIPANT_JOINED and
        everything downstream read an identified participant rather than an
        unknown that changes underneath them.

        Whether resolution is enabled at all — a resolver exists and this
        channel type is one it runs for — is asked before whether the roster
        knows the participant, even though the second is the question that
        matters: the first is free and the second reads the store, and a
        deployment that resolves nothing here would otherwise pay for a query on
        every arrival to reach a conclusion that was already settled.
        """
        if not self._identity.active:
            return None
        if await self._roster.knows(room_id, participant.participant_id):
            return None
        return await self._identity.resolve(room_id, participant)

    async def _on_participant_left(self, room_id: str, participant: ConferenceParticipant) -> None:
        room = self._attached_room(room_id)
        if room is None:
            return
        if self._is_own_bot(room_id, participant.participant_id):
            return
        async with self._participant_callback() as admitted:
            if not admitted:
                return
            await self._record_departure(room_id, room, participant)

    async def _record_departure(
        self, room_id: str, room: ConferenceRoomState, participant: ConferenceParticipant
    ) -> None:
        """Mark one departure on the roster and announce it. See the caller."""
        # Outside the drain, like an arrival's and for the same reason: it takes
        # the room lock, and a detach holds that lock across the drain it would
        # be waiting inside of. The check and the write are one step under it.
        async with self._roster_write(room_id) as writable:
            if not writable or not room.attached:
                return
            await self._roster.mark_left(room_id, participant.participant_id)
        # The same transaction shape as an arrival, for the same reason: the
        # hook and the event are two awaits, and a detach between them announces
        # a departure from a conference observers have already been told ended.
        async with self._activity.track(room_id):
            if not room.attached:
                return
            await self._fire(
                room_id,
                HookTrigger.ON_CONFERENCE_PARTICIPANT_LEFT,
                "conference_participant_left",
                f"Participant {participant.participant_id} left the conference",
                {"participant_id": participant.participant_id},
            )
            await self._emit_framework_event(
                "conference_participant_left", room_id, participant_id=participant.participant_id
            )

    # -------------------------------------------------------------------------
    # Public surface
    # -------------------------------------------------------------------------

    def may_interrupt(self, participant_id: str) -> bool:
        """Whether a participant is allowed to interrupt the bot."""
        return self._voice.may_interrupt(participant_id)

    # -------------------------------------------------------------------------
    # Channel contract
    # -------------------------------------------------------------------------

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
                external_id=message.external_id,
                provider=self._backend.name,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
            metadata=message.metadata,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Speak an event into the conference on the single bot track.

        What a conference is willing to read aloud is the channel's decision,
        and it is the whole of what happens here: a meeting is not a place to
        recite orchestration metadata, nor every message arriving from another
        channel. Synthesis, publication, and the barge-in that can stop it
        belong to ConferenceVoice.
        """
        if event.type is EventType.SYSTEM:
            return ChannelOutput.empty()
        if not isinstance(event.content, TextContent):
            return ChannelOutput.empty()
        if not self._speak_text_events and event.source.channel_type is not ChannelType.AI:
            return ChannelOutput.empty()
        if self._realtime_config is not None:
            # The realtime counterpart of speaking: the provider is the
            # room's voice, so a text event joins its conversation context
            # rather than being synthesized over it (RFC 12.10.12).
            await self._realtime.deliver_text(
                event.room_id,
                event.content.body,
                role=str(event.metadata.get("inject_role", "system")),
            )
            return ChannelOutput.empty()
        await self._voice.speak(event.room_id, event.content.body)
        return ChannelOutput.empty()

    async def close(self) -> None:
        """Run the channel's one shutdown, or join the one already running.

        There is one logical shutdown per channel (RFC 12.10.4): concurrent
        callers await the same shielded task, a caller cancelled mid-wait
        abandons only its own wait, and once the shutdown reaches its terminal
        result every later call replays that result — an immediate return
        after a success, the same ``ConferenceCloseError`` after a failure —
        rather than running the steps again.
        """
        await self._shutdown.close(self._close_once)

    async def _close_once(self) -> None:
        # Closing is a detach of every room at once, and it owes them the same
        # two steps in the same order. Clearing `attached` alone stops nothing
        # already in flight: a join suspended in the backend resumes, finds its
        # own generation unchanged because nothing bumped it, registers a bot on
        # a closed channel and announces a conference that starts after the
        # channel is gone. So admission closes by generation first.
        for room in self._rooms.values():
            if room.attached:
                room.bump()
                room.attached = False
            # Inventory every active session before the first external await.
            # If one room spends the whole departure budget, a later room's
            # bot is still on this ledger and therefore still visible and
            # named by the final failure.
            if room.bot is not None:
                room.start_closing(room.bot)
                room.bot = None
        # Participant callbacks are deliberately not part of the drain below —
        # they hold the framework's room lock, which a detach holds across that
        # drain — so closing has a barrier of its own. It is closed here, with
        # admission, and waited for at the very end: nothing else covers those
        # callbacks, since `close()` bumps generations without taking that lock
        # and `RoomKit.close()` releases the store and the lock manager once
        # the channels have closed and their resource leases are back.
        self._close_the_roster()
        # Playbacks are stopped before the bots leave, for the same reason the
        # detach path does it: a synthesis loop still running would publish on
        # a session that is on its way out. The synthesizer itself is closed at
        # the end, once nothing can still be drawing on it. The realtime rooms
        # come off the books at the same moment — the mixer stops feeding and
        # every response in flight is given its end — and their sessions are
        # disconnected below, before the bots leave the conferences their
        # audio was published into.
        self._voice.abandon_all()
        realtime_sessions = self._realtime.abandon_all()
        for room in self._rooms.values():
            room.cancel_tasks()
        # A detach may still be finishing on its own task — the deferred case —
        # and it ends in `leave()`. Cancelling it would strand the bot in the
        # conference, so it is waited for rather than cut off.
        unfinished_teardowns = await self._await_teardowns()
        if unfinished_teardowns:
            self._shutdown.record(
                component="channel",
                operation="detach",
                status=CloseStatus.ABANDONED,
                step="waiting for conference detaches",
                detail=f"{len(unfinished_teardowns)} teardown(s) still running after the budget",
            )
        await self._activity.drain_all()

        # Same reason the detach path does it, and the same deadline having
        # passed: a credential still being minted would be admission to a
        # conference this channel is closing out of.
        for room_id in list(self._rooms):
            self._abandon_mints(room_id)
        # Then the joins. A join suspended in the backend holds its room's lock
        # and has a `leave()` still ahead of it: the generation bump above makes
        # it abandon the session it just opened. Taking each lock is how this
        # waits for that to have happened — without it the backend is closed
        # first and the abandoning join calls into it afterwards.
        unsettled_joins = await self._settle_joins()
        if unsettled_joins:
            self._shutdown.record(
                component="backend",
                operation="join",
                status=CloseStatus.ABANDONED,
                step="waiting for conference joins",
                detail="join(s) still running for room(s) " + ", ".join(sorted(unsettled_joins)),
            )
        # The provider sessions go before the bots leave: each one's audio
        # publishes into a bot session, and disconnecting first is what stops
        # the source before the track it speaks on goes away. Best-effort per
        # session inside, on one bounded budget out here.
        if realtime_sessions and self._realtime_config is not None:
            await self._shutdown.spend(
                self._realtime.disconnect_sessions(
                    self._realtime_config.provider, realtime_sessions
                ),
                "disconnecting the realtime provider's sessions",
                component="realtime",
                operation="disconnect",
            )
        # Every session the channel still has in a conference, not only the one
        # in `bot`: a detach whose `leave()` the backend refused left its bot
        # sitting in the meeting and said so, and closing is the last moment
        # anything can be done about it. On one budget for the whole step —
        # `leave()` is a backend's network call, and one that never returns
        # must not hold every channel behind this one in its conference. A
        # leave the budget cancels puts its session on the books on the way
        # out, where the final raise below reports it.
        await self._shutdown.spend(
            self._leave_every_room(),
            "taking the bots out of their conferences",
            component="backend",
            operation="leave",
        )
        for room in self._rooms.values():
            room.forget_subscriptions()
        # All lanes receive cancellation together and share the same grace
        # period. A survivor keeps its lease on the pipeline — and a
        # transcription in flight its per-call lease on the STT — which is
        # what stops those providers being closed underneath a runaway task:
        # the leases are the retention; nothing here has to remember it.
        lanes = list(self._lanes.values())
        self._lanes.clear()
        lane_results = await asyncio.gather(
            *(self._close_lane_instance(lane) for lane in lanes),
            return_exceptions=True,
        )
        for lane, result in zip(lanes, lane_results, strict=True):
            if isinstance(result, BaseException):
                self._shutdown.record(
                    component="pipeline",
                    operation="close lane",
                    status=CloseStatus.FAILED,
                    step=f"closing conference lane {lane.track_id}",
                    detail=f"{type(result).__name__}: {result}",
                )
        # Finalized here rather than left to the recorder's own close: a caller
        # sharing the recorder across channels keeps it open, and the recordings
        # this channel started are still its own to finish. Not under a `spend`
        # of its own — every wait the recording subsystem makes is already
        # bounded internally, stage by stage, and a single outer budget would
        # cancel finalizations that were on their way to finishing inside
        # theirs. Everything the subsystem could not do is recorded against
        # the close rather than left in the log alone.
        if self._recorder is not None:
            await self._close_recordings()
        for room in self._rooms.values():
            room.track_epochs.clear()
            room.collision_reported = False
        # `leaving` is deliberately not cleared. What is left in it after the
        # pass above is a session the channel could not remove or a teardown
        # that outlived its budget — a bot still in a conference either way, and
        # forgetting it is how a closed channel came to report a meeting
        # unattended while the framework's bot was still listening to it.
        # The utterances a cancellation left to be closed publish their
        # terminal chunks on the backend, so they are settled before it goes.
        await self._voice.aclose(close_provider=False)
        # The backend closes only once nothing the channel admitted is still
        # using it. The leases are the authority — a wedged publish, a late
        # join, a leave that swallowed its cancellation all hold one — and the
        # teardown tasks stand in for work that spans several backend calls.
        # A backend still in use is retained and reported, and closes in the
        # background once the operations truly end (RFC 12.10.4).
        await self._shutdown.close_resource(
            ConferenceResource.BACKEND,
            self._close_backend,
            step="closing the conference backend",
            blockers=set(unfinished_teardowns),
        )
        if self._close_providers:
            await self._shutdown.close_resource(
                ConferenceResource.TTS,
                self._voice.close_tts,
                step="closing the TTS provider",
            )
            await self._shutdown.close_resource(
                ConferenceResource.REALTIME,
                self._realtime.close_provider,
                step="closing the realtime provider",
            )
        # Last, and after the media: the bookkeeping a participant callback was
        # in the middle of — waited for *here* rather than at the top because a
        # slow store holding this up must never be a bot held in a conference.
        # Bounded, because the framework closes channels in sequence and a wait
        # without a deadline here holds every channel behind this one in its
        # conference. What the budget cannot cover — a write the store already
        # has — sits under the framework's resource lease, and `RoomKit.close()`
        # waits for it before releasing the store and the lock manager.
        await self._settle_the_roster()
        if self._close_providers:
            await self._shutdown.close_resource(
                (ConferenceResource.PIPELINE, ConferenceResource.STT),
                self._close_owned_providers,
                step="closing the conference audio pipeline and STT",
                blockers=set(unfinished_teardowns),
            )
        # Last of all, and only once every step above has run: sessions and
        # resource failures are raised together. `RoomKit.close()` collects the
        # error and still closes every channel behind this one.
        stuck = {
            room_id: sorted(room.leaving) for room_id, room in self._rooms.items() if room.leaving
        }
        self._shutdown.raise_for_failures(stuck)

    async def _leave_every_room(self) -> None:
        """Take every room's sessions out concurrently on one shared budget."""
        await asyncio.gather(*(self._leave_all(room_id) for room_id in list(self._rooms)))

    async def _close_backend(self) -> None:
        """The backend's closer, run by the coordinator once its leases are back.

        The flag is set before the call rather than after, so work that
        outlived a budget reaches for a backend it can see is closed instead
        of one that only fails when called.
        """
        self._backend_closed = True
        await self._backend.close()

    async def _close_owned_providers(self) -> None:
        """Close pipeline and STT, each independently of the other's failure.

        The pipeline's own ``close()`` already closes every provider it holds
        and aggregates their failures rather than stopping at the first. It is
        synchronous and potentially blocking — a native provider's close is a
        real call — so it runs off the event loop; a close that blocks past
        the budget leaves its worker thread running and is reported, never
        waited for twice (the thread's future stays referenced by the
        executor, so nothing mistakes an abandoned close for a finished one).
        """
        if self._pipeline is not None:
            try:
                await asyncio.to_thread(self._pipeline.close)
            except BaseExceptionGroup as failures:
                for failure in failures.exceptions:
                    self._shutdown.record(
                        component="pipeline",
                        operation="close",
                        status=CloseStatus.FAILED,
                        step="closing the conference audio pipeline",
                        detail=f"{type(failure).__name__}: {failure}",
                    )
                logger.error(
                    "Conference channel %r could not close %d of its audio pipeline's "
                    "providers; the others were closed",
                    self.channel_id,
                    len(failures.exceptions),
                    exc_info=failures,
                )
        if self._stt is not None:
            await self._shutdown.spend(
                self._stt.close(),
                "closing the STT provider",
                component="stt",
            )

    async def _close_recordings(self) -> None:
        """Finalize the recordings, release the recorder, and report the rest.

        Everything the recording subsystem could not do — a finalization that
        failed or is still running inside the recorder, a recorder retained
        because calls the framework gave up on are still in it, a recorder
        whose own close failed — is recorded against the close. The
        *announcements* run hooks and read the store, neither of which is this
        channel's, so they get the drain budget; past it the recordings are
        finished and on disk, and what is lost is the notification.
        """
        assert self._recorder is not None
        report = await self._recorder.close(close_recorder=self._close_providers)
        for detail in report.unfinished:
            self._shutdown.record(
                component="recorder",
                operation="finalize",
                status=CloseStatus.FAILED,
                step="finalizing the conference recordings",
                detail=detail,
            )
        if report.recorder_retained:
            self._shutdown.record(
                component="recorder",
                operation="close",
                status=CloseStatus.RETAINED,
                step="releasing the conference recorder",
                detail="call(s) the framework gave up on are still running inside it",
            )
        if report.recorder_close_error is not None:
            self._shutdown.record(
                component="recorder",
                operation="close",
                status=CloseStatus.FAILED,
                step="releasing the conference recorder",
                detail=report.recorder_close_error,
            )
        try:
            async with asyncio.timeout(_conference_activity.DRAIN_TIMEOUT_S):
                await self._recording_events.stopped_all(report.finished)
        except TimeoutError:
            logger.error(
                "Conference channel %r closed without announcing %d finished "
                "recording(s): the announcements did not return within %.1fs. "
                "The recordings themselves are finalized",
                self.channel_id,
                len(report.finished),
                _conference_activity.DRAIN_TIMEOUT_S,
            )

    # -------------------------------------------------------------------------
    # Framework plumbing
    # -------------------------------------------------------------------------

    async def _fire(
        self,
        room_id: str,
        trigger: HookTrigger,
        code: str,
        message: str,
        data: dict[str, Any],
    ) -> None:
        if self._framework is None:
            return
        await self._framework._fire_lifecycle_hook(
            room_id,
            trigger,
            EventType.SYSTEM,
            code=code,
            message=message,
            data={"channel_id": self.channel_id, **data},
        )

    async def _emit_framework_event(self, name: str, room_id: str, **data: Any) -> None:
        if self._framework is None:
            return
        await self._framework._emit_framework_event(
            name, room_id=room_id, channel_id=self.channel_id, data=data
        )
