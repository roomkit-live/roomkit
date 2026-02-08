"""All string enums for RoomKit."""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class ChannelType(StrEnum):
    SMS = "sms"
    MMS = "mms"
    RCS = "rcs"
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    WHATSAPP_PERSONAL = "whatsapp_personal"
    WEBSOCKET = "websocket"
    AI = "ai"
    VOICE = "voice"
    REALTIME_VOICE = "realtime_voice"
    PUSH = "push"
    MESSENGER = "messenger"
    TELEGRAM = "telegram"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SYSTEM = "system"


@unique
class ChannelCategory(StrEnum):
    TRANSPORT = "transport"
    INTELLIGENCE = "intelligence"


@unique
class ChannelDirection(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BIDIRECTIONAL = "bidirectional"


@unique
class ChannelMediaType(StrEnum):
    TEXT = "text"
    RICH = "rich"
    MEDIA = "media"
    AUDIO = "audio"
    VIDEO = "video"
    LOCATION = "location"
    TEMPLATE = "template"


@unique
class EventType(StrEnum):
    MESSAGE = "message"
    SYSTEM = "system"
    TYPING = "typing"
    READ_RECEIPT = "read_receipt"
    DELIVERY_RECEIPT = "delivery_receipt"
    PRESENCE = "presence"
    REACTION = "reaction"
    EDIT = "edit"
    DELETE = "delete"
    # Participant lifecycle (RFC §5.1)
    PARTICIPANT_JOINED = "participant_joined"
    PARTICIPANT_LEFT = "participant_left"
    PARTICIPANT_IDENTIFIED = "participant_identified"
    # Channel lifecycle (RFC §3.7)
    CHANNEL_ATTACHED = "channel_attached"
    CHANNEL_DETACHED = "channel_detached"
    CHANNEL_MUTED = "channel_muted"
    CHANNEL_UNMUTED = "channel_unmuted"
    CHANNEL_UPDATED = "channel_updated"
    # Side effects
    TASK_CREATED = "task_created"
    OBSERVATION = "observation"
    # Voice pipeline events
    DTMF = "dtmf"
    RECORDING_STARTED = "recording_started"
    RECORDING_STOPPED = "recording_stopped"


@unique
class DeleteType(StrEnum):
    SENDER = "sender"
    SYSTEM = "system"
    ADMIN = "admin"


@unique
class EventStatus(StrEnum):
    PENDING = "pending"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    BLOCKED = "blocked"


@unique
class Access(StrEnum):
    READ_WRITE = "read_write"
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    NONE = "none"


@unique
class IdentificationStatus(StrEnum):
    IDENTIFIED = "identified"
    PENDING = "pending"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    CHALLENGE_SENT = "challenge_sent"
    REJECTED = "rejected"


@unique
class ParticipantRole(StrEnum):
    OWNER = "owner"
    AGENT = "agent"
    MEMBER = "member"
    OBSERVER = "observer"
    BOT = "bot"


@unique
class ParticipantStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    LEFT = "left"
    BANNED = "banned"


@unique
class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@unique
class RoomStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    ARCHIVED = "archived"


@unique
class DeliveryMode(StrEnum):
    BROADCAST = "broadcast"
    DIRECT = "direct"
    ROUND_ROBIN = "round_robin"


@unique
class HookTrigger(StrEnum):
    # Event pipeline (RFC §4.1)
    BEFORE_BROADCAST = "before_broadcast"
    AFTER_BROADCAST = "after_broadcast"
    # Channel lifecycle (RFC §4.1)
    ON_CHANNEL_ATTACHED = "on_channel_attached"
    ON_CHANNEL_DETACHED = "on_channel_detached"
    ON_CHANNEL_MUTED = "on_channel_muted"
    ON_CHANNEL_UNMUTED = "on_channel_unmuted"
    # Room lifecycle (RFC §4.1)
    ON_ROOM_CREATED = "on_room_created"
    ON_ROOM_PAUSED = "on_room_paused"
    ON_ROOM_CLOSED = "on_room_closed"
    # Identity (RFC §7.3)
    ON_IDENTITY_AMBIGUOUS = "on_identity_ambiguous"
    ON_IDENTITY_UNKNOWN = "on_identity_unknown"
    ON_PARTICIPANT_IDENTIFIED = "on_participant_identified"
    # Side effects
    ON_TASK_CREATED = "on_task_created"
    ON_ERROR = "on_error"
    # Delivery status (outbound message tracking)
    ON_DELIVERY_STATUS = "on_delivery_status"
    # Voice (RFC §18)
    ON_SPEECH_START = "on_speech_start"
    ON_SPEECH_END = "on_speech_end"
    ON_TRANSCRIPTION = "on_transcription"
    BEFORE_TTS = "before_tts"
    AFTER_TTS = "after_tts"
    # Voice - Enhanced (RFC §19)
    ON_BARGE_IN = "on_barge_in"
    ON_TTS_CANCELLED = "on_tts_cancelled"
    ON_PARTIAL_TRANSCRIPTION = "on_partial_transcription"
    ON_VAD_SILENCE = "on_vad_silence"
    ON_VAD_AUDIO_LEVEL = "on_vad_audio_level"
    ON_SPEAKER_CHANGE = "on_speaker_change"
    # Voice - Pipeline
    ON_DTMF = "on_dtmf"
    ON_TURN_COMPLETE = "on_turn_complete"
    ON_TURN_INCOMPLETE = "on_turn_incomplete"
    ON_BACKCHANNEL = "on_backchannel"
    ON_RECORDING_STARTED = "on_recording_started"
    ON_RECORDING_STOPPED = "on_recording_stopped"
    # Realtime Voice (RFC §20)
    ON_REALTIME_TOOL_CALL = "on_realtime_tool_call"
    ON_REALTIME_TEXT_INJECTED = "on_realtime_text_injected"


@unique
class HookExecution(StrEnum):
    SYNC = "sync"
    ASYNC = "async"
