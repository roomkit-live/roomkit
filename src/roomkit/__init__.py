"""RoomKit - Pure async Python library for multi-channel conversations."""

import contextlib

from roomkit._version import __version__
from roomkit.channels import (
    BuzzChannel,
    DiscordChannel,
    EmailChannel,
    HTTPChannel,
    MessengerChannel,
    RCSChannel,
    SMSChannel,
    TeamsChannel,
    TelegramChannel,
    WhatsAppChannel,
    WhatsAppPersonalChannel,
)
from roomkit.channels._acp_context import ACPContextContributor
from roomkit.channels._turn_config import AIChannelTurnConfig
from roomkit.channels.acp import ACPChannel
from roomkit.channels.acp_transport import ACPTransport, StdioACPTransport
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.av import AudioVideoChannel
from roomkit.channels.base import Channel, FrameworkAwareChannel
from roomkit.channels.cli import CLIChannel
from roomkit.channels.conference import (
    CONFERENCE_ADDRESS_KEYS,
    CONFERENCE_METADATA_KEY,
    CONFERENCE_UNASSERTED_METADATA_KEY,
    ConferenceBargeIn,
    ConferenceChannel,
    ConferenceRecordingStarted,
    ConferenceRecordingStopped,
    ConferenceTranscription,
    UtteranceTiming,
)
from roomkit.channels.realtime_av import RealtimeAudioVideoChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel, get_current_voice_session
from roomkit.channels.realtime_voice import ToolHandler as ToolHandler
from roomkit.channels.transport import TransportChannel
from roomkit.channels.video import VideoChannel
from roomkit.channels.voice import VoiceChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.conference import (
    BotSession,
    ConferenceAccess,
    ConferenceBackend,
    ConferenceCapability,
    ConferenceGrants,
    ConferenceInterruptionConfig,
    ConferenceInterruptionScope,
    ConferenceParticipant,
    ConferenceRealtimeConfig,
    ConferenceRecordingConfig,
    ConferenceRecordingMode,
    ConferenceToolHandler,
    ConferenceTrack,
    LiveKitConferenceBackend,
    LiveKitConfig,
    MockConferenceBackend,
    MockDelivery,
    MockFaults,
    MockTrackFormat,
    MockUtterance,
    TrackKind,
)
from roomkit.core.delivery import DeliveryStrategy, Immediate, Queued, WaitForIdle
from roomkit.core.exceptions import (
    ConferenceAlreadyAttachedError,
    ConferenceCapabilityError,
    ConferenceCloseError,
    ParticipantNotAdmittedError,
    ProviderDeliveryError,
    RoomNotAttachedError,
    VoiceSessionEndedError,
)
from roomkit.core.framework import (
    ChannelAlreadyRegisteredError,
    ChannelNotFoundError,
    ChannelNotRegisteredError,
    IdentityNotFoundError,
    ParticipantNotFoundError,
    RoomClosedError,
    RoomKit,
    RoomKitError,
    RoomNotFoundError,
    SourceAlreadyAttachedError,
    SourceNotFoundError,
    VoiceBackendNotConfiguredError,
    VoiceNotConfiguredError,
)
from roomkit.core.locks import InMemoryLockManager, RoomLockManager
from roomkit.delivery import (
    DeliveryBackend,
    DeliveryItem,
    DeliveryItemStatus,
    InMemoryDeliveryBackend,
)
from roomkit.memory import MemoryProvider
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import (
    DeliveryError,
    DeliveryHandle,
    DeliveryResult,
    DeliveryStatus,
    InboundMessage,
    InboundResult,
    ProviderResult,
)
from roomkit.models.enums import (
    Access,
    AgentResponsePolicy,
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
    RoomStatus,
    Visibility,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.participant import Participant
from roomkit.models.pending_input import PendingInput, PendingInputEvent, PendingInputStatus
from roomkit.models.plan_event import PlanUpdatedEvent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.room import Room, RoomTimers
from roomkit.models.session_event import SessionStartedEvent
from roomkit.models.store_filter import EventFilter, PersistencePolicy
from roomkit.models.thinking_event import ThinkingEvent
from roomkit.models.tool_call import (
    RESPONSE_SEGMENT_SEPARATOR,
    AfterResponseCallback,
    AIGenerationEvent,
    AIResponseEvent,
    BeforeGenerationCallback,
    ToolCallCallback,
    ToolCallEvent,
    response_transcript,
)
from roomkit.orchestration import (
    HANDOFF_TOOL,
    ConversationPhase,
    ConversationPipeline,
    ConversationRouter,
    ConversationState,
    HandoffHandler,
    HandoffRequest,
    HandoffResult,
    Loop,
    Orchestration,
    Pipeline,
    PipelineStage,
    RoutingConditions,
    RoutingRule,
    Supervisor,
    Swarm,
    get_conversation_state,
    set_conversation_state,
    setup_handoff,
)
from roomkit.providers.ai import ModelPricing
from roomkit.providers.image import ImageProvider, ImageResult, MockImageProvider
from roomkit.sandbox import SandboxExecutor, SandboxResult
from roomkit.skills import ScriptExecutor, Skill, SkillMetadata, SkillRegistry
from roomkit.store import ConversationStore, InMemoryStore, SQLiteSchemaError, SQLiteStore
from roomkit.telemetry.redaction import content_logging_enabled, set_content_logging
from roomkit.tools.base import Tool
from roomkit.tools.human_input import HumanInputHandler, HumanInputToolHandler
from roomkit.tools.policy import RoleOverride, ToolPolicy
from roomkit.video.events import VideoDetectionEvent
from roomkit.video.pipeline.filter import (
    FaceTouchConfig,
    FaceTouchFilter,
    FaceTouchSensitivity,
    FaceZone,
    MockFaceTouchFilter,
)
from roomkit.voice.pipeline.agc.simple import SimpleAGCProvider
from roomkit.voice.pipeline.denoiser.webrtc import WebRTCNoiseSuppressorProvider
from roomkit.voice.stt.language import STTLanguageLock

# Console (optional — requires `rich`)
with contextlib.suppress(ImportError):
    from roomkit.console import RoomKitConsole as RoomKitConsole

# AI documentation helpers (lazy import to avoid file I/O at import time)


def get_llms_txt() -> str:
    """Get the contents of llms.txt for LLM consumption."""
    from roomkit.ai_docs import get_llms_txt as _get_llms_txt

    return _get_llms_txt()


def get_agents_md() -> str:
    """Get the contents of AGENTS.md for AI coding assistants."""
    from roomkit.ai_docs import get_agents_md as _get_agents_md

    return _get_agents_md()


def get_llms_full_txt() -> str:
    """Get the contents of llms-full.txt (comprehensive documentation)."""
    from roomkit.ai_docs import get_llms_full_txt as _get_llms_full_txt

    return _get_llms_full_txt()


def get_ai_context() -> str:
    """Get combined AI context (AGENTS.md + llms.txt)."""
    from roomkit.ai_docs import get_ai_context as _get_ai_context

    return _get_ai_context()


__all__ = [
    "__version__",
    # Framework
    "RoomKit",
    # Errors
    "RoomKitError",
    "RoomClosedError",
    "RoomNotFoundError",
    "ChannelNotFoundError",
    "ChannelAlreadyRegisteredError",
    "ChannelNotRegisteredError",
    "ParticipantNotFoundError",
    "ParticipantNotAdmittedError",
    "ProviderDeliveryError",
    "IdentityNotFoundError",
    "SourceAlreadyAttachedError",
    "SourceNotFoundError",
    "VoiceBackendNotConfiguredError",
    "VoiceNotConfiguredError",
    "VoiceSessionEndedError",
    "ConferenceAlreadyAttachedError",
    "ConferenceCapabilityError",
    "ConferenceCloseError",
    "RoomNotAttachedError",
    # Delivery
    "DeliveryBackend",
    "DeliveryItem",
    "DeliveryItemStatus",
    "DeliveryStrategy",
    "Immediate",
    "InMemoryDeliveryBackend",
    "Queued",
    "WaitForIdle",
    # Channels
    "ACPChannel",
    "ACPContextContributor",
    "ACPTransport",
    "Agent",
    "AIChannel",
    "AIChannelTurnConfig",
    "AudioVideoChannel",
    "BuzzChannel",
    "Channel",
    "CLIChannel",
    "ConferenceChannel",
    "DiscordChannel",
    "EmailChannel",
    "FrameworkAwareChannel",
    "HTTPChannel",
    "MessengerChannel",
    "RCSChannel",
    "ResponseMetadata",
    "RealtimeAudioVideoChannel",
    "RealtimeVoiceChannel",
    "SMSChannel",
    "StdioACPTransport",
    "TeamsChannel",
    "TelegramChannel",
    "TransportChannel",
    "VideoChannel",
    "VoiceChannel",
    "SimpleAGCProvider",
    "WebRTCNoiseSuppressorProvider",
    "STTLanguageLock",
    "WebSocketChannel",
    "WhatsAppChannel",
    "WhatsAppPersonalChannel",
    # Enums (core)
    "Access",
    "AgentResponsePolicy",
    "ChannelCategory",
    "ChannelType",
    "EventStatus",
    "EventType",
    "HookExecution",
    "HookTrigger",
    "RoomStatus",
    "Visibility",
    # Orchestration
    "ConversationPhase",
    "ConversationPipeline",
    "ConversationRouter",
    "ConversationState",
    "get_conversation_state",
    "set_conversation_state",
    "HANDOFF_TOOL",
    "HandoffHandler",
    "HandoffRequest",
    "HandoffResult",
    "setup_handoff",
    "Loop",
    "Orchestration",
    "Pipeline",
    "PipelineStage",
    "RoutingConditions",
    "RoutingRule",
    "Supervisor",
    "Swarm",
    # Conference (SFU orchestration — RFC §12.10)
    "CONFERENCE_ADDRESS_KEYS",
    "CONFERENCE_METADATA_KEY",
    "CONFERENCE_UNASSERTED_METADATA_KEY",
    "BotSession",
    "ConferenceAccess",
    "ConferenceBackend",
    "ConferenceBargeIn",
    "ConferenceCapability",
    "ConferenceGrants",
    "ConferenceInterruptionConfig",
    "ConferenceInterruptionScope",
    "ConferenceParticipant",
    "ConferenceRealtimeConfig",
    "ConferenceRecordingConfig",
    "ConferenceRecordingMode",
    "ConferenceRecordingStarted",
    "ConferenceRecordingStopped",
    "ConferenceToolHandler",
    "ConferenceTrack",
    "ConferenceTranscription",
    "UtteranceTiming",
    "LiveKitConferenceBackend",
    "LiveKitConfig",
    "MockConferenceBackend",
    "MockDelivery",
    "MockFaults",
    "MockTrackFormat",
    "MockUtterance",
    "TrackKind",
    # Observability / privacy
    "content_logging_enabled",
    "set_content_logging",
    # Storage
    "ConversationStore",
    "EventFilter",
    "InMemoryStore",
    "SQLiteSchemaError",
    "SQLiteStore",
    "PersistencePolicy",
    # Locking (extension point — RFC §13.5)
    "InMemoryLockManager",
    "RoomLockManager",
    # Memory
    "MemoryProvider",
    # Sandbox
    "SandboxExecutor",
    "SandboxResult",
    # Skills
    "ScriptExecutor",
    "Skill",
    "SkillMetadata",
    "SkillRegistry",
    # Tools
    "HumanInputHandler",
    "HumanInputToolHandler",
    "RoleOverride",
    "ToolPolicy",
    # Human-in-the-loop models
    "PendingInput",
    "PendingInputEvent",
    "PendingInputStatus",
    # Models (core)
    "AIGenerationEvent",
    "AIResponseEvent",
    "AfterResponseCallback",
    "BeforeGenerationCallback",
    "ChannelBinding",
    "ChannelCapabilities",
    "ChannelOutput",
    "DeliveryError",
    "DeliveryHandle",
    "DeliveryResult",
    "DeliveryStatus",
    "EventSource",
    "FrameworkEvent",
    "HookResult",
    "ImageProvider",
    "ImageResult",
    "InjectedEvent",
    "InboundMessage",
    "InboundResult",
    "MockImageProvider",
    "ModelPricing",
    "Participant",
    "PlanUpdatedEvent",
    "ProviderResult",
    "RESPONSE_SEGMENT_SEPARATOR",
    "Room",
    "RoomContext",
    "RoomEvent",
    "RoomTimers",
    "SessionStartedEvent",
    "response_transcript",
    "TextContent",
    "ThinkingEvent",
    "Tool",
    "ToolCallCallback",
    "ToolCallContent",
    "ToolCallEvent",
    "ToolHandler",
    "VideoDetectionEvent",
    # Video pipeline filters
    "FaceTouchConfig",
    "FaceTouchFilter",
    "FaceTouchSensitivity",
    "FaceZone",
    "MockFaceTouchFilter",
    "get_current_voice_session",
    # Console (optional)
    "RoomKitConsole",
    # AI docs
    "get_agents_md",
    "get_ai_context",
    "get_llms_full_txt",
    "get_llms_txt",
]
