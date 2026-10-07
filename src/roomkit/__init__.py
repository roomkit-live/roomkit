"""RoomKit - Pure async Python library for multi-channel conversations."""

from __future__ import annotations

import contextlib

from roomkit._version import __version__
from roomkit.channels import (
    TOOL_FIND_TOOLS,
    TOOL_LIST_TOOLS,
    TOOL_SEARCH_INFRA_TOOL_NAMES,
    TURN_NOTES_HEADER,
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
    add_turn_note,
    split_turn_notes,
)
from roomkit.channels._acp_context import ACPContextContributor, acp_event_text
from roomkit.channels._turn_config import AIChannelTurnConfig
from roomkit.channels.acp import ACPChannel
from roomkit.channels.acp_transport import (
    ACPSessionInvalidatedError,
    ACPTransport,
    StdioACPTransport,
)
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel, EmptyEventDescriber
from roomkit.channels.av import AudioVideoChannel
from roomkit.channels.base import (
    Channel,
    FrameworkAwareChannel,
    RealtimeModelHost,
    hosts_realtime_model,
)
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
from roomkit.channels.skill_script_tool import RunSkillScriptTool
from roomkit.channels.transport import TransportChannel
from roomkit.channels.video import VideoChannel
from roomkit.channels.voice import VoiceChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.classifiers import (
    Answers,
    ChoiceAnswer,
    ChoiceQuestion,
    Classifier,
    ClassifierError,
    JevClassifier,
    LLMClassifier,
    MockClassifier,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    YesNoQuestion,
)
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
    ChannelRefusalError,
    ConferenceAlreadyAttachedError,
    ConferenceCapabilityError,
    ConferenceCloseError,
    HumanInputRejectedError,
    ParticipantNotAdmittedError,
    ProviderDeliveryError,
    RoomNotAttachedError,
    TaskCutShortError,
    TaskTurnFailedError,
    ToolFailedError,
    ToolNameCollisionError,
    ToolRefusedError,
    ToolTimeoutError,
    TurnCutShortError,
    UnservedToolCallError,
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
from roomkit.core.visibility import visible_events
from roomkit.delivery import (
    DeliveryBackend,
    DeliveryItem,
    DeliveryItemStatus,
    InMemoryDeliveryBackend,
)
from roomkit.memory import MemoryProvider, MemoryResult, TurnFootprint
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import (
    SUPERSEDED,
    DeliveryError,
    DeliveryHandle,
    DeliveryOutcome,
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
    ToolCallOutcome,
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
from roomkit.models.store_filter import EventFilter, PersistencePolicy, received_events
from roomkit.models.thinking_event import ThinkingEvent
from roomkit.models.tool_call import (
    RESPONSE_SEGMENT_SEPARATOR,
    AfterResponseCallback,
    AIGenerationEvent,
    AIResponseEvent,
    BeforeGenerationCallback,
    ContinuationPolicy,
    DeclaredTool,
    ToolCallCallback,
    ToolCallEvent,
    ToolCallObserver,
    ToolCallVerdict,
    ToolRoundEvent,
    response_transcript,
)
from roomkit.models.voice_delivery import VoiceDeliveryRecord
from roomkit.orchestration import (
    HANDOFF_TOOL,
    SUBMIT_RESULT,
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
    ResultTool,
    RoutingConditions,
    RoutingRule,
    Supervisor,
    Swarm,
    get_conversation_state,
    set_conversation_state,
    setup_handoff,
)
from roomkit.providers.ai import ModelPricing, ResponseSchemaError
from roomkit.providers.cerebras import CerebrasAIProvider, CerebrasConfig
from roomkit.providers.image import (
    ImageAttempt,
    ImageCapabilities,
    ImageGenerationError,
    ImageModelInfo,
    ImageOptions,
    ImageProgressCallback,
    ImageProvider,
    ImageResult,
    MockImageProvider,
)
from roomkit.sandbox import SandboxExecutor, SandboxResult
from roomkit.skills import RequiresMatch, ScriptExecutor, Skill, SkillMetadata, SkillRegistry
from roomkit.speaking import (
    AlwaysSpeak,
    ClassifierSpeakPolicy,
    LLMThinker,
    MockSpeakPolicy,
    MockThinker,
    SpeakDecision,
    SpeakDecisionEvent,
    SpeakPolicy,
    SpeakTurn,
    Thinker,
    Thought,
    ThoughtEvent,
)
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
from roomkit.voice.realtime.injection import VoiceInjectionResult
from roomkit.voice.realtime.reasoning import (
    AgentReasoningBackend,
    AIProviderReasoningBackend,
    ReasoningBackend,
    ReasoningCutShortError,
    ReasoningOutput,
    ReasoningRequest,
    ToolCallResult,
    TranscriptLine,
)
from roomkit.voice.stt.language import STTLanguageLock
from roomkit.voice.testing import PCMAudio, ScenarioVoiceBackend, TraceEntry, VoiceTrace
from roomkit.voice.tts.context import (
    ConversationTurn,
    TTSContext,
    TTSContextConfig,
    TTSContextLevel,
)
from roomkit.voice.tts.library import CustomVoice, VoiceConsentError, VoiceLibrary
from roomkit.voice.voices import DialogueTurn, VoiceInfo

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
    "VoiceDeliveryRecord",
    "VoiceInjectionResult",
    "CerebrasAIProvider",
    "CerebrasConfig",
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
    "HumanInputRejectedError",
    "RoomNotAttachedError",
    "ToolFailedError",
    "ToolRefusedError",
    "TaskCutShortError",
    "TaskTurnFailedError",
    "TurnCutShortError",
    "ToolTimeoutError",
    "ChannelRefusalError",
    "UnservedToolCallError",
    "ToolNameCollisionError",
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
    "acp_event_text",
    "ACPTransport",
    "ACPSessionInvalidatedError",
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
    "RealtimeModelHost",
    "hosts_realtime_model",
    "HTTPChannel",
    "MessengerChannel",
    "RCSChannel",
    "RequiresMatch",
    "ResponseMetadata",
    "RealtimeAudioVideoChannel",
    "RealtimeVoiceChannel",
    # Turn notes (RFC §6.4)
    "TURN_NOTES_HEADER",
    "add_turn_note",
    "split_turn_notes",
    # Tool Search
    "TOOL_FIND_TOOLS",
    "TOOL_LIST_TOOLS",
    "TOOL_SEARCH_INFRA_TOOL_NAMES",
    # Reasoning delegation (RFC §12.4.1)
    "AgentReasoningBackend",
    "AIProviderReasoningBackend",
    "ReasoningBackend",
    "ReasoningCutShortError",
    "ReasoningOutput",
    "ReasoningRequest",
    "ToolCallResult",
    "TranscriptLine",
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
    "ConversationTurn",
    "TTSContext",
    "TTSContextConfig",
    "TTSContextLevel",
    "DialogueTurn",
    "VoiceInfo",
    "CustomVoice",
    "VoiceConsentError",
    "VoiceLibrary",
    "PCMAudio",
    "ScenarioVoiceBackend",
    "TraceEntry",
    "VoiceTrace",
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
    "received_events",
    "visible_events",
    "set_conversation_state",
    "HANDOFF_TOOL",
    "SUBMIT_RESULT",
    "ResultTool",
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
    "MemoryResult",
    "TurnFootprint",
    # Sandbox
    "SandboxExecutor",
    "SandboxResult",
    # Skills
    "ScriptExecutor",
    "Skill",
    "SkillMetadata",
    "SkillRegistry",
    "Answers",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "Classifier",
    "ClassifierError",
    "JevClassifier",
    "LLMClassifier",
    "MockClassifier",
    "ScoreAnswer",
    "ScoreQuestion",
    "YesNoAnswer",
    "YesNoQuestion",
    "AlwaysSpeak",
    "ClassifierSpeakPolicy",
    "MockSpeakPolicy",
    "SpeakDecision",
    "SpeakDecisionEvent",
    "SpeakPolicy",
    "SpeakTurn",
    "LLMThinker",
    "MockThinker",
    "Thinker",
    "Thought",
    "ThoughtEvent",
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
    "DeclaredTool",
    "DeliveryError",
    "DeliveryHandle",
    "SUPERSEDED",
    "DeliveryOutcome",
    "DeliveryResult",
    "DeliveryStatus",
    "EventSource",
    "FrameworkEvent",
    "HookResult",
    "ImageProvider",
    "ImageAttempt",
    "ImageCapabilities",
    "ImageGenerationError",
    "ImageModelInfo",
    "ImageOptions",
    "ImageProgressCallback",
    "ImageResult",
    "InjectedEvent",
    "InboundMessage",
    "InboundResult",
    "MockImageProvider",
    "ModelPricing",
    "ResponseSchemaError",
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
    "ToolCallObserver",
    "ToolCallContent",
    "ToolCallOutcome",
    "ToolCallEvent",
    "ToolCallVerdict",
    "ToolRoundEvent",
    "ContinuationPolicy",
    "EmptyEventDescriber",
    "RunSkillScriptTool",
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
