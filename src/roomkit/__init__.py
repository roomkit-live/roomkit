"""RoomKit - Pure async Python library for multi-channel conversations."""

from roomkit._version import __version__
from roomkit.channels import (
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
from roomkit.channels.ai import AIChannel
from roomkit.channels.base import Channel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.channels.realtime_voice import ToolHandler as ToolHandler
from roomkit.channels.transport import TransportChannel
from roomkit.channels.voice import VoiceChannel
from roomkit.channels.websocket import (
    StreamChunk,
    StreamEnd,
    StreamMessage,
    StreamSendFn,
    StreamStart,
    WebSocketChannel,
)
from roomkit.core.framework import (
    ChannelNotFoundError,
    ChannelNotRegisteredError,
    IdentityNotFoundError,
    ParticipantNotFoundError,
    RoomKit,
    RoomKitError,
    RoomNotFoundError,
    SourceAlreadyAttachedError,
    SourceNotFoundError,
    VoiceBackendNotConfiguredError,
    VoiceNotConfiguredError,
)
from roomkit.core.hooks import HookEngine, HookRegistration
from roomkit.core.inbound_router import DefaultInboundRoomRouter, InboundRoomRouter
from roomkit.core.locks import InMemoryLockManager, RoomLockManager
from roomkit.identity.base import IdentityResolver
from roomkit.identity.mock import MockIdentityResolver
from roomkit.memory import MemoryProvider, MemoryResult, MockMemoryProvider, SlidingWindowMemory
from roomkit.models.channel import (
    ChannelBinding,
    ChannelCapabilities,
    ChannelOutput,
    RateLimit,
    RetryPolicy,
)
from roomkit.models.context import RoomContext
from roomkit.models.delivery import (
    DeliveryResult,
    DeliveryStatus,
    InboundMessage,
    InboundResult,
    ProviderResult,
)
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    DeleteType,
    DeliveryMode,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
    IdentificationStatus,
    ParticipantRole,
    ParticipantStatus,
    RoomStatus,
    TaskStatus,
)
from roomkit.models.event import (
    AudioContent,
    CompositeContent,
    DeleteContent,
    EditContent,
    EventContent,
    EventSource,
    LocationContent,
    MediaContent,
    RichContent,
    RoomEvent,
    SystemContent,
    TemplateContent,
    TextContent,
    VideoContent,
)
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.identity import Identity, IdentityHookResult, IdentityResult
from roomkit.models.participant import Participant
from roomkit.models.room import Room, RoomTimers
from roomkit.models.task import Observation, Task
from roomkit.models.trace import ProtocolTrace
from roomkit.orchestration.handoff import (
    HANDOFF_TOOL,
    HandoffHandler,
    HandoffMemoryProvider,
    HandoffRequest,
    HandoffResult,
    setup_handoff,
)
from roomkit.orchestration.router import (
    ConversationRouter,
    RoutingConditions,
    RoutingRule,
)
from roomkit.orchestration.state import (
    ConversationPhase,
    ConversationState,
    PhaseTransition,
    get_conversation_state,
    set_conversation_state,
)
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIProvider,
    AIResponse,
    AITextPart,
    AITool,
    AIToolCall,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.elasticemail.config import ElasticEmailConfig
from roomkit.providers.elasticemail.email import ElasticEmailProvider
from roomkit.providers.email.base import EmailProvider
from roomkit.providers.email.mock import MockEmailProvider
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.http.base import HTTPProvider
from roomkit.providers.http.config import HTTPProviderConfig
from roomkit.providers.http.mock import MockHTTPProvider
from roomkit.providers.http.provider import WebhookHTTPProvider
from roomkit.providers.http.webhook import parse_http_webhook
from roomkit.providers.messenger.base import MessengerProvider
from roomkit.providers.messenger.config import MessengerConfig
from roomkit.providers.messenger.facebook import FacebookMessengerProvider
from roomkit.providers.messenger.mock import MockMessengerProvider
from roomkit.providers.messenger.webhook import parse_messenger_webhook
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.rcs.base import RCSDeliveryResult, RCSProvider
from roomkit.providers.rcs.mock import MockRCSProvider
from roomkit.providers.sinch.config import SinchConfig
from roomkit.providers.sinch.sms import SinchSMSProvider, parse_sinch_webhook
from roomkit.providers.sms.base import SMSProvider
from roomkit.providers.sms.meta import WebhookMeta, extract_sms_meta
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.providers.sms.phone import is_valid_phone, normalize_phone
from roomkit.providers.teams.base import TeamsProvider
from roomkit.providers.teams.bot_framework import BotFrameworkTeamsProvider
from roomkit.providers.teams.config import TeamsConfig
from roomkit.providers.teams.conversation_store import (
    ConversationReferenceStore,
    InMemoryConversationReferenceStore,
)
from roomkit.providers.teams.mock import MockTeamsProvider
from roomkit.providers.teams.webhook import (
    is_bot_added,
    parse_teams_activity,
    parse_teams_webhook,
)
from roomkit.providers.telegram.base import TelegramProvider
from roomkit.providers.telegram.bot import TelegramBotProvider
from roomkit.providers.telegram.config import TelegramConfig
from roomkit.providers.telegram.mock import MockTelegramProvider
from roomkit.providers.telegram.webhook import parse_telegram_webhook
from roomkit.providers.telnyx.config import TelnyxConfig
from roomkit.providers.telnyx.rcs import (
    TelnyxRCSConfig,
    TelnyxRCSProvider,
    parse_telnyx_rcs_webhook,
)
from roomkit.providers.telnyx.sms import (
    TelnyxSMSProvider,
    parse_telnyx_webhook,
)
from roomkit.providers.twilio.config import TwilioConfig
from roomkit.providers.twilio.rcs import (
    TwilioRCSConfig,
    TwilioRCSProvider,
    parse_twilio_rcs_webhook,
)
from roomkit.providers.twilio.sms import TwilioSMSProvider, parse_twilio_webhook
from roomkit.providers.vllm import VLLMConfig, create_vllm_provider
from roomkit.providers.voicemeup.config import VoiceMeUpConfig
from roomkit.providers.voicemeup.sms import (
    VoiceMeUpSMSProvider,
    configure_voicemeup_mms,
    parse_voicemeup_webhook,
)
from roomkit.providers.whatsapp.base import WhatsAppProvider
from roomkit.providers.whatsapp.mock import MockWhatsAppProvider
from roomkit.providers.whatsapp.personal import WhatsAppPersonalProvider
from roomkit.realtime.base import (
    EphemeralCallback,
    EphemeralEvent,
    EphemeralEventType,
    RealtimeBackend,
)
from roomkit.realtime.memory import InMemoryRealtime
from roomkit.skills import (
    ScriptExecutor,
    ScriptResult,
    Skill,
    SkillMetadata,
    SkillParseError,
    SkillRegistry,
    SkillValidationError,
)
from roomkit.sources.base import (
    BaseSourceProvider,
    EmitCallback,
    SourceHealth,
    SourceProvider,
    SourceStatus,
)
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.telemetry import (
    Attr,
    ConsoleTelemetryProvider,
    MockTelemetryProvider,
    NoopTelemetryProvider,
    Span,
    SpanKind,
    TelemetryConfig,
    TelemetryProvider,
)
from roomkit.tools.compose import compose_tool_handlers
from roomkit.voice import (
    AudioChunk,
    AudioFrame,
    AudioPipeline,
    AudioPipelineConfig,
    AudioReceivedCallback,
    BargeInCallback,
    BargeInEvent,
    DenoiserProvider,
    DiarizationProvider,
    DiarizationResult,
    MockDenoiserProvider,
    MockDiarizationProvider,
    MockVADProvider,
    PartialTranscriptionEvent,
    SpeakerChangeEvent,
    STTProvider,
    TranscriptionResult,
    TTSCancelledEvent,
    TTSProvider,
    VADAudioLevelEvent,
    VADConfig,
    VADEvent,
    VADEventType,
    VADProvider,
    VADSilenceEvent,
    VoiceBackend,
    VoiceCapability,
    VoiceSession,
    VoiceSessionState,
    parse_voice_session,
)
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.realtime import (
    MockRealtimeProvider,
    MockRealtimeTransport,
    RealtimeErrorEvent,
    RealtimeSpeechEvent,
    RealtimeToolCallEvent,
    RealtimeTranscriptionEvent,
    RealtimeVoiceProvider,
)
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

# AI documentation helpers (lazy import to avoid file I/O at import time)


def get_llms_txt() -> str:
    """Get the contents of llms.txt for LLM consumption."""
    from roomkit.ai_docs import get_llms_txt as _get_llms_txt

    return _get_llms_txt()


def get_agents_md() -> str:
    """Get the contents of AGENTS.md for AI coding assistants."""
    from roomkit.ai_docs import get_agents_md as _get_agents_md

    return _get_agents_md()


def get_ai_context() -> str:
    """Get combined AI context (AGENTS.md + llms.txt)."""
    from roomkit.ai_docs import get_ai_context as _get_ai_context

    return _get_ai_context()


__all__ = [
    "__version__",
    # Core
    "RoomKit",
    "RoomKitError",
    "RoomNotFoundError",
    "ChannelNotFoundError",
    "ChannelNotRegisteredError",
    "ParticipantNotFoundError",
    "IdentityNotFoundError",
    "SourceAlreadyAttachedError",
    "SourceNotFoundError",
    "VoiceBackendNotConfiguredError",
    "VoiceNotConfiguredError",
    "RoomLockManager",
    "InMemoryLockManager",
    # Sources (event-driven)
    "BaseSourceProvider",
    "EmitCallback",
    "SourceHealth",
    "SourceProvider",
    "SourceStatus",
    # Routing
    "InboundRoomRouter",
    "DefaultInboundRoomRouter",
    # Channels
    "Channel",
    "TransportChannel",
    "AIChannel",
    "EmailChannel",
    "RCSChannel",
    "SMSChannel",
    "RealtimeVoiceChannel",
    "VoiceChannel",
    "WebSocketChannel",
    # WebSocket Streaming
    "StreamChunk",
    "StreamEnd",
    "StreamMessage",
    "StreamSendFn",
    "StreamStart",
    "MessengerChannel",
    "TelegramChannel",
    "TeamsChannel",
    "HTTPChannel",
    "WhatsAppChannel",
    "WhatsAppPersonalChannel",
    # Models - Enums
    "Access",
    "ChannelCategory",
    "ChannelDirection",
    "ChannelMediaType",
    "ChannelType",
    "DeliveryMode",
    "EventStatus",
    "EventType",
    "HookExecution",
    "HookTrigger",
    "IdentificationStatus",
    "ParticipantRole",
    "ParticipantStatus",
    "RoomStatus",
    "TaskStatus",
    # Orchestration
    "ConversationPhase",
    "ConversationRouter",
    "ConversationState",
    "HANDOFF_TOOL",
    "HandoffHandler",
    "HandoffMemoryProvider",
    "HandoffRequest",
    "HandoffResult",
    "PhaseTransition",
    "RoutingConditions",
    "RoutingRule",
    "get_conversation_state",
    "set_conversation_state",
    "setup_handoff",
    # Models - Data
    "AudioContent",
    "ChannelBinding",
    "ChannelCapabilities",
    "ChannelOutput",
    "CompositeContent",
    "DeleteContent",
    "DeleteType",
    "DeliveryResult",
    "DeliveryStatus",
    "EditContent",
    "EventContent",
    "EventSource",
    "FrameworkEvent",
    "HookResult",
    "Identity",
    "IdentityHookResult",
    "IdentityResult",
    "InboundMessage",
    "InboundResult",
    "InjectedEvent",
    "LocationContent",
    "MediaContent",
    "Observation",
    "Participant",
    "ProtocolTrace",
    "ProviderResult",
    "RateLimit",
    "RetryPolicy",
    "RichContent",
    "Room",
    "RoomContext",
    "RoomEvent",
    "RoomTimers",
    "SystemContent",
    "Task",
    "TemplateContent",
    "TextContent",
    "VideoContent",
    # Hooks
    "HookEngine",
    "HookRegistration",
    # Provider Errors
    "ProviderError",
    # Provider ABCs
    "AIProvider",
    "EmailProvider",
    "HTTPProvider",
    "MessengerProvider",
    "TelegramProvider",
    "TeamsProvider",
    "RCSProvider",
    "SMSProvider",
    "WhatsAppProvider",
    # AI
    "AIContext",
    "AIImagePart",
    "AIMessage",
    "AIResponse",
    "AITextPart",
    "AITool",
    "AIToolCall",
    "AIToolCallPart",
    "AIToolResultPart",
    "StreamDone",
    "StreamEvent",
    "StreamTextDelta",
    "StreamToolCall",
    "MockAIProvider",
    # AI – Anthropic
    "AnthropicAIProvider",
    "AnthropicConfig",
    # AI – Gemini
    "GeminiAIProvider",
    "GeminiConfig",
    # AI – OpenAI
    "OpenAIAIProvider",
    "OpenAIConfig",
    # AI – vLLM (local)
    "VLLMConfig",
    "create_vllm_provider",
    # HTTP – Generic Webhook
    "HTTPProviderConfig",
    "MockHTTPProvider",
    "WebhookHTTPProvider",
    "parse_http_webhook",
    # Email
    "ElasticEmailConfig",
    "ElasticEmailProvider",
    "MockEmailProvider",
    # Messenger
    "FacebookMessengerProvider",
    "MessengerConfig",
    "MockMessengerProvider",
    "parse_messenger_webhook",
    # Telegram
    "MockTelegramProvider",
    "TelegramBotProvider",
    "TelegramConfig",
    "parse_telegram_webhook",
    # Teams
    "BotFrameworkTeamsProvider",
    "ConversationReferenceStore",
    "InMemoryConversationReferenceStore",
    "MockTeamsProvider",
    "TeamsConfig",
    "is_bot_added",
    "parse_teams_activity",
    "parse_teams_webhook",
    # SMS
    "MockSMSProvider",
    "WebhookMeta",
    "extract_sms_meta",
    "is_valid_phone",
    "normalize_phone",
    # SMS - Sinch
    "SinchConfig",
    "SinchSMSProvider",
    "parse_sinch_webhook",
    # SMS - Telnyx
    "TelnyxConfig",
    "TelnyxSMSProvider",
    "parse_telnyx_webhook",
    # RCS - Telnyx
    "TelnyxRCSConfig",
    "TelnyxRCSProvider",
    "parse_telnyx_rcs_webhook",
    # SMS - Twilio
    "TwilioConfig",
    "TwilioSMSProvider",
    "parse_twilio_webhook",
    # SMS - VoiceMeUp
    "VoiceMeUpConfig",
    "VoiceMeUpSMSProvider",
    "configure_voicemeup_mms",
    "parse_voicemeup_webhook",
    # RCS
    "MockRCSProvider",
    "RCSDeliveryResult",
    # RCS - Twilio
    "TwilioRCSConfig",
    "TwilioRCSProvider",
    "parse_twilio_rcs_webhook",
    # WhatsApp
    "MockWhatsAppProvider",
    "WhatsAppPersonalProvider",
    # Identity
    "IdentityResolver",
    "MockIdentityResolver",
    # Memory
    "MemoryProvider",
    "MemoryResult",
    "MockMemoryProvider",
    "SlidingWindowMemory",
    # Store
    "ConversationStore",
    "InMemoryStore",
    "PostgresStore",
    # Realtime
    "EphemeralCallback",
    "EphemeralEvent",
    "EphemeralEventType",
    "InMemoryRealtime",
    "RealtimeBackend",
    # Voice
    "AudioChunk",
    "AudioFrame",
    "AudioPipeline",
    "AudioPipelineConfig",
    "AudioReceivedCallback",
    "BargeInCallback",
    "BargeInEvent",
    "DenoiserProvider",
    "DiarizationProvider",
    "DiarizationResult",
    "MockDenoiserProvider",
    "MockDiarizationProvider",
    "MockSTTProvider",
    "MockTTSProvider",
    "MockVADProvider",
    "MockVoiceBackend",
    "parse_voice_session",
    "PartialTranscriptionEvent",
    "SpeakerChangeEvent",
    "STTProvider",
    "TTSCancelledEvent",
    "TTSProvider",
    "TranscriptionResult",
    "VADConfig",
    "VADAudioLevelEvent",
    "VADEvent",
    "VADEventType",
    "VADProvider",
    "VADSilenceEvent",
    "VoiceBackend",
    "VoiceCapability",
    "VoiceSession",
    "VoiceSessionState",
    # Realtime Voice
    "MockRealtimeProvider",
    "MockRealtimeTransport",
    "RealtimeErrorEvent",
    "RealtimeSpeechEvent",
    "RealtimeToolCallEvent",
    "RealtimeTranscriptionEvent",
    "RealtimeVoiceProvider",
    # Skills
    "ScriptExecutor",
    "ScriptResult",
    "Skill",
    "SkillMetadata",
    "SkillParseError",
    "SkillRegistry",
    "SkillValidationError",
    # Telemetry
    "Attr",
    "ConsoleTelemetryProvider",
    "MockTelemetryProvider",
    "NoopTelemetryProvider",
    "OpenTelemetryProvider",
    "Span",
    "SpanKind",
    "TelemetryConfig",
    "TelemetryProvider",
    # Tools
    "MCPToolProvider",
    "compose_tool_handlers",
    # AI Docs
    "get_agents_md",
    "get_ai_context",
    "get_llms_txt",
]


def __getattr__(name: str) -> object:
    if name == "PostgresStore":
        from roomkit.store.postgres import PostgresStore

        return PostgresStore
    if name == "OpenTelemetryProvider":
        from roomkit.telemetry.opentelemetry import OpenTelemetryProvider

        return OpenTelemetryProvider
    if name == "MCPToolProvider":
        from roomkit.tools.mcp import MCPToolProvider

        return MCPToolProvider
    raise AttributeError(f"module 'roomkit' has no attribute {name}")
