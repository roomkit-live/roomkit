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
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.av import AudioVideoChannel
from roomkit.channels.base import Channel
from roomkit.channels.cli import CLIChannel
from roomkit.channels.realtime_av import RealtimeAudioVideoChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel, get_current_voice_session
from roomkit.channels.realtime_voice import ToolHandler as ToolHandler
from roomkit.channels.transport import TransportChannel
from roomkit.channels.video import VideoChannel
from roomkit.channels.voice import VoiceChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.delivery import DeliveryStrategy, Immediate, Queued, WaitForIdle
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
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
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
    ChannelType,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
    RoomStatus,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.participant import Participant
from roomkit.models.room import Room, RoomTimers
from roomkit.models.session_event import SessionStartedEvent
from roomkit.models.tool_call import ToolCallCallback, ToolCallEvent
from roomkit.orchestration import (
    Loop,
    Orchestration,
    Pipeline,
    Supervisor,
    Swarm,
)
from roomkit.tools.base import Tool

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
    "RoomNotFoundError",
    "ChannelNotFoundError",
    "ChannelNotRegisteredError",
    "ParticipantNotFoundError",
    "IdentityNotFoundError",
    "SourceAlreadyAttachedError",
    "SourceNotFoundError",
    "VoiceBackendNotConfiguredError",
    "VoiceNotConfiguredError",
    # Delivery
    "DeliveryStrategy",
    "Immediate",
    "Queued",
    "WaitForIdle",
    # Channels
    "Agent",
    "AIChannel",
    "AudioVideoChannel",
    "Channel",
    "CLIChannel",
    "EmailChannel",
    "HTTPChannel",
    "MessengerChannel",
    "RCSChannel",
    "RealtimeAudioVideoChannel",
    "RealtimeVoiceChannel",
    "SMSChannel",
    "TeamsChannel",
    "TelegramChannel",
    "TransportChannel",
    "VideoChannel",
    "VoiceChannel",
    "WebSocketChannel",
    "WhatsAppChannel",
    "WhatsAppPersonalChannel",
    # Enums (core)
    "Access",
    "ChannelCategory",
    "ChannelType",
    "EventStatus",
    "EventType",
    "HookExecution",
    "HookTrigger",
    "RoomStatus",
    # Orchestration strategies
    "Loop",
    "Orchestration",
    "Pipeline",
    "Supervisor",
    "Swarm",
    # Models (core)
    "ChannelBinding",
    "ChannelCapabilities",
    "ChannelOutput",
    "DeliveryResult",
    "DeliveryStatus",
    "EventSource",
    "FrameworkEvent",
    "HookResult",
    "InjectedEvent",
    "InboundMessage",
    "InboundResult",
    "Participant",
    "ProviderResult",
    "Room",
    "RoomContext",
    "RoomEvent",
    "RoomTimers",
    "SessionStartedEvent",
    "TextContent",
    "Tool",
    "ToolCallCallback",
    "ToolCallEvent",
    "ToolHandler",
    "get_current_voice_session",
    # AI docs
    "get_agents_md",
    "get_ai_context",
    "get_llms_full_txt",
    "get_llms_txt",
]
