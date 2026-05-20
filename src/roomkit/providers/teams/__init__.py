"""Microsoft Teams provider."""

from roomkit.providers.teams.base import TeamsProvider
from roomkit.providers.teams.bot_framework import BotFrameworkTeamsProvider
from roomkit.providers.teams.config import TeamsConfig
from roomkit.providers.teams.conversation_store import (
    ConversationReferenceStore,
    InMemoryConversationReferenceStore,
)
from roomkit.providers.teams.mock import MockTeamsProvider
from roomkit.providers.teams.models import TeamsMember
from roomkit.providers.teams.webhook import (
    is_bot_added,
    parse_teams_activity,
    parse_teams_reactions,
    parse_teams_webhook,
)

__all__ = [
    "BotFrameworkTeamsProvider",
    "ConversationReferenceStore",
    "InMemoryConversationReferenceStore",
    "MockTeamsProvider",
    "TeamsConfig",
    "TeamsMember",
    "TeamsProvider",
    "is_bot_added",
    "parse_teams_activity",
    "parse_teams_reactions",
    "parse_teams_webhook",
]
