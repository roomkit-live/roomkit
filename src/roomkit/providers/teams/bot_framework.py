"""Microsoft Teams provider using the Bot Framework SDK."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.teams.base import TeamsProvider
from roomkit.providers.teams.config import TeamsConfig
from roomkit.providers.teams.conversation_store import (
    ConversationReferenceStore,
    InMemoryConversationReferenceStore,
)

if TYPE_CHECKING:
    from botbuilder.core import BotFrameworkAdapter

logger = logging.getLogger("roomkit.providers.teams")


class BotFrameworkTeamsProvider(TeamsProvider):
    """Send messages via the Microsoft Bot Framework SDK."""

    def __init__(
        self,
        config: TeamsConfig,
        *,
        conversation_store: ConversationReferenceStore | None = None,
    ) -> None:
        try:
            from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
        except ImportError as exc:
            raise ImportError(
                "botbuilder-core is required for BotFrameworkTeamsProvider. "
                "Install it with: pip install roomkit[teams]"
            ) from exc

        self._config = config
        self._conversation_store = conversation_store or InMemoryConversationReferenceStore()
        settings_kwargs: dict[str, Any] = {
            "app_id": config.app_id,
            "app_password": config.app_password.get_secret_value(),
        }
        if config.tenant_id != "common":
            settings_kwargs["channel_auth_tenant"] = config.tenant_id
        self._adapter: BotFrameworkAdapter = BotFrameworkAdapter(
            BotFrameworkAdapterSettings(**settings_kwargs)
        )

    @property
    def adapter(self) -> BotFrameworkAdapter:
        """The underlying Bot Framework adapter."""
        return self._adapter

    @property
    def conversation_store(self) -> ConversationReferenceStore:
        """The conversation reference store."""
        return self._conversation_store

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        text = self._extract_text(event)
        if not text:
            return ProviderResult(success=False, error="empty_message")

        ref = await self._conversation_store.get(to)
        if ref is None:
            return ProviderResult(
                success=False,
                error="no_conversation_reference",
                metadata={"conversation_id": to},
            )

        try:
            import time

            from botbuilder.core import TurnContext
            from botbuilder.schema import Activity, ConversationReference

            conv_ref = ConversationReference().deserialize(ref)
            message_id: str | None = None

            async def _send_callback(turn_context: TurnContext) -> None:
                nonlocal message_id
                response = await turn_context.send_activity(Activity(type="message", text=text))
                if response and response.id:
                    message_id = response.id

            t0 = time.monotonic()
            await self._adapter.continue_conversation(
                conv_ref,
                _send_callback,
                self._config.app_id,
            )
            send_ms = (time.monotonic() - t0) * 1000

            from roomkit.telemetry.noop import NoopTelemetryProvider

            _tel = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
            _tel.record_metric(
                "roomkit.delivery.send_ms",
                send_ms,
                unit="ms",
                attributes={"provider": "BotFrameworkTeamsProvider"},
            )
        except Exception as exc:
            return ProviderResult(success=False, error=str(exc))

        return ProviderResult(success=True, provider_message_id=message_id)

    async def save_conversation_reference(self, activity_dict: dict[str, Any]) -> None:
        """Extract and store a ConversationReference from an inbound Activity dict."""
        from botbuilder.core import TurnContext
        from botbuilder.schema import Activity

        activity = Activity().deserialize(activity_dict)
        ref = TurnContext.get_conversation_reference(activity)
        conv_id = activity_dict.get("conversation", {}).get("id", "")
        if conv_id:
            await self._conversation_store.save(conv_id, ref.serialize())

    async def create_channel_conversation(
        self,
        service_url: str,
        channel_id: str,
        *,
        tenant_id: str | None = None,
    ) -> str:
        """Create a conversation in a Teams channel and store its reference.

        Use this to proactively message a Teams channel the bot has been
        installed in, even if no user has messaged the bot in that channel yet.

        Args:
            service_url: The Bot Framework service URL for the team
                (e.g. ``"https://smba.trafficmanager.net/amer/"``).
                Available from a prior ``conversationUpdate`` Activity or
                from :func:`parse_teams_activity`.
            channel_id: The Teams channel ID
                (e.g. ``"19:abc123@thread.tacv2"``).
            tenant_id: Azure AD tenant ID.  Falls back to
                :attr:`TeamsConfig.tenant_id` if not provided.

        Returns:
            The conversation ID for the newly created conversation.
            This ID can be used as the ``to`` parameter in :meth:`send`.

        Raises:
            RuntimeError: If the conversation could not be created.
        """
        from botbuilder.schema import (
            Activity,
            ChannelAccount,
            ConversationParameters,
            ConversationReference,
        )

        tid = tenant_id or self._config.tenant_id

        params = ConversationParameters(
            is_group=True,
            channel_data={"channel": {"id": channel_id}},
            tenant_id=tid,
            bot=ChannelAccount(id=self._config.app_id),
            activity=Activity(type="message", text=""),
        )

        response = await self._adapter.create_conversation(
            ConversationReference(service_url=service_url),
            params,
        )

        if not response or not response.id:
            msg = f"Failed to create conversation in channel {channel_id}"
            raise RuntimeError(msg)

        conv_id = response.id

        # Build and store a conversation reference so send() works immediately
        ref = ConversationReference(
            service_url=service_url,
            channel_id="msteams",
            conversation={"id": conv_id, "isGroup": True},
            bot=ChannelAccount(id=self._config.app_id),
        )
        await self._conversation_store.save(conv_id, ref.serialize())

        return str(conv_id)

    def verify_signature(
        self,
        payload: bytes,  # noqa: ARG002
        signature: str,
    ) -> bool:
        """Verify a Bot Framework JWT bearer token.

        Validates the JWT from the ``Authorization: Bearer <token>`` header
        using Microsoft's OpenID Connect metadata and signing keys.

        Requires the ``PyJWT`` and ``cryptography`` packages::

            pip install PyJWT cryptography

        Args:
            payload: Raw request body bytes (unused — Bot Framework signs
                the token, not the payload).
            signature: The full ``Authorization`` header value, including
                the ``Bearer `` prefix, OR just the raw JWT token.

        Returns:
            True if the JWT is valid (signature, issuer, audience, expiry).

        Raises:
            ValueError: If required dependencies are missing.
        """
        try:
            import jwt
            from jwt import PyJWKClient
        except ImportError as exc:
            raise ValueError(
                "PyJWT[crypto] is required for Teams JWT verification. "
                "Install it with: pip install 'PyJWT[crypto]'"
            ) from exc

        token = signature.removeprefix("Bearer ").strip()
        if not token:
            return False

        # Allowed issuers for Bot Framework v3.1 and v3.2
        allowed_issuers = {
            "https://api.botframework.com",
            "https://sts.windows.net/d6d49420-f39b-4df7-a1dc-d59a935871db/",
            "https://login.microsoftonline.com/d6d49420-f39b-4df7-a1dc-d59a935871db/v2.0",
            f"https://sts.windows.net/{self._config.tenant_id}/",
            f"https://login.microsoftonline.com/{self._config.tenant_id}/v2.0",
        }

        openid_urls = [
            "https://login.botframework.com/v1/.well-known/openidconfiguration",
            "https://login.microsoftonline.com/botframework.com/v2.0/.well-known/openid-configuration",
        ]

        for openid_url in openid_urls:
            try:
                jwk_client = PyJWKClient(openid_url, cache_jwk_set=True, lifespan=3600)
                signing_key = jwk_client.get_signing_key_from_jwt(token)
                claims = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["RS256"],
                    audience=self._config.app_id,
                    options={"verify_exp": True},
                )
                issuer = claims.get("iss", "")
                if issuer in allowed_issuers:
                    return True
            except Exception:  # nosec B112 — intentional: try next OpenID endpoint
                continue

        return False

    @staticmethod
    def _extract_text(event: RoomEvent) -> str:
        content = event.content
        if isinstance(content, TextContent):
            return content.body
        if hasattr(content, "body"):
            return str(content.body)
        return ""

    async def close(self) -> None:
        self._adapter = None
        self._conversation_store = None  # type: ignore[assignment]
