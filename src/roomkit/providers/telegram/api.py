"""The Telegram Bot API, as methods.

One base URL, one client, one error shape. An application talking to Telegram
needs more than the sends a room produces — it registers a webhook, identifies
its own bot, acknowledges a button press, rewrites a message it already sent.
Those calls are the same HTTP for everyone, and writing them a second time in
an application means a second base URL holding the same token, a second retry
policy and a second spelling of "Telegram said no".

:class:`TelegramBotProvider` is this class plus the translation of a
``RoomEvent`` into a Telegram message.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

from roomkit.models.delivery import ProviderResult
from roomkit.providers.telegram.config import TelegramConfig
from roomkit.telemetry.noop import NoopTelemetryProvider

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger("roomkit.providers.telegram")

_ResultShape = Literal["bot", "updates", "message", "true"]
_TRUE_RESULT_METHODS = frozenset(
    {
        "setWebhook",
        "deleteWebhook",
        "leaveChat",
        "sendChatAction",
        "answerCallbackQuery",
    }
)


class TelegramBotAPI:
    """Call the Telegram Bot API with a bot token.

    Every method answers with a :class:`ProviderResult`, so a caller reads
    success and failure the same way whichever call it made: ``error`` is
    ``telegram_<code>`` when Telegram refused, ``http_<status>`` when the
    refusal carried no Bot API body, and ``timeout`` when nothing came back.
    Telegram's own words for a refusal — the only text precise enough to tell a
    caller what to fix — arrive as ``metadata["description"]``.

    The two reads, :meth:`get_me` and :meth:`get_updates`, also carry Telegram's
    ``result`` payload under ``metadata["result"]``. Sends do not: their result
    is a Message object the caller already has, reduced to the
    ``provider_message_id`` it actually uses.
    """

    def __init__(self, config: TelegramConfig) -> None:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for TelegramBotProvider. "
                "Install it with: pip install roomkit[telegram]"
            ) from exc
        self._config = config
        self._webhook_secret = (
            config.webhook_secret.get_secret_value() if config.webhook_secret else None
        )
        self._httpx = _httpx
        self._client: httpx.AsyncClient = _httpx.AsyncClient(
            timeout=config.timeout,
        )

    # --- Identity and update source -------------------------------------

    async def get_me(self) -> ProviderResult:
        """Identify the bot behind the token — ``getMe``.

        The call that tells a good token from a typo. A wrong one comes back
        ``telegram_401``; an unreachable Telegram as ``timeout`` or an
        ``http_*``. On success ``metadata["result"]`` holds the bot object —
        ``id``, ``username``, ``first_name``.
        """
        return await self._request("getMe", json={}, want_result=True, result_shape="bot")

    async def get_updates(self, *, limit: int = 100, offset: int | None = None) -> ProviderResult:
        """Pull pending updates — ``getUpdates``.

        Telegram delivers updates one way or the other, never both: this
        returns nothing at all while a webhook is registered. Its use is the
        moment before one is — reading who has already written to the bot.

        Args:
            limit: How many updates to take, at most 100.
            offset: Skip past updates below this id, confirming them as read.

        Returns:
            A result whose ``metadata["result"]`` is the list of raw updates.
        """
        payload: dict[str, Any] = {"limit": limit}
        if offset is not None:
            payload["offset"] = offset
        return await self._request(
            "getUpdates", json=payload, want_result=True, result_shape="updates"
        )

    # --- Webhook lifecycle ----------------------------------------------

    async def set_webhook(
        self,
        url: str,
        *,
        secret: str | None = None,
        allowed_updates: list[str] | None = None,
        drop_pending_updates: bool = False,
    ) -> ProviderResult:
        """Register the URL Telegram POSTs updates to — ``setWebhook``.

        Args:
            url: A public HTTPS URL. Telegram refuses anything else, and
                refuses a URL it cannot reach — both come back as a failed
                result whose ``metadata["description"]`` says which.
            secret: Echoed back on every request in the
                ``X-Telegram-Bot-Api-Secret-Token`` header, which
                :meth:`TelegramBotProvider.verify_signature` checks. When
                omitted, :attr:`TelegramConfig.webhook_secret` is used. A
                successfully registered explicit value becomes the value this
                provider verifies, so registration and verification cannot
                silently diverge. Without either value, any host that learns
                the URL can post updates to it.
            allowed_updates: The update kinds to receive. Telegram's own
                default omits some kinds entirely, so a consumer that wants
                ``callback_query`` must ask for it by name.
            drop_pending_updates: Discard what queued up while no webhook was
                registered, rather than delivering it all at once.
        """
        webhook_secret = self._webhook_secret if secret is None else secret
        payload: dict[str, Any] = {"url": url}
        if webhook_secret:
            payload["secret_token"] = webhook_secret
        if allowed_updates is not None:
            payload["allowed_updates"] = allowed_updates
        if drop_pending_updates:
            payload["drop_pending_updates"] = True
        result = await self._api_call("setWebhook", payload)
        if result.success:
            # Telegram has accepted this exact value. Keep verification tied to
            # it even when the caller rotated the config's initial secret.
            self._webhook_secret = webhook_secret
        return result

    async def delete_webhook(self, *, drop_pending_updates: bool = False) -> ProviderResult:
        """Stop Telegram POSTing updates — ``deleteWebhook``."""
        payload: dict[str, Any] = {}
        if drop_pending_updates:
            payload["drop_pending_updates"] = True
        return await self._api_call("deleteWebhook", payload)

    async def leave_chat(self, chat_id: str) -> ProviderResult:
        """Leave a group, supergroup or channel — ``leaveChat``.

        The bot stops receiving that chat's updates. Being removed from a chat
        is the only way to stop them: a webhook is per-bot, not per-chat.
        """
        return await self._api_call("leaveChat", {"chat_id": chat_id})

    # --- Sends that are not a room's traffic ----------------------------

    async def send_message(self, chat_id: str, text: str) -> ProviderResult:
        """Send text as it stands — ``sendMessage``, no Markdown rendering.

        A room's outbound traffic goes through :meth:`TelegramBotProvider.send`,
        which renders a ``RoomEvent``. This is for text that is already final
        and belongs to no room — a connection confirmation, an acknowledgement.
        """
        return await self._api_call("sendMessage", {"chat_id": chat_id, "text": text})

    async def send_force_reply(self, chat_id: str, text: str) -> ProviderResult:
        """Send a message Telegram opens a reply box under — ``force_reply``.

        The answer arrives as an ordinary message carrying
        ``reply_to_message.message_id``. Matching that against the
        ``provider_message_id`` returned here is what ties an answer back to
        the question it answers.
        """
        return await self._api_call(
            "sendMessage",
            {"chat_id": chat_id, "text": text, "reply_markup": {"force_reply": True}},
        )

    async def send_chat_action(self, chat_id: str, action: str = "typing") -> ProviderResult:
        """Show a transient status such as "typing…" — ``sendChatAction``.

        Telegram clears it after about five seconds, or as soon as the bot
        sends a real message. Holding it up for a long generation therefore
        means re-sending it on a shorter cycle than that.
        """
        return await self._api_call("sendChatAction", {"chat_id": chat_id, "action": action})

    # --- Acting on a message already sent -------------------------------

    async def answer_callback_query(
        self, callback_query_id: str, text: str = ""
    ) -> ProviderResult:
        """Acknowledge an inline-button press — ``answerCallbackQuery``.

        Not optional: until it arrives the client keeps a spinner on the
        button. ``text``, when given, flashes to the user as a toast.
        """
        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        return await self._api_call("answerCallbackQuery", payload)

    async def edit_message_text(
        self,
        chat_id: str,
        message_id: int | str,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> ProviderResult:
        """Rewrite a message already in the chat — ``editMessageText``.

        Args:
            chat_id: The chat holding the message.
            message_id: The message to rewrite.
            text: Its new text.
            reply_markup: Its new keyboard. Left alone when omitted;
                ``{"inline_keyboard": []}`` drops the buttons, which is how a
                settled prompt stops being clickable.
        """
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return await self._api_call("editMessageText", payload)

    async def edit_message_reply_markup(
        self, chat_id: str, message_id: int | str, reply_markup: dict[str, Any]
    ) -> ProviderResult:
        """Replace a message's keyboard and nothing else — ``editMessageReplyMarkup``.

        What reflects a toggled selection back to the person who tapped it,
        without reprinting the message around it.
        """
        return await self._api_call(
            "editMessageReplyMarkup",
            {"chat_id": chat_id, "message_id": message_id, "reply_markup": reply_markup},
        )

    # --- Inbound files ---------------------------------------------------

    async def get_file(self, file_id: str) -> str | None:
        """Resolve an inbound ``file_id`` to a Bot API file path.

        The ``file_id`` arrives on an inbound update — both
        :func:`~roomkit.providers.telegram.webhook.parse_telegram_message` and
        :func:`~roomkit.providers.telegram.webhook.parse_telegram_webhook` put
        it in ``metadata["file_id"]`` for every media message. Pair the path
        this returns with :meth:`download_file` to get the bytes; the bot token
        needed for both lives here, not in the calling application.

        Telegram keeps the path valid for at least an hour, and refuses any
        file over 20 MB — the Bot API download ceiling — with a ``400``. An
        update carries ``metadata["file_size"]``, so a caller can tell which
        files are past that ceiling without spending the call.

        Args:
            file_id: Identifier from an inbound Telegram update.

        Returns:
            The file path to download, or None if Telegram refused the file or
            the call failed.
        """
        try:
            resp = await self._client.post(
                f"{self._config.base_url}/getFile", json={"file_id": file_id}
            )
            resp.raise_for_status()
            body = resp.json()
        except (self._httpx.HTTPError, ValueError) as exc:
            # ValueError covers a 2xx that is not JSON — an intercepting proxy's
            # error page reaches raise_for_status intact.
            logger.warning("getFile failed for file_id %s: %s", file_id, self._error_label(exc))
            return None
        if not isinstance(body, dict) or body.get("ok") is not True:
            return None
        result = body.get("result")
        if not isinstance(result, dict):
            return None
        file_path = result.get("file_path")
        return file_path if isinstance(file_path, str) and file_path else None

    async def download_file(self, file_path: str) -> bytes | None:
        """Download the bytes behind a path returned by :meth:`get_file`.

        Args:
            file_path: Path returned by :meth:`get_file`.

        Returns:
            The file content, or None if the download failed. Files over the
            Bot API's 20 MB ceiling never get this far — :meth:`get_file`
            already returned None for them.
        """
        try:
            resp = await self._client.get(f"{self._config.file_base_url}/{file_path}")
            resp.raise_for_status()
        except self._httpx.HTTPError as exc:
            logger.warning("download failed for %s: %s", file_path, self._error_label(exc))
            return None
        return resp.content

    # --- Transport -------------------------------------------------------

    @staticmethod
    def _error_label(exc: Exception) -> str:
        """Describe an httpx failure without its URL — every Bot API URL holds the token.

        ``str(HTTPStatusError)`` names the URL it failed on, which would put the
        bot token in the logs.
        """
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return f"{type(exc).__name__}({status})" if status else type(exc).__name__

    async def _api_call(self, method: str, payload: dict[str, Any]) -> ProviderResult:
        result_shape: _ResultShape = "true" if method in _TRUE_RESULT_METHODS else "message"
        return await self._request(method, json=payload, result_shape=result_shape)

    async def _api_upload(
        self, method: str, data: dict[str, Any], files: dict[str, Any]
    ) -> ProviderResult:
        """Multipart variant for sending file bytes (sendDocument/sendPhoto)."""
        return await self._request(method, data=data, files=files, result_shape="message")

    async def _request(
        self,
        method: str,
        *,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        want_result: bool = False,
        result_shape: _ResultShape | None = None,
    ) -> ProviderResult:
        url = f"{self._config.base_url}/{method}"
        try:
            t0 = time.monotonic()
            resp = await self._client.post(url, json=json, data=data, files=files)
            resp.raise_for_status()
            send_ms = (time.monotonic() - t0) * 1000
            body = resp.json()

            _tel = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
            _tel.record_metric(
                "roomkit.delivery.send_ms",
                send_ms,
                unit="ms",
                attributes={"provider": "TelegramBotProvider"},
            )
        except self._httpx.TimeoutException:
            return ProviderResult(success=False, error="timeout")
        except self._httpx.HTTPStatusError as exc:
            return self._parse_error(exc)
        except self._httpx.HTTPError as exc:
            # httpx error strings commonly include the request URL. Telegram
            # embeds the bot token in that URL, so expose only a safe label.
            return ProviderResult(success=False, error=self._error_label(exc))
        except ValueError:
            # A 2xx carrying something other than JSON — an intercepting proxy's
            # error page passes raise_for_status intact. Every call promises a
            # ProviderResult, so this cannot be allowed to escape as an exception.
            return ProviderResult(success=False, error="invalid_response")

        if not isinstance(body, dict):
            return ProviderResult(success=False, error="invalid_response")
        if body.get("ok") is not True:
            error_code = body.get("error_code")
            description = body.get("description")
            if not isinstance(error_code, int) or isinstance(error_code, bool):
                return ProviderResult(success=False, error="invalid_response")
            return ProviderResult(
                success=False,
                error=f"telegram_{error_code}",
                metadata={"description": description if isinstance(description, str) else ""},
            )
        if "result" not in body:
            return ProviderResult(success=False, error="invalid_response")

        # A send answers with a Message object; the rest answer with a bare
        # ``true``, a list of updates, or the bot itself. Only a Message has an
        # id to carry, so reach for one only when the result is shaped like one.
        result = body.get("result")
        if result_shape is not None and not self._valid_result(result, result_shape):
            return ProviderResult(success=False, error="invalid_response")
        message_id = str(result.get("message_id", "")) if isinstance(result, dict) else ""
        return ProviderResult(
            success=True,
            provider_message_id=message_id,
            metadata={"result": result} if want_result else {},
        )

    @staticmethod
    def _valid_result(result: Any, shape: _ResultShape) -> bool:
        """Validate the operation-specific success payload Telegram promises."""

        def is_int(value: Any) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        if shape == "true":
            return result is True
        if shape == "message":
            return isinstance(result, dict) and is_int(result.get("message_id"))
        if shape == "bot":
            return (
                isinstance(result, dict)
                and is_int(result.get("id"))
                and isinstance(result.get("is_bot"), bool)
                and isinstance(result.get("first_name"), str)
            )
        return isinstance(result, list) and all(
            isinstance(update, dict) and is_int(update.get("update_id")) for update in result
        )

    @staticmethod
    def _parse_error(exc: Any) -> ProviderResult:
        """Extract a Telegram Bot API error when available."""
        try:
            body = exc.response.json()
            error_code = body.get("error_code", exc.response.status_code)
            description = body.get("description", "")
            return ProviderResult(
                success=False,
                error=f"telegram_{error_code}",
                metadata={"description": description},
            )
        except Exception:
            return ProviderResult(
                success=False,
                error=f"http_{exc.response.status_code}",
            )

    async def close(self) -> None:
        await self._client.aclose()
