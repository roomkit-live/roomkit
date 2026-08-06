"""Tests for the Telegram Bot API surface — the calls an application makes
around its sends: identifying its bot, registering a webhook, acknowledging a
button press, rewriting a message it already sent."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from roomkit.providers.telegram import TelegramBotAPI, TelegramConfig


def _config(**overrides: Any) -> TelegramConfig:
    defaults: dict[str, Any] = {"bot_token": "123456:ABC-DEF"}
    defaults.update(overrides)
    return TelegramConfig(**defaults)


class _Recorder(httpx.AsyncBaseTransport):
    """Answers every call with ``result``, keeping the requests it was sent."""

    def __init__(self, result: Any = None, status: int = 200) -> None:
        self._result = result if result is not None else {"message_id": 7}
        self._status = status
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self._status, json={"ok": True, "result": self._result}, request=request
        )

    @property
    def last_method(self) -> str:
        return str(self.requests[-1].url).rsplit("/", 1)[-1]

    @property
    def last_body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


class _Refusing(httpx.AsyncBaseTransport):
    def __init__(self, status: int = 401, description: str = "Unauthorized") -> None:
        self._status = status
        self._description = description

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            self._status,
            json={"ok": False, "error_code": self._status, "description": self._description},
            request=request,
        )


class _TimingOut(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")


class _ConnectionFailure(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"could not connect to {request.url}", request=request)


class _WrongJsonShape(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["valid JSON", "wrong shape"], request=request)


def _api(transport: httpx.AsyncBaseTransport) -> TelegramBotAPI:
    api = TelegramBotAPI(_config())
    api._client = httpx.AsyncClient(transport=transport)
    return api


class TestReads:
    async def test_get_me_carries_the_bot_object(self) -> None:
        bot = {"id": 42, "is_bot": True, "username": "demo_bot", "first_name": "Demo"}
        transport = _Recorder(result=bot)

        result = await _api(transport).get_me()

        assert result.success is True
        assert result.metadata["result"] == bot
        assert transport.last_method == "getMe"

    async def test_get_me_on_a_bad_token_names_the_refusal(self) -> None:
        result = await _api(_Refusing()).get_me()

        assert result.success is False
        assert result.error == "telegram_401"
        assert result.metadata["description"] == "Unauthorized"

    async def test_get_me_on_an_unreachable_telegram_is_not_a_bad_token(self) -> None:
        """The distinction the caller acts on: retry, or tell the user to fix the token."""
        result = await _api(_TimingOut()).get_me()

        assert result.success is False
        assert result.error == "timeout"

    async def test_get_updates_carries_the_list(self) -> None:
        updates = [{"update_id": 1, "message": {"message_id": 9}}]
        transport = _Recorder(result=updates)

        result = await _api(transport).get_updates(limit=50, offset=3)

        assert result.success is True
        assert result.metadata["result"] == updates
        assert transport.last_method == "getUpdates"
        assert transport.last_body == {"limit": 50, "offset": 3}

    async def test_get_updates_omits_an_offset_it_was_not_given(self) -> None:
        transport = _Recorder(result=[])

        await _api(transport).get_updates()

        assert transport.last_body == {"limit": 100}

    async def test_get_file_ignores_valid_json_with_the_wrong_shape(self) -> None:
        assert await _api(_WrongJsonShape()).get_file("file-1") is None

    async def test_a_list_result_does_not_become_a_message_id(self) -> None:
        """``result`` is a Message only for a send; reaching into a list would raise."""
        result = await _api(_Recorder(result=[{"update_id": 1}])).get_updates()

        assert result.success is True
        assert result.provider_message_id == ""


class TestWebhookLifecycle:
    async def test_set_webhook_sends_url_secret_and_allowed_updates(self) -> None:
        transport = _Recorder(result=True)

        result = await _api(transport).set_webhook(
            "https://example.test/hook",
            secret="s3cret",
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
        )

        assert result.success is True
        assert transport.last_method == "setWebhook"
        assert transport.last_body == {
            "url": "https://example.test/hook",
            "secret_token": "s3cret",
            "allowed_updates": ["message", "callback_query"],
            "drop_pending_updates": True,
        }

    async def test_set_webhook_sends_only_the_url_by_default(self) -> None:
        transport = _Recorder(result=True)

        await _api(transport).set_webhook("https://example.test/hook")

        assert transport.last_body == {"url": "https://example.test/hook"}

    async def test_set_webhook_uses_the_configured_secret_by_default(self) -> None:
        transport = _Recorder(result=True)
        api = TelegramBotAPI(_config(webhook_secret="configured-secret"))
        api._client = httpx.AsyncClient(transport=transport)

        await api.set_webhook("https://example.test/hook")

        assert transport.last_body == {
            "url": "https://example.test/hook",
            "secret_token": "configured-secret",
        }

    async def test_a_bare_true_result_is_a_success_not_a_crash(self) -> None:
        """Every call but a send answers ``result: true``; only a Message has an id."""
        result = await _api(_Recorder(result=True)).set_webhook("https://example.test/hook")

        assert result.success is True
        assert result.provider_message_id == ""

    async def test_set_webhook_refused_says_why_in_telegrams_words(self) -> None:
        transport = _Refusing(400, "Bad Request: bad webhook: HTTPS url must be provided")

        result = await _api(transport).set_webhook("http://example.test/hook")

        assert result.success is False
        assert result.error == "telegram_400"
        assert "HTTPS" in result.metadata["description"]

    async def test_delete_webhook(self) -> None:
        transport = _Recorder(result=True)

        await _api(transport).delete_webhook(drop_pending_updates=True)

        assert transport.last_method == "deleteWebhook"
        assert transport.last_body == {"drop_pending_updates": True}

    async def test_delete_webhook_keeps_pending_updates_by_default(self) -> None:
        transport = _Recorder(result=True)

        await _api(transport).delete_webhook()

        assert transport.last_body == {}

    async def test_leave_chat(self) -> None:
        transport = _Recorder(result=True)

        await _api(transport).leave_chat("-100999")

        assert transport.last_method == "leaveChat"
        assert transport.last_body == {"chat_id": "-100999"}


class TestSends:
    async def test_send_message_sends_the_text_as_it_stands(self) -> None:
        transport = _Recorder(result={"message_id": 55})

        result = await _api(transport).send_message("12345", "**not** rendered")

        assert result.success is True
        assert result.provider_message_id == "55"
        assert transport.last_method == "sendMessage"
        assert transport.last_body == {"chat_id": "12345", "text": "**not** rendered"}

    async def test_send_force_reply_returns_the_prompts_id(self) -> None:
        """That id is what a later reply is matched against."""
        transport = _Recorder(result={"message_id": 88})

        result = await _api(transport).send_force_reply("12345", "Why?")

        assert result.provider_message_id == "88"
        assert transport.last_body["reply_markup"] == {"force_reply": True}

    async def test_send_chat_action_defaults_to_typing(self) -> None:
        transport = _Recorder(result=True)

        await _api(transport).send_chat_action("12345")

        assert transport.last_method == "sendChatAction"
        assert transport.last_body == {"chat_id": "12345", "action": "typing"}

    async def test_send_chat_action_takes_another_action(self) -> None:
        transport = _Recorder(result=True)

        await _api(transport).send_chat_action("12345", "upload_voice")

        assert transport.last_body["action"] == "upload_voice"


class TestActingOnASentMessage:
    async def test_answer_callback_query_without_a_toast(self) -> None:
        transport = _Recorder(result=True)

        await _api(transport).answer_callback_query("cq-1")

        assert transport.last_method == "answerCallbackQuery"
        assert transport.last_body == {"callback_query_id": "cq-1"}

    async def test_answer_callback_query_with_a_toast(self) -> None:
        transport = _Recorder(result=True)

        await _api(transport).answer_callback_query("cq-1", "Approved.")

        assert transport.last_body == {"callback_query_id": "cq-1", "text": "Approved."}

    async def test_edit_message_text_leaves_the_keyboard_alone_by_default(self) -> None:
        transport = _Recorder(result={"message_id": 12})

        await _api(transport).edit_message_text("12345", 12, "Approved")

        assert transport.last_method == "editMessageText"
        assert transport.last_body == {"chat_id": "12345", "message_id": 12, "text": "Approved"}

    async def test_edit_message_text_can_drop_the_buttons(self) -> None:
        transport = _Recorder(result={"message_id": 12})

        await _api(transport).edit_message_text(
            "12345", 12, "Approved", reply_markup={"inline_keyboard": []}
        )

        assert transport.last_body["reply_markup"] == {"inline_keyboard": []}

    async def test_edit_message_reply_markup_sends_no_text(self) -> None:
        transport = _Recorder(result={"message_id": 12})
        keyboard = {"inline_keyboard": [[{"text": "☑ One", "callback_data": "m:1"}]]}

        await _api(transport).edit_message_reply_markup("12345", 12, keyboard)

        assert transport.last_method == "editMessageReplyMarkup"
        assert transport.last_body == {
            "chat_id": "12345",
            "message_id": 12,
            "reply_markup": keyboard,
        }


class TestErrorShape:
    """One shape, whichever call produced it."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda api: api.get_me(),
            lambda api: api.set_webhook("https://example.test/hook"),
            lambda api: api.leave_chat("-1"),
            lambda api: api.send_message("1", "hi"),
            lambda api: api.answer_callback_query("cq-1"),
            lambda api: api.edit_message_text("1", 2, "x"),
        ],
    )
    async def test_every_call_reports_a_refusal_the_same_way(self, call: Any) -> None:
        result = await call(_api(_Refusing(403, "Forbidden: bot was blocked by the user")))

        assert result.success is False
        assert result.error == "telegram_403"
        assert result.metadata["description"] == "Forbidden: bot was blocked by the user"

    async def test_a_refusal_without_a_bot_api_body_falls_back_to_the_status(self) -> None:
        class _Html(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(502, content=b"<html>gateway</html>", request=request)

        result = await _api(_Html()).get_me()

        assert result.success is False
        assert result.error == "http_502"

    async def test_a_success_that_is_not_json_does_not_escape_as_an_exception(self) -> None:
        """An intercepting proxy's error page passes raise_for_status intact.

        Every call promises a ProviderResult; a JSONDecodeError raised out of
        the parse would break that promise for the whole surface at once.
        """

        class _Intercepted(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, content=b"<html>captive portal</html>", request=request)

        result = await _api(_Intercepted()).get_me()

        assert result.success is False
        assert result.error == "invalid_response"

    async def test_valid_json_with_the_wrong_shape_is_invalid(self) -> None:
        result = await _api(_WrongJsonShape()).get_me()

        assert result.success is False
        assert result.error == "invalid_response"

    async def test_success_without_a_result_is_invalid(self) -> None:
        result = await _api(
            httpx.MockTransport(
                lambda request: httpx.Response(200, json={"ok": True}, request=request)
            )
        ).get_me()

        assert result.success is False
        assert result.error == "invalid_response"

    @pytest.mark.parametrize(
        ("call", "wrong_result"),
        [
            (lambda api: api.get_me(), {"id": 42, "is_bot": True}),
            (lambda api: api.get_updates(), {"update_id": 1}),
            (lambda api: api.set_webhook("https://example.test/hook"), {"message_id": 1}),
            (lambda api: api.send_message("1", "hi"), True),
        ],
    )
    async def test_success_result_must_match_the_operation(
        self, call: Any, wrong_result: Any
    ) -> None:
        result = await call(_api(_Recorder(result=wrong_result)))

        assert result.success is False
        assert result.error == "invalid_response"

    async def test_updates_require_integer_update_ids(self) -> None:
        result = await _api(_Recorder(result=[{"update_id": True}])).get_updates()

        assert result.success is False
        assert result.error == "invalid_response"

    async def test_a_200_bot_api_refusal_is_still_a_failure(self) -> None:
        result = await _api(
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"ok": False, "error_code": 429, "description": "Too Many Requests"},
                    request=request,
                )
            )
        ).get_me()

        assert result.success is False
        assert result.error == "telegram_429"
        assert result.metadata["description"] == "Too Many Requests"

    async def test_a_transport_error_never_exposes_the_token_url(self) -> None:
        result = await _api(_ConnectionFailure()).get_me()

        assert result.success is False
        assert result.error == "ConnectError"
        assert "123456:ABC-DEF" not in result.error
