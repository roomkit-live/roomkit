"""A source's URL must not carry its token into logs and events (CWE-532).

``name`` is documented as being for logging and framework events, and it
reached both verbatim. Authenticating a WebSocket or SSE endpoint with a token
in the query string is ordinary — several providers document no other way — so
the URL as written is a credential.
"""

from __future__ import annotations

from roomkit.sources.sse import SSESource
from roomkit.sources.websocket import WebSocketSource
from roomkit.telemetry.redaction import safe_url


class TestSafeUrl:
    def test_query_string_is_dropped_and_marked(self) -> None:
        out = safe_url("wss://api.example.com/v1/listen?token=SECRET&model=x")
        assert "SECRET" not in out
        assert out == "wss://api.example.com/v1/listen?<redacted>"

    def test_no_query_means_no_marker(self) -> None:
        """A reader can tell 'no parameters' from 'parameters removed'."""
        assert safe_url("wss://example.com/stream") == "wss://example.com/stream"

    def test_userinfo_is_dropped(self) -> None:
        out = safe_url("wss://alice:hunter2@example.com:8443/socket")
        assert "hunter2" not in out
        assert out == "wss://<redacted>@example.com:8443/socket"

    def test_the_path_survives(self) -> None:
        """What makes a log line diagnosable is kept."""
        assert "/v1/listen" in safe_url("wss://h/v1/listen?k=v")

    def test_a_non_url_is_returned_unchanged(self) -> None:
        assert safe_url("not a url") == "not a url"

    def test_redaction_is_not_gated_on_content_logging(self) -> None:
        """A credential is not content — no debug flag should reveal it."""
        from roomkit.telemetry.redaction import set_content_logging

        set_content_logging(True)
        try:
            assert "SECRET" not in safe_url("wss://h/p?token=SECRET")
        finally:
            set_content_logging(False)


class TestSourceNamesAreSafe:
    def test_websocket_source_name_hides_the_token(self) -> None:
        src = WebSocketSource(channel_id="ws", url="wss://api.example.com/v1?token=SECRET")
        assert "SECRET" not in src.name
        assert src.name.startswith("websocket:wss://api.example.com/v1")

    def test_sse_source_name_hides_the_token(self) -> None:
        src = SSESource(channel_id="sse", url="https://api.example.com/events?apikey=SECRET")
        assert "SECRET" not in src.name
        assert src.name.startswith("sse:https://api.example.com/events")

    def test_the_connection_still_uses_the_real_url(self) -> None:
        """Only what is reported is redacted, never what is dialled."""
        url = "wss://api.example.com/v1?token=SECRET"
        src = WebSocketSource(channel_id="ws", url=url)
        assert src._url == url
