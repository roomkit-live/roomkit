"""Unit tests for SIP helper functions introduced for v0.7.0a17.

Covers three small but release-relevant helpers that are otherwise
exercised only indirectly through the BYE / disconnect paths:

* ``parse_bye_reason`` — RFC 3326 ``Reason`` header parser.
* ``_format_q850_reason`` — outbound Reason header builder.
* ``SIPAudioMixin._resolve_in_dialog_destination`` — Contact-vs-source
  destination picker for in-dialog requests on inbound calls.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

# Provide a minimal aiosipua stub so the SIP modules can be imported in
# the unit-test environment. ``parse_uri`` is patched per-test where the
# in-dialog destination tests need real parsing behaviour.
if "aiosipua" not in sys.modules:
    _fake = types.ModuleType("aiosipua")
    _fake.build_sdp = lambda **kwargs: None  # type: ignore[attr-defined]
    # ``parse_uri`` is imported unconditionally at the top of
    # ``_resolve_in_dialog_destination``, so a default stub is needed
    # even for tests that exercise the no-remote-target fallback.
    _fake.parse_uri = lambda _s: MagicMock(host="", port=None)  # type: ignore[attr-defined]
    sys.modules["aiosipua"] = _fake

from roomkit.voice.backends._sip_types import parse_bye_reason
from roomkit.voice.backends.sip_audio import SIPAudioMixin, _format_q850_reason

# ---------------------------------------------------------------------------
# parse_bye_reason
# ---------------------------------------------------------------------------


class TestParseByeReason:
    """RFC 3326 ``Reason: Q.850 ;cause=N ;text="…"`` parsing."""

    def test_none_input_returns_none(self) -> None:
        assert parse_bye_reason(None) is None

    def test_non_str_non_bytes_returns_none(self) -> None:
        # Defensive: the wire-bytes path is the documented contract, so
        # exotic types must not raise — they just signal "no reason".
        assert parse_bye_reason(12345) is None  # type: ignore[arg-type]

    def test_no_header_returns_none(self) -> None:
        msg = "BYE sip:bot@example.com SIP/2.0\r\nVia: SIP/2.0/UDP\r\n\r\n"
        assert parse_bye_reason(msg) is None

    def test_with_text_uses_carrier_text(self) -> None:
        msg = (
            "BYE sip:bot@example.com SIP/2.0\r\n"
            'Reason: Q.850 ;cause=21 ;text="Tenant declined"\r\n\r\n'
        )
        assert parse_bye_reason(msg) == {"cause": 21, "text": "Tenant declined"}

    def test_no_text_falls_back_to_canonical_label(self) -> None:
        # Carrier omits text="" — we look up the canonical Q.850 label
        # so callers always get something human-readable.
        msg = "Reason: Q.850 ;cause=17\r\n"
        assert parse_bye_reason(msg) == {"cause": 17, "text": "User busy"}

    def test_unknown_cause_uses_generic_label(self) -> None:
        # Cause 99 isn't in the canonical map and the carrier didn't
        # provide text — fall back to the generic "Q.850 cause N" form.
        msg = "Reason: Q.850 ;cause=99\r\n"
        assert parse_bye_reason(msg) == {"cause": 99, "text": "Q.850 cause 99"}

    def test_bytes_input_decoded(self) -> None:
        wire = b'Reason: Q.850 ;cause=16 ;text="Normal call clearing"\r\n'
        assert parse_bye_reason(wire) == {"cause": 16, "text": "Normal call clearing"}

    def test_invalid_utf8_bytes_replaced(self) -> None:
        # ``errors="replace"`` keeps the parser robust against carriers
        # that mix encodings — the regex still runs against the rest of
        # the message.
        wire = b"Reason: Q.850 ;cause=16\r\n\xff\xfe garbage"
        result = parse_bye_reason(wire)
        assert result is not None
        assert result["cause"] == 16

    def test_case_insensitive_header(self) -> None:
        # Carriers normalize header casing inconsistently; the regex
        # uses ``re.IGNORECASE`` so variants must match.
        msg = "reason: q.850 ;cause=16\r\n"
        assert parse_bye_reason(msg) == {"cause": 16, "text": "Normal call clearing"}


# ---------------------------------------------------------------------------
# _format_q850_reason
# ---------------------------------------------------------------------------


class TestFormatQ850Reason:
    """Outbound Reason header builder for ``disconnect(cause=…, text=…)``."""

    def test_none_cause_returns_none(self) -> None:
        assert _format_q850_reason(None, None) is None
        assert _format_q850_reason(None, "ignored") is None

    def test_cause_only(self) -> None:
        assert _format_q850_reason(16, None) == "Q.850 ;cause=16"

    def test_cause_and_text(self) -> None:
        assert _format_q850_reason(21, "Call rejected") == 'Q.850 ;cause=21 ;text="Call rejected"'

    def test_quote_in_text_stripped(self) -> None:
        # Quotes would close the header value early — the changelog
        # promises they're stripped.
        out = _format_q850_reason(21, 'bad "quote" inside')
        assert out == 'Q.850 ;cause=21 ;text="bad quote inside"'

    def test_crlf_in_text_replaced(self) -> None:
        # CR/LF in a header value would inject a new SIP header — must
        # be neutralized.
        out = _format_q850_reason(21, "first line\r\nInjected: evil")
        assert out is not None
        assert "\r" not in out
        assert "\n" not in out
        assert "Injected" in out  # text content preserved, just flattened

    def test_empty_text_drops_text_field(self) -> None:
        # Empty string is falsy → no ``text=""`` appended.
        assert _format_q850_reason(16, "") == "Q.850 ;cause=16"


# ---------------------------------------------------------------------------
# _resolve_in_dialog_destination
# ---------------------------------------------------------------------------


class _Mixin(SIPAudioMixin):
    """Concrete subclass so we can call the mixin method without the full SIP backend wiring."""


class TestResolveInDialogDestination:
    """Picks Contact URI over L3 source — central to the BYE-routing fix."""

    _default_parse_uri = staticmethod(lambda _s: MagicMock(host="", port=None))

    def _patch_parse_uri(self, host: str | None, port: int | None) -> Any:
        """Install a fake ``aiosipua.parse_uri`` returning *(host, port)*."""
        import aiosipua

        original = getattr(aiosipua, "parse_uri", None)
        result = MagicMock()
        result.host = host
        result.port = port
        aiosipua.parse_uri = lambda _s: result  # type: ignore[attr-defined]
        return original

    def _restore_parse_uri(self, original: Any) -> None:
        # Restore the previous ``parse_uri`` if the test captured one;
        # otherwise reinstall the module-level default so later tests in
        # the same process still see a callable. Never ``delattr`` —
        # subsequent tests import the SIP module unconditionally.
        import aiosipua

        aiosipua.parse_uri = original or self._default_parse_uri  # type: ignore[attr-defined]

    def test_uses_contact_when_remote_target_set(self) -> None:
        """Dialog has a Contact URI → BYE goes there, not to source_addr."""
        original = self._patch_parse_uri("203.0.113.5", 5061)
        try:
            mixin = _Mixin()
            call = MagicMock()
            call.dialog.remote_target = "<sip:carrier@203.0.113.5:5061>"
            call.source_addr = ("10.0.0.1", 5060)  # NAT outer — should be ignored

            assert mixin._resolve_in_dialog_destination(call) == ("203.0.113.5", 5061)
        finally:
            self._restore_parse_uri(original)

    def test_defaults_port_to_5060_when_uri_omits_it(self) -> None:
        original = self._patch_parse_uri("203.0.113.5", None)
        try:
            mixin = _Mixin()
            call = MagicMock()
            call.dialog.remote_target = "<sip:carrier@203.0.113.5>"
            call.source_addr = ("10.0.0.1", 5060)

            assert mixin._resolve_in_dialog_destination(call) == ("203.0.113.5", 5060)
        finally:
            self._restore_parse_uri(original)

    def test_falls_back_to_source_when_no_remote_target(self) -> None:
        """Dialog without remote target — defensive fallback."""
        # ``_resolve_in_dialog_destination`` imports ``parse_uri``
        # unconditionally, so the symbol must exist even when this test
        # never reaches the parse branch.
        original = self._patch_parse_uri("", None)
        try:
            mixin = _Mixin()
            call = MagicMock()
            call.dialog.remote_target = ""  # no Contact recorded
            call.source_addr = ("10.0.0.1", 5060)

            assert mixin._resolve_in_dialog_destination(call) == ("10.0.0.1", 5060)
        finally:
            self._restore_parse_uri(original)

    def test_falls_back_to_source_when_parse_fails(self) -> None:
        """Malformed Contact URI must not crash — fall back to L3 source."""
        import aiosipua

        original = getattr(aiosipua, "parse_uri", None)

        def _raise(_s: str) -> Any:
            raise ValueError("bad uri")

        aiosipua.parse_uri = _raise  # type: ignore[attr-defined]
        try:
            mixin = _Mixin()
            call = MagicMock()
            call.dialog.remote_target = "<not a uri>"
            call.source_addr = ("10.0.0.1", 5060)

            assert mixin._resolve_in_dialog_destination(call) == ("10.0.0.1", 5060)
        finally:
            self._restore_parse_uri(original)

    def test_falls_back_to_source_when_uri_has_no_host(self) -> None:
        """Parsed URI with empty host is treated like a parse failure."""
        original = self._patch_parse_uri("", 5060)
        try:
            mixin = _Mixin()
            call = MagicMock()
            call.dialog.remote_target = "<sip:>"
            call.source_addr = ("10.0.0.1", 5060)

            assert mixin._resolve_in_dialog_destination(call) == ("10.0.0.1", 5060)
        finally:
            self._restore_parse_uri(original)
