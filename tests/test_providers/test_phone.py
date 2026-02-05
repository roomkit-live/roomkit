"""Tests for phone number normalization utilities."""

from __future__ import annotations

import importlib.util

import pytest

from roomkit.providers.sms.phone import is_valid_phone, normalize_phone

# Check if phonenumbers is available
HAS_PHONENUMBERS = importlib.util.find_spec("phonenumbers") is not None

pytestmark = pytest.mark.skipif(
    not HAS_PHONENUMBERS,
    reason="phonenumbers library not installed",
)


class TestNormalizePhone:
    def test_already_e164(self) -> None:
        assert normalize_phone("+14185551234") == "+14185551234"

    def test_with_country_code_no_plus(self) -> None:
        assert normalize_phone("14185551234") == "+14185551234"

    def test_local_format_us(self) -> None:
        assert normalize_phone("418-555-1234", "US") == "+14185551234"

    def test_local_format_ca(self) -> None:
        assert normalize_phone("418-555-1234", "CA") == "+14185551234"

    def test_with_parentheses(self) -> None:
        assert normalize_phone("+1 (418) 555-1234") == "+14185551234"

    def test_with_spaces(self) -> None:
        assert normalize_phone("+1 418 555 1234") == "+14185551234"

    def test_french_number(self) -> None:
        assert normalize_phone("+33 6 12 34 56 78") == "+33612345678"

    def test_uk_number(self) -> None:
        assert normalize_phone("+44 7911 123456") == "+447911123456"

    def test_invalid_number(self) -> None:
        with pytest.raises(ValueError, match="Invalid phone number"):
            normalize_phone("123", "US")

    def test_unparseable_number(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse phone number"):
            normalize_phone("not-a-number", "US")


class TestIsValidPhone:
    def test_valid_e164(self) -> None:
        assert is_valid_phone("+14185551234") is True

    def test_valid_local(self) -> None:
        assert is_valid_phone("418-555-1234", "CA") is True

    def test_invalid(self) -> None:
        assert is_valid_phone("123") is False

    def test_not_a_number(self) -> None:
        assert is_valid_phone("hello") is False
