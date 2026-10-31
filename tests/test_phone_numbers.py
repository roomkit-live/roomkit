"""Phone numbers in one form, E.164, whatever spelling the provider or host used."""

from __future__ import annotations

import pytest

from roomkit.channels._phone import normalize_phone_number
from roomkit.channels.addressing import addressing_of
from roomkit.channels.transport import TransportChannel
from roomkit.models.enums import ChannelType


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("+15550000001", "+15550000001"),
        ("whatsapp:+15550000001", "+15550000001"),
        ("tel:+15550000001", "+15550000001"),
        ("0015550000001", "+15550000001"),
        ("+1 (555) 000-0001", "+15550000001"),
        ("+33 6 12 34 56 78", "+33612345678"),
    ],
)
def test_an_international_number_whatever_its_spelling(address: str, expected: str) -> None:
    assert normalize_phone_number(address) == expected


@pytest.mark.parametrize(
    ("address", "country_code", "expected"),
    [
        ("(555) 000-0001", "1", "+15550000001"),
        ("555-000-0001", "+1", "+15550000001"),
        ("15550000001", "1", "+15550000001"),
        ("06 12 34 56 78", "33", "+33612345678"),
        ("33612345678", "33", "+33612345678"),
    ],
)
def test_a_national_number_takes_the_default_country_code(
    address: str, country_code: str, expected: str
) -> None:
    assert normalize_phone_number(address, country_code) == expected


@pytest.mark.parametrize("address", ["5550000001", "15550000001", "13800138000"])
def test_digits_without_a_country_code_are_left_as_they_are(address: str) -> None:
    """No guess: 13800138000 is a national number in China, +13800138000 one in
    North America; reading one as the other would merge two people."""
    assert normalize_phone_number(address) == address


@pytest.mark.parametrize(
    "address",
    ["", "12345", "a@x.test", "15550000001@s.whatsapp.net", "system", "user-42"],
    ids=["empty", "short-code", "email", "whatsapp-jid", "system", "id"],
)
def test_anything_else_is_returned_unchanged(address: str) -> None:
    assert normalize_phone_number(address, "1") == address


@pytest.mark.parametrize(
    ("address", "country_code", "expected"),
    [
        ("+33 (0)6 12 34 56 78", None, "+33612345678"),
        ("0033 (0)6 12 34 56 78", None, "+33612345678"),
        ("tel:+33 (0)6 12 34 56 78", "33", "+33612345678"),
        ("callto:+15550000001", None, "+15550000001"),
        ("001234567890123", None, "+1234567890123"),
        ("00123456789012345", None, "+123456789012345"),
        ("0123456", "33", "+33123456"),
    ],
    ids=[
        "trunk-mark",
        "trunk-mark-after-00",
        "trunk-mark-after-a-scheme",
        "any-uri-scheme",
        "00-thirteen-digits",
        "00-fifteen-digits",
        "seven-digits-are-no-short-code",
    ],
)
def test_the_rfc_rules_write_a_number(
    address: str, country_code: str | None, expected: str
) -> None:
    """RFC §5.7: a scheme, the separators and the trunk prefix written ``(0)``
    after a country code are dropped; ``+`` or ``00`` is international; six
    digits or fewer are a short code."""
    assert normalize_phone_number(address, country_code) == expected


@pytest.mark.parametrize(
    ("address", "country_code"),
    [
        ("+06 12 34 56 78", None),
        ("000612345678", None),
        ("+1234567890123456", None),
        ("0012345678901234567", None),
        ("612345678901234", "33"),
        ("123456", "33"),
    ],
    ids=[
        "plus-then-zero",
        "00-then-zero",
        "sixteen-digits",
        "00-then-seventeen-digits",
        "code-makes-seventeen-digits",
        "six-digit-short-code",
    ],
)
def test_what_is_no_e164_number_is_kept_as_written(address: str, country_code: str | None) -> None:
    """RFC §5.7: a result of more than 15 digits, or one whose first digit is
    ``0``, which no country code is, is kept as written, and so is a short
    code."""
    assert normalize_phone_number(address, country_code) == address


@pytest.mark.parametrize("code", ["0x", "0", "033", "1234", "", "+", "+0", "x1", "١"])
def test_a_country_code_that_is_none_is_refused(code: str) -> None:
    """A country code is one to three digits, the first not 0, with or
    without its ``+``, as roomkit-rs's ``CountryCode`` (RMK-1021 item 4): a
    channel given another, or a normalization asked with one, raises. The
    number was kept unchanged, the mistake unseen."""
    with pytest.raises(ValueError, match="is not a country code"):
        addressing_of(ChannelType.SMS).normalize("06 12 34 56 78", code)
    with pytest.raises(ValueError, match="is not a country code"):
        TransportChannel("sms", ChannelType.SMS, provider=None, country_code=code)


@pytest.mark.parametrize("code", ["1", "+33", "44", "+1", "999"])
def test_a_country_code_with_or_without_its_plus_is_taken(code: str) -> None:
    number = addressing_of(ChannelType.SMS).normalize("514 555 0100", code)
    assert number.startswith("+" + code.removeprefix("+"))
