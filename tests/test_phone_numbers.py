"""Phone numbers in one form, E.164, whatever spelling the provider or host used."""

from __future__ import annotations

import pytest

from roomkit.channels._phone import normalize_phone_number


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
