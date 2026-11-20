"""Phone numbers in one form, E.164, whatever the provider wrote.

Providers hand the same number over in different shapes — Twilio ``+15550000001``,
Twilio WhatsApp ``whatsapp:+15550000001``, Sinch ``15550000001``, a host typing
``(514) 555-0100`` — and a room is found by comparing the sender's number with
the one its binding names (RFC §10.4). Compared as strings, the same person
is a stranger to their own room. Every channel addressed by phone number
therefore reads numbers through :func:`normalize_phone_number`.

It never raises and needs no dependency: it runs on every inbound sender.
Validating a number a person typed is another job, done by
:func:`roomkit.providers.sms.normalize_phone` (the ``phonenumbers`` extra).
"""

from __future__ import annotations

import re

__all__ = ["normalize_phone_number"]

# A URI scheme (``whatsapp:``, ``tel:``, ``sms:``...): letters, then a colon.
_SCHEME = re.compile(r"^[A-Za-z]+:")
_SEPARATORS = re.compile(r"[\s\-.()]")
# The trunk prefix written "(0)" after a country code: "+33 (0)6 12 34 56 78".
_TRUNK_MARK = "(0)"
# A number of this many digits or fewer, written without "+" or "00", is a
# short code (an SMS service number), not a subscriber's: it is left as it is
# rather than given a country code.
_MAX_SHORT_CODE_DIGITS = 6
# With a country code, digits that start with it and are longer than this are
# the international form without its "+" ("15145550100" for code 1).
_MAX_NATIONAL_DIGITS = 10
# The most digits of an E.164 number.
_MAX_E164_DIGITS = 15


def normalize_phone_number(address: str, default_country_code: str | None = None) -> str:
    """The E.164 form of *address*, or *address* unchanged when it is not a
    phone number (RFC §5.7).

    - A URI scheme (``whatsapp:``, ``tel:``, ``sms:``), the separators people
      type (spaces, dashes, dots, parentheses) and the trunk prefix written
      ``(0)`` after a country code are dropped: ``+33 (0)6 12 34 56 78`` is
      ``+33612345678``.
    - ``+15550000001`` and ``0015550000001`` are international.
    - Digits without ``+`` or ``00``: with *default_country_code* (``"1"``,
      ``"+44"``), six digits or fewer are a short code, kept as written; a
      number that starts with the code and is longer than a national number
      already carries it; any other is national, its trunk prefix ``0`` replaced
      by the code, or the code put before it. Without a code they are left as
      they are: guessing would merge people, since ``13800138000`` is a
      national number in China and ``+13800138000`` one in North America. A
      provider whose senders come international without ``+`` adds it in its
      parser.
    - A result of more than 15 digits, or whose first digit is ``0`` (no
      country code is), is no E.164 number: *address* is kept as written, and
      so is an email address, a WhatsApp JID or any other id.
    """
    if not address:
        return address
    dialled = _dialled(_SCHEME.sub("", address.strip(), count=1).lstrip())
    if dialled is None:
        return address
    international, digits = dialled
    if international:
        return _e164(digits) or address
    if not default_country_code or len(digits) <= _MAX_SHORT_CODE_DIGITS:
        return address
    code = default_country_code.lstrip("+")
    if digits.startswith("0"):
        number = code + digits[1:]
    elif digits.startswith(code) and len(digits) > _MAX_NATIONAL_DIGITS:
        number = digits
    else:
        number = code + digits
    return _e164(number) or address


def _dialled(value: str) -> tuple[bool, str] | None:
    """Whether *value* is written international (``+`` or ``00``), and its
    digits; ``None`` when anything but digits and separators is written.
    The separators are dropped first: ``(+44) 20 7946 0958`` is written with
    ``+``."""
    international = _SEPARATORS.sub("", value).startswith(("+", "00"))
    if international:
        value = value.replace(_TRUNK_MARK, "", 1)
    compact = _SEPARATORS.sub("", value)
    if compact.startswith("+"):
        digits = compact[1:]
    elif compact.startswith("00"):
        digits = compact[2:]
    else:
        digits = compact
    if not digits.isascii() or not digits.isdigit():
        return None
    return international, digits


def _e164(digits: str) -> str | None:
    """``+`` and *digits*, or ``None`` when they are no E.164 number: more
    than 15 digits, or a first digit ``0``, which no country code is."""
    if not digits or len(digits) > _MAX_E164_DIGITS or digits.startswith("0"):
        return None
    return f"+{digits}"
