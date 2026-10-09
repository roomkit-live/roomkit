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

_SCHEMES = ("whatsapp:", "tel:", "sms:", "mms:", "rcs:")
_SEPARATORS = re.compile(r"[\s\-.()]")
_PHONE = re.compile(r"\+?\d{3,15}")
# A national number shorter than this is a short code (an SMS service number),
# not a subscriber's: it is left as it is rather than given a country code.
_MIN_NATIONAL_DIGITS = 7
# With a country code, digits that start with it and are longer than this are
# the international form without its "+" ("15145550100" for code 1).
_MAX_NATIONAL_DIGITS = 10


def normalize_phone_number(address: str, default_country_code: str | None = None) -> str:
    """The E.164 form of *address*, or *address* unchanged when it is not a phone number.

    - A scheme (``whatsapp:``, ``tel:``, ``sms:``, ``mms:``, ``rcs:``) and the
      separators people type (spaces, dashes, dots, parentheses) are dropped.
    - ``+15550000001`` and ``0015550000001`` are already international.
    - Digits without ``+``: with *default_country_code* (``"1"``, ``"+44"``),
      a number that starts with it and is longer than a national number is
      international, any other is national (its leading ``0`` trunk prefix
      dropped) and takes the code. Without it they are left as they are:
      guessing would merge people, since ``13800138000`` is a national
      number in China and ``+13800138000`` one in North America. A provider
      whose senders come international without ``+`` adds it in its parser.
    - A short code, an email address, a WhatsApp JID or any other id is
      returned unchanged.
    """
    if not address:
        return address
    value = address.strip()
    lowered = value.lower()
    for scheme in _SCHEMES:
        if lowered.startswith(scheme):
            value = value[len(scheme) :]
            break
    compact = _SEPARATORS.sub("", value)
    if not _PHONE.fullmatch(compact):
        return address
    if compact.startswith("+"):
        return compact
    if compact.startswith("00"):
        return "+" + compact[2:]
    if default_country_code:
        code = default_country_code.lstrip("+")
        if compact.startswith(code) and len(compact) > _MAX_NATIONAL_DIGITS:
            return "+" + compact
        national = compact.lstrip("0")
        if len(national) < _MIN_NATIONAL_DIGITS:
            return address
        return f"+{code}{national}"
    return address
