"""Email addresses in one form, whatever case or wrapping the sender used.

A room is found by comparing the sender's address with the one its binding
names (RFC §10.4). ``Alice@Example.com`` and ``alice@example.com`` reach the
same mailbox at every mainstream provider, and a mail client writes the
sender as ``Alice Martin <alice@example.com>``: compared as strings, one
person opens several rooms. The email channel therefore reads addresses
through :func:`normalize_email_address`.
"""

from __future__ import annotations

import re

__all__ = ["normalize_email_address"]

# One address: something before and after a single "@", no spaces.
_ADDRESS = re.compile(r"[^\s@<>]+@[^\s@<>]+")
# The address of a "Display Name <address>" form.
_ANGLE = re.compile(r"<([^<>]+)>\s*$")


def normalize_email_address(address: str) -> str:
    """The lower-case address of *address*, or *address* unchanged when it is not one.

    ``Alice Martin <Alice@Example.com>`` and ``mailto:alice@example.com`` are
    ``alice@example.com``. The whole address is lower-cased: RFC 5321 lets a
    server treat the local part's case as significant, but no mainstream
    provider does, and a sender writing their address in another case is the
    common case, not two people.
    """
    if not address:
        return address
    value = address.strip()
    angle = _ANGLE.search(value)
    if angle:
        value = angle.group(1).strip()
    if value.lower().startswith("mailto:"):
        value = value[len("mailto:") :]
    if not _ADDRESS.fullmatch(value):
        return address
    return value.lower()
