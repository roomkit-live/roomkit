"""An idempotency key and an id as the models take them (RFC §13.4, §5)."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, Field


def _no_blank_key(key: str | None) -> str | None:
    """*key*, or None when it is blank: a blank key names no message, so the
    pipeline looks none up and no store indexes one (RFC §13.4)."""
    if key is not None and not key.strip():
        return None
    return key


IdempotencyKey = Annotated[str | None, AfterValidator(_no_blank_key)]
"""An optional idempotency key, a blank one read as none."""


Id = Annotated[str, Field(min_length=1)]
"""A room's, a channel's, an event's, a participant's or an identity's id:
never empty, as roomkit-rs's ids are (RMK-1021 item 29)."""
