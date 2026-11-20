"""A channel type's addressing (README, "Calls": ``addressing_of.normalize``;
RFC §5.7).

``addressing_of.normalize`` writes an address with the addressing the
implementation gives a channel of a type by default, a country code the
implementation refuses being the call's ``invalid_argument``. A scripted
channel's own ``addressing`` is built in ``_script.addressed_transport``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit, addressing_of
from roomkit.models.enums import ChannelType
from tests.conformance._calls import Call, read

if TYPE_CHECKING:
    from tests.conformance._harness import Harness


async def _normalize(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> str:
    channel_type = read(lambda: ChannelType(spec["channel_type"]))
    return addressing_of(channel_type).normalize(spec["address"], spec.get("country_code"))


ADDRESSING_CALLS: dict[str, Call] = {"addressing_of.normalize": _normalize}

ADDRESSING_ARGUMENTS: dict[str, set[str]] = {
    "addressing_of.normalize": {"channel_type", "address", "country_code"},
}
