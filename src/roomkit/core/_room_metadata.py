"""The keys of a room's metadata only the framework writes (RFC §5.1, §14.1).

Three kinds:

- the implementation's namespace, ``roomkit:`` (RFC §5.1): what the
  framework keeps for the room, a detach's revocation
  (``roomkit:detached_channels``, §7.5 rule 7) and what the room keeps for a
  channel (``roomkit:channel_state:<id>``, §10.6.2);
- a key of the room's private state (RFC §14.1,
  :mod:`roomkit.core._private_state`): the register of authors, a
  discussion's results to post;
- what orchestration keeps for the room, written through the store: its
  conversation state (RFC §19.2), a discussion's configuration and speak queue
  (RFC §19.7.5 rule 16).

Every writer the kit hands a room's metadata from outside has them dropped
with a warning, the rest written (:func:`without_framework_keys`): the kit's
``create_room()`` and ``update_room_metadata()``, called from a hook's body
too, a turn's ``TurnHost.update_room_metadata``, which the owner serves
through the latter, and a channel's ``metadata_updates`` (§10.1 step 15). So a
read-modify-write of a room's metadata (``{**room.metadata, ...}``) writes,
and the forged key never lands. The store's ``update_room()`` and
``patch_room_metadata()`` stay open: they sit below the kit's API, where
migrations and the runtime write these keys (a detach's revocation, a
channel's state, the base store's private state), and no channel output or
hook result reaches them.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from roomkit.core._private_state import PRIVATE_STATE_KEYS

logger = logging.getLogger("roomkit.room_metadata")

FRAMEWORK_NAMESPACE = "roomkit:"
"""The prefix of the room-metadata keys the framework keeps for the room."""

DISCUSSION_CONFIG = "_discussion"
"""The room metadata key a discussion's configuration is stored under
(RFC §19.7.5 rule 16)."""

SPEAK_QUEUE = "_speak_queue"
"""The room metadata key a discussion's speak queue is stored under
(RFC §19.7.5 rule 16)."""

CONVERSATION_STATE = "_conversation_state"
"""The room metadata key a room's conversation state is stored under (RFC §19.2):
``save_conversation_state`` writes it, a room's metadata being no way to (§19.2)."""

_ORCHESTRATION_KEYS = frozenset({CONVERSATION_STATE, DISCUSSION_CONFIG, SPEAK_QUEUE})


def the_framework_s(key: str) -> bool:
    """Whether only the framework writes *key* of a room's metadata."""
    return (
        key.startswith(FRAMEWORK_NAMESPACE)
        or key in PRIVATE_STATE_KEYS
        or key in _ORCHESTRATION_KEYS
    )


def without_framework_keys(
    metadata: Mapping[str, Any], *, room_id: str | None, writer: str
) -> dict[str, Any]:
    """*metadata*, a write of room *room_id*'s metadata by *writer*, without
    the keys only the framework writes, each dropped with a warning naming
    the writer; the rest is written (RFC §5.1, §14.1, §10.1 step 15)."""
    kept: dict[str, Any] = {}
    for key, value in metadata.items():
        if the_framework_s(key):
            logger.warning(
                "%s drops the metadata key %r of room %s: only the framework writes it",
                writer,
                key,
                room_id or "being created",
                extra={"room_id": room_id},
            )
            continue
        kept[key] = value
    return kept
