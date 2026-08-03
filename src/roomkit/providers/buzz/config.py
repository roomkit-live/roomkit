"""Buzz (Nostr relay) provider configuration."""

from __future__ import annotations

import os
import string
from typing import Any

from pydantic import BaseModel, SecretStr, field_validator

_HEX_LOWER = set(string.digits + "abcdef")


class BuzzConfig(BaseModel):
    """Buzz relay agent configuration.

    ``private_key`` is the agent's Nostr secret (``nsec…`` or hex); it signs the
    agent's events and authenticates it to the relay (NIP-42/98).
    """

    relay_url: str
    private_key: SecretStr
    # Drop the agent's own events so its outbound messages don't echo back in
    # through the inbound stream.
    ignore_own: bool = True
    # Self-join the channel (NIP-29 kind 9000, role=bot) on connect, so the
    # agent's messages reach other channel members and it resolves in mention
    # autocomplete. Requires buzzkit>=0.1.1.
    auto_join: bool = True
    # Announce presence (kind 20001 "online") on connect + periodic heartbeat,
    # so the agent shows as online while it runs; a deliberate stop() publishes
    # "offline" so the agent's dot flips immediately instead of waiting out the
    # relay-side presence TTL. Requires buzzkit>=0.1.2.
    announce_presence: bool = True
    # Optional NIP-OA owner-attestation tag JSON (``["auth", <owner>, …]``) — makes
    # the relay record the agent's owner. Generate it with the owner's key via
    # ``buzzkit.compute_auth_tag``. Requires buzzkit>=0.1.2.
    auth_tag: str | None = None
    # Leave the channel (NIP-29 kind 9022) when the source stops. Off by
    # default: on a private channel the membership was granted by an admin and
    # self-join cannot get it back, so leaving on every shutdown would lock the
    # agent out. Enable only for open channels where auto_join can re-enter.
    # Requires buzzkit>=0.2.0.
    leave_on_stop: bool = False
    # The agent's owner as a 64-hex Nostr pubkey — gates the relay's owner
    # control commands (see obey_owner_commands). Normally left None: the
    # owner is derived from the VERIFIED auth_tag (which wins when both are
    # set). Set it only for an agent without an attestation, mirroring
    # buzz-acp's --agent-owner fallback. Requires buzzkit>=0.3.0.
    owner_pubkey: str | None = None
    # Honor Buzz's owner control commands: a kind-9 message whose trimmed
    # content is exactly "!shutdown" (or "!cancel"/"!rotate"), mentioning the
    # agent and authored by the PROVEN owner, is consumed — never routed to
    # the pipeline, so the AI cannot answer its own stop command — and
    # "!shutdown" stops the source gracefully (presence "offline", socket
    # closed) unless an on_owner_command callback takes over. Inert while no
    # owner is provable (no verified auth_tag and no owner_pubkey): commands
    # then remain regular messages, as does any command from a non-owner.
    # Requires buzzkit>=0.3.0.
    obey_owner_commands: bool = True

    @classmethod
    def from_env(cls, **overrides: Any) -> BuzzConfig:
        """Build a config from Buzz's reserved identity environment variables.

        Reads ``BUZZ_PRIVATE_KEY`` (or its alias ``NOSTR_PRIVATE_KEY``),
        ``BUZZ_RELAY_URL``, and the optional ``BUZZ_AUTH_TAG`` — the exact
        env triplet every Buzz launcher hands its agents, so a RoomKit agent
        is launchable by the same bash script, systemd unit, or container
        entrypoint as any other Buzz agent. Identity is fail-closed: a
        missing or empty key or relay URL raises instead of building an
        identityless agent. ``overrides`` are passed through to the model
        (e.g. ``owner_pubkey=...``, ``leave_on_stop=True``).
        """
        private_key = os.environ.get("BUZZ_PRIVATE_KEY") or os.environ.get("NOSTR_PRIVATE_KEY")
        if not private_key:
            raise ValueError(
                "BUZZ_PRIVATE_KEY (or NOSTR_PRIVATE_KEY) is not set — "
                "refusing to build an identityless agent"
            )
        relay_url = os.environ.get("BUZZ_RELAY_URL")
        if not relay_url:
            raise ValueError("BUZZ_RELAY_URL is not set")
        auth_tag = os.environ.get("BUZZ_AUTH_TAG") or None
        return cls(relay_url=relay_url, private_key=private_key, auth_tag=auth_tag, **overrides)

    @field_validator("owner_pubkey")
    @classmethod
    def _validate_owner_pubkey(cls, value: str | None) -> str | None:
        """Normalize to lowercase hex; refuse anything that is not a pubkey.

        Fail-closed: a malformed owner would silently disarm (or worse,
        mis-arm) the owner-command gate, so it is rejected at construction.
        """
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or not set(normalized) <= _HEX_LOWER:
            raise ValueError("owner_pubkey must be a 64-character hex Nostr pubkey")
        return normalized
