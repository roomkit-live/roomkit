"""Buzz (Nostr relay) provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


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
