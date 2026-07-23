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
