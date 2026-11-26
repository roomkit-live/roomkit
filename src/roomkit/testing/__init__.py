"""Channels for tests and examples: a transport that records what it
delivers, and an agent that answers each event it is asked. And the contract
of a turn backend's two ports, as checks a service's own tests call."""

from __future__ import annotations

from roomkit.testing.channels import Delivery, EchoAgent, RecordingTransport
from roomkit.testing.turn_ports import check_turn_broker, check_turn_registry

__all__ = [
    "Delivery",
    "EchoAgent",
    "RecordingTransport",
    "check_turn_broker",
    "check_turn_registry",
]
