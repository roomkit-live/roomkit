"""Channels for tests and examples: a transport that records what it
delivers, and an agent that answers each event it is asked."""

from __future__ import annotations

from roomkit.testing.channels import Delivery, EchoAgent, RecordingTransport

__all__ = ["Delivery", "EchoAgent", "RecordingTransport"]
