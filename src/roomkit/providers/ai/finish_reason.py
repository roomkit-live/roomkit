"""Why a generation stopped, in the one vocabulary of the provider contract.

Each vendor reports how a response ended in its own words. A provider maps
them onto this closed set in its adapter, so the core reads one vocabulary and
knows no vendor's word (RFC §6.7).
"""

from __future__ import annotations

from enum import StrEnum


class FinishReason(StrEnum):
    """Why a generation stopped (RFC §6.7).

    A provider maps its vendor's words onto these, and ends a generation on
    ``None`` when its vendor gave no reason or one the provider does not
    know.
    """

    STOP = "stop"
    """The model ended its answer itself."""
    TOOL_CALLS = "tool_calls"
    """The model stopped to call tools."""
    LENGTH = "length"
    """The response ran out of room: its output cap, or its context window
    filling up mid-answer."""
    CONTENT_FILTER = "content_filter"
    """A filter or the provider's safeguards withheld or stopped the answer."""
    REFUSAL = "refusal"
    """The model refused to answer."""
    MALFORMED_CALL = "malformed_call"
    """The model called a tool, and the provider could not parse the call, so
    it handed none over."""
    UNEXPECTED_CALL = "unexpected_call"
    """The model called a tool the request did not enable, and the provider
    handed no call over."""
    ERROR = "error"
    """The generation failed before its end."""
