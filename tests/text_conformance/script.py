"""What a scenario asks a provider to answer, in no vendor's words, and how a
driver reports what a request replayed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Finish = Literal["stop", "tool", "cut", "context", "filtered", "malformed", "unexpected", "none"]
"""How the response ended: on its own, on tool calls, cut by the output cap,
cut by the context window filling up, stopped by a content filter or a
refusal, on a call the vendor could not parse, on a call to a tool the request
did not enable, or a stream that stopped with no stop reason."""


@dataclass(frozen=True)
class Call:
    """A tool call the model writes, its arguments as the text it streams."""

    name: str
    arguments: str
    id: str | None = None
    index: int | None = None
    fragments: int = 1
    """How many pieces the arguments stream in."""
    as_object: bool = False
    """The server sends the arguments as the JSON object they spell, in one
    piece, not as text (a wire whose arguments are always objects ignores it)."""
    functionless: bool = False
    """The entry carries no function at all (a custom tool's call, or one a
    server's tool parser lost): nothing the loop can run."""


@dataclass(frozen=True)
class Reasoning:
    """A block of reasoning: text with its signature, or a redacted block."""

    text: str = ""
    signature: str | None = None
    redacted: str | None = None


@dataclass(frozen=True)
class Usage:
    input: int = 11
    output: int = 7
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0


@dataclass(frozen=True)
class Script:
    """One generation, streamed or not."""

    text: str = ""
    calls: tuple[Call, ...] = ()
    reasoning: tuple[Reasoning, ...] = ()
    finish: Finish = "stop"
    usage: Usage = field(default_factory=Usage)
    calls_in_one_chunk: bool = False
    """Stream every call's first fragment in the same chunk."""
    answered_by: str = "served-model"
    """The model the response names as the one that answered: no provider's
    configured model, so a reader that reports the asked one shows."""
    usage_alone: bool = False
    """Stream the usage on a chunk of its own, with no choice, after the stop."""


# What a request replayed of a round, in a driver's normalized words:
#   ("thinking", text, signature) — a signed reasoning block
#   ("redacted", data)            — a redacted reasoning block
#   ("inline", text)              — reasoning sent inline in the text (<think>)
#   ("field", text)               — reasoning sent in a dedicated field
#   ("text", text)                — what the assistant said
#   ("call", ref, arguments)      — a call; ref is its id, or its name where the
#                                   wire pairs results by name
#   ("signature", ref, signature) — the signature a call goes back with
#   ("result", ref, text, error)  — a tool result; error is the wire's error flag,
#                                   None where the wire has none
Item = tuple[Any, ...]
