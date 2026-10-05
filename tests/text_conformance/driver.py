"""What a wire family's driver gives the conformance suite."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

import pytest

from roomkit.providers.ai.base import AIProvider
from tests.text_conformance.script import Item, Script

# What a wire may be unable to express; a driver names each one it cannot, with
# the reason, and a scenario needing it is skipped with that reason.
ARGUMENT_TEXT = "argument_text"
"""Arguments arrive as text the model wrote, so they can be invalid JSON or cut."""
WRITTEN_UNREADABLE = "written_unreadable"
"""The model can write arguments that are not a JSON object on an ordinary stop."""
CALL_INDEX = "call_index"
"""Streamed call fragments carry an index, which may be absent."""
COMPOSITION = "composition"
"""The stream announces a call's arguments as they are composed."""
CALLS_IN_ONE_CHUNK = "calls_in_one_chunk"
"""Several calls can start in the same stream chunk."""
REPEATED_ID = "repeated_id"
"""Two calls can carry the same server id and still be two calls."""
RESPONSE_CALL_WITHOUT_ID = "response_call_without_id"
"""A response that is not streamed can carry a call with no id."""
STREAM_WITHOUT_FINISH = "stream_without_finish"
"""A stream can stop mid-call with no stop reason."""
SIGNED_REASONING = "signed_reasoning"
"""Reasoning comes in blocks, each signed."""
REDACTED_REASONING = "redacted_reasoning"
"""A reasoning block can come redacted, as opaque data."""
CACHE_USAGE = "cache_usage"
"""Usage reports cache reads."""
CACHE_WRITE_USAGE = "cache_write_usage"
"""Usage reports cache writes."""
REASONING_USAGE = "reasoning_usage"
"""Usage reports reasoning tokens."""
STREAM_USAGE = "stream_usage"
"""A stream reports usage."""
SCHEMA_AS_GIVEN = "schema_as_given"
"""A tool's parameter schema is declared as the tool gave it."""
IMAGE_RESULTS = "image_results"
"""A tool result can carry an image."""
MALFORMED_CALL = "malformed_call"
"""The response can end on a call the vendor would not hand over: one it could
not parse, or one to a tool the request did not enable."""
FILTER_STOP = "filter_stop"
"""A content filter or a refusal can stop the response mid-answer."""
OBJECT_ARGUMENTS_RESPONSE = "object_arguments_response"
"""A response that is not streamed can carry a call's arguments as an object."""
THINK_TAGS = "think_tags"
"""Reasoning can come inline in the answer's text, as ``<think>`` tags."""
FUNCTIONLESS_CALL = "functionless_call"
"""A response can carry a tool-call entry with no function at all."""
USAGE_ALONE = "usage_alone"
"""A stream can report its usage on a chunk of its own, with no choice."""
SERVER_ID = "server_id"
"""A call carries the id its server gave it."""

ReasoningConvention = Literal["blocks", "call_signature", "inline", "field", "dropped"]
"""How a wire replays earlier reasoning: as signed blocks, as one signature on
each call of the round (Gemini), inline in the text, in a field of its own,
or not at all."""


class Driver(ABC):
    """One wire family: a provider answering a script, and its requests read back.

    ``provider()`` returns a provider whose next generation, streamed or not,
    answers the script with the vendor SDK's own objects; every request it
    sends lands in ``requests``.
    """

    label: str
    covers: tuple[type[AIProvider], ...] = ()
    """The provider classes this driver stands for."""
    cannot: Mapping[str, str] = {}
    reasoning: ReasoningConvention = "inline"
    """How the wire replays a model's earlier reasoning."""
    error_flag: bool = False
    """The wire flags a failed tool result."""
    calls_by_name: bool = False
    """The wire pairs a result with its call by the tool's name, not an id."""
    refused_names: tuple[str, ...] = ()
    """Tool names the provider refuses before the request."""
    accepted_names: tuple[str, ...] = ("lookup",)

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def __repr__(self) -> str:
        return self.label

    def require(self, *capabilities: str) -> None:
        """Skip the scenario, saying why, when the wire cannot express it."""
        for capability in capabilities:
            if capability in self.cannot:
                pytest.skip(f"{self.label}: {self.cannot[capability]}")

    @abstractmethod
    def provider(self, script: Script) -> AIProvider:
        """A provider whose next generation answers *script*."""

    @abstractmethod
    def declared(self, request: Any) -> dict[str, dict[str, Any]]:
        """Each tool the request declared: its name and parameter schema."""

    @abstractmethod
    def replayed(self, request: Any) -> list[Item]:
        """The assistant and tool turns of the request, in order, normalized."""
