"""How a provider declares the turn's tools (RFC §6.7).

What every vendor shares: the schema of a tool that takes no parameters, and
the Chat Completions declaration the OpenAI-shaped providers send. What each
vendor decides on its own, the tool names it accepts, is a :class:`ToolNameRule`
its provider holds and checks before the request.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from roomkit.providers.ai.base import AITool, ProviderError


def declared_parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """The schema a tool is declared with: its own, its root an object, or for
    a tool that takes no parameters an object with none, which every vendor
    accepts.

    An empty map is not one: Anthropic refuses it, and Anthropic and Mistral
    refuse a declaration without a schema (measured 2026-10-02). A root
    without a type is an object's, said so for every vendor: Anthropic and
    OpenAI refuse it untyped (measured 2026-10-03).
    """
    if not parameters:
        return {"type": "object", "properties": {}}
    if parameters.get("type") is None:
        return {**parameters, "type": "object"}
    return dict(parameters)


def chat_tool_declarations(tools: Sequence[AITool]) -> list[dict[str, Any]]:
    """The tools as Chat Completions declares them, the shape every
    OpenAI-compatible server reads."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": declared_parameters(tool.parameters),
            },
        }
        for tool in tools
    ]


@dataclass(frozen=True)
class ToolNameRule:
    """The tool names one vendor accepts.

    A provider checks the turn's tools against its vendor's rule before the
    request, so a name the vendor would refuse fails at once with an error
    naming the tool and the rule, not mid-turn with the vendor's 400.
    """

    vendor: str
    pattern: str
    """The names the vendor accepts, as a regular expression matched whole."""
    _compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", re.compile(self.pattern))

    def check(self, names: Iterable[str]) -> None:
        """Raise a non-retryable :class:`ProviderError` for the first name the
        vendor does not accept."""
        for name in names:
            if self._compiled.fullmatch(name) is None:
                raise ProviderError(
                    f"tool {name!r}: {self.vendor} accepts tool names matching {self.pattern}",
                    provider=self.vendor,
                    context_overflow=False,
                    before_request=True,
                )
