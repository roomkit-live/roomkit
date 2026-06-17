"""Reusable example tools.

A :class:`Tool` bundles a schema (``definition``) with its
implementation (``handler``), so an ``AIChannel`` can accept it directly
via ``tools=[...]`` — no separate ``tool_handler`` argument needed.

:class:`WebSearchTool` does a real, key-free internet lookup through
DuckDuckGo's Instant Answer API. It is intentionally simple: good enough
to show a model calling a tool and grounding its reply on the result.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

_DDG_ENDPOINT = "https://api.duckduckgo.com/"
_TOOL_NAME = "web_search"


class WebSearchTool:
    """Simple internet search via DuckDuckGo's Instant Answer API.

    Satisfies the :class:`roomkit.tools.Tool` protocol (``definition`` +
    ``handler``), so ``AIChannel(..., tools=[WebSearchTool()])`` wires
    both the schema and the implementation automatically.
    """

    def __init__(self, *, timeout: float = 10.0, max_topics: int = 3) -> None:
        self._timeout = timeout
        self._max_topics = max_topics

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": _TOOL_NAME,
            "description": (
                "Search the public internet for a short, factual answer to a "
                "query. Returns an instant-answer summary and a few related "
                "snippets. Use it for current facts, definitions, or lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'capital of Canada'.",
                    },
                },
                "required": ["query"],
            },
        }

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        # Let composed handlers fall through to the next tool on a mismatch.
        if name != _TOOL_NAME:
            return json.dumps({"error": f"Unknown tool: {name}"})

        query = str(arguments.get("query", "")).strip()
        if not query:
            return "No query provided."

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    _DDG_ENDPOINT,
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1",
                    },
                    headers={"User-Agent": "roomkit-example/1.0"},
                )
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            # Hand the failure back to the model as the tool result rather
            # than crashing the loop — it can apologize or try again.
            return f"Search failed for {query!r}: {exc}"

        return self._format(query, data)

    def _format(self, query: str, data: dict[str, Any]) -> str:
        """Reduce the DDG payload to a compact, model-friendly string.

        DDG's Instant Answer API returns an ``AbstractText`` for entity /
        topic / definition queries (e.g. "Ottawa", "speed of light"). When
        present we return just that — it is the clean, useful signal. For
        queries with no abstract we fall back to a few related snippets.
        """
        abstract = (
            data.get("AbstractText") or data.get("Answer") or data.get("Definition") or ""
        ).strip()
        if abstract:
            heading = (data.get("Heading") or "").strip()
            head = f" ({heading})" if heading and heading.lower() != query.lower() else ""
            return f"Results for {query!r}{head}:\n{abstract}"

        topics: list[str] = []
        for topic in data.get("RelatedTopics", []):
            text = topic.get("Text") if isinstance(topic, dict) else None
            if text:
                topics.append(text.strip())
            if len(topics) >= self._max_topics:
                break
        if topics:
            related = "\n".join(f"- {t}" for t in topics)
            return f"Results for {query!r}:\nRelated:\n{related}"

        return f"No instant answer found for {query!r}."
