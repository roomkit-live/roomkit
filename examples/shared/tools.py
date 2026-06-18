"""Reusable example tools.

A :class:`Tool` bundles a schema (``definition``) with its
implementation (``handler``), so an ``AIChannel`` can accept it directly
via ``tools=[...]`` — no separate ``tool_handler`` argument needed.

:class:`WebSearchTool` is a small internet-search tool with two backends:

- **Wikipedia** (default, no API key) — a full-text search followed by an
  article summary. It handles the natural-language queries a model tends
  to generate ("Coca-Cola company history and facts" → the Coca-Cola
  article), which a bare entity-lookup API does not. Encyclopedic only.
- **Tavily** (when ``TAVILY_API_KEY`` is set) — real web search built for
  LLM agents. Finds niche companies and current info, not just
  encyclopedic topics. Get a free key at https://tavily.com.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger("roomkit.examples.tools")

_TOOL_NAME = "web_search"
_UA = "roomkit-example/1.0 (https://github.com/roomkit-live/roomkit)"
_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_TAVILY_API = "https://api.tavily.com/search"


class WebSearchTool:
    """Internet search satisfying the :class:`roomkit.tools.Tool` protocol.

    ``AIChannel(..., tools=[WebSearchTool()])`` wires both the schema and
    the handler automatically and runs the whole loop. Set
    ``TAVILY_API_KEY`` for real web results; otherwise it falls back to a
    key-free Wikipedia lookup.
    """

    def __init__(self, *, timeout: float = 10.0, max_results: int = 3) -> None:
        self._timeout = timeout
        self._max_results = max_results
        self._tavily_key = os.environ.get("TAVILY_API_KEY")

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": _TOOL_NAME,
            "description": (
                "Search the internet for information about a topic, entity, "
                "company, or current fact. Returns a short summary and a few "
                "related results. Use it whenever a question needs information "
                "you don't already know."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, e.g. 'history of the Coca-Cola company'.",
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

        if self._tavily_key:
            try:
                return await self._search_tavily(query)
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
                # Degrade to Wikipedia rather than break the loop, but make
                # the failure visible so a real Tavily problem isn't hidden.
                logger.warning("Tavily search failed (%s); falling back to Wikipedia", exc)
        return await self._search_wikipedia(query)

    async def _search_tavily(self, query: str) -> str:
        """Real web search via Tavily (requires TAVILY_API_KEY)."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                _TAVILY_API,
                json={
                    "api_key": self._tavily_key,
                    "query": query,
                    "max_results": self._max_results,
                    "include_answer": True,
                    "search_depth": "basic",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        lines = [f"Results for {query!r} (web search):"]
        answer = (data.get("answer") or "").strip()
        if answer:
            lines.append(answer)
        for res in data.get("results", [])[: self._max_results]:
            title = (res.get("title") or "").strip()
            url = (res.get("url") or "").strip()
            content = (res.get("content") or "").strip()
            lines.append(f"- {title} ({url})\n  {content[:160]}")
        return "\n".join(lines)

    async def _search_wikipedia(self, query: str) -> str:
        """Key-free encyclopedic lookup: full-text search then summary."""
        async with httpx.AsyncClient(timeout=self._timeout, headers={"User-Agent": _UA}) as client:
            try:
                search = await client.get(
                    _WIKI_API,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": self._max_results,
                        "format": "json",
                    },
                )
                search.raise_for_status()
                hits = search.json().get("query", {}).get("search", [])
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                return f"Search failed for {query!r}: {exc}"

            if not hits:
                return (
                    f"No encyclopedic result for {query!r}. "
                    "(Tip: set TAVILY_API_KEY for full web search of niche or current topics.)"
                )

            title = hits[0]["title"]
            extract = ""
            try:
                summary = await client.get(_WIKI_SUMMARY + quote(title.replace(" ", "_")))
                if summary.status_code == 200:
                    extract = (summary.json().get("extract") or "").strip()
            except (httpx.HTTPError, json.JSONDecodeError):
                extract = ""

        lines = [f"Results for {query!r} (Wikipedia: {title}):"]
        if extract:
            lines.append(extract)
        others = [h["title"] for h in hits[1:]]
        if others:
            lines.append("Related: " + ", ".join(others))
        return "\n".join(lines)
