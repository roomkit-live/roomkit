"""Playwright MCP setup for the screen assistant example."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _clean_schema(schema: dict[str, object]) -> dict[str, object]:
    """Deep-clean a JSON Schema for Gemini compatibility.

    Gemini FunctionDeclaration rejects non-standard keys like
    $schema, additionalProperties, additional_properties, etc.
    """
    _STRIP_KEYS = {
        "$schema",
        "additionalProperties",
        "additional_properties",
        "default",
        "title",
    }
    out: dict[str, object] = {}
    for k, v in schema.items():
        if k in _STRIP_KEYS:
            continue
        if isinstance(v, dict):
            out[k] = _clean_schema(v)
        elif isinstance(v, list):
            out[k] = [_clean_schema(item) if isinstance(item, dict) else item for item in v]
        else:
            out[k] = v
    return out


async def setup_playwright_mcp(
    voice_choice: str,
    browser_mode: str,
) -> tuple[str, object | None, list[dict[str, object]], set[str], list[object]]:
    """Set up the optional Playwright MCP server.

    Returns:
        (browser_mode, mcp_session, tools, tool_names, cleanup_list)
        *browser_mode* may be downgraded to ``"vision"`` if setup fails.
    """
    playwright_mcp: object | None = None
    playwright_tools: list[dict[str, object]] = []
    playwright_tool_names: set[str] = set()
    pw_cleanup: list[object] = []

    if browser_mode == "playwright" and voice_choice != "openai":
        print("WARNING: Playwright mode requires OpenAI voice.")
        print("Gemini Live cannot handle Playwright's tool declarations.")
        print("Falling back to vision mode. Use VOICE_PROVIDER=openai for Playwright.\n")
        browser_mode = "vision"

    if browser_mode == "playwright":
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            logger.info("Starting Playwright MCP server via npx...")

            pw_params = StdioServerParameters(
                command="npx",
                args=["@playwright/mcp"],
            )
            pw_transport_ctx = stdio_client(pw_params)
            pw_streams = await pw_transport_ctx.__aenter__()
            pw_read, pw_write = pw_streams[0], pw_streams[1]
            pw_session = ClientSession(pw_read, pw_write)
            await pw_session.__aenter__()
            await pw_session.initialize()

            pw_result = await pw_session.list_tools()
            for tool in pw_result.tools:
                params = _clean_schema(dict(tool.inputSchema)) if tool.inputSchema else {}
                playwright_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": params,
                    }
                )
                playwright_tool_names.add(tool.name)

            playwright_mcp = pw_session
            pw_cleanup.extend([pw_session, pw_transport_ctx])
            logger.info(
                "Playwright MCP ready — %d tools: %s",
                len(playwright_tools),
                sorted(playwright_tool_names),
            )
        except Exception:
            logger.exception("Failed to start Playwright MCP — falling back to vision mode")
            browser_mode = "vision"

    return browser_mode, playwright_mcp, playwright_tools, playwright_tool_names, pw_cleanup
