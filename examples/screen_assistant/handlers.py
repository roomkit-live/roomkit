"""Tool-call dispatcher for the screen assistant example.

A single ``ScreenToolDispatcher.handle(name, arguments)`` method takes
the place of a 250-line if/elif tree in the example. Returns the tool
result string, or ``None`` when the tool is not handled here (let the
hook fall through to ``HookResult.allow()``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess

from .omniview import OmniViewClient
from .screens import list_screens, switch_screen
from .state import ScreenAssistantState
from .tools import ACTION_TOOLS
from .verify import assess_action_result, build_verify_question
from .vision import analyze_with_cost

logger = logging.getLogger("screen_assistant.handlers")

_PLAYWRIGHT_MAX_RESULT = 4000
_VERIFY_PAUSE_S = 0.5
_OPEN_APP_PAUSE_S = 2.0
_OPEN_APP_FOCUS_PAUSE_S = 0.5

# Tools that mutate the keyboard/mouse/clipboard. Subject to the
# "describe_screen after a FAILED action" gate.
_MUTATING_TOOLS: frozenset[str] = frozenset({"open_app", "click_result", *ACTION_TOOLS})


class ScreenToolDispatcher:
    """Route realtime tool calls to the right handler.

    The constructor wires up optional integrations (OmniView, Playwright MCP);
    each ``handle()`` call returns the tool result (a string) or ``None`` when
    the tool is not one we own.
    """

    def __init__(
        self,
        state: ScreenAssistantState,
        *,
        playwright_mcp: object | None = None,
        playwright_tool_names: set[str] | None = None,
    ) -> None:
        self.state = state
        self.playwright_mcp = playwright_mcp
        self.playwright_tool_names = playwright_tool_names or set()
        # Serialize tool dispatch so that batched tool calls from the LLM
        # don't race (e.g. command+l + type_text in the same turn would
        # otherwise pile keystrokes on top of each other).
        self._lock = asyncio.Lock()
        # Tracks the last action's STATUS so we can force describe_screen
        # before another mutating action when the previous one failed.
        self._last_failed = False

    async def handle(self, name: str, arguments: dict[str, object]) -> str | None:
        async with self._lock:
            # Gate: if the previous mutating action failed, block the next
            # one until the agent calls describe_screen (or list_screens).
            if self._last_failed and name in _MUTATING_TOOLS:
                return (
                    "BLOCKED: the previous action returned STATUS: FAILED. "
                    "Call describe_screen first to understand what happened, "
                    "tell the user, then choose a different approach. "
                    "Do NOT retry the same action."
                )

            # Mark "tool in progress" so the vision observer can suppress
            # injections during this window — vision arriving mid-tool
            # would confuse the model into firing a second response on top
            # of the (forthcoming) tool result.
            self.state.tool_in_progress = True
            try:
                result = await self._dispatch(name, arguments)
            finally:
                self.state.tool_in_progress = False

            if result is not None and name in _MUTATING_TOOLS:
                self._last_failed = "STATUS: FAILED" in result
            elif name in ("describe_screen", "list_screens"):
                # A read-only check clears the gate.
                self._last_failed = False
            return result

    async def _dispatch(self, name: str, arguments: dict[str, object]) -> str | None:
        if name == "list_screens":
            return await list_screens(self.state)
        if name == "switch_screen":
            return switch_screen(self.state, int(arguments.get("monitor", 1)))
        if name == "open_app":
            return await self._open_app(str(arguments.get("app_name", "")))
        if name == "describe_screen":
            return await self._describe_screen(str(arguments.get("query", "")))
        if name == "observe" and self.state.omniview is not None:
            return await self._observe(self.state.omniview)
        if name == "click_result" and self.state.omniview is not None:
            return await self._click_result(self.state.omniview, arguments)
        if name in ACTION_TOOLS:
            return await self._action_tool(name, arguments)
        if self.playwright_mcp is not None and name in self.playwright_tool_names:
            return await self._playwright(name, arguments)
        return None

    # -- open_app -----------------------------------------------------------

    async def _open_app(self, app_name: str) -> str:
        logger.info("open_app(%r)", app_name)
        try:
            subprocess.run(  # noqa: ASYNC221
                ["open", "-a", app_name],
                check=True,
                capture_output=True,
                timeout=5,
            )
            await asyncio.sleep(_OPEN_APP_PAUSE_S)
            subprocess.run(  # noqa: ASYNC221
                ["osascript", "-e", f'tell application "{app_name}" to activate'],
                capture_output=True,
                timeout=3,
            )
            await asyncio.sleep(_OPEN_APP_FOCUS_PAUSE_S)
        except subprocess.CalledProcessError as exc:
            return f"Failed to open {app_name}: {exc.stderr.decode().strip()}"
        except FileNotFoundError:
            return f"Application '{app_name}' not found."

        verify = await analyze_with_cost(
            self.state,
            f"Which application is now in the foreground? Is {app_name} focused?",
        )
        self.state.record_description(verify)
        is_focused = app_name.lower() in verify.lower() or "chrome" in verify.lower()
        status = "OK" if is_focused else "FAILED"
        verdict = f"{app_name} {'is' if is_focused else 'is NOT'} in the foreground."
        out = (
            f"ACTION: open_app({app_name})\nSTATUS: {status}\n"
            f"SCREEN: {verify[:200]}\nVERDICT: {verdict}"
        )
        if not is_focused:
            out += (
                f"\nSUGGESTION: {app_name} may not have opened. Try again or check the app name."
            )
        return out

    # -- describe_screen ----------------------------------------------------

    async def _describe_screen(self, query: str) -> str:
        if not query:
            query = "Describe the current screen: foreground app, visible content, URLs."
        result = await analyze_with_cost(self.state, query)
        self.state.record_description(result)
        return result

    # -- OmniView observe ---------------------------------------------------

    async def _observe(self, omniview: OmniViewClient) -> str:
        logger.info("observe: calling OmniView /parse")
        self.state.cost_telemetry.totals["vision_calls"] += 1
        result_data = await omniview.parse()
        elements = result_data.get("elements", [])
        logger.info("observe: OmniView returned %d elements", len(elements))
        img_h = result_data.get("height", 2160)

        cleaned: list[dict[str, object]] = []
        for el in elements:
            content = omniview.clean_ocr(str(el.get("content", "")))
            if len(content) < 3:
                continue
            center = el.get("center", [0, 0])
            cleaned.append(
                {
                    "id": el.get("id"),
                    "type": el.get("element_type"),
                    "text": content[:80],
                    "center": center,
                    "interactable": el.get("interactable", False),
                }
            )

        # Content area first (middle 60%), text before icons, then by Y.
        top_band = img_h * 0.15
        bottom_band = img_h * 0.85

        def _priority(e: dict[str, object]) -> tuple[bool, bool, float]:
            cy = e["center"][1]  # type: ignore[index]
            in_content = top_band < cy < bottom_band
            is_text = e["type"] == "text"
            return (not in_content, not is_text, cy)

        cleaned.sort(key=_priority)
        return json.dumps(
            {
                "status": "ok",
                "elements": cleaned[:40],
                "total": len(cleaned),
                "note": (
                    "Use click_result(element_id=N) to click an element. "
                    "Elements are sorted by relevance — content area first."
                ),
            }
        )

    # -- OmniView click_result ---------------------------------------------

    async def _click_result(
        self,
        omniview: OmniViewClient,
        arguments: dict[str, object],
    ) -> str:
        element_id = int(arguments.get("element_id", -1))
        el = omniview.get_element_by_id(element_id)
        logger.info(
            "click_result: element_id=%d found=%s content=%r",
            element_id,
            el is not None,
            str(el.get("content", ""))[:60] if el else "N/A",
        )
        if el is None:
            return json.dumps(
                {
                    "status": "failed",
                    "error": f"Element {element_id} not found. Run observe first.",
                }
            )

        cx, cy = int(el["center"][0]), int(el["center"][1])  # type: ignore[index]
        content = str(el.get("content", ""))
        button = str(arguments.get("button", "left"))
        double = bool(arguments.get("double", False))
        omniview.click_at(cx, cy, button=button, clicks=2 if double else 1)
        await asyncio.sleep(_VERIFY_PAUSE_S)

        if not self.state.auto_verify:
            return json.dumps({"status": "ok", "clicked": content[:60], "center": [cx, cy]})

        return await self._format_verified_result(
            tool_name="click_result",
            arguments=arguments,
            action_label=f'click_result({element_id}) -> "{content[:40]}"',
        )

    # -- Action tools (click_element / type_text / press_key / scroll) -----

    async def _action_tool(self, name: str, arguments: dict[str, object]) -> str:
        # Ensure the target app actually has keyboard focus before sending
        # modifier-key combos. Vision sees the *rendered* foreground, which
        # can differ from the OS keyboard-focused app — without this,
        # ``command+t`` can land in iTerm instead of Chrome.
        if name == "press_key" and "+" in str(arguments.get("key", "")):
            await self._activate_target_app()
        action_result = await self._run_action(name, arguments)
        if not self.state.auto_verify:
            return action_result
        try:
            await asyncio.sleep(_VERIFY_PAUSE_S)
            return await self._format_verified_result(
                tool_name=name,
                arguments=arguments,
                action_label=f"{name}({json.dumps(dict(arguments), default=str)})",
                action_fallback=action_result,
            )
        except Exception:
            logger.exception("Auto-verify failed after %s", name)
            return action_result

    async def _activate_target_app(self) -> None:
        """Force the target app to grab keyboard focus via osascript.

        Only fires when the latest screen description suggests the target
        app is rendered on screen — avoids stealing focus from whatever
        the user is actually doing.
        """
        target = self.state.target_app
        if not target:
            return
        if target.lower() not in self.state.latest_description.lower():
            return
        try:
            subprocess.run(  # noqa: ASYNC221
                ["osascript", "-e", f'tell application "{target}" to activate'],
                check=False,
                capture_output=True,
                timeout=2,
            )
            # Let focus settle before the keystroke.
            await asyncio.sleep(0.2)
        except Exception:
            logger.exception("Failed to activate %r before keystroke", target)

    async def _run_action(self, name: str, arguments: dict[str, object]) -> str:
        """Execute the action, preferring OmniView /locate for click_element."""
        if name != "click_element":
            return await self.state.input_tools.handler(name, arguments)

        omniview = self.state.omniview
        if omniview is None:
            logger.info("click_element: OmniView not available, using vision")
            return await self.state.input_tools.handler(name, arguments)

        element_desc = str(arguments.get("element", ""))
        logger.info("click_element: trying OmniView /locate for %r", element_desc)
        try:
            locate_result = await omniview.locate(element_desc)
        except Exception:
            logger.exception("click_element: OmniView /locate failed, falling back to vision")
            return await self.state.input_tools.handler(name, arguments)

        el = locate_result.get("element") or {}
        center = el.get("center")
        logger.info(
            "click_element: OmniView /locate result: found=%s center=%s content=%r score=%s",
            locate_result.get("found"),
            center,
            str(el.get("content", ""))[:60],
            locate_result.get("match_score", "?"),
        )
        if locate_result.get("found") and center and center[0] and center[1]:
            cx, cy = int(center[0]), int(center[1])
            omniview.click_at(cx, cy)
            logger.info("click_element via OmniView: %r → click at (%d,%d)", element_desc, cx, cy)
            return f"Clicked '{element_desc}' via OmniView at ({cx},{cy})."

        reason = "not found" if not locate_result.get("found") else "no center coords"
        logger.info(
            "click_element: OmniView miss for %r (%s), falling back to vision",
            element_desc,
            reason,
        )
        return await self.state.input_tools.handler(name, arguments)

    # -- Playwright MCP -----------------------------------------------------

    async def _playwright(self, name: str, arguments: dict[str, object]) -> str:
        logger.info("Playwright MCP tool: %s(%s)", name, arguments)
        pw_result = await self.playwright_mcp.call_tool(name, arguments)  # type: ignore[union-attr]
        texts = [c.text for c in pw_result.content if hasattr(c, "text")]
        raw = "\n".join(texts) if texts else json.dumps({"status": "ok"})
        if len(raw) <= _PLAYWRIGHT_MAX_RESULT:
            return raw
        return (
            raw[:_PLAYWRIGHT_MAX_RESULT]
            + f"\n... [truncated {len(raw)} → {_PLAYWRIGHT_MAX_RESULT} chars]"
        )

    # -- Shared verify formatting ------------------------------------------

    async def _format_verified_result(
        self,
        *,
        tool_name: str,
        arguments: dict[str, object],
        action_label: str,
        action_fallback: str | None = None,
    ) -> str:
        try:
            verify_q = build_verify_question(tool_name, arguments)
            screen_desc = await analyze_with_cost(self.state, verify_q)
        except Exception:
            if action_fallback is not None:
                logger.exception("Auto-verify failed after %s", tool_name)
                return action_fallback
            raise

        self.state.record_description(screen_desc)
        verdict = assess_action_result(tool_name, arguments, screen_desc)
        out = (
            f"ACTION: {action_label}\nSTATUS: {verdict['status']}\n"
            f"SCREEN: {screen_desc[:200]}\nVERDICT: {verdict['verdict']}"
        )
        if verdict["status"] == "FAILED":
            out += f"\nSUGGESTION: {verdict['suggestion']}"
            if tool_name in ACTION_TOOLS:
                out += "\nDo NOT proceed with the next step — fix this first."
        if tool_name in ACTION_TOOLS:
            logger.info(
                "Auto-verify %s: %s → %s", tool_name, verdict["status"], verdict["verdict"]
            )
        return out
