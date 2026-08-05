"""What a tool call produced, in the handful of lines a reader wants.

ACP fixes the *envelope* — ``content`` blocks and a free-form ``raw_output``
— and leaves the payload inside it to each agent. Claude Code fills it with
text and diff blocks. Codex fills it with ``{"formatted_output": …,
"exit_code": N}``, and for shell commands puts a ``terminal`` block in
``content`` that carries no inline text at all, the output arriving only in
``raw_output``. Rendered naively, one agent reads like a transcript and the
other like a JSON dump cut mid-string::

    ⎿ ✓ Read file '/…/SKILL.md' · 6 ms
      {"formatted_output": "---\\nname: deploy-tools\\ndescription: Drive the

So this module reads the envelope rather than printing it: known payload
shapes are unwrapped down to their text, media is named instead of dumped
(a base64 image is bytes, not a preview), and a display payload that turns
out to be empty falls back to the raw result.

Display only — it never decides whether a call succeeded; that is the
status line's job. Anything it cannot recognise still renders as compact
JSON, so a new agent shape degrades to what the console did before.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from roomkit.models.event import ToolCallContent

PreviewLine = tuple[str, str]
"""``(kind, text)`` — kind is ``dim`` (output), ``add``/``del`` (diff)."""

_PREVIEW_MAX_LINES = 5
_PREVIEW_HARD_CAP = 200  # lines collected before slicing — bounds huge diffs
_LINE_MAX = 200  # a preview line is a glance, not a paste buffer

_PAYLOAD_KEYS = ("formatted_output", "output", "text", "content", "result", "error")
"""Keys agents wrap their real output in, most specific first.

Also the test for "is this string an envelope?" — see :func:`_decoded`.
"""


def tool_result_preview(
    content: ToolCallContent,
    *,
    max_lines: int = _PREVIEW_MAX_LINES,
) -> list[PreviewLine]:
    """Extract display lines from a tool result — what Claude Code shows.

    Returns ``(kind, line)`` pairs capped at *max_lines* with a trailing
    ``… +N lines`` marker.
    """
    collected: list[PreviewLine] = []
    for source in _preview_sources(content):
        _collect_preview(source, collected)
        if collected:
            break
    if not collected:
        return []
    if len(collected) > max_lines:
        hidden = len(collected) - max_lines
        collected = collected[:max_lines]
        collected.append(("dim", f"… +{hidden} lines"))
    return collected


def _preview_sources(content: ToolCallContent) -> Iterator[Any]:
    """Where the text may live, most display-worthy first.

    The error text wins on failure, then ACP's display-intended payload
    (``structured_content["acp_content"]``, where file diffs live), then the
    raw result. Each is tried in turn *until one yields lines*: a
    ``terminal`` block is a handle on live output, not output, and behind
    one the command's text is in the raw result.
    """
    if content.status == "failed" and content.error:
        # Built by the ACP channel as ``_result_text(raw_output)`` — the
        # payload's JSON, not a message, whenever the agent sent structure.
        yield _decoded(content.error)
    if isinstance(content.structured_content, Mapping):
        acp_content = content.structured_content.get("acp_content")
        if acp_content is not None:
            yield acp_content
    if content.result is not None:
        yield content.result


def _collect_preview(value: Any, out: list[PreviewLine]) -> None:
    if value is None or len(out) >= _PREVIEW_HARD_CAP:
        return
    if isinstance(value, str):
        out.extend(("dim", _clip(line)) for line in value.splitlines() if line.strip())
        return
    if isinstance(value, Mapping):
        _collect_mapping(value, out)
        return
    if isinstance(value, list):
        for item in value:
            if len(out) >= _PREVIEW_HARD_CAP:
                return
            _collect_preview(item, out)
        return
    out.append(("dim", _clip(str(value))))


def _collect_mapping(value: Mapping[str, Any], out: list[PreviewLine]) -> None:
    """A content block if it declares a type, otherwise a payload wrapper."""
    collector = _BLOCK_COLLECTORS.get(str(value.get("type") or ""))
    if collector is not None:
        collector(value, out)
        return
    if "formatted_output" in value:
        _collect_exec_payload(value, out)
        return
    for key in _PAYLOAD_KEYS:
        nested = value.get(key)
        if nested is not None:  # ``{"result": null, "error": "…"}`` reads on
            _collect_preview(_decoded(nested), out)
            return
    out.append(("dim", _clip(json.dumps(value, ensure_ascii=False, default=str))))


def _collect_exec_payload(block: Mapping[str, Any], out: list[PreviewLine]) -> None:
    """Codex's command payload: aggregated output beside a POSIX exit code.

    The code is only worth a line when the command failed *silently* — a
    status line already says the call failed, but ``✗ tool failed`` with no
    output under it leaves nothing to act on.
    """
    before = len(out)
    _collect_preview(_decoded(block.get("formatted_output")), out)
    exit_code = block.get("exit_code")
    if len(out) == before and isinstance(exit_code, int) and exit_code != 0:
        out.append(("dim", f"exit code {exit_code}"))


def _collect_content_block(block: Mapping[str, Any], out: list[PreviewLine]) -> None:
    _collect_preview(block.get("content"), out)


def _collect_text_block(block: Mapping[str, Any], out: list[PreviewLine]) -> None:
    _collect_preview(block.get("text"), out)


def _collect_terminal_block(block: Mapping[str, Any], out: list[PreviewLine]) -> None:
    """Nothing: a terminal block names a live stream, it does not carry it."""
    return


def _collect_media_block(block: Mapping[str, Any], out: list[PreviewLine]) -> None:
    """Name the medium — the payload is base64 bytes, never a preview."""
    kind = str(block.get("type") or "media")
    mime = str(block.get("mimeType") or block.get("mime_type") or "")
    out.append(("dim", f"[{kind} {mime}]" if mime else f"[{kind}]"))


def _collect_link_block(block: Mapping[str, Any], out: list[PreviewLine]) -> None:
    label = block.get("name") or block.get("uri")
    if label:
        out.append(("dim", _clip(str(label))))


def _collect_resource_block(block: Mapping[str, Any], out: list[PreviewLine]) -> None:
    """An embedded resource: its text if it has one, else its URI.

    Never its ``blob`` — that field is base64 by definition.
    """
    resource = block.get("resource")
    if not isinstance(resource, Mapping):
        return
    text = resource.get("text")
    if isinstance(text, str) and text.strip():
        _collect_preview(text, out)
        return
    uri = resource.get("uri")
    if uri:
        out.append(("dim", _clip(str(uri))))


def _collect_diff_block(block: Mapping[str, Any], out: list[PreviewLine]) -> None:
    path = block.get("path")
    if path:
        out.append(("dim", _clip(str(path))))
    # ACP dumps use camelCase aliases (oldText/newText); tolerate snake_case.
    old_text = block.get("oldText", block.get("old_text")) or ""
    new_text = block.get("newText", block.get("new_text")) or ""
    diff = difflib.unified_diff(
        str(old_text).splitlines(),
        str(new_text).splitlines(),
        lineterm="",
        n=1,
    )
    for line in diff:
        if len(out) >= _PREVIEW_HARD_CAP:
            return
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("+"):
            out.append(("add", _clip(line)))
        elif line.startswith("-"):
            out.append(("del", _clip(line)))
        else:
            out.append(("dim", _clip(line)))


_BLOCK_COLLECTORS: dict[str, Callable[[Mapping[str, Any], list[PreviewLine]], None]] = {
    "content": _collect_content_block,
    "text": _collect_text_block,
    "terminal": _collect_terminal_block,
    "image": _collect_media_block,
    "audio": _collect_media_block,
    "resource_link": _collect_link_block,
    "resource": _collect_resource_block,
    "diff": _collect_diff_block,
}


def _decoded(value: Any) -> Any:
    """Parse an envelope that arrived as a JSON string; leave text alone.

    Only applied where the protocol puts an envelope — the error text and
    the value of a payload key — and only when the parse yields a wrapper
    we recognise. A tool whose output *is* JSON (``cat package.json``) keeps
    its lines instead of being taken apart.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped.startswith("{"):
        return value
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return value
    if isinstance(parsed, Mapping) and any(key in parsed for key in _PAYLOAD_KEYS):
        return parsed
    return value


def _clip(line: str) -> str:
    return line if len(line) <= _LINE_MAX else f"{line[: _LINE_MAX - 1]}…"


__all__ = ["PreviewLine", "tool_result_preview"]
