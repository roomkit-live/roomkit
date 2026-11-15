"""Render and bound the complete escaped skill manifest (RFC §24.1)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from html import escape

from roomkit._text import quoted
from roomkit.skills.errors import SkillManifestTooLargeError
from roomkit.skills.models import SkillMetadata

_QUOTED_MAX = 1100
_MANIFEST_MAX = 65_536


def render_manifest(
    listed: list[SkillMetadata],
    unavailable: Mapping[str, str],
    origin_of: Callable[[str], str | None],
) -> str:
    """The manifest of *listed* skills and *unavailable* ones."""
    if not listed and not unavailable:
        return ""

    lines: list[str] = []
    if listed:
        lines.append("<available_skills>")
        for meta in listed:
            lines.extend(_manifest_lines(meta, origin_of(meta.name)))
        lines.append("</available_skills>")
    if unavailable:
        lines.append("<unavailable_skills>")
        for name, reason in unavailable.items():
            lines.append(f'  <skill name="{escape(name)}">')
            lines.append(f"    <reason>{escape(reason)}</reason>")
            lines.append("  </skill>")
        lines.append("</unavailable_skills>")
    manifest = "\n".join(lines)
    characters = len(manifest)
    if characters > _MANIFEST_MAX:
        raise SkillManifestTooLargeError(characters, _MANIFEST_MAX)
    return manifest


def _manifest_lines(meta: SkillMetadata, label: str | None) -> list[str]:
    """One skill of the manifest. A served skill names its origin, and what
    its server wrote is quoted as a text from outside (RFC §6.4, §24.7)."""
    if label is None:
        opening = f'  <skill name="{escape(meta.name)}">'
        fields = [meta.description, meta.license, meta.compatibility]
    else:
        opening = f'  <skill name="{escape(meta.name)}" origin="{escape(label)}">'
        fields = [
            quoted(value, _QUOTED_MAX) if value else None
            for value in (meta.description, meta.license, meta.compatibility)
        ]
    lines = [opening]
    for tag, value in zip(("description", "license", "compatibility"), fields, strict=True):
        if value:
            # A quote made every double quote mark a single one: element
            # text needs no quote escaped, only its brackets and ampersands.
            text = escape(value) if label is None else escape(value, quote=False)
            lines.append(f"    <{tag}>{text}</{tag}>")
    lines.append("  </skill>")
    return lines
