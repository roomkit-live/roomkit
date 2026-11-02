"""How an MCP ``CallToolResult`` is checked, then rendered: as a string, or for the model."""

from __future__ import annotations

import json
from typing import Any

from roomkit.providers.ai.base import AIImagePart, AITextPart
from roomkit.providers.utils import parse_data_uri
from roomkit.tools.compose import ToolResult
from roomkit.tools.result import bounded_result

# The raster formats every vision vendor accepts. Anything else (an SVG, a
# TIFF) would fail the whole request at the vendor, so it reaches the model as
# a note instead of an image.
_MODEL_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

# The bounds of a result the model sees an image of (RFC §21.2): its texts,
# the notes of the images held back among them, within this many characters
# in all; at most this many of its images, within this many bytes of data URL
# in all. Four is an agent's default ``max_author_images``.
_RESULT_TEXT_LIMIT = 100_000
_MODEL_IMAGES_MAX = 4
_MODEL_IMAGE_BYTES_MAX = 5 * 1024 * 1024


def _content_text(content: Any) -> str:
    return content.text if hasattr(content, "text") else str(content)


def _join(texts: list[str]) -> str:
    """One text as itself, several as a JSON array: the string contract."""
    if len(texts) == 1:
        return texts[0]
    return json.dumps(texts)


def _decodes(url: str) -> bool:
    try:
        parse_data_uri(url)
    except ValueError:
        return False
    return True


def _model_part(content: Any) -> AITextPart | AIImagePart:
    """One content of a result as the model reads it.

    Text stays text, and an image the vendors accept becomes an image part.
    Other binary content (an image in another format or with a corrupt
    payload, audio, a blob resource) becomes a one-line note: its repr would
    hand the model kilobytes of base64, and a bad image would fail the request.
    Anything else keeps the text it has in :func:`text_body`.
    """
    if hasattr(content, "text"):
        return AITextPart(text=content.text)
    kind = getattr(content, "type", None)
    if kind == "image":
        mime = getattr(content, "mimeType", None) or "image/png"
        url = f"data:{mime};base64,{getattr(content, 'data', '')}"
        if mime in _MODEL_IMAGE_TYPES and _decodes(url):
            return AIImagePart(url=url, mime_type=mime)
    elif kind == "audio":
        mime = getattr(content, "mimeType", None) or "audio"
    elif kind == "resource" and hasattr(getattr(content, "resource", None), "blob"):
        mime = getattr(content.resource, "mimeType", None) or "binary"
    else:
        return AITextPart(text=_content_text(content))
    return _note(kind, mime)


def _note(kind: str, mime: str) -> AITextPart:
    return AITextPart(text=f"[{kind} content ({mime}) not shown to the model]")


def _bounded_parts(parts: list[AITextPart | AIImagePart], name: str) -> ToolResult:
    """*parts* within the bounds of a result the model sees an image of.

    An image past the count or the bytes reads as its note, which counts as
    a text. Each text is kept whole while it fits; the one past the bound is
    cut with a note saying how long it was, and the texts after it are left
    out. The images that fit stay in place.
    """
    left = _RESULT_TEXT_LIMIT
    images = image_bytes = 0
    kept: list[AITextPart | AIImagePart] = []
    for part in parts:
        if isinstance(part, AIImagePart):
            size = len(part.url)
            if images < _MODEL_IMAGES_MAX and image_bytes + size <= _MODEL_IMAGE_BYTES_MAX:
                images += 1
                image_bytes += size
                kept.append(part)
                continue
            part = _note("image", part.mime_type or "image/png")
        if left == 0:
            continue
        text = bounded_result(part.text, left, name)
        left = left - len(text) if text == part.text else 0
        kept.append(AITextPart(text=text))
    return kept


def error_text(result: Any) -> str:
    """The words of a result the server flagged ``isError``."""
    return " ".join(_content_text(c) for c in result.content)


def text_body(result: Any) -> str:
    """A successful result as :meth:`MCPToolProvider.call_tool` returns it."""
    return _join([_content_text(c) for c in result.content])


def handler_result(result: Any, name: str) -> ToolResult:
    """A successful result of the tool *name* as the tool handler returns it.

    Content parts when an image survives, so the model sees it, within the
    bounds of :func:`_bounded_parts`; otherwise one string, the shape
    :func:`text_body` gives, with binary content noted rather than spelled
    out in base64.
    """
    parts = [_model_part(c) for c in result.content]
    if any(isinstance(p, AIImagePart) for p in parts):
        return _bounded_parts(parts, name)
    return _join([p.text for p in parts if isinstance(p, AITextPart)])


def check_structured_content(name: str, schema: dict[str, Any], result: Any) -> None:
    """Raise when a tool's result breaks the output schema it was listed with.

    The MCP SDK's own check (``ClientSession.call_tool``), kept here so that a
    call never lists the server's tools to find a schema (RFC §21.2): the same
    ``RuntimeError`` messages, and a ``$ref`` resolved within the schema only,
    never fetched.
    """
    from jsonschema import SchemaError, ValidationError, validate
    from referencing import Registry
    from referencing.exceptions import Unresolvable

    if result.structuredContent is None:
        raise RuntimeError(
            f"Tool {name} has an output schema but did not return structured content"
        )
    try:
        validate(result.structuredContent, schema, registry=Registry())
    except ValidationError as e:
        raise RuntimeError(f"Invalid structured content returned by tool {name}: {e}") from e
    except (SchemaError, Unresolvable) as e:
        raise RuntimeError(f"Invalid schema for tool {name}: {e}") from e
