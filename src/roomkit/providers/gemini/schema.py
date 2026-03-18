"""Gemini-compatible JSON Schema cleaning.

Gemini's ``FunctionDeclaration`` rejects extra JSON Schema fields that
are common in OpenAPI / MCP tool definitions (``$schema``,
``additionalProperties``, ``default``, ``title``).  This module
provides a recursive cleaner that strips unsupported fields while
preserving the structure Gemini expects.
"""

from __future__ import annotations

from typing import Any

# Fields that Gemini accepts in a function parameter schema.
_GEMINI_ALLOWED_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "description",
        "enum",
        "items",
        "format",
        "nullable",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
    }
)


def clean_gemini_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Recursively strip fields that Gemini rejects from a JSON Schema.

    Removes ``$schema``, ``additionalProperties``, ``additional_properties``,
    ``default``, and ``title`` at every nesting level.  Preserves all keys
    in :data:`_GEMINI_ALLOWED_KEYS`.

    Args:
        schema: A JSON Schema dict (e.g. tool parameter schema), or None.

    Returns:
        A cleaned copy, or None if input was None.
    """
    if schema is None:
        return None
    return _clean(schema)


def _clean(obj: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in obj.items():
        if key not in _GEMINI_ALLOWED_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            result[key] = {k: _clean(v) for k, v in value.items() if isinstance(v, dict)}
        elif key == "items" and isinstance(value, dict):
            result[key] = _clean(value)
        else:
            result[key] = value
    return result
