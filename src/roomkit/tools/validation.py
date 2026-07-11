"""Dependency-free validation of tool-call arguments against a declared schema.

Manual and intentionally minimal (no ``jsonschema`` dependency): it enforces
required properties and primitive JSON types only. Complex JSON Schema features
($ref, anyOf/oneOf, format, pattern, nested object/array validation) are NOT
enforced — this is a first-boundary sanity gate that stops obviously malformed
tool calls before execution, not a full validator.
"""

from __future__ import annotations

from typing import Any


def _matches_type(value: Any, json_type: str) -> bool:
    """Return whether *value* matches a primitive JSON Schema ``type``.

    ``bool`` is excluded from the numeric types because in Python ``bool`` is a
    subclass of ``int`` — a boolean must not satisfy ``integer``/``number``.
    Unknown type names are not enforced (treated as a match).
    """
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "null":
        return value is None
    return True  # unknown type — do not enforce


def validate_tool_arguments(parameters: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """Validate *arguments* against a JSON-Schema-style *parameters* object.

    Checks that every ``required`` property is present and that each supplied
    argument whose property declares a primitive ``type`` matches it.

    Returns a human-readable error string on the first violation, or ``None`` if
    the arguments pass (or the schema is empty / not enforceable).
    """
    if not isinstance(parameters, dict):
        return None
    if not isinstance(arguments, dict):
        return f"expected an object of arguments, got {type(arguments).__name__}"

    required = parameters.get("required")
    if isinstance(required, list):
        for field in required:
            if field not in arguments:
                return f"missing required argument '{field}'"

    properties = parameters.get("properties")
    if isinstance(properties, dict):
        for key, value in arguments.items():
            spec = properties.get(key)
            if not isinstance(spec, dict):
                continue  # additional / unknown property — not enforced
            json_type = spec.get("type")
            if isinstance(json_type, str) and not _matches_type(value, json_type):
                return f"argument '{key}' must be of type {json_type}"
    return None
