"""Dependency-free validation of tool-call arguments against a declared schema.

Manual and intentionally minimal (no ``jsonschema`` dependency): it enforces
required properties and primitive JSON types only. Complex JSON Schema features
($ref, anyOf/oneOf, format, pattern, nested object/array validation) are NOT
enforced — this is a first-boundary sanity gate that stops obviously malformed
tool calls before execution, not a full validator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Primitive JSON Schema type name -> predicate. ``bool`` is excluded from the
# numeric types because in Python ``bool`` is a subclass of ``int``, so a
# boolean must not satisfy ``integer``/``number``.
_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def _matches_type(value: Any, json_type: str) -> bool:
    """Return whether *value* matches a primitive JSON Schema ``type``.

    Unknown type names are not enforced (treated as a match).
    """
    check = _TYPE_CHECKS.get(json_type)
    return check is None or check(value)


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
