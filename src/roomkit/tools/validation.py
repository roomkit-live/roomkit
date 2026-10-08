"""Dependency-free validation of tool-call arguments against a declared schema.

Manual and intentionally minimal (no ``jsonschema`` dependency): it enforces
required properties, primitive JSON types, and unknown arguments against a
closed schema. Complex JSON Schema features ($ref, anyOf/oneOf, format,
pattern, nested object/array validation) are NOT enforced — this is a
first-boundary sanity gate that stops obviously malformed tool calls before
execution, not a full validator.

:func:`repair_tool_arguments` sits beside the gate rather than inside it: it
repairs the known, unambiguous mismatches *before* validation runs (a hub
tool's hoisted arguments, a primitive the model's server quoted), so the
validator itself stays a pure predicate over (schema, arguments).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
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

    Checks that every ``required`` property is present, that each supplied
    argument whose property declares a primitive ``type`` matches it, and —
    when the schema declares ``additionalProperties: false`` — that no unknown
    argument was supplied.

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
        closed = parameters.get("additionalProperties") is False
        for key, value in arguments.items():
            spec = properties.get(key)
            if not isinstance(spec, dict):
                # An unknown argument is a violation only when the schema
                # closed itself (``additionalProperties: false`` — what FastMCP
                # emits for a typed tool function). Answering here is what makes
                # the failure actionable: the model invented the argument, so
                # the reply has to name the real ones. Left to the tool, the
                # same call comes back as an opaque framework error the model
                # cannot correct from, and it re-issues the call unchanged.
                if closed:
                    known = ", ".join(sorted(properties)) or "none"
                    return f"unknown argument '{key}' (this tool accepts: {known})"
                continue  # open schema — additional properties are allowed
            json_type = spec.get("type")
            if isinstance(json_type, str) and not _matches_type(value, json_type):
                return f"argument '{key}' must be of type {json_type}"
    return None


# The container property a hub tool declares for its own arguments. A hub tool
# exposes one tool per domain behind a ``{action, params}`` signature, and a
# model trained mostly on flat schemas (one tool = its arguments) hoists the
# inner keys one level up. The name is fixed rather than inferred ("the
# schema's only object property") because guessing the container is how a real
# typo lands inside an unrelated object argument.
#
# The name alone is not enough, though: ``params`` is an ordinary name for an
# ordinary options object. What separates a hub container from one is that a
# hub container declares no shape of its own — it cannot, its shape varies with
# ``action``. See the free-form check in :func:`fold_hoisted_arguments`.
_PARAMS_PROPERTY = "params"


def fold_hoisted_arguments(
    parameters: dict[str, Any], arguments: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Fold root-level arguments back into a hub tool's ``params`` container.

    A model calling ``{action, params}`` flat — ``{"action": "list_columns",
    "board_id": "…"}`` — is refused by :func:`validate_tool_arguments` against
    the closed schema, costing a round-trip while the model corrects itself.
    Repairing the shape before validation spends that round-trip on work
    instead. Opening the schema (``additionalProperties: true``) would do the
    same job and silence real typos with it, so the schema stays closed and the
    repair is explicit and narrow.

    Folds only when every condition holds:

    - the schema closed itself (``additionalProperties: false``) — an open
      schema already accepts root keys, so there is nothing to repair;
    - it declares a ``params`` property of type ``object``;
    - that ``params`` declares no ``properties`` of its own — a hub container
      cannot, its shape varies with ``action``, and a declared shape means the
      property is an ordinary options object that merely shares the name;
    - at least one supplied root key is undeclared;
    - ``params`` is absent or empty.

    Returns ``(folded_arguments, None)`` when repaired, ``(None, None)`` when
    the call is none of its business (the caller keeps the original arguments),
    and ``(None, error)`` when both forms are present at once — ambiguous, so
    it is refused with a message naming which one to keep.
    """
    if not isinstance(parameters, dict) or not isinstance(arguments, dict):
        return None, None
    if parameters.get("additionalProperties") is not False:
        return None, None

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return None, None
    params_spec = properties.get(_PARAMS_PROPERTY)
    if not isinstance(params_spec, dict) or params_spec.get("type") != "object":
        return None, None
    if params_spec.get("properties"):
        # Declaring its own shape is what disqualifies it: a hub container
        # cannot declare one, so this is an ordinary options object that shares
        # the name. Folding into it would relocate an undeclared root key —
        # a misspelt *root* property, most often — inside the container instead
        # of naming it back to the model, and the validator cannot catch it
        # there because it does not recurse (see the module docstring). Left
        # alone, the same call is refused by name.
        return None, None

    hoisted = [key for key in arguments if key not in properties]
    if not hoisted:
        return None, None

    existing = arguments.get(_PARAMS_PROPERTY)
    if existing is not None and not isinstance(existing, dict):
        # A non-object ``params`` is a type error, not a hoisted call — leave it
        # to validation, which names the expected type.
        return None, None
    if existing:
        names = ", ".join(f"'{key}'" for key in hoisted)
        return None, (
            f"{names} passed at the root while '{_PARAMS_PROPERTY}' is already set — "
            f"pass every tool argument inside '{_PARAMS_PROPERTY}'"
        )

    folded = {key: value for key, value in arguments.items() if key not in hoisted}
    folded[_PARAMS_PROPERTY] = {key: arguments[key] for key in hoisted}
    return folded, None


@dataclass(frozen=True)
class RepairedArguments:
    """A model's call arguments after the repairs every gate makes before it
    validates them, and what each repair touched."""

    arguments: dict[str, Any]
    folded: tuple[str, ...] = ()
    """Root keys folded into a hub tool's ``params``."""
    unquoted: tuple[str, ...] = ()
    """Arguments read off their quotes as the primitive their property declares."""
    error: str | None = None
    """Why the call is refused unrepaired: both hub forms at once."""


def repair_tool_arguments(parameters: Any, arguments: Any) -> RepairedArguments:
    """The repairs every gate runs on a model's call before validating it, in
    order: a hub tool's hoisted arguments folded back into ``params``
    (:func:`fold_hoisted_arguments`), then the primitives the model's server
    quoted read as their declared types (:func:`unquote_primitive_arguments`).
    Arguments a BEFORE_TOOL_USE hook rewrote are never repaired
    (:func:`rewritten_arguments_error`)."""
    folded, fold_error = fold_hoisted_arguments(parameters, arguments)
    if fold_error is not None:
        return RepairedArguments(arguments, error=fold_error)
    hoisted = tuple(sorted(set(arguments) - set(folded))) if folded is not None else ()
    unquoted, names = unquote_primitive_arguments(parameters, folded or arguments)
    return RepairedArguments(unquoted, folded=hoisted, unquoted=names)


# A literal's spelling, JSON's own: what a quoted primitive must be read as.
_INTEGER_TEXT = re.compile(r"-?\d+")
_NUMBER_TEXT = re.compile(r"-?\d+(\.\d+)?([eE][+-]?\d+)?")
_NOT_A_LITERAL = object()


def _read_literal(text: str, json_type: str) -> Any:
    """What *text* is as a literal of *json_type*, or ``_NOT_A_LITERAL``."""
    text = text.strip()
    if json_type == "integer" and _INTEGER_TEXT.fullmatch(text):
        return int(text)
    if json_type == "number" and _NUMBER_TEXT.fullmatch(text):
        number = float(text)
        return number if math.isfinite(number) else _NOT_A_LITERAL
    if json_type == "boolean" and text.lower() in ("true", "false"):
        return text.lower() == "true"
    return _NOT_A_LITERAL


def unquote_primitive_arguments(parameters: Any, arguments: Any) -> tuple[Any, tuple[str, ...]]:
    """Read a primitive the model's call quoted as the type its property
    declares: ``"0"`` for an ``integer`` is ``0``, ``"0.5"`` for a ``number``
    is ``0.5``, ``"true"`` for a ``boolean`` is ``true``.

    Some servers parse a model's tool call into JSON by the schema the request
    declared (vLLM's ``qwen3_xml`` and ``qwen3_coder``): for a tool missing from
    the request, a catalogue tool recovered at call time, every value arrives
    as a string whatever the model wrote, and an error cannot teach the model
    to send what its server will not deliver. The reading is narrow: only a
    property declaring a primitive ``type``, which the validator would refuse
    the string for, and only a string that spells that type's literal exactly;
    anything else is left to the validator, which names the expected type.

    Returns the arguments, repaired or as given, and the names it read.
    """
    if not isinstance(parameters, dict) or not isinstance(arguments, dict):
        return arguments, ()
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return arguments, ()
    read: dict[str, Any] = {}
    for key, value in arguments.items():
        spec = properties.get(key)
        json_type = spec.get("type") if isinstance(spec, dict) else None
        if isinstance(value, str) and isinstance(json_type, str):
            literal = _read_literal(value, json_type)
            if literal is not _NOT_A_LITERAL:
                read[key] = literal
    if not read:
        return arguments, ()
    return {**arguments, **read}, tuple(read)


def rewritten_arguments_error(
    name: str, schema: dict[str, Any] | None, arguments: dict[str, Any]
) -> str | None:
    """Why the arguments BEFORE_TOOL_USE left for a call to *name* fail its
    *schema*, as the model reads it; ``None`` when they fit or nothing is
    declared.

    Run after the hooks whether they returned arguments or not: the event's
    arguments are a mutable dict a hook may edit in place. The model's own
    call was validated, and folded, before the hooks ran, so a failure here
    is always a hook's, and the words say the arguments were rewritten. No
    fold here, deliberately: a hook's arguments are user code, and repairing
    them would hide its bug instead of naming it.
    """
    if schema is None:
        return None
    error = validate_tool_arguments(schema, arguments)
    return None if error is None else f"Invalid rewritten arguments for '{name}': {error}"
