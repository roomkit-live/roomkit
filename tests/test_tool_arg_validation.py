"""Tests for dependency-free tool-argument validation."""

from __future__ import annotations

from roomkit.tools.validation import validate_tool_arguments

_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer"},
        "metric": {"type": "boolean"},
        "ratio": {"type": "number"},
    },
    "required": ["city", "days"],
}


def test_valid_arguments_pass() -> None:
    assert validate_tool_arguments(_SCHEMA, {"city": "Montréal", "days": 3}) is None
    assert (
        validate_tool_arguments(
            _SCHEMA, {"city": "Laval", "days": 1, "metric": True, "ratio": 1.5}
        )
        is None
    )


def test_missing_required_argument_rejected() -> None:
    err = validate_tool_arguments(_SCHEMA, {"city": "Québec"})
    assert err is not None
    assert "days" in err


def test_wrong_primitive_type_rejected() -> None:
    err = validate_tool_arguments(_SCHEMA, {"city": "Québec", "days": "three"})
    assert err is not None
    assert "days" in err and "integer" in err


def test_bool_is_not_an_integer() -> None:
    # bool is a subclass of int in Python — must not satisfy `integer`.
    err = validate_tool_arguments(_SCHEMA, {"city": "Québec", "days": True})
    assert err is not None
    assert "days" in err


def test_integer_satisfies_number() -> None:
    schema = {"type": "object", "properties": {"ratio": {"type": "number"}}}
    assert validate_tool_arguments(schema, {"ratio": 2}) is None


def test_non_dict_arguments_rejected() -> None:
    err = validate_tool_arguments(_SCHEMA, ["not", "a", "dict"])  # type: ignore[arg-type]
    assert err is not None


def test_empty_or_unknown_schema_permissive() -> None:
    assert validate_tool_arguments({}, {"anything": 1}) is None
    # Unknown/unenforced types don't reject.
    schema = {"type": "object", "properties": {"x": {"type": "weird"}}}
    assert validate_tool_arguments(schema, {"x": object()}) is None


def test_additional_properties_not_enforced() -> None:
    # Properties not declared in the schema pass through untouched.
    assert validate_tool_arguments(_SCHEMA, {"city": "A", "days": 1, "extra": 9}) is None
