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


def test_additional_properties_allowed_on_an_open_schema() -> None:
    # The schema does not close itself, so undeclared properties pass through.
    assert validate_tool_arguments(_SCHEMA, {"city": "A", "days": 1, "extra": 9}) is None


_CLOSED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "query": {"type": "string"},
        "max_results": {"type": "integer"},
        "region": {"type": "string"},
    },
    "required": ["query"],
}


def test_unknown_argument_rejected_on_a_closed_schema() -> None:
    # The real regression: a model invents a plausible parameter it knows from
    # another vendor's API. Without this the call reaches the tool and comes
    # back as an opaque framework error the model cannot correct from.
    err = validate_tool_arguments(_CLOSED_SCHEMA, {"query": "x", "web_search_depth": "2"})
    assert err is not None
    assert "web_search_depth" in err


def test_unknown_argument_error_names_the_real_arguments() -> None:
    err = validate_tool_arguments(_CLOSED_SCHEMA, {"query": "x", "web_search_depth": "2"})
    assert err is not None
    for name in ("query", "max_results", "region"):
        assert name in err


def test_closed_schema_still_accepts_its_own_arguments() -> None:
    assert validate_tool_arguments(_CLOSED_SCHEMA, {"query": "x"}) is None
    assert (
        validate_tool_arguments(
            _CLOSED_SCHEMA, {"query": "x", "max_results": 5, "region": "fr-fr"}
        )
        is None
    )


def test_closed_schema_reports_missing_required_before_unknown() -> None:
    # A call that is both incomplete and polluted names the missing argument
    # first: supplying it is what unblocks the call.
    err = validate_tool_arguments(_CLOSED_SCHEMA, {"web_search_depth": "2"})
    assert err is not None
    assert "query" in err
