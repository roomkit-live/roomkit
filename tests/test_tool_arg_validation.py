"""Tests for dependency-free tool-argument validation."""

from __future__ import annotations

from roomkit.tools.validation import (
    fold_hoisted_arguments,
    repair_tool_arguments,
    unquote_primitive_arguments,
    validate_tool_arguments,
)

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


# A hub tool: one tool per domain, ``{action, params}``, closed by the schema
# generator. This is the shape a flat-schema-trained model flattens.
_HUB_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string"},
        "params": {"type": "object"},
    },
    "required": ["action"],
}


def test_hoisted_arguments_are_folded_into_params() -> None:
    folded, err = fold_hoisted_arguments(
        _HUB_SCHEMA, {"action": "list_columns", "board_id": "board-1"}
    )
    assert err is None
    assert folded == {"action": "list_columns", "params": {"board_id": "board-1"}}
    # And the repaired call is what the gate would have accepted all along.
    assert validate_tool_arguments(_HUB_SCHEMA, folded) is None


def test_fold_keeps_declared_root_arguments_at_the_root() -> None:
    folded, err = fold_hoisted_arguments(_HUB_SCHEMA, {"action": "x", "a": 1, "b": 2})
    assert err is None
    assert folded == {"action": "x", "params": {"a": 1, "b": 2}}


def test_fold_into_an_empty_params_container() -> None:
    # ``params: {}`` carries no argument, so folding into it is unambiguous.
    folded, err = fold_hoisted_arguments(_HUB_SCHEMA, {"action": "x", "params": {}, "a": 1})
    assert err is None
    assert folded == {"action": "x", "params": {"a": 1}}


def test_both_forms_at_once_is_refused_and_names_the_survivor() -> None:
    folded, err = fold_hoisted_arguments(_HUB_SCHEMA, {"action": "x", "params": {"a": 1}, "b": 2})
    assert folded is None
    assert err is not None
    assert "'b'" in err and "params" in err


def test_nothing_to_fold_when_the_call_is_already_well_formed() -> None:
    assert fold_hoisted_arguments(_HUB_SCHEMA, {"action": "x", "params": {"a": 1}}) == (
        None,
        None,
    )
    assert fold_hoisted_arguments(_HUB_SCHEMA, {"action": "x"}) == (None, None)


def test_open_schema_is_left_alone() -> None:
    # Undeclared root keys are legal there — there is nothing to repair, and
    # moving them would change what the tool receives.
    open_hub = {**_HUB_SCHEMA}
    del open_hub["additionalProperties"]
    assert fold_hoisted_arguments(open_hub, {"action": "x", "board_id": "1"}) == (None, None)


def test_tool_without_a_params_property_is_left_to_the_validator() -> None:
    # A genuinely unknown argument on a flat tool must still be refused.
    assert fold_hoisted_arguments(_CLOSED_SCHEMA, {"query": "x", "web_search_depth": "2"}) == (
        None,
        None,
    )
    assert validate_tool_arguments(_CLOSED_SCHEMA, {"query": "x", "web_search_depth": "2"})


def test_params_declared_as_a_non_object_is_not_a_container() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"action": {"type": "string"}, "params": {"type": "string"}},
    }
    assert fold_hoisted_arguments(schema, {"action": "x", "a": 1}) == (None, None)


# Not a hub tool: ``params`` is an ordinary options object with a shape of its
# own. It only shares the name.
_OPTIONS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "params": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"width": {"type": "integer"}, "height": {"type": "integer"}},
        },
    },
}


def test_params_declaring_its_own_shape_is_not_a_hub_container() -> None:
    # A hub container declares no properties — its shape varies with ``action``.
    # One that declares them is an options object, so the call is left alone.
    assert fold_hoisted_arguments(_OPTIONS_SCHEMA, {"title": "Q3", "width": 800}) == (None, None)


def test_a_misspelt_root_property_is_named_rather_than_buried() -> None:
    # The reason for the check above. Folding would produce
    # ``{"params": {"titel": "Q3"}}``, which passes validation — nothing
    # recurses into ``params`` — handing the tool a bogus key under a missing
    # title. Refusing names the mistake the model can actually correct.
    call = {"titel": "Q3"}
    assert fold_hoisted_arguments(_OPTIONS_SCHEMA, call) == (None, None)
    err = validate_tool_arguments(_OPTIONS_SCHEMA, call)
    assert err is not None
    assert "titel" in err and "title" in err


def test_an_empty_properties_declaration_is_still_free_form() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string"},
            "params": {"type": "object", "properties": {}},
        },
    }
    folded, err = fold_hoisted_arguments(schema, {"action": "x", "a": 1})
    assert err is None
    assert folded == {"action": "x", "params": {"a": 1}}


def test_params_supplied_as_a_non_object_is_left_to_the_type_check() -> None:
    folded, err = fold_hoisted_arguments(_HUB_SCHEMA, {"action": "x", "params": "a=1", "b": 2})
    assert (folded, err) == (None, None)
    type_error = validate_tool_arguments(_HUB_SCHEMA, {"action": "x", "params": "a=1", "b": 2})
    assert type_error is not None
    assert "object" in type_error


# -- Primitives a server quoted ----------------------------------------------


def test_a_quoted_primitive_is_read_as_its_declared_type() -> None:
    """A server that types a call by the declared schema leaves every value of
    an undeclared tool a string: the reading puts the literal back."""
    arguments = {"city": "Laval", "days": "0", "ratio": " 0.5 ", "metric": "True"}

    unquoted, names = unquote_primitive_arguments(_SCHEMA, arguments)

    assert unquoted == {"city": "Laval", "days": 0, "ratio": 0.5, "metric": True}
    assert names == ("days", "ratio", "metric")
    assert validate_tool_arguments(_SCHEMA, unquoted) is None


def test_what_does_not_spell_the_literal_is_left_for_the_validator() -> None:
    """Guessing is worse than a refusal: only a value spelling the declared
    type's literal exactly is read."""
    for key, sent in (
        ("days", "top"),
        ("days", "1.5"),
        ("days", "1_000"),
        ("ratio", "nan"),
        ("ratio", "1e999"),
        ("metric", "yes"),
        ("metric", "1"),
    ):
        arguments = {"city": "Laval", "days": 1, key: sent}
        unquoted, names = unquote_primitive_arguments(_SCHEMA, arguments)
        assert (unquoted, names) == (arguments, ()), key
        assert validate_tool_arguments(_SCHEMA, unquoted) is not None


def test_a_string_property_keeps_its_digits() -> None:
    arguments = {"city": "3", "days": 1}
    assert unquote_primitive_arguments(_SCHEMA, arguments) == (arguments, ())


def test_the_repair_folds_a_hub_call_then_reads_its_root_primitives() -> None:
    schema = {
        **_HUB_SCHEMA,
        "properties": {**_HUB_SCHEMA["properties"], "limit": {"type": "integer"}},
    }

    repair = repair_tool_arguments(
        schema, {"action": "list_columns", "board_id": "board-1", "limit": "5"}
    )

    assert repair.error is None
    assert repair.arguments == {
        "action": "list_columns",
        "limit": 5,
        "params": {"board_id": "board-1"},
    }
    assert (repair.folded, repair.unquoted) == (("board_id",), ("limit",))


def test_an_ambiguous_hub_call_is_refused_unrepaired() -> None:
    arguments = {"action": "x", "board_id": "b", "params": {"a": 1}}

    repair = repair_tool_arguments(_HUB_SCHEMA, arguments)

    assert repair.error is not None and repair.arguments is arguments
