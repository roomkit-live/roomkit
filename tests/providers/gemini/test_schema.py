"""Tests for Gemini JSON Schema cleaning."""

from __future__ import annotations

from roomkit.providers.gemini.schema import clean_gemini_schema


class TestCleanGeminiSchema:
    def test_strips_unsupported_fields(self) -> None:
        """Should remove $schema, additionalProperties, default, title."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "SearchArgs",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "title": "Query",
                    "default": "",
                },
            },
            "required": ["query"],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert "$schema" not in cleaned
        assert "title" not in cleaned
        assert "additionalProperties" not in cleaned
        assert cleaned["type"] == "object"
        assert cleaned["required"] == ["query"]
        # Nested property should also be cleaned
        assert "title" not in cleaned["properties"]["query"]
        assert "default" not in cleaned["properties"]["query"]
        assert cleaned["properties"]["query"]["type"] == "string"

    def test_preserves_valid_keys(self) -> None:
        """Valid Gemini schema keys should be preserved."""
        schema = {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of results",
                    "minimum": 1,
                    "maximum": 100,
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["count"],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == schema  # unchanged — all keys are valid

    def test_nested_schemas(self) -> None:
        """Should recursively clean nested property schemas."""
        schema = {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "title": "Filter",
                    "additionalProperties": True,
                    "properties": {
                        "field": {
                            "type": "string",
                            "default": "name",
                            "title": "Field",
                        },
                    },
                },
            },
        }
        cleaned = clean_gemini_schema(schema)
        assert "title" not in cleaned["properties"]["filter"]
        assert "additionalProperties" not in cleaned["properties"]["filter"]
        nested = cleaned["properties"]["filter"]["properties"]["field"]
        assert "title" not in nested
        assert "default" not in nested
        assert nested["type"] == "string"

    def test_items_cleaned(self) -> None:
        """Array items schema should also be cleaned."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "title": "Item",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "default": ""},
                },
            },
        }
        cleaned = clean_gemini_schema(schema)
        assert "title" not in cleaned["items"]
        assert "additionalProperties" not in cleaned["items"]

    def test_none_input(self) -> None:
        """None input should return None."""
        assert clean_gemini_schema(None) is None

    def test_empty_schema(self) -> None:
        """Empty dict should return empty dict."""
        assert clean_gemini_schema({}) == {}


class TestUnionCollapse:
    """Pydantic / OpenAPI union shapes collapse to a single nullable branch.

    Without this collapse, ``anyOf`` / ``oneOf`` / ``allOf`` would be
    stripped by the unknown-key pass and the property would emerge
    typeless — Gemini silently refuses to invoke such tools.
    """

    def test_pydantic_optional_to_nullable(self) -> None:
        """``Optional[str]`` from Pydantic becomes ``{type: string, nullable: True}``."""
        schema = {
            "type": "object",
            "properties": {
                "note": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Optional note",
                },
            },
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        note = cleaned["properties"]["note"]
        assert note["type"] == "string"
        assert note["nullable"] is True
        assert note["description"] == "Optional note"
        assert "anyOf" not in note

    def test_one_of_handled_like_any_of(self) -> None:
        schema = {
            "oneOf": [{"type": "integer"}, {"type": "null"}],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {"type": "integer", "nullable": True}

    def test_all_of_handled_like_any_of(self) -> None:
        schema = {
            "allOf": [{"type": "boolean"}, {"type": "null"}],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {"type": "boolean", "nullable": True}

    def test_wider_union_keeps_first_non_null(self) -> None:
        """Multiple non-null branches → keep first; nullable=True if any null."""
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
                {"type": "null"},
            ],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert cleaned["type"] == "string"
        assert cleaned["nullable"] is True

    def test_union_without_null_no_nullable(self) -> None:
        """Pure non-null union → first branch, no nullable flag."""
        schema = {
            "anyOf": [{"type": "string"}, {"type": "integer"}],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert cleaned["type"] == "string"
        assert "nullable" not in cleaned

    def test_pure_null_union_falls_back_to_string(self) -> None:
        """Degenerate {anyOf: [{type: null}]} stays valid."""
        schema = {"anyOf": [{"type": "null"}]}
        cleaned = clean_gemini_schema(schema)
        assert cleaned == {"type": "string", "nullable": True}

    def test_collapse_inside_array_items(self) -> None:
        """Pydantic Optional inside array items must also collapse."""
        schema = {
            "type": "array",
            "items": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
            },
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        items = cleaned["items"]
        assert items["type"] == "string"
        assert items["nullable"] is True

    def test_collapse_inside_nested_properties(self) -> None:
        """Pydantic Optional inside nested object properties must collapse."""
        schema = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "description": "An optional int",
                        },
                    },
                },
            },
        }
        cleaned = clean_gemini_schema(schema)
        value = cleaned["properties"]["inner"]["properties"]["value"]  # type: ignore[index]
        assert value["type"] == "integer"
        assert value["nullable"] is True
        assert value["description"] == "An optional int"

    def test_description_preserved_when_collapsing(self) -> None:
        """Parent-level description must survive the collapse."""
        schema = {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "A field description.",
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert cleaned["description"] == "A field description."

    def test_branch_description_preserved(self) -> None:
        """A non-null branch's own description survives."""
        schema = {
            "anyOf": [
                {"type": "string", "description": "branch desc"},
                {"type": "null"},
            ],
        }
        cleaned = clean_gemini_schema(schema)
        assert cleaned is not None
        assert cleaned["description"] == "branch desc"

    def test_property_with_collapsed_optional_has_type(self) -> None:
        """Regression guard: properties never emerge typeless from a collapse."""
        schema = {
            "type": "object",
            "properties": {
                "field_a": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "field_b": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
            },
        }
        cleaned = clean_gemini_schema(schema)
        for prop in cleaned["properties"].values():  # type: ignore[union-attr]
            assert "type" in prop, f"Property emerged typeless: {prop}"
