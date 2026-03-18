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
