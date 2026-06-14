"""Tests for the Gemini-on-Vertex provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from roomkit.providers.gemini.vertex import GeminiVertexConfig


def _mock_genai_module() -> MagicMock:
    """Return a MagicMock that behaves like the google.genai module."""
    mod = MagicMock()
    mod.types = MagicMock()
    return mod


def _genai_modules(mock_genai: MagicMock) -> dict[str, Any]:
    """Build the sys.modules patch dict so the google.genai imports resolve."""
    return {
        "google": MagicMock(genai=mock_genai),
        "google.genai": mock_genai,
    }


def _vconfig(**overrides: Any) -> GeminiVertexConfig:
    defaults: dict[str, Any] = {
        "project": "my-proj",
        "location": "northamerica-northeast1",
    }
    defaults.update(overrides)
    return GeminiVertexConfig(**defaults)


class TestGeminiVertexConfig:
    def test_project_required(self) -> None:
        with pytest.raises(ValidationError):
            GeminiVertexConfig(location="us-central1")  # type: ignore[call-arg]

    def test_location_required(self) -> None:
        # Location is intentionally required (no default) to keep data in-region.
        with pytest.raises(ValidationError):
            GeminiVertexConfig(project="p")  # type: ignore[call-arg]

    def test_api_key_optional(self) -> None:
        # Vertex authenticates via ADC, not an API key.
        assert _vconfig().api_key is None

    def test_inherits_gemini_defaults(self) -> None:
        cfg = _vconfig()
        assert cfg.model == "gemini-3.1-flash-lite"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 1.0

    def test_custom_values(self) -> None:
        cfg = _vconfig(project="p2", location="europe-west1", model="gemini-3.5-flash")
        assert cfg.project == "p2"
        assert cfg.location == "europe-west1"
        assert cfg.model == "gemini-3.5-flash"


class TestGeminiVertexProvider:
    def test_client_built_in_vertex_mode(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.vertex import GeminiVertexProvider

            GeminiVertexProvider(_vconfig())

            mock_genai.Client.assert_called_once_with(
                vertexai=True,
                project="my-proj",
                location="northamerica-northeast1",
            )

    def test_no_api_key_passed_to_client(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.vertex import GeminiVertexProvider

            GeminiVertexProvider(_vconfig(api_key="should-be-ignored"))

            _, kwargs = mock_genai.Client.call_args
            assert "api_key" not in kwargs

    def test_inherits_gemini_provider(self) -> None:
        mock_genai = _mock_genai_module()
        with patch.dict("sys.modules", _genai_modules(mock_genai)):
            from roomkit.providers.gemini.vertex import GeminiVertexProvider

            provider = GeminiVertexProvider(_vconfig(model="gemini-3.5-flash"))
            # Subclasses GeminiAIProvider — name-based so it survives the
            # importlib.reload other Gemini tests do to the parent module.
            assert "GeminiAIProvider" in {c.__name__ for c in type(provider).__mro__}
            # Inherited behaviour works unchanged.
            assert provider.model_name == "gemini-3.5-flash"
            assert provider.supports_vision is True
