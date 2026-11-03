"""How a provider declares a turn's tools (RFC §6.7, RMK-309).

A tool without parameters is declared as an object with none, which every
vendor accepts; a name no vendor accepts is refused when the tool is defined,
and a name one vendor refuses is refused by its provider before the request.
The vendor rules are the ones measured on the wire on 2026-10-02.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types as genai_types
from pydantic import ValidationError

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, ProviderError
from roomkit.providers.ai.tool_declaration import (
    ToolNameRule,
    chat_tool_declarations,
    declared_parameters,
)
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.anthropic.request import build_kwargs
from roomkit.providers.deepgram.settings import format_functions
from roomkit.providers.gemini.realtime_config import _live_tools
from roomkit.providers.gemini.realtime_models import live_model_profile
from roomkit.providers.gemini.schema import function_declaration
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.openai.live_events import format_backend_tools
from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.tools.mcp import _definition

NO_PARAMETERS = {"type": "object", "properties": {}}
_ANTHROPIC = AnthropicConfig(api_key="k", model="claude-haiku-4-5")


def _context(*names: str, parameters: dict[str, Any] | None = None) -> AIContext:
    return AIContext(
        messages=[AIMessage(role="user", content="hi")],
        tools=[AITool(name=name, description="d", parameters=parameters or {}) for name in names],
    )


class TestDeclaredParameters:
    @pytest.mark.parametrize("parameters", [{}, None])
    def test_a_tool_without_parameters_is_an_object_with_none(self, parameters: Any) -> None:
        assert declared_parameters(parameters) == NO_PARAMETERS

    def test_a_tool_keeps_its_own_schema(self) -> None:
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        assert declared_parameters(schema) == schema

    @pytest.mark.parametrize(
        "schema",
        [{"properties": {"q": {"type": "string"}}}, {"type": None, "properties": {}}],
        ids=["absent", "null"],
    )
    def test_a_root_without_a_type_is_an_objects(self, schema: dict[str, Any]) -> None:
        """Anthropic and OpenAI refuse it untyped (RMK-398, measured)."""
        assert declared_parameters(schema) == {**schema, "type": "object"}

    def test_the_chat_declaration_carries_it(self) -> None:
        [declaration] = chat_tool_declarations([AITool(name="now", description="d")])
        assert declaration == {
            "type": "function",
            "function": {"name": "now", "description": "d", "parameters": NO_PARAMETERS},
        }


class TestToolNameAnyVendorAccepts:
    @pytest.mark.parametrize("name", ["lookup", "files.read", "ns:tool", "mcp__s__t", "1abc"])
    def test_a_name_some_vendor_accepts_is_kept(self, name: str) -> None:
        assert AITool(name=name, description="d").name == name

    @pytest.mark.parametrize("name", ["", "a b", "a/b", "é"])
    def test_a_name_no_vendor_accepts_is_refused(self, name: str) -> None:
        with pytest.raises(ValidationError, match="accepted by no provider"):
            AITool(name=name, description="d")

    def test_a_rule_names_the_tool_and_itself(self) -> None:
        rule = ToolNameRule("acme", r"[a-z]+")
        with pytest.raises(ProviderError) as raised:
            rule.check(["ok", "Not.Ok"])
        assert str(raised.value) == "tool 'Not.Ok': acme accepts tool names matching [a-z]+"
        assert raised.value.retryable is False
        assert raised.value.context_overflow is False


def _openai(**config: Any) -> Any:
    module = MagicMock()
    module.APIStatusError = type("APIStatusError", (Exception,), {})
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    with patch.dict(sys.modules, {"openai": module}):
        from roomkit.providers.openai.ai import OpenAIAIProvider

        provider = OpenAIAIProvider(OpenAIConfig(api_key="k", model="gpt-4.1-mini", **config))
    provider._client = MagicMock()
    provider._client.chat.completions.create = AsyncMock()
    return provider


class TestOpenAI:
    async def test_a_name_openai_refuses_fails_before_the_request(self) -> None:
        provider = _openai()

        with pytest.raises(ProviderError, match="tool 'files.read': openai accepts"):
            await provider.generate(_context("files.read"))

        provider._client.chat.completions.create.assert_not_awaited()

    async def test_a_server_behind_a_base_url_decides_its_names(self) -> None:
        provider = _openai(base_url="http://localhost:8000/v1")

        assert provider._declare_tools(_context("files.read").tools)[0]["function"]["name"] == (
            "files.read"
        )

    def test_a_tool_without_parameters_is_declared_as_an_object(self) -> None:
        [declaration] = _openai()._declare_tools(_context("now").tools)
        assert declaration["function"]["parameters"] == NO_PARAMETERS


class TestAnthropic:
    def test_a_name_anthropic_refuses_fails_before_the_request(self) -> None:
        with pytest.raises(ProviderError, match="tool 'files.read': anthropic accepts"):
            build_kwargs(_ANTHROPIC, _context("files.read"))

    def test_a_tool_without_parameters_is_declared_as_an_object(self) -> None:
        kwargs = build_kwargs(_ANTHROPIC, _context("now"))
        assert kwargs["tools"][0]["input_schema"] == NO_PARAMETERS


class TestGemini:
    def test_a_dotted_name_is_declared(self) -> None:
        declaration = function_declaration(
            genai_types, name="files.read", description="d", parameters=None
        )
        assert declaration.name == "files.read"

    def test_a_name_gemini_refuses_fails_before_the_request(self) -> None:
        with pytest.raises(ProviderError, match="tool '1abc': gemini accepts"):
            function_declaration(genai_types, name="1abc", description="d", parameters=None)

    @pytest.mark.parametrize("parameters", [{}, None])
    def test_a_tool_without_parameters_is_declared_as_an_object(self, parameters: Any) -> None:
        declaration = function_declaration(
            genai_types, name="now", description="d", parameters=parameters
        )
        assert declaration.parameters is not None
        assert declaration.parameters.properties == {}


def _mistral() -> Any:
    module = MagicMock()
    with patch.dict(sys.modules, {"mistralai": module, "mistralai.client": module}):
        from roomkit.providers.mistral.ai import MistralAIProvider

        return MistralAIProvider(MistralConfig(api_key="k"))


class TestMistral:
    def test_a_dotted_name_is_declared(self) -> None:
        kwargs = _mistral()._build_kwargs(_context("files.read"))
        assert kwargs["tools"][0]["function"]["name"] == "files.read"

    def test_a_name_mistral_refuses_fails_before_the_request(self) -> None:
        with pytest.raises(ProviderError, match="tool 'ns:tool': mistral accepts"):
            _mistral()._build_kwargs(_context("ns:tool"))


class TestRealtimeDeclarations:
    def test_the_live_api_declares_a_tool_without_parameters_as_an_object(self) -> None:
        [tool] = format_backend_tools([{"name": "now", "description": "d"}])
        assert tool["parameters"] == NO_PARAMETERS

    def test_deepgram_declares_a_tool_without_parameters_as_an_object(self) -> None:
        [function] = format_functions([{"name": "now", "description": "d", "parameters": {}}])
        assert function["parameters"] == NO_PARAMETERS

    def test_a_realtime_root_without_a_type_is_an_objects(self) -> None:
        untyped = {"properties": {"q": {"type": "string"}}}
        [tool] = format_backend_tools(
            [{"name": "find", "description": "d", "parameters": untyped}]
        )
        [function] = format_functions(
            [{"name": "find", "description": "d", "parameters": untyped}]
        )
        assert tool["parameters"]["type"] == function["parameters"]["type"] == "object"


class TestMCP:
    def test_a_tool_no_provider_accepts_is_skipped_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        listed = SimpleNamespace(name="read file", description="d", inputSchema={}, meta=None)

        assert _definition(listed) is None
        assert "'read file' skipped" in caplog.text

    def test_a_dotted_mcp_tool_is_kept(self) -> None:
        listed = SimpleNamespace(name="files.read", description="d", inputSchema={}, meta=None)

        definition = _definition(listed)

        assert definition is not None and definition.name == "files.read"


class TestAServerBehindAURLDecides:
    def test_anthropic_behind_a_base_url_checks_no_name(self) -> None:
        config = AnthropicConfig(api_key="k", model="claude-haiku-4-5", base_url="http://gw")
        kwargs = build_kwargs(config, _context("files.read"))
        assert kwargs["tools"][0]["name"] == "files.read"

    def test_mistral_behind_a_server_url_checks_no_name(self) -> None:
        module = MagicMock()
        with patch.dict(sys.modules, {"mistralai": module, "mistralai.client": module}):
            from roomkit.providers.mistral.ai import MistralAIProvider

            provider = MistralAIProvider(MistralConfig(api_key="k", server_url="http://gw"))
        kwargs = provider._build_kwargs(_context("ns:tool"))
        assert kwargs["tools"][0]["function"]["name"] == "ns:tool"


class TestEveryDeclarationDeclaresAnEmptyObject:
    def test_gemini_live(self) -> None:
        model = "gemini-live-2.5-flash-preview"
        profile = live_model_profile(model)
        [tool] = _live_tools(genai_types, model, profile, [{"name": "now"}], set())
        assert tool.function_declarations[0].parameters.properties == {}

    def test_polargrid_and_ollama(self) -> None:
        tools = [AITool(name="now", description="d")]
        polargrid = PolarGridAIProvider._build_tools(SimpleNamespace(), tools)
        ollama = OllamaAIProvider._build_tools(SimpleNamespace(), tools)
        assert polargrid == ollama == chat_tool_declarations(tools)
        assert polargrid[0]["function"]["parameters"] == NO_PARAMETERS


def test_an_mcp_tool_rejected_for_another_reason_is_not_blamed_on_its_name() -> None:
    listed = SimpleNamespace(name="lookup", description="d", inputSchema="not-a-schema", meta=None)

    with pytest.raises(ValidationError, match="parameters"):
        _definition(listed)
