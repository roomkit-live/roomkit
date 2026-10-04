"""The turn's reasoning settings outrank the provider's configuration (RFC §6.7).

One table of every provider whose configuration carries a reasoning setting
under the name the turn uses, and of where each puts it on the request: a
provider that reads its configuration alone fails here. A vendor setting of
the provider's own (Ollama's ``think``, Gemini's ``thinking_level``,
PolarGrid's ``thinking``) yields to the turn on what the turn states.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, AITool
from roomkit.providers.ai.reasoning import nearest_level, thinking_switch, turn_setting
from roomkit.providers.anthropic import AnthropicConfig
from roomkit.providers.anthropic.request import build_kwargs
from roomkit.providers.cerebras.ai import CerebrasAIProvider
from roomkit.providers.cerebras.config import CerebrasConfig
from roomkit.providers.deepseek.ai import DeepSeekAIProvider
from roomkit.providers.deepseek.config import DeepSeekConfig
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.litellm.ai import LiteLLMAIProvider
from roomkit.providers.litellm.config import LiteLLMConfig
from roomkit.providers.meta.ai import MetaAIProvider
from roomkit.providers.meta.config import MetaConfig
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.ollama.ai import OllamaAIProvider
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.openrouter.ai import OpenRouterAIProvider
from roomkit.providers.openrouter.config import OpenRouterConfig
from roomkit.providers.polargrid.ai import PolarGridAIProvider
from roomkit.providers.polargrid.config import PolarGridConfig
from roomkit.providers.qwen.ai import QwenAIProvider
from roomkit.providers.qwen.config import QwenConfig
from roomkit.providers.vllm import _openai_config, _VLLMProvider
from roomkit.providers.vllm.config import VLLMConfig
from roomkit.providers.xai.ai import XAIAIProvider
from roomkit.providers.xai.config import XAIConfig

_TOOLS = [AITool(name="lookup", description="x", parameters={})]


def _provider(cls: type, config: Any) -> Any:
    provider = cls.__new__(cls)
    provider._config = config
    return provider


def _sampled(provider: Any, context: AIContext) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    provider._apply_sampling_kwargs(kwargs, context)
    return kwargs


def _top_level(provider: Any, context: AIContext) -> Any:
    return _sampled(provider, context).get("reasoning_effort")


def _openrouter(provider: Any, context: AIContext) -> Any:
    return _sampled(provider, context).get("extra_body", {}).get("reasoning", {}).get("effort")


def _deepseek(provider: Any, context: AIContext) -> Any:
    thinking = _sampled(provider, context).get("extra_body", {}).get("thinking", {})
    return thinking.get("reasoning_effort")


def _mistral(provider: Any, context: AIContext) -> Any:
    return provider._resolve_reasoning_effort(context)


Effort = tuple[Callable[[str], Any], Callable[[Any, AIContext], Any]]


def _with(cls: type, config: type, **fields: Any) -> Callable[[str], Any]:
    """Build *cls* with a configured effort."""
    return lambda effort: _provider(cls, config(api_key="k", reasoning_effort=effort, **fields))


# Each provider built with a configured effort, and how to read the effort it sends.
_EFFORT: dict[str, Effort] = {
    "openai": (_with(OpenAIAIProvider, OpenAIConfig, model="gpt-5-mini"), _top_level),
    "openrouter": (_with(OpenRouterAIProvider, OpenRouterConfig, model="m"), _openrouter),
    "xai": (_with(XAIAIProvider, XAIConfig), _top_level),
    "cerebras": (_with(CerebrasAIProvider, CerebrasConfig, model="m"), _top_level),
    "meta": (_with(MetaAIProvider, MetaConfig), _top_level),
    "litellm": (_with(LiteLLMAIProvider, LiteLLMConfig, model="m"), _top_level),
    "deepseek": (_with(DeepSeekAIProvider, DeepSeekConfig, model="m"), _deepseek),
    "mistral": (_with(MistralAIProvider, MistralConfig), _mistral),
}

# Where a turn with tools does not carry an enabling effort, and why (RFC §6.7):
# LiteLLM cannot know the model behind an alias.
_NOT_ON_TOOL_TURNS = {"litellm"}


def _context(**fields: Any) -> AIContext:
    return AIContext(messages=[AIMessage(role="user", content="hi")], **fields)


def test_a_turn_value_outranks_the_configured_one_even_when_falsy() -> None:
    assert turn_setting(False, True) is False
    assert turn_setting(None, True) is True
    assert turn_setting("low", "high") == "low"
    assert turn_setting(None, None) is None


@pytest.mark.parametrize("name", sorted(_EFFORT))
def test_the_turn_effort_outranks_the_configured_one(name: str) -> None:
    build, read = _EFFORT[name]

    assert read(build("high"), _context(reasoning_effort="low")) == "low"
    assert read(build("high"), _context()) == "high"


@pytest.mark.parametrize("name", sorted(set(_EFFORT) - _NOT_ON_TOOL_TURNS))
def test_a_tool_turn_carries_the_effort_as_a_turn_without_does(name: str) -> None:
    build, read = _EFFORT[name]

    assert read(build("high"), _context(tools=_TOOLS, reasoning_effort="low")) == "low"


_SWITCHED_ON = {"api_key": "k", "model": "m", "enable_thinking": True}


@pytest.mark.parametrize(
    ("provider", "read"),
    [
        (
            _provider(DeepSeekAIProvider, DeepSeekConfig(**_SWITCHED_ON)),
            lambda kwargs: kwargs["extra_body"]["thinking"],
        ),
        (
            _provider(QwenAIProvider, QwenConfig(**_SWITCHED_ON)),
            lambda kwargs: kwargs["extra_body"]["enable_thinking"],
        ),
    ],
    ids=["deepseek", "qwen"],
)
def test_the_turn_switch_off_outranks_a_configured_switch_on(
    provider: Any, read: Callable[[dict[str, Any]], Any]
) -> None:
    sent = read(_sampled(provider, _context(tools=_TOOLS, enable_thinking=False)))

    assert sent in ({"type": "disabled"}, False)


@pytest.mark.parametrize(
    ("provider", "read"),
    [
        (
            _provider(DeepSeekAIProvider, DeepSeekConfig(**_SWITCHED_ON)),
            lambda kwargs: kwargs["extra_body"]["thinking"],
        ),
        (
            _provider(QwenAIProvider, QwenConfig(**_SWITCHED_ON)),
            lambda kwargs: kwargs["extra_body"]["enable_thinking"],
        ),
    ],
    ids=["deepseek", "qwen"],
)
def test_an_effort_of_none_turns_a_configured_switch_off(
    provider: Any, read: Callable[[dict[str, Any]], Any]
) -> None:
    sent = read(_sampled(provider, _context(tools=_TOOLS, reasoning_effort="none")))

    assert sent in ({"type": "disabled"}, False)


# A vendor setting of the provider's own yields to the turn on what the turn
# states, whether the model reasons or how much, and supplies the rest (RFC
# §6.7).


def _think(think: Any, **turn: Any) -> Any:
    provider = _provider(OllamaAIProvider, OllamaConfig(model="m", think=think))
    return provider._resolve_think(_context(tools=_TOOLS, **turn))


@pytest.mark.parametrize(
    ("think", "turn", "sent"),
    [
        ("high", {"thinking_budget": 4096}, "high"),
        ("high", {"reasoning_effort": "low"}, "low"),
        ("high", {"reasoning_effort": "xhigh"}, "high"),
        ("high", {"enable_thinking": False}, False),
        ("high", {"reasoning_effort": "none"}, False),
        (True, {"thinking_budget": 0}, False),
        (False, {"enable_thinking": True}, True),
        # A model with no configured level may take none: Ollama refuses a
        # level there, so the turn's is not sent.
        (None, {"reasoning_effort": "low"}, None),
        (None, {}, None),
    ],
)
def test_ollama_think_yields_to_the_turn(think: Any, turn: dict[str, Any], sent: Any) -> None:
    assert _think(think, **turn) == sent


def _gemini_thinking(model: str, level: str | None, **turn: Any) -> Any:
    genai_types = pytest.importorskip("google.genai.types")
    provider = _provider(
        GeminiAIProvider, GeminiConfig(api_key="k", model=model, thinking_level=level)
    )
    provider._types = genai_types
    return provider._build_gen_config(_context(tools=_TOOLS, **turn)).thinking_config


def test_gemini_thinking_level_yields_to_the_turn() -> None:
    off = _gemini_thinking("gemini-3.8-flash", "high", thinking_budget=0)
    kept = _gemini_thinking("gemini-3.8-flash", "high", thinking_budget=4096)
    low = _gemini_thinking("gemini-3.7-flash", None, reasoning_effort="low")

    assert (off.thinking_level, off.thinking_budget) == (None, 0)
    assert kept.thinking_level.value == "HIGH"
    assert low.thinking_level.value == "LOW"


def test_gemini_reads_the_levels_a_model_takes_from_the_catalogue() -> None:
    minimal = _gemini_thinking("gemini-3.5-flash", None, reasoning_effort="minimal")
    nearest = _gemini_thinking("gemini-3.8-flash", None, reasoning_effort="minimal")

    assert minimal.thinking_level.value == "MINIMAL"
    assert nearest.thinking_level.value == "LOW"
    # Gemini 2.5 refuses a level: the turn's is not sent.
    assert _gemini_thinking("gemini-2.5-flash", None, reasoning_effort="low") is None


@pytest.mark.parametrize(
    ("configured", "turn", "sent"),
    [
        (True, {"enable_thinking": False}, False),
        (True, {"thinking_budget": 0}, False),
        (False, {"thinking_budget": 2048}, True),
        (None, {"reasoning_effort": "none"}, False),
        (None, {"reasoning_effort": "high"}, None),
    ],
)
def test_polargrid_thinking_yields_to_the_turn(
    configured: bool | None, turn: dict[str, Any], sent: bool | None
) -> None:
    provider = _provider(PolarGridAIProvider, PolarGridConfig(api_key="k", thinking=configured))
    request = provider._build_request(_context(tools=_TOOLS, **turn), stream=False)

    assert request.get("enable_thinking") == sent


def test_the_switch_reads_the_budget_then_enable_thinking_then_an_effort_of_none() -> None:
    assert thinking_switch(_context(thinking_budget=0, enable_thinking=True)) is False
    assert thinking_switch(_context(thinking_budget=512, reasoning_effort="none")) is True
    assert thinking_switch(_context(enable_thinking=True, reasoning_effort="none")) is True
    assert thinking_switch(_context(reasoning_effort="none"), configured=True) is False
    assert thinking_switch(_context(thinking_budget=-1)) is False
    assert thinking_switch(_context(reasoning_effort="low"), configured=None) is None


def test_an_effort_goes_as_the_nearest_level_the_model_takes() -> None:
    three = ("low", "medium", "high")
    assert nearest_level("minimal", three) == "low"
    assert nearest_level("xhigh", three) == "high"
    assert nearest_level("minimal", ("minimal", *three)) == "minimal"
    assert nearest_level("none", three) is None and nearest_level("max", three) is None
    assert nearest_level("low", ()) is None


def _off_gemini(provider: Any, context: AIContext) -> bool:
    genai_types = pytest.importorskip("google.genai.types")
    provider._types = genai_types
    return provider._build_gen_config(context).thinking_config.thinking_budget == 0


# Every implementation of "whether the model reasons, as the turn states it",
# each configured to reason, and what it sends when the turn says off.
_SWITCHES_OFF: dict[str, tuple[Any, Callable[[Any, AIContext], bool]]] = {
    "qwen": (
        _provider(QwenAIProvider, QwenConfig(**_SWITCHED_ON)),
        lambda p, c: _sampled(p, c)["extra_body"]["enable_thinking"] is False,
    ),
    "deepseek": (
        _provider(DeepSeekAIProvider, DeepSeekConfig(**_SWITCHED_ON)),
        lambda p, c: _sampled(p, c)["extra_body"]["thinking"] == {"type": "disabled"},
    ),
    "vllm": (
        _provider(_VLLMProvider, _openai_config(VLLMConfig(model="m", enable_thinking=True))),
        lambda p, c: (
            _sampled(p, c)["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        ),
    ),
    "ollama": (
        _provider(OllamaAIProvider, OllamaConfig(model="m", think="high")),
        lambda p, c: p._resolve_think(c) is False,
    ),
    "polargrid": (
        _provider(PolarGridAIProvider, PolarGridConfig(api_key="k", thinking=True)),
        lambda p, c: p._build_request(c, stream=False)["enable_thinking"] is False,
    ),
    "gemini": (
        _provider(
            GeminiAIProvider,
            GeminiConfig(api_key="k", model="gemini-3.8-flash", thinking_level="high"),
        ),
        _off_gemini,
    ),
    "mistral": (
        _provider(MistralAIProvider, MistralConfig(api_key="k", reasoning_effort="high")),
        lambda p, c: p._resolve_reasoning_effort(c) == "none",
    ),
    "openrouter": (
        _provider(
            OpenRouterAIProvider, OpenRouterConfig(api_key="k", model="m", reasoning_effort="high")
        ),
        lambda p, c: _sampled(p, c)["extra_body"]["reasoning"] == {"enabled": False},
    ),
    "litellm": (
        _provider(
            LiteLLMAIProvider, LiteLLMConfig(api_key="k", model="m", reasoning_effort="high")
        ),
        lambda p, c: "reasoning_effort" not in _sampled(p, c),
    ),
    "anthropic": (
        None,
        lambda _p, c: (
            "thinking"
            not in build_kwargs(AnthropicConfig(api_key="k", model="claude-opus-4-8"), c)
        ),
    ),
    # A model OpenAI's catalogue says reasons takes ``none``, its off value.
    "openai": (
        _provider(OpenAIAIProvider, OpenAIConfig(api_key="k", model="gpt-6-luna")),
        lambda p, c: _sampled(p, c)["reasoning_effort"] == "none",
    ),
    # Muse cannot stop reasoning: off asks for the least of it.
    "meta": (
        _provider(MetaAIProvider, MetaConfig(api_key="k", reasoning_effort="high")),
        lambda p, c: _sampled(p, c)["reasoning_effort"] == "minimal",
    ),
}


@pytest.mark.parametrize("name", sorted(_SWITCHES_OFF))
@pytest.mark.parametrize(
    "off",
    [{"thinking_budget": 0}, {"enable_thinking": False}, {"reasoning_effort": "none"}],
    ids=["budget_0", "enable_thinking_false", "effort_none"],
)
def test_the_turn_switch_off_reaches_every_implementation(name: str, off: dict[str, Any]) -> None:
    provider, is_off = _SWITCHES_OFF[name]

    assert is_off(provider, _context(**off))


@pytest.mark.parametrize(
    ("model", "level"),
    [("gemini-3.1-pro-preview", "LOW"), ("gemini-3.5-flash-lite", "MINIMAL")],
)
def test_gemini_sends_its_lowest_level_for_off_where_the_model_cannot_stop(
    model: str, level: str
) -> None:
    """A budget of 0 answers 400 on these two (measured 2026-09-27)."""
    thinking = _gemini_thinking(model, "high", enable_thinking=False)

    assert thinking.thinking_budget is None
    assert thinking.thinking_level.value == level


def test_anthropic_turns_adaptive_thinking_on_from_enable_thinking() -> None:
    """RFC §6.7: enable_thinking states the switch; a model without adaptive
    thinking needs its budget, so the switch alone leaves it off there."""
    adaptive = AnthropicConfig(api_key="k", model="claude-opus-4-8")
    budgeted = AnthropicConfig(api_key="k", model="claude-opus-4-8", use_adaptive_thinking=False)

    assert build_kwargs(adaptive, _context(enable_thinking=True))["thinking"]["type"] == "adaptive"
    assert "thinking" not in build_kwargs(budgeted, _context(enable_thinking=True))
    budget = build_kwargs(budgeted, _context(enable_thinking=True, thinking_budget=2048))
    assert budget["thinking"] == {"type": "enabled", "budget_tokens": 2048}


@pytest.mark.parametrize("off", [{"thinking_budget": 0}, {"enable_thinking": False}])
def test_a_model_that_does_not_reason_is_sent_no_off_value(off: dict[str, Any]) -> None:
    """``reasoning_effort`` is refused by a model that does not reason
    (gpt-4.1), and one behind a base_url is no model the catalogue knows."""
    plain = _provider(OpenAIAIProvider, OpenAIConfig(api_key="k", model="gpt-4.1"))
    proxied = _provider(
        OpenAIAIProvider,
        OpenAIConfig(api_key="k", model="gpt-6-luna", base_url="http://local.test/v1"),
    )

    assert "reasoning_effort" not in _sampled(plain, _context(**off))
    assert "reasoning_effort" not in _sampled(proxied, _context(**off))
