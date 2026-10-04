"""A vendor's official URL written out is the vendor's own endpoint (RMK-484,
RFC §6.7).

Every provider that applies its vendor's rules on the vendor's endpoint only
(the tool-name rule, the catalogue's profile, a modern model's defaults)
reads that endpoint the same way: no ``base_url``, or one of the vendor's
official URLs, whatever its case of scheme and host and its trailing slash.
Any other URL is a server that decides for itself.
"""

from __future__ import annotations

import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, AITool, ProviderError
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.anthropic.request import build_kwargs
from roomkit.providers.deepseek.ai import DeepSeekAIProvider
from roomkit.providers.deepseek.config import DeepSeekConfig
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from roomkit.providers.openai.live import OpenAILiveProvider
from roomkit.providers.openai.live_config import HostedReasoning
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.providers.vendor_endpoint import is_vendor_endpoint

DOTTED = AITool(name="files.read", description="d")
COLON = AITool(name="files:read", description="d")
PROXY = "https://gateway.example.test/v1"


def _openai(base_url: str | None) -> OpenAIAIProvider:
    return OpenAIAIProvider(OpenAIConfig(api_key="k", model="gpt-6-luna", base_url=base_url))


def _anthropic(base_url: str | None) -> AnthropicConfig:
    return AnthropicConfig(api_key="k", model="claude-sonnet-5-5", base_url=base_url)


def _realtime(base_url: str | None) -> OpenAIRealtimeProvider:
    return OpenAIRealtimeProvider(api_key="k", base_url=base_url)


def _live(base_url: str | None) -> OpenAILiveProvider:
    return OpenAILiveProvider(
        api_key="k", base_url=base_url, delegation=HostedReasoning(model="gpt-5.6-terra")
    )


def _deepseek(base_url: str) -> DeepSeekAIProvider:
    config = DeepSeekConfig(api_key="k", model="deepseek-chat", base_url=base_url)
    return DeepSeekAIProvider(config)


def _mistral_names_checked(server_url: str | None) -> bool:
    config = MistralConfig(api_key="k", model="mistral-large-latest", server_url=server_url)
    context = AIContext(messages=[AIMessage(role="user", content="go")], tools=[COLON])
    try:
        MistralAIProvider(config)._build_kwargs(context)
    except ProviderError:
        return True
    return False


def _anthropic_names_checked(base_url: str | None) -> bool:
    context = AIContext(messages=[AIMessage(role="user", content="go")], tools=[DOTTED])
    try:
        build_kwargs(_anthropic(base_url), context)
    except ProviderError:
        return True
    return False


OWN = {
    "openai-text": (
        lambda url: _openai(url)._tool_name_rule is not None,
        [None, "https://api.openai.com/v1", "https://API.openai.com/v1/"],
    ),
    "openai-text-profile": (
        lambda url: _openai(url)._is_openai_endpoint,
        [None, "https://api.openai.com/v1"],
    ),
    "anthropic-names": (
        _anthropic_names_checked,
        [None, "https://api.anthropic.com", "https://api.anthropic.com/v1/messages"],
    ),
    "anthropic-defaults": (
        lambda url: _anthropic(url).use_adaptive_thinking,
        [None, "https://api.anthropic.com/"],
    ),
    "anthropic-deferral": (
        lambda url: AnthropicAIProvider(_anthropic(url)).supports_deferred_tools,
        [None, "https://api.anthropic.com"],
    ),
    "openai-realtime": (
        lambda url: _realtime(url)._tool_name_rule is not None,
        [None, "wss://api.openai.com/v1/realtime", "wss://api.openai.com/v1/realtime/"],
    ),
    "gpt-live": (
        lambda url: _live(url)._tool_name_rule is not None,
        [None, "wss://api.openai.com/v1/live/sessions"],
    ),
    "mistral": (
        _mistral_names_checked,
        [None, "https://api.mistral.ai", "https://api.mistral.ai:443/"],
    ),
    "deepseek": (
        lambda url: _deepseek(url)._tool_name_rule is not None,
        ["https://api.deepseek.com/v1", "https://api.deepseek.com"],
    ),
}

CASES = [(site, url) for site, (_, urls) in OWN.items() for url in urls]


@pytest.mark.parametrize(("site", "base_url"), CASES)
def test_the_vendor_s_url_written_out_gets_the_vendor_s_rules(
    site: str, base_url: str | None
) -> None:
    applies, _ = OWN[site]

    assert applies(base_url)


@pytest.mark.parametrize("site", list(OWN))
def test_another_server_decides_for_itself(site: str) -> None:
    applies, urls = OWN[site]
    scheme = "wss" if any(url and url.startswith("wss") for url in urls) else "https"

    assert not applies(PROXY.replace("https", scheme))


def test_the_url_is_compared_on_its_scheme_host_and_path() -> None:
    assert is_vendor_endpoint(" HTTPS://Api.OpenAI.com/v1/ ", "https://api.openai.com/v1")
    assert not is_vendor_endpoint("https://api.openai.com/v2", "https://api.openai.com/v1")
    assert not is_vendor_endpoint("http://api.openai.com/v1", "https://api.openai.com/v1")
    assert is_vendor_endpoint("https://api.openai.com:443/v1", "https://api.openai.com/v1")
    assert not is_vendor_endpoint("https://api.openai.com:8443/v1", "https://api.openai.com/v1")
    assert not is_vendor_endpoint("https://u:p@api.openai.com/v1", "https://api.openai.com/v1")
    assert not is_vendor_endpoint("https://api.openai.com:x/v1", "https://api.openai.com/v1")
