"""Whether a provider talks to its vendor's own endpoint (RFC §6.7).

A provider knows its vendor's rules (the tool names it accepts, the
capabilities its catalogue states, the defaults a modern model needs) on the
vendor's own endpoint only: behind another ``base_url`` the server decides.
The vendor's own endpoint is the SDK's default, or one of the vendor's official
URLs written out: a configuration naming ``https://api.openai.com/v1`` talks to
OpenAI, not to a proxy, and gets OpenAI's rules.
"""

from __future__ import annotations

from urllib.parse import urlsplit

OPENAI_BASE_URL = "https://api.openai.com/v1"
"""OpenAI's REST endpoint, the SDK's default."""

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
"""Anthropic's API, the SDK's default."""

DEEPSEEK_BASE_URLS = ("https://api.deepseek.com/v1", "https://api.deepseek.com")
"""DeepSeek's endpoint, under the two bases its documentation gives."""

MISTRAL_BASE_URL = "https://api.mistral.ai"
"""Mistral's API, the SDK's default server."""

_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def is_vendor_endpoint(base_url: str | None, *official: str) -> bool:
    """Whether *base_url* is the vendor's own endpoint: none given (the SDK's
    default), or one of its *official* URLs, whatever its case of scheme and
    host, its scheme's default port written out, and its trailing slash."""
    if base_url is None:
        return True
    return _comparable(base_url) in {_comparable(url) for url in official}


def _comparable(url: str) -> tuple[str, str, int | None, str]:
    """*url* reduced to what names an endpoint: scheme, host, port (none for
    the scheme's default) and path. Credentials in it make it another
    endpoint: the vendor's takes none there."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    try:
        port = parts.port
    except ValueError:  # a port that is no number: no vendor's endpoint
        return scheme, parts.netloc.lower(), -1, parts.path
    port = port if port != _DEFAULT_PORTS.get(scheme) else None
    host = parts.netloc.lower() if parts.username or parts.password else parts.hostname
    return scheme, host or "", port, parts.path.rstrip("/")
