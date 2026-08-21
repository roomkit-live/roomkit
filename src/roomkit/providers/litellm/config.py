"""LiteLLM proxy provider configuration."""

from __future__ import annotations

from pydantic import SecretStr

from roomkit.providers.openai.config import OpenAIConfig


class LiteLLMConfig(OpenAIConfig):
    """LiteLLM proxy (AI gateway) provider configuration.

    A LiteLLM proxy fronts 100+ upstream providers behind one OpenAI-compatible
    endpoint, adding virtual keys, per-key budgets, and central routing — the
    self-hosted gateway pattern. It speaks the OpenAI Chat Completions API
    verbatim, so this **subclasses** :class:`OpenAIConfig` and inherits every
    request field (``temperature``, ``reasoning_effort``,
    ``include_stream_usage``, ``use_max_completion_tokens``, ``extra_body`` …).
    Inheriting — rather than re-declaring those fields — keeps the two configs
    from drifting apart: any field the inherited
    :class:`~roomkit.providers.openai.ai.OpenAIAIProvider` reads is guaranteed
    to exist here.

    Only the gateway's endpoint and key semantics differ.
    """

    base_url: str = "http://localhost:4000"
    """Base URL of the LiteLLM proxy. The default is the proxy's own default
    port; point it at your deployment. With or without a ``/v1`` suffix — the
    proxy mounts the API at both roots."""

    api_key: SecretStr
    """LiteLLM virtual key or master key (``sk-...``), sent as
    ``Authorization: Bearer <key>``. Required because a gateway's whole point
    is per-key auth and budgets; for a dev proxy running without
    authentication, pass any placeholder."""

    model: str
    """Public model name as the proxy configures it — the ``model_name`` from
    the proxy's ``config.yaml``, often an alias like ``"gpt-5.5"`` or
    ``"claude-sonnet"``. Required (which alias routes where is the deployment's
    decision, not roomkit's). Browse the deployed set with
    :meth:`~roomkit.providers.litellm.ai.LiteLLMAIProvider.list_models`."""
