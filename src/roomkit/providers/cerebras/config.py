"""Cerebras chat provider configuration."""

from __future__ import annotations

from typing import Literal

from roomkit.providers.openai.config import OpenAIConfig


class CerebrasConfig(OpenAIConfig):
    """Configuration for Cerebras's OpenAI-compatible Chat Completions API.

    Inherits connection settings, sampling, retries and ``extra_body`` from
    :class:`OpenAIConfig`. ``model`` is required; upgrading RoomKit never
    silently selects another model. Install with ``roomkit[cerebras]``.
    """

    base_url: str = "https://api.cerebras.ai/v1"
    """Cerebras endpoint. May be overridden for a dedicated endpoint or proxy."""

    use_max_completion_tokens: bool = True
    """The output cap includes both reasoning and final-answer tokens."""

    include_stream_usage: bool = False
    """Cerebras sends usage in the final chunk without ``stream_options``.
    Leave False for Cerebras; the inherited decoder still collects usage.
    """

    reasoning_effort: str | None = None
    """Reasoning effort, also sent on tool turns. Supported values depend on
    the model: GPT OSS accepts low/medium/high; Qwen 3.8 and Gemma 4 also
    accept none. A per-turn ``AIContext.reasoning_effort`` takes precedence.
    Token-based ``thinking_budget`` is not mapped to an effort level.
    """

    reasoning_format: Literal["parsed", "raw", "hidden", "none"] | None = "parsed"
    """Use parsed reasoning to keep thinking separate from user-visible text.
    Raw output cannot always be separated (GPT OSS concatenates it without
    delimiters). Supported formats vary by model; None omits the parameter.
    """

    clear_thinking: bool | None = None
    """Whether to remove historical reasoning before prompting Qwen 3.8.
    None leaves the server default. Only set for models supporting this field.
    """
