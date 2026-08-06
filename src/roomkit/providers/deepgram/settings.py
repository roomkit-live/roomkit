"""Translation of RoomKit's session arguments into Deepgram's ``Settings`` payload.

Pure functions over a :class:`DeepgramAgentConfig` and the per-session
``provider_config`` dict: no sockets, no session state. The provider calls
:func:`build_settings` once at connect time, and :func:`build_think` /
:func:`build_speak` again when a live session is reconfigured.
"""

from __future__ import annotations

from typing import Any

from roomkit.providers.deepgram.config import DeepgramAgentConfig


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def format_functions(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Project RoomKit tool dicts to Deepgram's ``think.functions`` shape.

    Tool dicts reaching the provider carry extra keys the caller uses elsewhere
    (notably ``tags``, added for cross-lingual Tool Search); Deepgram rejects
    unknown fields, so only name/description/parameters survive. A function
    carrying an ``endpoint`` is executed by Deepgram server-side; without one it
    comes back as a ``client_side`` call for RoomKit to run.
    """
    functions: list[dict[str, Any]] = []
    for tool in tools or []:
        name = tool.get("name")
        if not name:
            continue
        function: dict[str, Any] = {"name": name, "description": tool.get("description", "")}
        if tool.get("parameters") is not None:
            function["parameters"] = tool["parameters"]
        if tool.get("endpoint"):
            function["endpoint"] = tool["endpoint"]
        functions.append(function)
    return functions


def build_think(
    cfg: DeepgramAgentConfig,
    *,
    system_prompt: str | None,
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    pc: dict[str, Any],
) -> dict[str, Any]:
    """Build the ``agent.think`` block — the LLM stage."""
    provider: dict[str, Any] = {
        "type": pc.get("think_provider", cfg.think_provider),
        "model": pc.get("think_model", cfg.think_model),
    }
    if temperature is not None:
        provider["temperature"] = temperature

    think: dict[str, Any] = {"provider": provider}
    if pc.get("think_endpoint"):
        think["endpoint"] = pc["think_endpoint"]
    if pc.get("context_length") is not None:
        think["context_length"] = pc["context_length"]
    if system_prompt:
        think["prompt"] = system_prompt
    functions = format_functions(tools)
    if functions:
        think["functions"] = functions
    return think


def build_speak(
    cfg: DeepgramAgentConfig, *, voice: str | None, pc: dict[str, Any]
) -> dict[str, Any]:
    """Build the ``agent.speak`` block — the TTS stage."""
    provider: dict[str, Any] = {
        "type": "deepgram",
        "model": voice or pc.get("speak_model", cfg.speak_model),
    }
    language = pc.get("speak_language", cfg.speak_language)
    if language:
        provider["language"] = language
    return {"provider": provider}


def build_listen(cfg: DeepgramAgentConfig, *, pc: dict[str, Any]) -> dict[str, Any]:
    """Build the ``agent.listen`` block — the speech-to-text stage."""
    provider: dict[str, Any] = {
        "type": "deepgram",
        "model": pc.get("listen_model", cfg.listen_model),
    }
    version = pc.get("listen_version", cfg.listen_version)
    if version:
        provider["version"] = version
    language = pc.get("listen_language", cfg.listen_language)
    if language:
        provider["language"] = language
    if pc.get("keyterms"):
        provider["keyterms"] = list(pc["keyterms"])
    if pc.get("smart_format") is not None:
        provider["smart_format"] = bool(pc["smart_format"])
    return {"provider": provider}


def build_settings(
    cfg: DeepgramAgentConfig,
    *,
    system_prompt: str | None,
    voice: str | None,
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    input_sample_rate: int,
    output_sample_rate: int,
    pc: dict[str, Any],
) -> dict[str, Any]:
    """Build the full ``Settings`` message sent once, right after connecting.

    ``pc["settings"]`` is deep-merged last, so an integrator can reach a field
    this module does not model without losing the rest of the payload.
    """
    output: dict[str, Any] = {
        "encoding": pc.get("output_encoding", "linear16"),
        "sample_rate": output_sample_rate,
        "container": pc.get("output_container", "none"),
    }
    if pc.get("output_bitrate") is not None:
        output["bitrate"] = int(pc["output_bitrate"])

    agent: dict[str, Any] = {
        "listen": build_listen(cfg, pc=pc),
        "think": build_think(
            cfg, system_prompt=system_prompt, tools=tools, temperature=temperature, pc=pc
        ),
        "speak": build_speak(cfg, voice=voice, pc=pc),
    }
    greeting = pc.get("greeting", cfg.greeting)
    if greeting:
        agent["greeting"] = greeting

    settings: dict[str, Any] = {
        "type": "Settings",
        "audio": {
            "input": {
                "encoding": pc.get("input_encoding", "linear16"),
                "sample_rate": input_sample_rate,
            },
            "output": output,
        },
        "agent": agent,
    }
    if pc.get("tags"):
        settings["tags"] = list(pc["tags"])
    if pc.get("settings"):
        settings = deep_merge(settings, pc["settings"])
    return settings
