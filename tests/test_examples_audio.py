"""Regression tests for local realtime example audio defaults."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from examples.shared.audio import build_denoiser


def test_openai_local_example_defaults_to_continuous_webrtc_ns() -> None:
    """Open-speaker OpenAI sessions must not silently fall back to no NS."""
    example = Path(__file__).parents[1] / "examples" / "realtime_voice_local_openai.py"
    tree = ast.parse(example.read_text())

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_denoiser"
    ]

    assert len(calls) == 1
    defaults = [keyword.value for keyword in calls[0].keywords if keyword.arg == "default"]
    assert len(defaults) == 1
    assert isinstance(defaults[0], ast.Constant)
    assert defaults[0].value == "webrtc"


def test_openai_local_example_guards_initial_provider_vad_input() -> None:
    """Open-speaker playback gives AEC time to converge before server VAD."""
    example = Path(__file__).parents[1] / "examples" / "realtime_voice_local_openai.py"
    tree = ast.parse(example.read_text())

    env_gets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "BARGE_IN_GUARD_MS"
    ]
    guard_configs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "InterruptionConfig"
    ]

    assert len(env_gets) == 1
    assert isinstance(env_gets[0].args[1], ast.Constant)
    assert env_gets[0].args[1].value == "600"
    assert len(guard_configs) == 1
    assert any(keyword.arg == "allow_during_first_ms" for keyword in guard_configs[0].keywords)


def test_webrtc_denoiser_factory_constructs_before_logging_enabled(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.delenv("DENOISE", raising=False)
    denoiser = MagicMock()
    constructor = MagicMock(return_value=denoiser)

    with (
        patch(
            "roomkit.voice.pipeline.denoiser.webrtc.WebRTCNoiseSuppressorProvider",
            constructor,
        ),
        caplog.at_level(logging.INFO, logger="examples.shared.audio"),
    ):
        result = build_denoiser(24000, default="webrtc")

    assert result is denoiser
    constructor.assert_called_once_with(sample_rate=24000)
    assert "Denoiser enabled (WebRTC NS)" in caplog.messages


def test_failed_denoiser_initialization_is_not_logged_as_enabled(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("DENOISE", "rnnoise")

    with (
        patch(
            "roomkit.voice.pipeline.denoiser.rnnoise.RNNoiseDenoiserProvider",
            side_effect=ImportError("librnnoise missing"),
        ),
        caplog.at_level(logging.INFO, logger="examples.shared.audio"),
    ):
        result = build_denoiser(24000)

    assert result is None
    assert "Denoiser enabled (RNNoise)" not in caplog.messages
    assert "RNNoise not installed — denoiser disabled" in caplog.messages
