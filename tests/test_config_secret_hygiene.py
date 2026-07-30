"""A config's secret must not appear in its ``repr()`` (RFC §17.7).

Nothing in the library logs a config object today, so this is latent rather
than active — but a traceback renders every local it passes, and an operator
adding ``logger.debug("config=%s", config)`` should not thereby publish a key.
The repo already had the convention on both sides (``SecretStr`` for pydantic
models, ``field(repr=False)`` for dataclasses); these are the ones that had
been missed.
"""

from __future__ import annotations

import dataclasses

import pytest

from roomkit.conference.livekit import LiveKitConfig
from roomkit.providers.anam.config import AnamConfig
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.video.vision.gemini import GeminiVisionConfig
from roomkit.video.vision.openai import OpenAIVisionConfig
from roomkit.voice.pipeline.denoiser.aicoustics import AICousticsDenoiserConfig
from roomkit.voice.stt.deepgram import DeepgramConfig
from roomkit.voice.stt.gradium import GradiumSTTConfig
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig
from roomkit.voice.tts.gradium import GradiumTTSConfig
from roomkit.voice.tts.grok import GrokTTSConfig

_CANARY = "s3cr3t-canary-value"


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: DeepgramConfig(api_key=_CANARY), id="deepgram-stt"),
        pytest.param(lambda: GradiumSTTConfig(api_key=_CANARY), id="gradium-stt"),
        pytest.param(lambda: ElevenLabsConfig(api_key=_CANARY), id="elevenlabs-tts"),
        pytest.param(lambda: GrokTTSConfig(api_key=_CANARY), id="grok-tts"),
        pytest.param(lambda: GradiumTTSConfig(api_key=_CANARY), id="gradium-tts"),
        pytest.param(lambda: OpenAIVisionConfig(api_key=_CANARY), id="openai-vision"),
        pytest.param(lambda: GeminiVisionConfig(api_key=_CANARY), id="gemini-vision"),
        pytest.param(lambda: AICousticsDenoiserConfig(license_key=_CANARY), id="aicoustics"),
        pytest.param(
            lambda: LiveKitConfig(url="wss://x", api_key=_CANARY, api_secret=_CANARY),
            id="livekit",
        ),
    ],
)
def test_dataclass_config_hides_its_secret(factory) -> None:
    config = factory()
    assert _CANARY not in repr(config)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: AnamConfig(api_key=_CANARY), id="anam"),
        pytest.param(
            lambda: ElevenLabsRealtimeConfig(api_key=_CANARY, agent_id="a1"),
            id="elevenlabs-realtime",
        ),
    ],
)
def test_pydantic_config_hides_its_secret(factory) -> None:
    config = factory()
    assert _CANARY not in repr(config)
    assert _CANARY not in str(config)
    # Still reachable on purpose — redaction is about display, not access.
    assert config.api_key.get_secret_value() == _CANARY


def test_the_secret_is_still_usable_on_dataclasses() -> None:
    """repr=False hides the field from display; it does not remove it."""
    config = DeepgramConfig(api_key=_CANARY)
    assert config.api_key == _CANARY
    assert any(f.name == "api_key" for f in dataclasses.fields(config))
