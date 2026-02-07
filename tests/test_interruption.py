"""Tests for InterruptionHandler — all 4 strategies + backwards compat."""

from __future__ import annotations

from roomkit.voice.interruption import (
    InterruptionConfig,
    InterruptionHandler,
    InterruptionStrategy,
)
from roomkit.voice.pipeline.backchannel_detector import BackchannelDecision
from roomkit.voice.pipeline.mock import MockBackchannelDetector


class TestDisabledStrategy:
    def test_never_interrupts(self):
        config = InterruptionConfig(strategy=InterruptionStrategy.DISABLED)
        handler = InterruptionHandler(config)
        decision = handler.evaluate(playback_position_ms=5000, speech_duration_ms=3000)
        assert not decision.should_interrupt
        assert decision.reason == "interruptions disabled"


class TestImmediateStrategy:
    def test_always_interrupts(self):
        config = InterruptionConfig(strategy=InterruptionStrategy.IMMEDIATE)
        handler = InterruptionHandler(config)
        decision = handler.evaluate(playback_position_ms=500)
        assert decision.should_interrupt
        assert decision.reason == "immediate strategy"

    def test_respects_allow_during_first_ms(self):
        config = InterruptionConfig(
            strategy=InterruptionStrategy.IMMEDIATE,
            allow_during_first_ms=1000,
        )
        handler = InterruptionHandler(config)
        # Too early in playback
        decision = handler.evaluate(playback_position_ms=500)
        assert not decision.should_interrupt
        assert decision.reason == "playback too early"

        # After threshold
        decision = handler.evaluate(playback_position_ms=1500)
        assert decision.should_interrupt


class TestConfirmedStrategy:
    def test_interrupts_after_min_speech(self):
        config = InterruptionConfig(
            strategy=InterruptionStrategy.CONFIRMED,
            min_speech_ms=300,
        )
        handler = InterruptionHandler(config)
        decision = handler.evaluate(
            playback_position_ms=500, speech_duration_ms=400
        )
        assert decision.should_interrupt
        assert decision.reason == "speech confirmed"

    def test_no_interrupt_if_speech_too_short(self):
        config = InterruptionConfig(
            strategy=InterruptionStrategy.CONFIRMED,
            min_speech_ms=300,
        )
        handler = InterruptionHandler(config)
        decision = handler.evaluate(
            playback_position_ms=500, speech_duration_ms=100
        )
        assert not decision.should_interrupt
        assert decision.reason == "speech too short"

    def test_exact_threshold(self):
        config = InterruptionConfig(
            strategy=InterruptionStrategy.CONFIRMED,
            min_speech_ms=200,
        )
        handler = InterruptionHandler(config)
        decision = handler.evaluate(
            playback_position_ms=500, speech_duration_ms=200
        )
        assert decision.should_interrupt


class TestSemanticStrategy:
    def test_backchannel_detected(self):
        bc = MockBackchannelDetector(
            decisions=[BackchannelDecision(is_backchannel=True, confidence=0.95)]
        )
        config = InterruptionConfig(strategy=InterruptionStrategy.SEMANTIC)
        handler = InterruptionHandler(config, backchannel_detector=bc)

        decision = handler.evaluate(
            playback_position_ms=500,
            speech_duration_ms=300,
            speech_text="uh-huh",
        )
        assert not decision.should_interrupt
        assert decision.is_backchannel

    def test_not_backchannel_interrupts(self):
        bc = MockBackchannelDetector(
            decisions=[BackchannelDecision(is_backchannel=False)]
        )
        config = InterruptionConfig(strategy=InterruptionStrategy.SEMANTIC)
        handler = InterruptionHandler(config, backchannel_detector=bc)

        decision = handler.evaluate(
            playback_position_ms=500,
            speech_duration_ms=300,
            speech_text="Actually wait",
        )
        assert decision.should_interrupt
        assert not decision.is_backchannel

    def test_no_detector_falls_back_to_confirmed(self):
        config = InterruptionConfig(
            strategy=InterruptionStrategy.SEMANTIC,
            min_speech_ms=200,
        )
        handler = InterruptionHandler(config)  # No backchannel_detector

        # Long enough speech
        decision = handler.evaluate(
            playback_position_ms=500, speech_duration_ms=300
        )
        assert decision.should_interrupt
        assert "fallback" in decision.reason

        # Short speech
        decision = handler.evaluate(
            playback_position_ms=500, speech_duration_ms=100
        )
        assert not decision.should_interrupt


class TestBackwardsCompat:
    def test_legacy_enable_barge_in_true(self):
        """Legacy enable_barge_in=True maps to IMMEDIATE with allow_during_first_ms."""
        from roomkit.channels.voice import VoiceChannel

        channel = VoiceChannel(
            "ch1", enable_barge_in=True, barge_in_threshold_ms=300
        )
        assert channel._enable_barge_in is True
        assert (
            channel._interruption_handler.config.strategy
            == InterruptionStrategy.IMMEDIATE
        )
        assert channel._interruption_handler.config.allow_during_first_ms == 300

    def test_legacy_enable_barge_in_false(self):
        """Legacy enable_barge_in=False maps to DISABLED strategy."""
        from roomkit.channels.voice import VoiceChannel

        channel = VoiceChannel("ch1", enable_barge_in=False)
        assert channel._enable_barge_in is False
        assert (
            channel._interruption_handler.config.strategy
            == InterruptionStrategy.DISABLED
        )

    def test_explicit_interruption_overrides_legacy(self):
        """Explicit interruption param takes precedence over legacy."""
        from roomkit.channels.voice import VoiceChannel

        config = InterruptionConfig(
            strategy=InterruptionStrategy.IMMEDIATE,
            min_speech_ms=100,
        )
        channel = VoiceChannel(
            "ch1",
            enable_barge_in=False,  # Should be ignored
            interruption=config,
        )
        assert channel._enable_barge_in is True  # IMMEDIATE != DISABLED
        assert (
            channel._interruption_handler.config.strategy
            == InterruptionStrategy.IMMEDIATE
        )
