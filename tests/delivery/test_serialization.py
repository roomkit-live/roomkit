"""Tests for strategy serialization and deserialization."""

from __future__ import annotations

from roomkit.core.delivery import Immediate, Queued, WaitForIdle
from roomkit.delivery.serialization import deserialize_strategy, serialize_strategy


class TestSerializeStrategy:
    def test_none_returns_immediate(self) -> None:
        assert serialize_strategy(None) == {"type": "immediate", "params": {}}

    def test_immediate(self) -> None:
        result = serialize_strategy(Immediate())
        assert result == {"type": "immediate", "params": {}}

    def test_wait_for_idle_defaults(self) -> None:
        result = serialize_strategy(WaitForIdle())
        assert result == {
            "type": "wait_for_idle",
            "params": {"buffer": 1.0, "playback_timeout": 15.0},
        }

    def test_wait_for_idle_custom(self) -> None:
        result = serialize_strategy(WaitForIdle(buffer=3.0, playback_timeout=30.0))
        assert result == {
            "type": "wait_for_idle",
            "params": {"buffer": 3.0, "playback_timeout": 30.0},
        }

    def test_queued_defaults(self) -> None:
        result = serialize_strategy(Queued())
        assert result == {
            "type": "queued",
            "params": {"buffer": 1.0, "playback_timeout": 15.0, "separator": "\n\n"},
        }

    def test_queued_custom(self) -> None:
        result = serialize_strategy(Queued(buffer=2.0, separator=" | "))
        assert result["params"]["buffer"] == 2.0
        assert result["params"]["separator"] == " | "


class TestDeserializeStrategy:
    def test_immediate(self) -> None:
        s = deserialize_strategy({"type": "immediate", "params": {}})
        assert isinstance(s, Immediate)

    def test_wait_for_idle(self) -> None:
        s = deserialize_strategy(
            {"type": "wait_for_idle", "params": {"buffer": 2.5, "playback_timeout": 20.0}}
        )
        assert isinstance(s, WaitForIdle)
        assert s.buffer == 2.5
        assert s.playback_timeout == 20.0

    def test_queued(self) -> None:
        params = {"buffer": 1.0, "playback_timeout": 15.0, "separator": "---"}
        s = deserialize_strategy({"type": "queued", "params": params})
        assert isinstance(s, Queued)
        assert s.separator == "---"

    def test_unknown_type_falls_back(self) -> None:
        s = deserialize_strategy({"type": "unknown_strategy", "params": {}})
        assert isinstance(s, Immediate)

    def test_empty_dict(self) -> None:
        s = deserialize_strategy({})
        assert isinstance(s, Immediate)


class TestRoundtrip:
    def test_immediate_roundtrip(self) -> None:
        original = Immediate()
        restored = deserialize_strategy(serialize_strategy(original))
        assert isinstance(restored, Immediate)

    def test_wait_for_idle_roundtrip(self) -> None:
        original = WaitForIdle(buffer=5.0, playback_timeout=60.0)
        restored = deserialize_strategy(serialize_strategy(original))
        assert isinstance(restored, WaitForIdle)
        assert restored.buffer == 5.0
        assert restored.playback_timeout == 60.0

    def test_queued_roundtrip(self) -> None:
        original = Queued(buffer=2.0, separator=" // ")
        restored = deserialize_strategy(serialize_strategy(original))
        assert isinstance(restored, Queued)
        assert restored.buffer == 2.0
        assert restored.separator == " // "
