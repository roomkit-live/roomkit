"""Unit tests for SkillActivationMemory — per-room skill activation state."""

from __future__ import annotations

import json
from pathlib import Path

from roomkit.channels._skill_activation import SkillActivationMemory
from roomkit.skills.registry import SkillRegistry


def _registry(tmp_path: Path, *names: str) -> SkillRegistry:
    for name in names:
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: A test skill\n---\nRules for {name}.",
            encoding="utf-8",
        )
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


def _call(name: str, result: str = '{"ok": true}') -> dict[str, object]:
    return {"name": "activate_skill", "arguments": {"name": name}, "result": result}


class TestActivate:
    def test_first_activation_is_new_then_not(self) -> None:
        mem = SkillActivationMemory()
        assert mem.activate("r1", "alpha") is True
        assert mem.activate("r1", "alpha") is False
        assert mem.is_active("r1", "alpha") is True

    def test_rooms_are_isolated(self) -> None:
        mem = SkillActivationMemory()
        mem.activate("r1", "alpha")
        assert mem.is_active("r2", "alpha") is False
        assert mem.activate("r2", "alpha") is True
        assert mem.active_names("r1") == {"alpha"}
        assert mem.active_names("r2") == {"alpha"}

    def test_missing_room_or_name_is_never_active(self) -> None:
        mem = SkillActivationMemory()
        assert mem.activate(None, "alpha") is False
        assert mem.activate("r1", "") is False
        assert mem.is_active(None, "alpha") is False
        assert mem.active_names(None) == set()

    def test_oldest_skill_is_evicted_past_the_cap(self) -> None:
        mem = SkillActivationMemory(max_active=2)
        mem.activate("r1", "alpha")
        mem.activate("r1", "beta")
        mem.activate("r1", "gamma")

        assert mem.active_names("r1") == {"beta", "gamma"}
        # Evicted, so the model gets the body back rather than a phantom rule.
        assert mem.activate("r1", "alpha") is True

    def test_reactivating_refreshes_recency(self) -> None:
        mem = SkillActivationMemory(max_active=2)
        mem.activate("r1", "alpha")
        mem.activate("r1", "beta")
        mem.activate("r1", "alpha")  # alpha is in use again
        mem.activate("r1", "gamma")

        # beta is the stale one now, not alpha
        assert mem.active_names("r1") == {"alpha", "gamma"}

    def test_rooms_are_capped_fifo(self) -> None:
        mem = SkillActivationMemory()
        for i in range(120):
            mem.activate(f"room-{i}", "alpha")
        assert mem.is_active("room-0", "alpha") is False
        assert mem.is_active("room-119", "alpha") is True


class TestRenderPrompt:
    def test_none_when_nothing_active(self, tmp_path: Path) -> None:
        registry = _registry(tmp_path, "alpha")
        mem = SkillActivationMemory()
        assert mem.render_prompt("r1", registry) is None
        assert mem.render_prompt(None, registry) is None

    def test_renders_bodies_in_activation_order(self, tmp_path: Path) -> None:
        registry = _registry(tmp_path, "alpha", "beta")
        mem = SkillActivationMemory()
        mem.activate("r1", "alpha")
        mem.activate("r1", "beta")

        block = mem.render_prompt("r1", registry)
        assert block is not None
        assert "Active skill instructions" in block
        assert "Rules for alpha." in block
        assert "Rules for beta." in block
        assert block.index("alpha") < block.index("beta")

    def test_skips_skills_the_registry_no_longer_serves(self, tmp_path: Path) -> None:
        registry = _registry(tmp_path, "alpha")
        mem = SkillActivationMemory()
        mem.activate("r1", "alpha")
        mem.activate("r1", "vanished")

        block = mem.render_prompt("r1", registry)
        assert block is not None
        assert "Rules for alpha." in block
        assert "vanished" not in block

    def test_none_when_no_active_skill_resolves(self, tmp_path: Path) -> None:
        registry = _registry(tmp_path, "alpha")
        mem = SkillActivationMemory()
        mem.activate("r1", "vanished")
        assert mem.render_prompt("r1", registry) is None


class TestHydration:
    def test_needs_hydration_until_seeded(self) -> None:
        mem = SkillActivationMemory()
        assert mem.needs_hydration("r1") is True
        mem.seed("r1", [])
        assert mem.needs_hydration("r1") is False

    def test_room_with_live_activations_is_not_hydrated(self) -> None:
        mem = SkillActivationMemory()
        mem.activate("r1", "alpha")
        assert mem.needs_hydration("r1") is False

    def test_seeds_successful_activations_only(self) -> None:
        mem = SkillActivationMemory()
        mem.seed(
            "r1",
            [
                _call("alpha"),
                _call("beta", result=json.dumps({"error": "Skill 'beta' not found"})),
                _call("gamma", result="Result too large (9000 tokens). Full output saved"),
                {"name": "find_tools", "arguments": {"query": "alpha"}, "result": "{}"},
            ],
        )
        # Only the call that actually put rules in front of the model counts.
        assert mem.active_names("r1") == {"alpha"}

    def test_seeded_skill_answers_as_already_active(self) -> None:
        mem = SkillActivationMemory()
        mem.seed("r1", [_call("alpha")])
        assert mem.activate("r1", "alpha") is False

    def test_seed_ignores_missing_room(self) -> None:
        mem = SkillActivationMemory()
        mem.seed(None, [_call("alpha")])
        assert mem.active_names(None) == set()
