"""Scripted registries discovered on disk or populated through public add."""

from __future__ import annotations

import json
import shutil
import tempfile
import weakref
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from roomkit.skills import Skill, SkillMetadata, SkillRegistry


def scripted_skill_registry(
    skills: list[Mapping[str, Any]], requires_match: str = "exact"
) -> SkillRegistry:
    """Keep each registry's files alive until that registry is collected."""
    root = Path(tempfile.mkdtemp(prefix="roomkit-conformance-skills-"))
    try:
        directories = [_write_skill(root, index, spec) for index, spec in enumerate(skills)]
        registry = _registry_with_matcher(requires_match)
        registry.discover(root)
        for spec, directory in zip(skills, directories, strict=True):
            if spec.get("in_memory", False):
                registry.add(_memory_skill(spec, directory))
        for spec in skills:
            if "unavailable" in spec:
                registry.mark_unavailable(spec["name"], spec["unavailable"])
            if spec.get("unlisted", False):
                registry.mark_unlisted(spec["name"])
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    weakref.finalize(registry, shutil.rmtree, root, ignore_errors=True)
    return registry


def _write_skill(root: Path, index: int, spec: Mapping[str, Any]) -> Path:
    in_memory = spec.get("in_memory", False)
    if not isinstance(in_memory, bool):
        raise AssertionError("in_memory: expected a boolean")
    # An in-memory skill's public name does not choose its filesystem path.
    directory = root / "_in_memory" / str(index) if in_memory else root / spec["name"]
    (directory / "references").mkdir(parents=True)
    if not in_memory:
        _write_frontmatter(directory, spec)
    for name, text in spec.get("references", {}).items():
        (directory / "references" / name).write_text(text, encoding="utf-8")
    if scripts := spec.get("scripts"):
        (directory / "scripts").mkdir()
        for name, text in scripts.items():
            (directory / "scripts" / name).write_text(text, encoding="utf-8")
    return directory


def _write_frontmatter(directory: Path, spec: Mapping[str, Any]) -> None:
    requirements = "".join(
        f"{key}: {json.dumps(spec[key])}\n" for key in ("allowed_tools", "requires") if key in spec
    )
    (directory / "SKILL.md").write_text(
        f"---\nname: {spec['name']}\ndescription: {spec['description']}\n"
        f"{requirements}---\n{spec.get('instructions', '')}",
        encoding="utf-8",
    )


def _memory_skill(spec: Mapping[str, Any], directory: Path) -> Skill:
    extra: dict[str, str] = {}
    if "requires" in spec:
        required = spec["requires"]
        extra["requires"] = ", ".join(required) if isinstance(required, list) else required
    metadata = SkillMetadata(
        name=spec["name"],
        description=spec["description"],
        allowed_tools=spec.get("allowed_tools"),
        extra_metadata=extra,
    )
    return Skill(metadata, spec.get("instructions", ""), directory)


def _registry_with_matcher(mode: str) -> SkillRegistry:
    if mode == "exact":
        return SkillRegistry()
    if mode not in ("hub", "fail"):
        raise AssertionError(f"Unknown skill matcher: {mode}")

    def match_requirement(required: str, offered: Collection[str]) -> bool:
        if mode == "fail":
            raise RuntimeError("Scripted skill matcher failed")
        return any(name == required or name.startswith(f"{required}_") for name in offered)

    return SkillRegistry(requires_match=match_requirement)
