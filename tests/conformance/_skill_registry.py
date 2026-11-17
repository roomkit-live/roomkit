"""Exercise public registry operations on scripted MCP skill-server replies."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from roomkit.skills import MCPSkill, Skill, SkillError, SkillRegistry
from roomkit.skills.mcp import ServedOrigin
from tests.conformance._calls import UnreadableError
from tests.conformance._skill_registry_shape import registry_problems
from tests.conformance._skill_server import ScriptedSkillServer

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._calls import Call
    from tests.conformance._harness import Harness


def _skill(skill: Skill | None) -> Any:
    if skill is None:
        return None
    value: dict[str, Any] = {
        "metadata": asdict(skill.metadata),
        "instructions": skill.instructions,
        "has_scripts": skill.has_scripts,
        "references": skill.list_references(),
    }
    if isinstance(skill, MCPSkill):
        value |= {
            "origin": skill.origin,
            "uri": skill.held.uri,
            "read_total": skill.read_total,
            "files": {name: content.hex() for name, content in skill.files.items()},
            "retirements": skill.retirements,
        }
    return value


def _origin(origin: ServedOrigin | None, servers: Mapping[str, ScriptedSkillServer]) -> Any:
    if origin is None:
        return None
    return {
        "label": origin.label,
        "server": next(name for name, server in servers.items() if origin.server is server),
        "allow_dynamic": origin.allow_dynamic,
        "approve_execution": origin.approve_execution is not None,
        "execution_tools": list(origin.execution_tools),
    }


def _snapshot(registry: SkillRegistry) -> dict[str, Any]:
    return {
        "metadata": [asdict(item) for item in registry.all_metadata()],
        "skill_names": registry.skill_names,
        "listed_names": registry.listed_names,
        "skill_count": registry.skill_count,
        "has_entries": registry.has_entries,
        "has_served": registry.has_served,
        "unavailable_skills": registry.unavailable_skills,
        "execution_tool_patterns": sorted(registry.execution_tool_patterns),
        "gated_tool_names": sorted(registry.gated_tool_names()),
        "manifest": registry.to_prompt_xml(),
    }


class RegistrySession:
    """Each operation selects a real registry; copies retain their real catalogue."""

    def __init__(self, servers: Mapping[str, Mapping[str, Any]]) -> None:
        self.servers = {name: ScriptedSkillServer(spec) for name, spec in servers.items()}
        self.registries = {"main": SkillRegistry()}

    def _copy(self, registry: SkillRegistry, operation: Mapping[str, Any]) -> None:
        target = operation["target"]
        if target in self.registries:
            raise UnreadableError(f"duplicate skill registry {target}")
        self.registries[target] = registry.copy(
            operation.get("names"), marks=operation.get("marks", True)
        )

    async def operation(self, operation: Mapping[str, Any]) -> dict[str, Any]:
        registry = self.registries.get(operation.get("registry", "main"))
        if registry is None:
            raise UnreadableError("unknown skill registry")
        method = operation["method"]
        if method == "add_mcp_server" and operation["server"] not in self.servers:
            raise UnreadableError("unknown skill server")
        name = operation.get("name", "")
        calls = {
            "add_mcp_server": lambda: registry.add_mcp_server(
                self.servers[operation["server"]],
                operation["label"],
                allow_dynamic=operation.get("allow_dynamic", False),
                execution_tools=operation.get("execution_tools", []),
            ),
            "load_skill": lambda: registry.load_skill(name),
            "get_skill": lambda: registry.get_skill(name),
            "read_file": lambda: registry.read_file(name, operation["filename"]),
            "is_served": lambda: registry.is_served(name),
            "origin": lambda: _origin(registry.origin(name), self.servers),
            "copy": lambda: self._copy(registry, operation),
            "mark_unavailable": lambda: registry.mark_unavailable(name, operation["reason"]),
            "mark_unlisted": lambda: registry.mark_unlisted(name),
            "snapshot": lambda: _snapshot(registry),
        }
        try:
            value = calls[method]()
            if inspect.isawaitable(value):
                value = await value
        except SkillError as error:
            return {"error": type(error).__name__, "message": str(error)}
        except KeyError as error:
            if method != "read_file" or error.args != (name,):
                raise
            return {"error": "KeyError", "message": name}
        if method in {"get_skill", "load_skill"}:
            value = _skill(value)
        return {"result": value}


async def session(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Calls use public SDK methods; every unexpected fixture interaction fails."""
    if problems := registry_problems(spec, "skill_registry_session"):
        raise UnreadableError("; ".join(problems))
    state = RegistrySession(spec["servers"])
    try:
        results = [await state.operation(operation) for operation in spec["operations"]]
    finally:
        problems = [problem for server in state.servers.values() for problem in server.finish()]
        if problems:
            raise UnreadableError("; ".join(problems))
    return {
        "results": results,
        "requests": {name: server.requests for name, server in state.servers.items()},
    }


SKILL_REGISTRY_CALLS: dict[str, Call] = {"skill_registry_session": session}
