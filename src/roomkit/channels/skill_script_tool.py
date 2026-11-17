"""``run_skill_script`` as a tool a realtime channel serves (RFC §24).

An AI channel given ``skills=`` and a script executor declares and serves
``run_skill_script`` itself. A realtime voice channel whose skills belong to
another agent (a reasoning backend's) still has to run their scripts behind
its own execution gate: this tool, passed in its ``tools=``, runs them through
the one handler every channel uses.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar

from roomkit.channels._skill_constants import RUN_SCRIPT_SCHEMA, TOOL_RUN_SCRIPT
from roomkit.channels._skill_handlers import handle_run_script

if TYPE_CHECKING:
    from roomkit.skills import ScriptExecutor, SkillRegistry


class RunSkillScriptTool:
    """Runs a skill's script, as ``run_skill_script``, with *executor*.

    Satisfies :class:`roomkit.tools.Tool`: the schema every channel declares
    and the handler every channel calls. A script outside its skill answers
    an error; a skill the registry does not offer refuses the call
    (:class:`~roomkit.core.exceptions.ToolRefusedError`), as on any channel.
    """

    name: ClassVar[str] = TOOL_RUN_SCRIPT
    """The name every channel declares and calls it under."""

    def __init__(self, skills: SkillRegistry, executor: ScriptExecutor) -> None:
        self._skills = skills
        self._executor = executor

    @property
    def definition(self) -> dict[str, Any]:
        """An independent copy of the ``run_skill_script`` schema."""
        return deepcopy(RUN_SCRIPT_SCHEMA)

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        """Run the script *arguments* names; the script's result as JSON."""
        return await handle_run_script(arguments, self._skills, self._executor)
