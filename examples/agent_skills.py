"""Agent Skills integration with AIChannel.

Demonstrates how to use the Agent Skills standard (https://agentskills.io)
to give AI channels specialized knowledge packages. Shows:
- SkillRegistry discovery from a directory of SKILL.md packages
- AIChannel auto-registration of skill tools (activate_skill, read_skill_reference)
- ScriptExecutor ABC for integrator-defined script execution
- Combining skills with user-defined tools

Run with:
    uv run python examples/agent_skills.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    ScriptExecutor,
    ScriptResult,
    Skill,
    SkillRegistry,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AIToolCall,
)
from roomkit.providers.ai.mock import MockAIProvider

# ---------------------------------------------------------------------------
# 1. Create sample skill directories (in production, these already exist)
# ---------------------------------------------------------------------------


def create_sample_skills(base_dir: Path) -> Path:
    """Create sample skill directories for the demo."""
    skills_dir = base_dir / "skills"
    skills_dir.mkdir()

    # Skill 1: code-review
    cr = skills_dir / "code-review"
    cr.mkdir()
    (cr / "SKILL.md").write_text(
        "---\n"
        "name: code-review\n"
        "description: Review code for bugs, security issues, and style problems\n"
        "license: MIT\n"
        "---\n"
        "\n"
        "# Code Review Skill\n"
        "\n"
        "When reviewing code, check for:\n"
        "1. Security vulnerabilities (SQL injection, XSS, etc.)\n"
        "2. Logic errors and edge cases\n"
        "3. Performance issues\n"
        "4. Style consistency with the project\n"
        "\n"
        "Use the style-guide reference for project-specific conventions.\n"
        "Run the lint script to check for common issues.\n",
    )
    (cr / "references").mkdir()
    (cr / "references" / "style-guide.md").write_text(
        "# Style Guide\n\n"
        "- Use snake_case for functions and variables\n"
        "- Use PascalCase for classes\n"
        "- Max line length: 99 characters\n"
        "- Always use type hints on public methods\n"
    )
    (cr / "scripts").mkdir()
    (cr / "scripts" / "lint.sh").write_text("#!/bin/bash\necho 'Linting passed'\n")

    # Skill 2: test-writer
    tw = skills_dir / "test-writer"
    tw.mkdir()
    (tw / "SKILL.md").write_text(
        "---\n"
        "name: test-writer\n"
        "description: Generate comprehensive test cases from source code\n"
        "---\n"
        "\n"
        "# Test Writer Skill\n"
        "\n"
        "Generate tests using pytest with these patterns:\n"
        "- One test class per module under test\n"
        "- Use fixtures for shared setup\n"
        "- Test both happy path and error cases\n"
        "- Use parametrize for multiple inputs\n",
    )
    (tw / "references").mkdir()
    (tw / "references" / "patterns.md").write_text(
        "# Test Patterns\n\n"
        "## Fixture pattern\n"
        "```python\n"
        "@pytest.fixture\n"
        "def client():\n"
        "    return TestClient(app)\n"
        "```\n"
    )

    return skills_dir


# ---------------------------------------------------------------------------
# 2. Mock AI provider that simulates tool calls
# ---------------------------------------------------------------------------


class SkillDemoProvider(MockAIProvider):
    """Provider that calls activate_skill on first turn, then responds."""

    def __init__(self) -> None:
        super().__init__(responses=["I've reviewed the style guide and can help!"])
        self._turn = 0

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        self._turn += 1

        if self._turn == 1:
            # First turn: AI decides to activate the code-review skill
            return AIResponse(
                content="",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 50, "completion_tokens": 10},
                tool_calls=[
                    AIToolCall(
                        id="call_1",
                        name="activate_skill",
                        arguments={"name": "code-review"},
                    )
                ],
            )

        if self._turn == 2:
            # Second turn: AI reads the style guide reference
            return AIResponse(
                content="",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 100, "completion_tokens": 10},
                tool_calls=[
                    AIToolCall(
                        id="call_2",
                        name="read_skill_reference",
                        arguments={
                            "skill_name": "code-review",
                            "filename": "style-guide.md",
                        },
                    )
                ],
            )

        # Third turn: AI responds with the review
        return AIResponse(
            content=(
                "Based on the code review skill and your project's style guide, "
                "here's my review:\n"
                "1. Line 42: Use snake_case for `getUserName` -> `get_user_name`\n"
                "2. Line 58: Missing type hint on `process()` return value\n"
                "3. Line 73: Consider extracting the nested loop into a helper"
            ),
            finish_reason="stop",
            usage={"prompt_tokens": 200, "completion_tokens": 50},
        )


# ---------------------------------------------------------------------------
# 3. Simple ScriptExecutor example
# ---------------------------------------------------------------------------


class DemoScriptExecutor(ScriptExecutor):
    """Demo executor that just prints what would be run."""

    async def execute(
        self,
        skill: Skill,
        script_name: str,
        arguments: dict[str, str] | None = None,
    ) -> ScriptResult:
        print(f"    [executor] Would run: {skill.name}/scripts/{script_name}")
        if arguments:
            print(f"    [executor] Arguments: {arguments}")
        return ScriptResult(
            exit_code=0,
            stdout="Linting passed - 0 issues found",
            success=True,
        )


# ---------------------------------------------------------------------------
# 4. Main demo
# ---------------------------------------------------------------------------


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp)
        skills_dir = create_sample_skills(base_dir)

        # --- Discover skills ---
        print("=== Discovering Skills ===")
        registry = SkillRegistry()
        count = registry.discover(skills_dir)
        print(f"  Found {count} skills: {registry.skill_names}")

        for meta in registry.all_metadata():
            print(f"  - {meta.name}: {meta.description}")

        # --- Show prompt XML ---
        print("\n=== Prompt XML ===")
        print(registry.to_prompt_xml())

        # --- Set up RoomKit with skills-enabled AIChannel ---
        print("\n=== Running Conversation ===")
        kit = RoomKit()

        ws = WebSocketChannel("ws-user")
        ai = AIChannel(
            "ai-assistant",
            provider=SkillDemoProvider(),
            system_prompt="You are a senior code reviewer.",
            skills=registry,
            script_executor=DemoScriptExecutor(),
        )

        kit.register_channel(ws)
        kit.register_channel(ai)

        inbox: list[RoomEvent] = []

        async def on_recv(_conn: str, event: RoomEvent) -> None:
            inbox.append(event)

        ws.register_connection("user-conn", on_recv)

        await kit.create_room(room_id="review-room")
        await kit.attach_channel("review-room", "ws-user")
        await kit.attach_channel(
            "review-room",
            "ai-assistant",
            category=ChannelCategory.INTELLIGENCE,
        )

        # User sends a code review request
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="dev",
                content=TextContent(body="Please review my Python module for style issues."),
            )
        )

        # Show what the AI responded
        for ev in inbox:
            if ev.source.channel_id == "ai-assistant":
                print(f"\n  AI response:\n  {ev.content.body}")  # type: ignore[union-attr]

        # --- Inspect what happened ---
        print("\n=== AI Provider Calls ===")
        provider: SkillDemoProvider = ai._provider  # type: ignore[assignment]
        for i, call in enumerate(provider.calls):
            print(f"  Call {i + 1}:")
            if call.tools:
                print(f"    Tools available: {[t.name for t in call.tools]}")

        # --- Show skills in system prompt ---
        if provider.calls:
            prompt = provider.calls[0].system_prompt or ""
            if "<available_skills>" in prompt:
                print("\n=== Skills in System Prompt ===")
                start = prompt.index("<available_skills>")
                end = prompt.index("</available_skills>") + len("</available_skills>")
                print(f"  {prompt[start:end]}")

        # --- Also demonstrate combining with user tools ---
        print("\n=== Skills + User Tools ===")
        inbox.clear()

        await kit.detach_channel("review-room", "ws-user")
        await kit.detach_channel("review-room", "ai-assistant")

        await kit.create_room(room_id="combined-room")
        await kit.attach_channel("combined-room", "ws-user")
        await kit.attach_channel(
            "combined-room",
            "ai-assistant",
            category=ChannelCategory.INTELLIGENCE,
            metadata={
                "tools": [
                    {
                        "name": "fetch_pr",
                        "description": "Fetch a pull request from GitHub",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "pr_number": {"type": "integer"},
                            },
                            "required": ["pr_number"],
                        },
                    }
                ]
            },
        )

        # Verify the combined tools
        provider2: MockAIProvider = ai._provider  # type: ignore[assignment]
        provider2.calls.clear()  # reset for clean inspection

        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="dev",
                content=TextContent(body="Check PR #42"),
            )
        )

        if provider2.calls:
            tools = provider2.calls[0].tools
            print(f"  All tools: {[t.name for t in tools]}")
            print("  (user tools + skill tools combined)")

        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
