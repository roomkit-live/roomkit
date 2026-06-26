"""Integration tests for Tool Search on the text/HTTP AIChannel.

Mirror of the realtime Tool Search behaviour, adapted to the text tool
loop: the model first sees only ``find_tools``/``list_tools`` + pinned
tools; calling ``find_tools`` reveals matches that become visible on the
NEXT tool-loop round (no ``provider.reconfigure`` — the text loop re-sends
its tool list every round). Covers streaming + non-streaming.
"""

from __future__ import annotations

import json

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.sandbox.executor import SandboxExecutor
from roomkit.tools.policy import ToolPolicy
from tests.conftest import make_event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx() -> RoomContext:
    return RoomContext(room=Room(id="r1"))


def _binding(tools: list[dict] | None = None) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": tools} if tools is not None else {},
    )


def _catalogue(n: int) -> list[dict]:
    """``n`` distinct noise tools (no lexical overlap with 'send sms')."""
    return [
        {"name": f"widget_{i}", "description": f"Operate widget number {i}."} for i in range(n)
    ]


_SMS_TOOL = {
    "name": "send_sms",
    "description": "Send an SMS text message to a phone number.",
    "parameters": {
        "type": "object",
        "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
        "required": ["to", "body"],
    },
}


async def _noop_handler(name: str, arguments: dict) -> str:
    """User tool handler so the channel wires a real tool loop (``has_any_tools``).

    The discovery tools (find_tools/list_tools) are channel-managed and never
    reach this; revealed tools would, but the reveal tests stop after the
    discovery call so this is only here to enable the loop.
    """
    return json.dumps({"ok": True, "tool": name})


class UnknownWindowProvider(MockAIProvider):
    """A provider whose active model is absent from the catalog → no known
    context window (custom / local model). Forces the count-based fallback."""

    @property
    def model_name(self) -> str:
        return "some-custom-local-model"


def _tool_names(context) -> set[str]:
    return {t.name for t in context.tools}


async def _run(ch: AIChannel, binding: ChannelBinding):
    return await ch.on_event(make_event(body="go", channel_id="sms1"), binding, _ctx())


def _tool_result(context) -> dict:
    """Parse the single tool-result message in a provider call's messages."""
    tool_msgs = [m for m in context.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    return json.loads(tool_msgs[0].content[0].result)


# ---------------------------------------------------------------------------
# Activation gating (the count-based seam)
# ---------------------------------------------------------------------------


class TestActivationGating:
    async def test_under_window_pct_is_noop(self) -> None:
        """Deferrable tools under the window-% budget → no search tools, all visible.

        Mock window is 8192; a high pct (50%) keeps a small catalogue under budget.
        """
        provider = MockAIProvider(responses=["hi"])
        ch = AIChannel(
            "ai1", provider=provider, system_prompt="Be nice.", tool_search_threshold_pct=50.0
        )
        await _run(ch, _binding(_catalogue(3)))

        names = _tool_names(provider.calls[0])
        assert "find_tools" not in names
        assert "list_tools" not in names
        assert names == {"widget_0", "widget_1", "widget_2"}
        assert provider.calls[0].system_prompt == "Be nice."

    async def test_auto_activates_above_window_pct(self) -> None:
        """Deferrable tools over the window-% budget auto-activate Tool Search.

        A low pct (0.5% of 8192 ≈ 41 tokens) is exceeded by a few tools, so the
        catalogue is deferred — self-tuning to the model's window, not a count.
        """
        provider = MockAIProvider(responses=["hi"])
        ch = AIChannel("ai1", provider=provider, tool_search_threshold_pct=0.5)
        await _run(ch, _binding(_catalogue(5)))

        names = _tool_names(provider.calls[0])
        assert "find_tools" in names and "list_tools" in names
        assert not any(n.startswith("widget_") for n in names)

    async def test_count_fallback_when_window_unknown(self) -> None:
        """No known context window → fall back to the tool-count threshold."""
        # 5 tools > threshold 3 → activate; the % path is unavailable.
        on = AIChannel(
            "ai1", provider=UnknownWindowProvider(responses=["hi"]), tool_search_threshold=3
        )
        await _run(on, _binding(_catalogue(5)))
        assert "find_tools" in _tool_names(on._provider.calls[0])  # type: ignore[attr-defined]

        # 2 tools <= threshold 3 → no activation.
        off = AIChannel(
            "ai2", provider=UnknownWindowProvider(responses=["hi"]), tool_search_threshold=3
        )
        await _run(off, _binding(_catalogue(2)))
        assert "find_tools" not in _tool_names(off._provider.calls[0])  # type: ignore[attr-defined]

    async def test_forced_off_never_activates(self) -> None:
        provider = MockAIProvider(responses=["hi"])
        ch = AIChannel("ai1", provider=provider, tool_search=False, tool_search_threshold=3)
        await _run(ch, _binding(_catalogue(10)))

        names = _tool_names(provider.calls[0])
        assert "find_tools" not in names
        assert len([n for n in names if n.startswith("widget_")]) == 10


# ---------------------------------------------------------------------------
# Round 0 surface
# ---------------------------------------------------------------------------


class TestInitialSurface:
    async def test_round0_only_infra_and_pinned(self) -> None:
        provider = MockAIProvider(responses=["hi"])
        ch = AIChannel(
            "ai1",
            provider=provider,
            system_prompt="Base.",
            tool_search=True,
            tool_search_pinned=["widget_1"],
        )
        await _run(ch, _binding(_catalogue(5)))

        names = _tool_names(provider.calls[0])
        assert names == {"find_tools", "list_tools", "widget_1"}
        # Preamble appended to the system prompt.
        assert "Base." in provider.calls[0].system_prompt
        assert "find_tools" in provider.calls[0].system_prompt


# ---------------------------------------------------------------------------
# find_tools reveals matches on the next round (streaming + non-streaming)
# ---------------------------------------------------------------------------


class TestFindToolsReveal:
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_find_tools_reveals_next_round(self, streaming: bool) -> None:
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        AIToolCall(id="t1", name="find_tools", arguments={"query": "send sms"})
                    ],
                ),
                AIResponse(content="done", finish_reason="stop"),
            ],
            streaming=streaming,
        )
        ch = AIChannel("ai1", provider=provider, tool_search=True, tool_handler=_noop_handler)
        binding = _binding([*_catalogue(5), _SMS_TOOL])
        output = await _run(ch, binding)
        assert output.responded is True
        # Streaming providers run the tool loop lazily as the stream is drained.
        if output.response_stream is not None:
            async for _ in output.response_stream:
                pass

        # Round 0: send_sms hidden behind find_tools.
        assert "send_sms" not in _tool_names(provider.calls[0])

        # find_tools result lists the match compactly (name + description, NO
        # inline schema — the full schema arrives via the next round's tool list,
        # so the result can't overflow on verbose tools).
        result = _tool_result(provider.calls[1])
        assert [m["name"] for m in result["matches"]] == ["send_sms"]
        assert "parameters" not in result["matches"][0]

        # Round 1: send_sms is now directly invocable; noise stays hidden.
        round1 = _tool_names(provider.calls[1])
        assert "send_sms" in round1
        assert "find_tools" in round1 and "list_tools" in round1
        assert not any(n.startswith("widget_") for n in round1)

    async def test_list_tools_reveals_nothing(self) -> None:
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[AIToolCall(id="t1", name="list_tools", arguments={})],
                ),
                AIResponse(content="done", finish_reason="stop"),
            ],
        )
        ch = AIChannel("ai1", provider=provider, tool_search=True, tool_handler=_noop_handler)
        binding = _binding([*_catalogue(3), _SMS_TOOL])
        await _run(ch, binding)

        # list_tools enumerates the catalogue...
        result = _tool_result(provider.calls[1])
        listed = {t["name"] for t in result["tools"]}
        assert "send_sms" in listed and "widget_0" in listed
        assert "find_tools" not in listed  # infra excluded from the listing

        # ...but reveals nothing — round 1 surface is still just the infra.
        round1 = _tool_names(provider.calls[1])
        assert round1 == {"find_tools", "list_tools"}


# ---------------------------------------------------------------------------
# Infra tools bypass a restrictive tool policy (exec-guard exemption)
# ---------------------------------------------------------------------------


class TestPolicyExemption:
    async def test_find_tools_runs_under_whitelist_policy(self) -> None:
        """A whitelist policy that omits find_tools must not block it."""
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        AIToolCall(id="t1", name="find_tools", arguments={"query": "send sms"})
                    ],
                ),
                AIResponse(content="done", finish_reason="stop"),
            ],
        )
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_search=True,
            tool_policy=ToolPolicy(allow=["send_sms"]),
            tool_handler=_noop_handler,
        )
        binding = _binding([*_catalogue(5), _SMS_TOOL])
        await _run(ch, binding)

        result = _tool_result(provider.calls[1])
        # Not a policy-denied error — the match came through.
        assert "error" not in result
        assert [m["name"] for m in result["matches"]] == ["send_sms"]
        # send_sms is both revealed and policy-allowed → visible round 1.
        assert "send_sms" in _tool_names(provider.calls[1])


# ---------------------------------------------------------------------------
# Sandbox tools defer behind Tool Search (but skip the user policy / gating)
# ---------------------------------------------------------------------------


class _FakeSandbox(SandboxExecutor):
    """Declares two ``sandbox_*`` tools without a real container.

    ``execute`` is never reached here — the model calls find_tools and the
    reveal round ends with a plain text response, so no sandbox tool runs.
    """

    def __init__(self, names: tuple[str, ...] = ("sandbox_read", "sandbox_grep")) -> None:
        self._names = names

    async def execute(self, command: str, arguments: dict | None = None):
        raise AssertionError("sandbox execute must not run in these tests")

    def tool_definitions(self) -> list[dict]:
        descriptions = {
            "sandbox_read": "Read file contents from the sandbox.",
            "sandbox_grep": "Run a regex pattern match across the workspace.",
        }
        return [
            {
                "name": n,
                "description": descriptions.get(n, f"Sandbox {n} tool."),
                "parameters": {"type": "object", "properties": {}},
            }
            for n in self._names
        ]


class TestSandboxToolSearch:
    async def test_sandbox_tools_deferred_then_revealed_under_policy(self) -> None:
        """Sandbox tools defer behind find_tools AND skip a whitelist policy.

        With Tool Search active they are hidden round 0 (not pinned), even
        though a sandbox is attached. find_tools surfaces the match, and round 1
        exposes it — the restrictive tool policy (which omits every sandbox
        tool) never blocks it, because sandbox tools are policy-exempt.
        """
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        AIToolCall(
                            id="t1", name="find_tools", arguments={"query": "read file contents"}
                        )
                    ],
                ),
                AIResponse(content="done", finish_reason="stop"),
            ],
        )
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_search=True,
            tool_policy=ToolPolicy(allow=["send_sms"]),  # omits every sandbox tool
            sandbox=_FakeSandbox(),
            tool_handler=_noop_handler,
        )
        await _run(ch, _binding())

        # Round 0: sandbox tools deferred behind the discovery tools.
        round0 = _tool_names(provider.calls[0])
        assert round0 == {"find_tools", "list_tools"}

        # find_tools surfaced sandbox_read — not a policy-denied error.
        result = _tool_result(provider.calls[1])
        assert "error" not in result
        assert [m["name"] for m in result["matches"]] == ["sandbox_read"]

        # Round 1: revealed + visible despite the whitelist policy; the
        # unmatched sandbox tool stays deferred.
        round1 = _tool_names(provider.calls[1])
        assert "sandbox_read" in round1
        assert "sandbox_grep" not in round1

    async def test_sandbox_tools_visible_when_search_off(self) -> None:
        """Tool Search off → sandbox tools stay visible AND policy-exempt."""
        provider = MockAIProvider(responses=["hi"])
        ch = AIChannel(
            "ai1",
            provider=provider,
            tool_search=False,
            tool_policy=ToolPolicy(allow=["send_sms"]),  # omits every sandbox tool
            sandbox=_FakeSandbox(),
            tool_handler=_noop_handler,
        )
        await _run(ch, _binding())

        names = _tool_names(provider.calls[0])
        assert {"sandbox_read", "sandbox_grep"} <= names
        assert "find_tools" not in names
