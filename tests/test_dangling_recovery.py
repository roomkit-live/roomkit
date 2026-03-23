"""Tests for dangling tool-call recovery (channels/_dangling_recovery.py)."""

from __future__ import annotations

from roomkit.channels._dangling_recovery import (
    _CANCEL_MSG,
    _build_patched_list,
    _collect_result_ids,
    _has_dangling_calls,
    patch_dangling_tool_calls,
)
from roomkit.providers.ai.base import (
    AIMessage,
    AITextPart,
    AIToolCallPart,
    AIToolResultPart,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_call(tc_id: str, name: str = "some_tool") -> AIToolCallPart:
    return AIToolCallPart(id=tc_id, name=name, arguments={})


def _tool_result(tc_id: str, name: str = "some_tool", result: str = "ok") -> AIToolResultPart:
    return AIToolResultPart(tool_call_id=tc_id, name=name, result=result)


def _assistant_msg(parts: list) -> AIMessage:
    return AIMessage(role="assistant", content=parts)


def _tool_msg(parts: list) -> AIMessage:
    return AIMessage(role="tool", content=parts)


def _user_msg(text: str) -> AIMessage:
    return AIMessage(role="user", content=text)


# ===========================================================================
# _collect_result_ids
# ===========================================================================


class TestCollectResultIds:
    def test_empty_messages(self) -> None:
        assert _collect_result_ids([]) == set()

    def test_no_tool_results(self) -> None:
        msgs = [_user_msg("hello"), _assistant_msg([AITextPart(text="hi")])]
        assert _collect_result_ids(msgs) == set()

    def test_collects_result_ids(self) -> None:
        msgs = [
            _tool_msg([_tool_result("tc-1"), _tool_result("tc-2")]),
        ]
        assert _collect_result_ids(msgs) == {"tc-1", "tc-2"}

    def test_skips_string_content(self) -> None:
        msgs = [AIMessage(role="user", content="plain string")]
        assert _collect_result_ids(msgs) == set()


# ===========================================================================
# _has_dangling_calls
# ===========================================================================


class TestHasDanglingCalls:
    def test_no_messages(self) -> None:
        assert _has_dangling_calls([], set()) is False

    def test_all_calls_resolved(self) -> None:
        msgs = [_assistant_msg([_tool_call("tc-1")])]
        assert _has_dangling_calls(msgs, {"tc-1"}) is False

    def test_dangling_call_detected(self) -> None:
        msgs = [_assistant_msg([_tool_call("tc-1")])]
        assert _has_dangling_calls(msgs, set()) is True

    def test_non_assistant_skipped(self) -> None:
        msgs = [_user_msg("hello")]
        assert _has_dangling_calls(msgs, set()) is False

    def test_string_content_skipped(self) -> None:
        msgs = [AIMessage(role="assistant", content="plain text")]
        assert _has_dangling_calls(msgs, set()) is False


# ===========================================================================
# patch_dangling_tool_calls (integration)
# ===========================================================================


class TestPatchDanglingToolCalls:
    def test_empty_messages(self) -> None:
        result = patch_dangling_tool_calls([])
        assert result == []

    def test_no_dangling_returns_same_list(self) -> None:
        msgs = [
            _user_msg("hello"),
            _assistant_msg([_tool_call("tc-1")]),
            _tool_msg([_tool_result("tc-1")]),
        ]
        result = patch_dangling_tool_calls(msgs)
        assert result is msgs  # Same object — no copy

    def test_dangling_call_gets_cancellation(self) -> None:
        msgs = [
            _assistant_msg([_tool_call("tc-1", "my_tool")]),
            _user_msg("interrupted"),
        ]
        result = patch_dangling_tool_calls(msgs)

        assert len(result) == 3
        # First: the assistant message (unchanged)
        assert result[0].role == "assistant"
        # Second: synthetic tool result
        assert result[1].role == "tool"
        assert isinstance(result[1].content, list)
        assert len(result[1].content) == 1
        part = result[1].content[0]
        assert isinstance(part, AIToolResultPart)
        assert part.tool_call_id == "tc-1"
        assert part.name == "my_tool"
        assert part.result == _CANCEL_MSG
        # Third: user message
        assert result[2].role == "user"

    def test_dangling_merged_into_following_tool_message(self) -> None:
        """When a tool message follows the dangling assistant, cancellations
        are merged into it rather than inserted as a separate message."""
        msgs = [
            _assistant_msg([_tool_call("tc-1"), _tool_call("tc-2")]),
            _tool_msg([_tool_result("tc-1")]),
        ]
        result = patch_dangling_tool_calls(msgs)

        assert len(result) == 2
        # The tool message should now have both the original result and the cancellation
        tool_msg = result[1]
        assert tool_msg.role == "tool"
        assert isinstance(tool_msg.content, list)
        assert len(tool_msg.content) == 2

        # First part is the cancellation for tc-2
        cancel_part = tool_msg.content[0]
        assert isinstance(cancel_part, AIToolResultPart)
        assert cancel_part.tool_call_id == "tc-2"
        assert cancel_part.result == _CANCEL_MSG

        # Second part is the existing result for tc-1
        existing_part = tool_msg.content[1]
        assert isinstance(existing_part, AIToolResultPart)
        assert existing_part.tool_call_id == "tc-1"

    def test_mixed_dangling_and_complete(self) -> None:
        """Mix of completed and dangling tool calls across messages."""
        msgs = [
            _user_msg("do stuff"),
            # First assistant turn — resolved
            _assistant_msg([_tool_call("tc-1")]),
            _tool_msg([_tool_result("tc-1")]),
            # Second assistant turn — dangling
            _assistant_msg([_tool_call("tc-2")]),
            _user_msg("cancel that"),
        ]
        result = patch_dangling_tool_calls(msgs)

        # Should be 6 messages: user, assistant, tool, assistant, synth-tool, user
        assert len(result) == 6
        assert result[4].role == "tool"
        cancel_part = result[4].content[0]
        assert isinstance(cancel_part, AIToolResultPart)
        assert cancel_part.tool_call_id == "tc-2"

    def test_multiple_dangling_calls_in_one_message(self) -> None:
        msgs = [
            _assistant_msg(
                [
                    _tool_call("tc-1", "tool_a"),
                    _tool_call("tc-2", "tool_b"),
                ]
            ),
        ]
        result = patch_dangling_tool_calls(msgs)

        assert len(result) == 2
        tool_msg = result[1]
        assert tool_msg.role == "tool"
        assert len(tool_msg.content) == 2
        ids = {p.tool_call_id for p in tool_msg.content}
        assert ids == {"tc-1", "tc-2"}

    def test_trailing_dangling_flushed_at_end(self) -> None:
        """Dangling calls at end of list get flushed as a final tool message."""
        msgs = [
            _assistant_msg([_tool_call("tc-1")]),
        ]
        result = patch_dangling_tool_calls(msgs)

        assert len(result) == 2
        assert result[1].role == "tool"
        assert result[1].content[0].tool_call_id == "tc-1"


# ===========================================================================
# _build_patched_list (lower-level edge cases)
# ===========================================================================


class TestBuildPatchedList:
    def test_pending_flushed_before_non_tool(self) -> None:
        """Pending cancellations flush before a non-tool message."""
        msgs = [
            _assistant_msg([_tool_call("tc-1")]),
            _user_msg("hi"),
        ]
        result = _build_patched_list(msgs, set())

        # assistant, synth-tool, user
        assert len(result) == 3
        assert result[1].role == "tool"
        assert result[2].role == "user"
