"""Text logging observes the call before gates and its answer before hooks."""

from __future__ import annotations

import logging

import pytest

from roomkit import (
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolFailedError,
    ToolRefusedError,
    UnservedToolCallError,
    content_logging_enabled,
    set_content_logging,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AITextPart, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.policy import ToolPolicy
from tests.test_framework import SimpleChannel


def lines(caplog):
    return [record for record in caplog.records if record.name == "roomkit.channels.ai"]


@pytest.fixture(autouse=True)
def content_logging(caplog):
    original = content_logging_enabled()
    set_content_logging(True)
    caplog.set_level("DEBUG", logger="roomkit.channels.ai")
    yield
    set_content_logging(original)


async def run(
    streaming,
    handler,
    *,
    arguments=None,
    duplicate=False,
    policy=None,
    hooks=(),
    requested=None,
    skills=None,
):
    calls = requested or [AIToolCall(id="c1", name="lookup", arguments=arguments or {})]
    if duplicate:
        calls.append(AIToolCall(id="c1", name="lookup", arguments={"duplicate": "synthetic-held"}))
    model = MockAIProvider(
        ai_responses=[AIResponse(content="", tool_calls=calls), AIResponse(content="done")],
        streaming=streaming,
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel(
            "ai1",
            provider=model,
            tools=[
                AITool(
                    name="lookup", description="Synthetic lookup", parameters={"type": "object"}
                )
            ],
            tool_handler=handler,
            tool_policy=policy,
            skills=skills,
        )
    )
    for n, (trigger, callback) in enumerate(hooks):
        kit.hook(trigger, name=f"logging_{n}")(callback)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1")
    try:
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="sms1", sender_id="synthetic-user", content=TextContent(body="run")
            ),
            room_id="r1",
        )
        assert result.error is None
    finally:
        await kit.close()
    return [
        part
        for message in model.calls[1].messages
        if message.role == "tool"
        for part in message.content
    ]


async def test_arguments_are_logged_before_a_policy_refuses_them(caplog, streaming):
    ran = []

    async def handler(name, arguments):
        ran.append(name)
        return "synthetic-answer"

    await run(
        streaming,
        handler,
        arguments={"z": "synthetic-z", "a": "synthetic-a"},
        policy=ToolPolicy(deny=["lookup"]),
    )
    messages = [r.getMessage() for r in lines(caplog)]
    at = messages.index("Executing tool lookup (call c1) with a, z")
    assert messages[at + 1] == 'Tool lookup arguments: {"z": "synthetic-z", "a": "synthetic-a"}'
    assert messages.index("Tool lookup blocked by policy") > at + 1
    assert ran == []
    assert not any("returned" in message or "result:" in message for message in messages)


@pytest.mark.parametrize("block", [False, True])
async def test_logs_precede_argument_and_result_hooks(caplog, streaming, block):
    seen = []

    async def before(event, context):
        assert any(
            r.getMessage() == 'Tool lookup arguments: {"query": "synthetic-model"}'
            for r in lines(caplog)
        )
        seen.append("before")
        return HookResult(action="allow", metadata={"arguments": {"query": "synthetic-rewritten"}})

    async def handler(name, arguments):
        assert arguments == {"query": "synthetic-rewritten"}
        seen.append("handler")
        return "synthetic-original"

    async def after(event, context):
        assert event.result == "synthetic-original"
        assert any(
            r.getMessage() == "Tool lookup result: synthetic-original" for r in lines(caplog)
        )
        seen.append("after")
        return (
            HookResult.block("synthetic-block")
            if block
            else HookResult(action="allow", metadata={"result": "synthetic-replacement"})
        )

    parts = await run(
        streaming,
        handler,
        arguments={"query": "synthetic-model"},
        hooks=[(HookTrigger.BEFORE_TOOL_USE, before), (HookTrigger.ON_TOOL_CALL, after)],
    )
    assert seen == ["before", "handler", "after"]
    assert (
        "synthetic-block" in parts[0].as_text()
        if block
        else parts[0].as_text() == "synthetic-replacement"
    )
    messages = [r.getMessage() for r in lines(caplog)]
    assert not any("Tool lookup arguments: " in m and "synthetic-rewritten" in m for m in messages)
    assert not any("Tool lookup result: " in m and "synthetic-replacement" in m for m in messages)


async def test_a_held_duplicate_has_no_second_execution_log(caplog, streaming):
    ran = []

    async def handler(name, arguments):
        ran.append(name)
        return "synthetic-result"

    parts = await run(streaming, handler, duplicate=True)
    assert len(parts) == 2 and ran == ["lookup"]
    messages = [r.getMessage() for r in lines(caplog)]
    assert messages.count("Executing tool lookup (call c1) with no arguments") == 1
    assert not any("synthetic-held" in m for m in messages)


@pytest.mark.parametrize(
    ("answer", "size", "preview"),
    [
        (None, 4, "null"),
        ({"x": "é"}, 10, '{"x": "é"}'),
        ([AITextPart(text="synthetic-text")], -1, "[\"type='text' text='synthetic-text'\"]"),
    ],
)
async def test_json_answers_and_text_parts_keep_their_logging_shape(
    caplog, streaming, answer, size, preview
):
    async def handler(name, arguments):
        return answer

    await run(streaming, handler)
    messages = [r.getMessage() for r in lines(caplog)]
    assert any(m.startswith(f"Tool lookup returned {size} chars in ") for m in messages)
    assert f"Tool lookup result: {preview}" in messages


@pytest.mark.parametrize(
    ("error", "level", "prefix"),
    [
        (
            UnservedToolCallError("synthetic-decline"),
            logging.INFO,
            "Tool lookup: nothing served it (",
        ),
        (
            ToolRefusedError("synthetic-refusal"),
            logging.INFO,
            "Tool lookup refused: synthetic-refusal",
        ),
        (
            ToolFailedError("synthetic-failure"),
            logging.WARNING,
            "Tool lookup failed: synthetic-failure",
        ),
        (
            ValueError("synthetic-error"),
            logging.WARNING,
            "Tool lookup raised ValueError: synthetic-error",
        ),
    ],
)
async def test_handler_errors_have_their_source_level_and_no_result_preview(
    caplog, streaming, error, level, prefix
):
    async def handler(name, arguments):
        raise error

    await run(streaming, handler)
    records = lines(caplog)
    matches = [r for r in records if r.getMessage().startswith(prefix)]
    assert len(matches) == 1 and matches[0].levelno == level
    assert not any("Tool lookup result:" in r.getMessage() for r in records)
    if isinstance(error, UnservedToolCallError):
        assert not any("synthetic-decline" in r.getMessage() for r in records)


@pytest.mark.parametrize("gate", ["unknown", "cut", "garbled", "withdrawn", "skill", "before"])
async def test_gate_diagnostics_are_once_at_their_source_level(caplog, streaming, gate, tmp_path):
    from tests.test_toolset_edges import _skills

    requested = AIToolCall(
        id="c1",
        name="unknown" if gate == "unknown" else "lookup",
        partial=gate in ("cut", "garbled"),
        garbled=gate == "garbled",
    )
    hooks = []
    skills = _skills(tmp_path, "synthetic", "lookup") if gate == "skill" else None
    if gate == "withdrawn":

        async def before_generation(event, context):
            event.ai_context.tools = []
            return HookResult.allow()

        hooks.append((HookTrigger.BEFORE_AI_GENERATION, before_generation))
    if gate == "before":

        async def before_tool(event, context):
            return HookResult.block("synthetic-policy")

        hooks.append((HookTrigger.BEFORE_TOOL_USE, before_tool))
    ran = []

    async def handler(name, arguments):
        ran.append(name)
        return "synthetic-result"

    await run(streaming, handler, requested=[requested], hooks=hooks, skills=skills)
    assert ran == []
    expected = {
        "unknown": "Provider requested undeclared tool unknown",
        "cut": "Provider cut tool call lookup (c1): it does not run",
        "garbled": "Provider sent unreadable arguments for tool call lookup (c1): it does not run",
        "withdrawn": "Provider called lookup, withdrawn for this turn",
        "skill": "Tool lookup blocked by skill gating",
        "before": "Tool lookup denied by BEFORE_TOOL_USE hook",
    }[gate]
    matches = [record for record in lines(caplog) if record.getMessage() == expected]
    assert len(matches) == 1
    assert matches[0].levelno == (logging.INFO if gate == "before" else logging.WARNING)
