"""Explicit functional contracts for chat E2E scenarios.

All inputs and tool results are synthetic. Only ScriptExecutor's allowlisted
fixture executes a process. No scenario mutates an external service.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.chat.harness import Harness
from roomkit import (
    Access,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomEvent,
    ToolCallEvent,
)
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.channel import RetryPolicy
from roomkit.providers.ai.base import AIContext, AIMessage, AITool, StreamTextDelta
from roomkit.realtime.base import EphemeralEventType
from roomkit.skills import ScriptExecutor, ScriptResult, Skill, SkillRegistry
from roomkit.tools import current_tool_actor_id, current_tool_room_id

PROMPT = "Reply with exactly the word pong."


@dataclass
class Scenario:
    name: str
    features: tuple[str, ...]
    description: str
    run: Callable[[Harness], Awaitable[None]]
    streaming: bool = True
    options: dict[str, Any] = field(default_factory=dict)
    options_factory: Callable[[], dict[str, Any]] | None = None
    mock_supported: bool = False
    fault_injection: bool = False

    def make_options(self) -> dict[str, Any]:
        """Give each sample fresh stateful skills/memory/executor instances."""
        return {**self.options, **(self.options_factory() if self.options_factory else {})}


def tool(name: str, description: str, properties: dict[str, Any]) -> AITool:
    return AITool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    )


class FixtureExecutor(ScriptExecutor):
    """Run only the bundled calculation fixture, with no inherited environment."""

    def __init__(self) -> None:
        self.calls = 0
        self.expected = (
            Path(__file__).parent / "fixtures/quote-policy/scripts/total.py"
        ).resolve()

    async def execute(
        self, skill: Skill, script_name: str, arguments: dict[str, str] | None = None
    ) -> ScriptResult:
        path = await asyncio.to_thread(skill.resolve_script, script_name)
        if await asyncio.to_thread(path.resolve) != self.expected:
            raise ValueError("Only the benchmark calculation fixture is executable")
        self.calls += 1
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(path),
            json.dumps(arguments or {}),
            env={},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), 3)
        except BaseException:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        return ScriptResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            success=proc.returncode == 0,
        )


def record_handler(
    h: Harness, handler: Callable[[str, dict[str, Any]], Awaitable[str]]
) -> Callable[[str, dict[str, Any]], Awaitable[str]]:
    async def wrapped(name: str, arguments: dict[str, Any]) -> str:
        entry = {
            "name": name,
            "arguments": dict(arguments),
            "room": current_tool_room_id(),
            "actor": current_tool_actor_id(),
            "start": time.perf_counter(),
            "end": 0.0,
        }
        h.executions.append(entry)
        try:
            return await handler(name, arguments)
        finally:
            entry["end"] = time.perf_counter()

    return wrapped


async def direct_text(h: Harness) -> None:
    result = await h.provider.generate(
        AIContext(messages=[AIMessage(role="user", content=PROMPT)], temperature=0)
    )
    h.check("answer_pong", result.content.strip().lower().rstrip(".") == "pong")
    h.check("finished", result.finish_reason == "stop")


async def direct_stream(h: Harness) -> None:
    text = ""
    async for event in h.provider.generate_structured_stream(
        AIContext(messages=[AIMessage(role="user", content=PROMPT)], temperature=0)
    ):
        if isinstance(event, StreamTextDelta):
            text += event.text
    h.check("answer_pong", text.strip().lower().rstrip(".") == "pong")


async def chat(h: Harness) -> None:
    result = await h.ask(PROMPT)
    h.check("inbound_success", not result.blocked and not result.error)
    h.check("answer_pong", h.answer().strip().lower().rstrip(".") == "pong")
    h.check("response_hook", len(h.responses) == 1)


async def reasoning(h: Harness) -> None:
    await h.ask("Compute 17 times 19 carefully. Reply with the number only.")
    h.check("answer_323", "323" in h.answer())
    h.check("thinking_observed", any(response.thinking for response in h.responses))
    await asyncio.sleep(0)


async def parallel_tools(h: Harness) -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        await asyncio.sleep(0.08)
        return json.dumps({"warehouse": arguments["warehouse"], "stock": 21})

    h.handler = record_handler(h, handler)
    await h.ask(
        "Call stock for north and south warehouses together in one tool round. Both "
        "lookups are independent. Then reply with their combined stock only."
    )
    h.check(
        "two_lookups",
        {e["arguments"].get("warehouse") for e in h.executions} == {"north", "south"},
    )
    h.check("answer_42", "42" in h.answer())
    h.check(
        "parallel_execution",
        len(h.executions) == 2
        and min(e["end"] for e in h.executions) > max(e["start"] for e in h.executions),
    )
    h.details["execution_ms"] = [(e["end"] - e["start"]) * 1000 for e in h.executions]


async def chained_tools(h: Harness) -> None:
    async def handler(name: str, args: dict[str, Any]) -> str:
        if name == "lookup":
            return json.dumps({"receipt": "REC-812", "amount": 21})
        if args.get("receipt") != "REC-812":
            raise ValueError("A receipt from lookup is required")
        return json.dumps({"result": 42})

    h.handler = record_handler(h, handler)
    await h.ask(
        "Use lookup for order ORD-42. Then pass its receipt to double_amount. Reply "
        "with the result only."
    )
    h.check(
        "ordered_dependencies", [e["name"] for e in h.executions] == ["lookup", "double_amount"]
    )
    h.check("answer_42", "42" in h.answer())


async def tool_recovery(h: Harness) -> None:
    attempts = 0

    async def handler(name: str, args: dict[str, Any]) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Temporary inventory outage. Retry this lookup once.")
        return '{"stock":42}'

    h.handler = record_handler(h, handler)
    await h.ask(
        "Use inventory to read stock. If there is a temporary failure, retry once. "
        "Reply with stock only."
    )
    h.check("tool_retried", attempts == 2)
    h.check("answer_42", "42" in h.answer())


async def tool_hooks(h: Harness) -> None:
    async def handler(name: str, args: dict[str, Any]) -> str:
        return json.dumps({"sum": args["a"] + args["b"]})

    h.handler = record_handler(h, handler)

    @h.kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC)
    async def rewrite(event: ToolCallEvent, context: RoomContext) -> HookResult:
        return HookResult(action="allow", metadata={"arguments": {"a": 20, "b": 22}})

    @h.kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, priority=20)
    async def override(event: ToolCallEvent, context: RoomContext) -> HookResult:
        return HookResult(action="allow", metadata={"result": '{"sum":99}'})

    await h.ask(
        "Use add with a=17 and b=25. The tool is authoritative; repeat its returned sum only."
    )
    h.check(
        "rewritten_arguments_executed",
        len(h.executions) == 1 and h.executions[0]["arguments"] == {"a": 20, "b": 22},
    )
    h.check("override_reached_model", "99" in h.answer())


async def block_input(h: Harness) -> None:
    @h.kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def block(event: RoomEvent, context: RoomContext) -> HookResult:
        return HookResult.block("Synthetic benchmark rejection")

    result = await h.ask(PROMPT)
    h.check("inbound_blocked", result.blocked)
    h.check("provider_not_called", not h.provider.calls)


async def skill_workflow(h: Harness) -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return '{"quantity":3,"unit_price":11}'

    h.handler = record_handler(h, handler)
    await h.ask(
        "Activate quote-policy and follow every step to produce a benchmark quote. "
        "Include its policy marker and computed total."
    )
    h.check("skill_activated", "quote-policy" in h.ai.active_skill_names("main"))
    names = [e.name for e in h.tool_events]
    h.check("skill_reference_read", "read_skill_reference" in names)
    h.check("skill_script_executed", "run_skill_script" in names)
    h.check("gated_inventory_executed", "inventory" in names)
    h.check("inventory_hidden_before_activation", "inventory" not in h.provider.calls[0].tools)
    h.check(
        "inventory_exposed_after_activation",
        any("inventory" in call.tools for call in h.provider.calls[1:]),
    )
    h.check("quote_correct", "40" in h.answer() and "COBALT-72" in h.answer())
    calls_before = names.count("activate_skill")
    await h.ask(
        "What is the policy marker you just used? Reply with that marker only; no "
        "need to reactivate the skill."
    )
    h.check(
        "activation_reused",
        [e.name for e in h.tool_events].count("activate_skill") == calls_before,
    )
    h.check("skill_room_isolation", not h.ai.active_skill_names("other"))
    h.details["tool_sequence"] = [e.name for e in h.tool_events]


async def concurrent_rooms(h: Harness) -> None:
    async def handler(name: str, args: dict[str, Any]) -> str:
        await asyncio.sleep(0.03)
        return json.dumps({"room": current_tool_room_id(), "actor": current_tool_actor_id()})

    h.handler = record_handler(h, handler)
    for i in range(4):
        room = f"room-{i}"
        await h.add_room(room)
        await h.kit.add_member(room, "user", f"actor-{i}")
    await asyncio.gather(
        *[
            h.ask(
                "Use whoami. Return its room and actor exactly, separated by a space.",
                room=f"room-{i}",
                actor=f"actor-{i}",
            )
            for i in range(4)
        ]
    )
    for i in range(4):
        answer = h.answer(f"room-{i}")
        h.check("correct_room_actor", f"room-{i}" in answer and f"actor-{i}" in answer)
        h.check("no_cross_room_text", all(f"room-{j}" not in answer for j in range(4) if j != i))
    h.check(
        "tool_context_isolation",
        all(
            e["room"] == "room-" + str(i) and e["actor"] == "actor-" + str(i)
            for i in range(4)
            for e in h.executions
            if e["room"] == f"room-{i}"
        ),
    )


async def idempotency(h: Harness) -> None:
    await asyncio.gather(*[h.ask(PROMPT, idempotency_key="duplicate") for _ in range(4)])
    h.check("one_generation_for_duplicate", len(h.provider.calls) == 1)
    h.check("answer_pong", "pong" in h.answer().lower())


async def muted_tools(h: Harness) -> None:
    async def handler(name: str, args: dict[str, Any]) -> str:
        return '{"stock":42}'

    h.handler = record_handler(h, handler)
    await h.kit.mute("main", "assistant")
    await h.ask("Use inventory, then answer with stock only.")
    h.check("muting_keeps_tool_effects", len(h.executions) == 1)
    h.check("muting_suppresses_response", not h.answer())


async def read_only(h: Harness) -> None:
    await h.kit.set_access("main", "user", Access.READ_ONLY)
    result = await h.ask(PROMPT)
    h.check("read_only_cannot_write", result.blocked)
    h.check("provider_not_called", not h.provider.calls)


async def retry(h: Harness) -> None:
    h.provider.fail_next = 1
    await chat(h)
    h.check("one_injected_failure", sum(c.injected for c in h.provider.calls) == 1)
    h.check(
        "recovered_on_second_attempt",
        len(h.provider.calls) == 2 and not h.provider.calls[-1].error,
    )


async def memory_window(h: Harness) -> None:
    await h.ask("Remember the synthetic marker OBSOLETE-581. Reply OK.")
    await h.ask("Reply with exactly the word pong.")
    await h.ask("Reply with exactly the word pong.")
    final = h.provider.contexts[-1]
    h.check("old_history_trimmed", "OBSOLETE-581" not in str(final.messages))
    h.check("recent_turn_retained", any("pong" in str(m.content) for m in final.messages))
    h.details["message_counts"] = [c.messages for c in h.provider.calls]


async def tool_search(h: Harness) -> None:
    async def handler(name: str, args: dict[str, Any]) -> str:
        return '{"stock":42}'

    h.handler = record_handler(h, handler)
    await h.ask(
        "Use find_tools to find the inventory stock lookup, then call that tool and "
        "return the stock only."
    )
    h.check("search_used", any(e.name == "find_tools" for e in h.tool_events))
    h.check("deferred_tool_executed", any(e["name"] == "inventory" for e in h.executions))
    h.check("answer_42", "42" in h.answer())


async def capped_loop(h: Harness) -> None:
    async def handler(name: str, args: dict[str, Any]) -> str:
        return '{"done":false,"next_action":"call inventory again"}'

    h.handler = record_handler(h, handler)
    await h.ask(
        "Keep calling inventory until its done field is true. When done is false, "
        "call inventory again. Do not finish early."
    )
    h.check("round_limit_reported", any(e["reason"] == "max_rounds" for e in h.ai.ends))
    h.check("at_most_one_tool_execution", len(h.executions) <= 1)


async def muted_stream(h: Harness) -> None:
    await h.kit.mute("main", "assistant")
    await h.ask("Remember SILENT-842.")
    h.check("muted_stream_not_generated", not h.provider.calls and not h.answer())
    await h.kit.unmute("main", "assistant")
    await h.ask(PROMPT)
    h.check("muted_input_in_history", "SILENT-842" in str(h.provider.contexts[-1].messages))


async def visibility(h: Harness) -> None:
    await h.kit.set_visibility("main", "user", "none")
    before = len(await h.kit.store.list_events("main"))
    await h.ask(PROMPT)
    h.check("hidden_input_stored", len(await h.kit.store.list_events("main")) == before + 1)
    h.check("hidden_input_not_broadcast", not h.provider.calls)


async def ui_events(h: Harness) -> None:
    before = await h.kit.store.list_events("main")
    await h.kit.publish_typing("main", "alice", is_typing=True)
    await h.kit.publish_typing("main", "alice", is_typing=False)
    await h.kit.publish_presence("main", "alice", "online")
    await h.kit.publish_reaction("main", "alice", "synthetic-event", "+1")
    await h.kit.publish_read_receipt("main", "alice", "synthetic-event")
    expected = {
        EphemeralEventType.TYPING_START,
        EphemeralEventType.TYPING_STOP,
        EphemeralEventType.PRESENCE_ONLINE,
        EphemeralEventType.REACTION,
        EphemeralEventType.READ_RECEIPT,
    }
    async with asyncio.timeout(2):
        while not expected <= {event.type for event in h.ephemeral}:
            await asyncio.sleep(0.001)
    h.check("all_ui_events_delivered", expected <= {event.type for event in h.ephemeral})
    h.check("ephemeral_not_in_history", await h.kit.store.list_events("main") == before)


async def denied_tool(h: Harness) -> None:
    async def handler(name: str, args: dict[str, Any]) -> str:
        return '{"stock":42}'

    h.handler = record_handler(h, handler)

    @h.kit.hook(HookTrigger.BEFORE_TOOL_USE)
    async def deny(event: ToolCallEvent, context: RoomContext) -> HookResult:
        return HookResult.block("Synthetic policy denies inventory. Answer DENIED.")

    await h.ask("Call inventory once. If access is denied, answer DENIED and do not retry.")
    h.check("denied_tool_not_executed", not h.executions)
    h.check("denial_reached_model", "DENIED" in h.answer())


async def long_stream(h: Harness) -> None:
    await h.ask(
        "Print integers 1 through 100 in ascending order, separated by single spaces. "
        "No other text."
    )
    h.check("complete_sequence", h.answer().split() == [str(i) for i in range(1, 101)])


def scenarios() -> list[Scenario]:
    inventory = tool("inventory", "Read inventory quantity, unit price and stock", {})

    def skill_options() -> dict[str, Any]:
        registry = SkillRegistry()
        registry.discover(Path(__file__).parent / "fixtures")
        return {"skills": registry, "script_executor": FixtureExecutor()}

    integers = {"a": {"type": "integer"}, "b": {"type": "integer"}}
    catalog = [
        Scenario(
            "direct_text",
            ("provider",),
            "Direct provider, buffered response",
            direct_text,
            streaming=False,
            mock_supported=True,
        ),
        Scenario(
            "direct_stream",
            ("provider",),
            "Direct provider, streamed response",
            direct_stream,
            mock_supported=True,
        ),
        Scenario(
            "chat_text",
            ("pipeline", "store", "hooks"),
            "Full buffered chat turn",
            chat,
            streaming=False,
            mock_supported=True,
        ),
        Scenario(
            "chat_stream",
            ("pipeline", "streaming", "events"),
            "Full streamed chat turn",
            chat,
            mock_supported=True,
        ),
        Scenario(
            "reasoning",
            ("reasoning", "streaming"),
            "Thinking remains distinct from visible text",
            reasoning,
            options={"reasoning_effort": "low"},
        ),
        Scenario(
            "parallel_tools",
            ("tools", "parallel", "events"),
            "Two independent 80ms lookups in one round",
            parallel_tools,
            options={
                "tools": [
                    tool("stock", "Read one warehouse's stock", {"warehouse": {"type": "string"}})
                ]
            },
        ),
        Scenario(
            "chained_tools",
            ("tools", "history"),
            "Dependent tools with receipt propagation",
            chained_tools,
            options={
                "tools": [
                    tool(
                        "lookup", "Lookup order and return receipt", {"order": {"type": "string"}}
                    ),
                    tool(
                        "double_amount",
                        "Double amount for a receipt returned by lookup",
                        {"receipt": {"type": "string"}},
                    ),
                ]
            },
        ),
        Scenario(
            "tool_recovery",
            ("tools", "errors"),
            "One injected tool exception, then real-model recovery",
            tool_recovery,
            options={"tools": [inventory]},
            fault_injection=True,
        ),
        Scenario(
            "tool_hooks",
            ("hooks", "tools"),
            "Rewrite arguments and override a tool result",
            tool_hooks,
            options={"tools": [tool("add", "Add two integers", integers)]},
        ),
        Scenario(
            "block_input",
            ("hooks", "permissions"),
            "Blocked inbound avoids provider calls",
            block_input,
            mock_supported=True,
        ),
        Scenario(
            "skills",
            ("skills", "references", "scripts", "gating", "history"),
            "Activate a skill, reference, gated tool, script and follow-up",
            skill_workflow,
            options={
                "tools": [inventory],
            },
            options_factory=skill_options,
        ),
        Scenario(
            "concurrent_rooms",
            ("concurrency", "identity", "isolation", "tools"),
            "Four rooms share one AI channel and provider",
            concurrent_rooms,
            options={
                "tools": [
                    tool(
                        "whoami",
                        "Read the current authenticated tool context's room and actor",
                        {},
                    )
                ]
            },
        ),
        Scenario(
            "idempotency",
            ("idempotency", "concurrency", "indexing"),
            "Four concurrent copies generate once",
            idempotency,
            mock_supported=True,
        ),
        Scenario(
            "muted_tools",
            ("permissions", "tools"),
            "Muted buffered AI executes tools without speaking",
            muted_tools,
            streaming=False,
            options={"tools": [inventory]},
        ),
        Scenario(
            "read_only",
            ("permissions",),
            "Read-only transport cannot inject",
            read_only,
            mock_supported=True,
        ),
        Scenario(
            "provider_retry",
            ("resilience",),
            "Injected transient 503 then live retry",
            retry,
            options={
                "retry_policy": RetryPolicy(
                    max_retries=1, base_delay_seconds=0.05, max_delay_seconds=0.05
                )
            },
            mock_supported=True,
            fault_injection=True,
        ),
        Scenario(
            "memory_window",
            ("memory", "history"),
            "Sliding window drops old content",
            memory_window,
            options_factory=lambda: {"memory": SlidingWindowMemory(max_events=2)},
        ),
        Scenario(
            "tool_search",
            ("tool_search", "tools"),
            "Search and execute from a deferred 13-tool catalog",
            tool_search,
            options={
                "tool_search": True,
                "tool_search_threshold": 2,
                "tools": [inventory]
                + [
                    tool(f"unrelated_{i}", f"Unrelated synthetic maintenance action {i}", {})
                    for i in range(12)
                ],
            },
        ),
        Scenario(
            "round_limit",
            ("limits", "streaming", "events"),
            "One-round tool cap is explicit",
            capped_loop,
            options={"tools": [inventory], "max_tool_rounds": 1},
        ),
    ]
    catalog.extend(
        [
            Scenario(
                "long_stream",
                ("streaming", "backpressure", "store"),
                "Deliver and persist a 100-number streamed answer",
                long_stream,
            ),
            Scenario(
                "muted_stream",
                ("permissions", "memory", "streaming"),
                "Muted stream skips generation but retains inbound history",
                muted_stream,
                mock_supported=True,
            ),
            Scenario(
                "visibility",
                ("visibility", "store"),
                "Hidden inbound is stored without reaching AI",
                visibility,
                mock_supported=True,
            ),
            Scenario(
                "ui_events",
                ("realtime", "typing", "presence", "reactions", "receipts"),
                "Five UI events delivered without persistence",
                ui_events,
                mock_supported=True,
            ),
            Scenario(
                "denied_tool",
                ("permissions", "hooks", "tools"),
                "A denied tool never executes and the model sees the denial",
                denied_tool,
                options={"tools": [inventory]},
            ),
        ]
    )
    for name in ("parallel_tools", "chained_tools", "tool_hooks"):
        original = next(s for s in catalog if s.name == name)
        catalog.append(
            Scenario(
                name + "_buffered",
                original.features,
                original.description + " (buffered)",
                original.run,
                streaming=False,
                options=original.options,
            )
        )
    return catalog
