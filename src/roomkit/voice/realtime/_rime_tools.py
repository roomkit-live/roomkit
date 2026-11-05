"""Prism tool rounds: exactly one result and an acknowledged continuation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.providers.ai.tool_calls import realtime_call_arguments

if TYPE_CHECKING:
    from roomkit.voice.realtime._rime_session import PrismSession


@dataclass
class _Round:
    calls: set[str] = field(default_factory=set)
    acknowledged: set[str] = field(default_factory=set)
    finished: bool = False


class PrismTools:
    def __init__(self, state: PrismSession) -> None:
        self.state = state
        self.rounds: dict[str, _Round] = {}
        self.parents: dict[str, str] = {}

    async def call(self, event: dict[str, Any]) -> None:
        state = self.state
        call_id, response_id = event.get("call_id", ""), event.get("response_id", "")
        if not response_id:
            await state.fire("error", "invalid_tool_call", "Prism tool call has no response ID")
            return
        if not state.owner._book_tool_call(state.session, call_id, response_id):
            return
        self.parents[call_id] = response_id
        self.rounds.setdefault(response_id, _Round()).calls.add(call_id)
        arguments = realtime_call_arguments(event.get("arguments", ""), cut=False)
        # A handler may await submit_tool_result(), which awaits a wire ack.
        # The receive loop must remain free to read that acknowledgement.
        state.spawn(state.fire("tool_call", call_id, event.get("name", ""), arguments), "tool")

    async def result(self, call_id: str, result: str) -> None:
        state = self.state
        answerable, response_id = state.owner._answerable_tool_call(state.session, call_id)
        if not answerable:
            return
        try:
            await state.request(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    },
                },
                "conversation.item.created",
            )
        except BaseException:
            await self.abandon(response_id)
            raise

    async def acknowledged(self, item: dict[str, Any]) -> None:
        if item.get("type") != "function_call_output":
            return
        call_id = item.get("call_id", "")
        response_id = self.parents.get(call_id)
        if response_id in self.rounds:
            self.rounds[response_id].acknowledged.add(call_id)
            await self.continue_ready()

    async def finished(self, response: dict[str, Any]) -> None:
        response_id = response.get("id", "")
        if response_id in self.rounds:
            # A cancelled reply does not undo calls Prism already issued.
            self.rounds[response_id].finished = True
            await self.continue_ready()

    async def continue_ready(self) -> None:
        state = self.state
        if state.closed or state.speaking or state.response_id or state.response_requested:
            return
        for response_id, turn in tuple(self.rounds.items()):
            if not turn.finished or turn.calls != turn.acknowledged:
                continue
            event_id = uuid4().hex
            state.response_requested = event_id
            state.ready.clear()
            self._remove(response_id)
            await state.send(
                {
                    "type": "response.create",
                    "event_id": event_id,
                    "response": {
                        "metadata": {
                            "prism_cause": "tool_continuation",
                            "prism_parent_response_id": response_id,
                        },
                    },
                }
            )
            return

    async def abandon(self, response_id: str, call_ids: list[str] | None = None) -> None:
        turn = self.rounds.get(response_id)
        ids = call_ids if call_ids is not None else list(turn.calls) if turn else []
        await self.state.owner._abandon_open_tool_calls(self.state.session, ids)
        self._remove(response_id)

    def _remove(self, response_id: str) -> None:
        turn = self.rounds.pop(response_id, None)
        if turn:
            for call_id in turn.calls:
                self.parents.pop(call_id, None)

    def clear(self) -> None:
        self.rounds.clear()
        self.parents.clear()
