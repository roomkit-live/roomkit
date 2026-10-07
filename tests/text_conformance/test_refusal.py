"""A refusal reads the same streamed or not on the OpenAI wire (RMK-531,
RFC §6.4): its text in the end's ``metadata["refusal"]``, the answer empty."""

from __future__ import annotations

import httpx
import pytest

from roomkit.providers.ai.base import AIContext, AIMessage
from tests.text_conformance.http_wire import HttpWire, http_wires, sse
from tests.text_conformance.openai_wire import wires

REFUSAL = "I cannot help with that."
CONTEXT = AIContext(messages=[AIMessage(role="user", content="go")])
LABELS = {wire.label for wire in wires()}
OPENAI = [wire for wire in http_wires() if wire.label in LABELS]


def _completion() -> dict[str, object]:
    message = {"role": "assistant", "content": None, "refusal": REFUSAL}
    return {
        "id": "chatcmpl-0",
        "object": "chat.completion",
        "created": 0,
        "model": "served-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


def _chunk(delta: dict[str, object], finish: str | None = None) -> dict[str, object]:
    return {
        "id": "chatcmpl-0",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "served-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


@pytest.mark.parametrize("wire", OPENAI, ids=[wire.label for wire in OPENAI])
@pytest.mark.parametrize("mode", ["generate", "stream"])
async def test_a_refusal_reads_the_same_streamed_or_not(wire: HttpWire, mode: str) -> None:
    if mode == "generate":
        provider = wire.build(lambda request: httpx.Response(200, json=_completion()))
        response = await provider.generate(CONTEXT)
        text, metadata = response.content, response.metadata
    else:
        body = sse([_chunk({"refusal": REFUSAL}), _chunk({}, "stop")]) + b"data: [DONE]\n\n"
        provider = wire.build(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=body
            )
        )
        events = [event async for event in provider.generate_structured_stream(CONTEXT)]
        text = "".join(getattr(event, "text", "") for event in events[:-1])
        metadata = events[-1].metadata
    await provider.close()

    assert text == ""
    assert metadata.get("refusal") == REFUSAL
