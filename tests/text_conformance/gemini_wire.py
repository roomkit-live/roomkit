"""The Gemini wire (Google AI and Vertex): parts in candidates."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from google.genai import types

from roomkit.providers.ai.base import AIProvider
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.gemini.vertex import GeminiVertexConfig, GeminiVertexProvider
from tests.text_conformance.chat_wire import iterate
from tests.text_conformance.driver import (
    ARGUMENT_TEXT,
    CACHE_WRITE_USAGE,
    CALL_INDEX,
    COMPOSITION,
    FUNCTIONLESS_CALL,
    REDACTED_REASONING,
    REPEATED_ID,
    SCHEMA_AS_GIVEN,
    SIGNED_REASONING,
    THINK_TAGS,
    WRITTEN_UNREADABLE,
    Driver,
    ReasoningConvention,
)
from tests.text_conformance.script import Item, Reasoning, Script

# Gemini has no stop reason of its own for a round of calls: it ends STOP.
_FINISH = {
    "stop": types.FinishReason.STOP,
    "tool": types.FinishReason.STOP,
    "cut": types.FinishReason.MAX_TOKENS,
    # Gemini caps the output at what the context window has left.
    "context": types.FinishReason.MAX_TOKENS,
    "filtered": types.FinishReason.SAFETY,
    "malformed": types.FinishReason.MALFORMED_FUNCTION_CALL,
    "unexpected": types.FinishReason.UNEXPECTED_TOOL_CALL,
    "none": None,
}

_CANNOT = {
    ARGUMENT_TEXT: "function-call arguments arrive parsed, as an object, never as text",
    WRITTEN_UNREADABLE: (
        "FunctionCall.args is an object by type; a call Gemini cannot parse never "
        "arrives, the response ends MALFORMED_FUNCTION_CALL"
    ),
    CALL_INDEX: "a function call carries no stream index",
    REPEATED_ID: "a FunctionCall id is unique; the same id again is that call re-emitted",
    SIGNED_REASONING: (
        "one thought_signature signs the round, on its first function call, not each thought part"
    ),
    REDACTED_REASONING: "Gemini has no redacted reasoning",
    THINK_TAGS: "reasoning comes in thought parts; text is the answer's",
    CACHE_WRITE_USAGE: "Gemini's usage counts no cache writes",
    FUNCTIONLESS_CALL: "a function-call part always carries its FunctionCall",
    SCHEMA_AS_GIVEN: (
        "the provider declares Gemini's OpenAPI subset (parameters), not "
        "parameters_json_schema, which is unmeasured (RMK-386)"
    ),
}


def _thought(block: Reasoning) -> types.Part:
    if block.redacted is not None:
        raise ValueError("Gemini has no redacted reasoning")
    return types.Part(text=block.text, thought=True)


def _call_parts(script: Script) -> list[types.Part]:
    """Each call whole, its arguments parsed."""
    ids = [call.id for call in script.calls if call.id]
    if len(ids) != len(set(ids)):
        raise ValueError(f"a Gemini function call id is unique; the script repeats one: {ids}")
    return [
        types.Part(
            function_call=types.FunctionCall(
                id=call.id, name=call.name, args=json.loads(call.arguments)
            )
        )
        for call in script.calls
    ]


def _pieces(script: Script) -> list[list[types.Part]]:
    """The response's parts, one list per stream chunk: a thought per chunk,
    the text, then each call or all of them at once.

    Gemini signs one part of a round: its first function call (measured, see
    ``roomkit.providers.gemini.request``), or with no call its last part.
    """
    pieces = [[_thought(block)] for block in script.reasoning]
    if script.text:
        pieces.append([types.Part(text=script.text)])
    calls = _call_parts(script)
    if calls and script.calls_in_one_chunk:
        pieces.append(calls)
    else:
        pieces.extend([part] for part in calls)
    signatures = [block.signature for block in script.reasoning if block.signature]
    if signatures and pieces:
        signed = calls[0] if calls else pieces[-1][-1]
        signed.thought_signature = signatures[-1].encode()
    return pieces


def _usage(script: Script) -> types.GenerateContentResponseUsageMetadata:
    # The prompt count includes the cached prefix, the candidates count leaves
    # thinking out, and Gemini counts no cache writes.
    usage = script.usage
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage.input + usage.cache_read,
        cached_content_token_count=usage.cache_read,
        candidates_token_count=usage.output - usage.reasoning,
        thoughts_token_count=usage.reasoning,
        total_token_count=usage.input + usage.cache_read + usage.output,
    )


def _chunk(
    script: Script,
    parts: list[types.Part],
    finish: types.FinishReason | None = None,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
) -> types.GenerateContentResponse:
    candidate = types.Candidate(
        content=types.Content(role="model", parts=parts), finish_reason=finish, index=0
    )
    return types.GenerateContentResponse(
        candidates=[candidate], usage_metadata=usage, model_version=script.answered_by
    )


def _stream(script: Script) -> list[types.GenerateContentResponse]:
    """The last chunk carries the stop reason and the usage, or the usage
    comes after it on a chunk with no candidate."""
    pieces = _pieces(script)
    last = pieces.pop() if pieces else []
    usage = _usage(script)
    chunks = [
        *(_chunk(script, parts) for parts in pieces),
        _chunk(script, last, _FINISH[script.finish], None if script.usage_alone else usage),
    ]
    if script.usage_alone:
        chunks.append(
            types.GenerateContentResponse(usage_metadata=usage, model_version=script.answered_by)
        )
    return chunks


def _response(script: Script) -> types.GenerateContentResponse:
    parts = [part for parts in _pieces(script) for part in parts]
    return _chunk(script, parts, _FINISH[script.finish], _usage(script))


def _json_spelling(node: Any) -> Any:
    """A dumped Gemini ``Schema`` in JSON Schema's spelling: its types lowercase."""
    if isinstance(node, list):
        return [_json_spelling(item) for item in node]
    if not isinstance(node, dict):
        return node
    spelled: dict[str, Any] = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            spelled[key] = value.lower()
        elif key == "properties" and isinstance(value, dict):
            spelled[key] = {name: _json_spelling(schema) for name, schema in value.items()}
        elif key in ("items", "anyOf"):
            spelled[key] = _json_spelling(value)
        else:
            spelled[key] = value
    return spelled


def _declared_schema(declaration: types.FunctionDeclaration) -> dict[str, Any]:
    """``parameters_json_schema`` as given, else ``parameters``, Gemini's
    OpenAPI subset."""
    if declaration.parameters_json_schema is not None:
        return declaration.parameters_json_schema
    if declaration.parameters is None:
        return {}
    dumped = declaration.parameters.model_dump(mode="json", exclude_none=True, by_alias=True)
    return _json_spelling(dumped)


def _result_text(value: Any) -> str:
    """What the model reads of a response value. RoomKit wraps its text as
    ``{"result": text}``, which Gemini reads as the whole output."""
    if isinstance(value, dict) and set(value) == {"result"}:
        value = value["result"]
    return value if isinstance(value, str) else json.dumps(value)


def _b64(signature: bytes | None) -> str | None:
    return base64.b64encode(signature).decode("ascii") if signature else None


def _signature(signature: bytes | str | None) -> str | None:
    """A call's signature as the script wrote it."""
    return signature.decode() if isinstance(signature, bytes) else signature


class GeminiWire(Driver):
    """A provider on google-genai's ``models.generate_content*`` calls."""

    # The round's thought_signature goes back on each of its calls; the
    # thought parts themselves are not replayed.
    reasoning: ReasoningConvention = "call_signature"
    # FunctionResponse.response takes an "error" key for a failed call.
    error_flag = True
    calls_by_name = True
    refused_names = ("1lookup", "-lookup")
    accepted_names = ("lookup", "files.read", "fs:read")

    def __init__(
        self,
        provider_cls: type[AIProvider],
        build: Callable[[], AIProvider],
        *,
        label: str,
        cannot: dict[str, str],
    ) -> None:
        super().__init__()
        self._build = build
        self.label = label
        self.covers = (provider_cls,)
        self.cannot = cannot

    def provider(self, script: Script) -> AIProvider:
        provider = self._build()
        chunks = _stream(script)

        async def generate_content_stream(**kwargs: Any) -> Any:
            self.requests.append(kwargs)
            return iterate(chunks)

        async def generate_content(**kwargs: Any) -> Any:
            self.requests.append(kwargs)
            return _response(script)

        models = SimpleNamespace(
            generate_content_stream=generate_content_stream, generate_content=generate_content
        )
        provider._client = SimpleNamespace(aio=SimpleNamespace(models=models))  # type: ignore[attr-defined]
        return provider

    def declared(self, request: Any) -> dict[str, dict[str, Any]]:
        return {
            declaration.name or "": _declared_schema(declaration)
            for tool in request["config"].tools or []
            for declaration in tool.function_declarations or []
        }

    def replayed(self, request: Any) -> list[Item]:
        items: list[Item] = []
        for content in request["contents"]:
            for part in content.parts or []:
                items.extend(self._part_items(content.role, part))
        return items

    def _part_items(self, role: str | None, part: types.Part) -> list[Item]:
        if part.function_response is not None:
            return [self._result(part.function_response)]
        if role == "user":
            return [("image",)] if part.inline_data or part.file_data else []
        if part.function_call is not None:
            call = part.function_call
            ref = call.name if self.calls_by_name else call.id
            items: list[Item] = [("call", ref, dict(call.args or {}))]
            if part.thought_signature:
                items.append(("signature", ref, _signature(part.thought_signature)))
            return items
        if part.thought:
            return [("thinking", part.text or "", _b64(part.thought_signature))]
        return [("text", part.text)] if part.text else []

    def _result(self, response: types.FunctionResponse) -> Item:
        body = response.response or {}
        ref = response.name if self.calls_by_name else response.id
        failed = "error" in body
        text = _result_text(body["error"] if failed else body.get("output", body))
        return ("result", ref, text, failed if self.error_flag else None)


def wires() -> list[Driver]:
    """Google AI and Vertex: one provider, two clients."""
    return [
        GeminiWire(
            GeminiAIProvider,
            lambda: GeminiAIProvider(GeminiConfig(api_key="k")),
            label="gemini",
            cannot={
                **_CANNOT,
                COMPOSITION: "the Gemini API streams each call whole; partial_args is Vertex only",
            },
        ),
        GeminiWire(
            GeminiVertexProvider,
            lambda: GeminiVertexProvider(GeminiVertexConfig(project="p", location="europe-west1")),
            label="vertex",
            cannot={
                **_CANNOT,
                COMPOSITION: (
                    "the provider does not ask Vertex to stream arguments "
                    "(stream_function_call_arguments), so each call arrives whole"
                ),
            },
        ),
    ]
