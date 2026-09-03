"""Translate RoomKit messages and context into a Gemini request.

The provider in ``ai.py`` owns the client, the call and the stream. This
module owns the other direction: how an ``AIMessage`` history, its images,
its tool calls and their thought signatures become the ``Content`` list and
the ``GenerateContentConfig`` the Gemini API accepts. It sits beside
``schema.py`` for the same reason that one does — request shaping reads on
its own, away from stream parsing.

``types`` is the SDK's ``google.genai.types`` module, handed in by the
provider rather than imported here: the SDK is an optional dependency that
``sdk.build_genai_client`` loads lazily, and the provider is what holds it.
"""

from __future__ import annotations

import base64
from typing import Any, cast

from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
)
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.gemini.schema import clean_gemini_schema


def _part_signature(part: AIToolCallPart) -> bytes | None:
    """Decode the thought_signature a tool call carries, if any.

    Stored base64 in the part's metadata (a portable str), sent to Gemini
    as the raw bytes it emitted.
    """
    sig = part.metadata.get("thought_signature")
    return base64.b64decode(sig) if sig else None


def _round_signature(parts: list[Any]) -> bytes | None:
    """Decode the first thought_signature carried by a round's tool calls.

    ``None`` when no call in the round carries one — nothing to borrow,
    and the request will be rejected if the model was thinking.
    """
    for p in parts:
        if isinstance(p, AIToolCallPart):
            sig = _part_signature(p)
            if sig is not None:
                return sig
    return None


def format_messages(types: Any, messages: list[AIMessage]) -> list[Any]:
    """Convert AIMessage list to Gemini Content format."""
    contents = []
    for msg in messages:
        if isinstance(msg.content, list) and any(
            isinstance(p, AIToolCallPart) for p in msg.content
        ):
            # Model message with function calls
            contents.append(
                types.Content(role="model", parts=_model_call_parts(types, msg.content))
            )
        elif isinstance(msg.content, list) and any(
            isinstance(p, AIToolResultPart) for p in msg.content
        ):
            contents.extend(_tool_result_contents(types, msg.content))
        else:
            role = "model" if msg.role == "assistant" else "user"
            parts = format_content(types, msg.content)
            contents.append(types.Content(role=role, parts=parts))
    return contents


def _model_call_parts(types: Any, content: list[Any]) -> list[Any]:
    """Parts of a model turn that carries function calls, replayed signed.

    The round's signature is lent to the calls Gemini left unsigned: the API
    emits the thought_signature on the FIRST functionCall part of a parallel
    group only, but its validator rejects any history functionCall part
    without one ("Function call is missing a thought_signature", observed
    live on gemini-3.5-flash with a 2-call round). Reusing the group's
    signature satisfies it. Resolved by scanning the whole message up front
    rather than carried forward as the loop meets it: the signature is not
    guaranteed to reach the part that ends up first, and a call preceding
    the signed one would otherwise replay bare — the very 400 this borrowing
    exists to prevent.
    """
    round_sig = _round_signature(content)
    parts = []
    for p in content:
        if isinstance(p, AITextPart):
            parts.append(types.Part.from_text(text=p.text))
        elif isinstance(p, AIToolCallPart):
            own_sig = _part_signature(p)
            sig = own_sig if own_sig is not None else round_sig
            if sig is None:
                parts.append(types.Part.from_function_call(name=p.name, args=p.arguments))
                continue
            # No ``thought=True`` here — a signed functionCall part is NOT a
            # thought part, and flagging it as one desyncs Google's validator
            # from the shape it originally streamed.
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(name=p.name, args=p.arguments),
                    thought_signature=sig,
                )
            )
    return parts


def _tool_result_contents(types: Any, content: list[Any]) -> list[Any]:
    """Contents replaying a tool-result turn: the responses, then any images.

    A FunctionResponse.response is a JSON Struct — it can't carry image
    bytes, so an image tool result keeps the function response text-only and
    the image is decoded onto a following user Content via ``_image_part``
    (inline bytes the model can actually see). Text results are unchanged.
    """
    parts = []
    image_parts: list[Any] = []
    for p in content:
        if isinstance(p, AIToolResultPart):
            text, images = p.split_for_message()
            parts.append(types.Part.from_function_response(name=p.name, response={"result": text}))
            image_parts.extend(_image_part(types, img) for img in images)
    contents = [types.Content(role="user", parts=parts)]
    if image_parts:
        contents.append(types.Content(role="user", parts=image_parts))
    return contents


def format_content(
    types: Any,
    content: (
        str | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart | AIThinkingPart]
    ),
) -> list[Any]:
    """Convert content to Gemini Parts."""
    if isinstance(content, str):
        return [types.Part.from_text(text=content)]

    parts = []
    for item in content:
        if isinstance(item, AITextPart):
            parts.append(types.Part.from_text(text=item.text))
        elif isinstance(item, AIImagePart):
            parts.append(_image_part(types, item))
    return parts


def _image_part(types: Any, item: AIImagePart) -> Any:
    """Build a Gemini image Part from an ``AIImagePart``.

    RoomKit carries images as ``data:<media_type>;base64,<data>``
    URIs — the convention the Anthropic and OpenAI providers consume.
    Gemini's ``from_uri`` expects a fetchable file URI (``gs://`` or
    ``https://``); handed a data URI it ships a broken reference and
    the model never sees the image. Decode a data URI to inline bytes
    via ``from_bytes``; pass a real URI through to ``from_uri``.
    """
    url = item.url
    if url.startswith("data:"):
        header, _, b64data = url.partition(",")
        media_type = header[len("data:") :].split(";", 1)[0] or (item.mime_type or "image/jpeg")
        return types.Part.from_bytes(
            data=base64.b64decode(b64data),
            mime_type=media_type,
        )
    return types.Part.from_uri(
        file_uri=url,
        mime_type=item.mime_type or "image/jpeg",
    )


def build_gen_config(types: Any, config: GeminiConfig, context: AIContext) -> Any:
    """Build Gemini generation config from AIContext."""
    gen_config = types.GenerateContentConfig(
        temperature=context.temperature,
        max_output_tokens=context.max_tokens or config.max_tokens,
    )

    # Thinking config. ``include_thoughts=True`` is required for Gemini to
    # stream thought summaries — without it the model still reasons but the
    # reasoning never reaches the response. ``thinking_level`` targets Gemini
    # 3.x; ``thinking_budget`` (from the per-turn context) targets 2.5.
    thinking_level = config.thinking_level
    thinking_budget = context.thinking_budget
    if thinking_level:
        gen_config.thinking_config = types.ThinkingConfig(
            thinking_level=thinking_level,
            include_thoughts=True,
        )
    elif thinking_budget:
        gen_config.thinking_config = types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=True,
        )

    if context.system_prompt:
        gen_config.system_instruction = context.system_prompt

    if context.tools:
        func_decls = [
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=cast(Any, clean_gemini_schema(t.parameters)) if t.parameters else None,
            )
            for t in context.tools
        ]
        gen_config.tools = [types.Tool(function_declarations=func_decls)]

    return gen_config


def reject_model_turn_tail(contents: list[Any]) -> None:
    """Refuse a request whose history ends with the model speaking.

    Gemini answers a user turn; handed a history ending on a model one it
    replies ``400 "Requests ending with a model turn are not supported."``,
    which reaches the conversation as an opaque provider rejection. The
    cause is upstream — a turn generated with no new input to answer, most
    often concurrent turns on one room each rebuilding a history that ends
    on another's reply. Raised here so the report names that, rather than
    appending a continuation turn the application never wrote: doing so
    would answer a prompt nobody sent and hide the race that produced it.
    """
    tail = 0
    for content in reversed(contents):
        if getattr(content, "role", None) != "model":
            break
        tail += 1
    if not tail:
        return
    raise ProviderError(
        "Gemini requires the request to end with a user turn; the history ends "
        f"with a model turn ({tail} trailing model content(s)). Append the "
        "triggering user message, or drop the trailing model turn, before "
        "generating.",
        retryable=False,
        provider="gemini",
    )
