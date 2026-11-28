"""Fast approximate token estimation for context budget management."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from roomkit._media import content_text, is_vision_image, media_label
from roomkit.models.event import (
    AudioContent,
    CompositeContent,
    MediaContent,
    TextContent,
    VideoContent,
)
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AITextPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.utils import extract_event_text as _transport_text

if TYPE_CHECKING:
    from roomkit.models.event import RoomEvent


# Flat cost of one image: vendors bill an image by its (downscaled) size, a
# thousand-odd tokens for a screenshot, whatever its base64 length.
_IMAGE_TOKENS = 1000


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ~ 4 characters for English text."""
    return len(text) // 4 + 1


def estimate_tool_tokens(tool: AITool) -> int:
    """Estimate tokens for a single tool definition sent to the model.

    Counts the name, description, and the JSON schema of the parameters —
    the parts a provider serializes into the request's tool list.
    """
    total = estimate_tokens(tool.name) + estimate_tokens(tool.description)
    if tool.parameters:
        total += estimate_tokens(json.dumps(tool.parameters))
    return total


def estimate_message_tokens(message: AIMessage) -> int:
    """Estimate tokens for a complete message including role overhead."""
    overhead = 4  # role, delimiters
    if isinstance(message.content, str):
        return overhead + estimate_tokens(message.content)
    total = overhead
    for part in message.content:
        if isinstance(part, AITextPart):
            total += estimate_tokens(part.text)
        elif isinstance(part, AIToolCallPart):
            args_str = (
                json.dumps(part.arguments)
                if isinstance(part.arguments, dict)
                else str(part.arguments)
            )
            total += estimate_tokens(part.name) + estimate_tokens(args_str)
        elif isinstance(part, AIToolResultPart):
            # as_text() stands a "[image]" for each image; the images are
            # billed as images all the same.
            total += estimate_tokens(part.as_text())
            if isinstance(part.result, list):
                images = sum(isinstance(p, AIImagePart) for p in part.result)
                total += images * _IMAGE_TOKENS
        elif isinstance(part, AIImagePart):
            total += _IMAGE_TOKENS
    return total


def extract_event_text(event: RoomEvent) -> str:
    """The text of ``event`` as the memory layer reads it.

    The transports' extraction (:func:`roomkit.providers.utils.extract_event_text`),
    with two differences: a media reads as its label and a composite as its
    parts, never a ``data:`` URI or a reference (RFC §6.6); other content that
    carries no text, a location, a tool call, is rendered with ``str()``
    rather than dropped, because it still costs tokens. A summary reads
    every content but a tool call's row by its text form instead, and a tool
    call's row by this rendering (:func:`roomkit.memory._summary.summary_text`).
    """
    text = _transport_text(event)
    if text or isinstance(event.content, TextContent):
        return text
    labelled = content_text(event.content)
    return labelled if labelled is not None else str(event.content)


def estimate_event_tokens(event: RoomEvent, *, text_only: bool = False) -> int:
    """Estimate a RoomEvent without billing media payloads as text.

    ``text_only`` omits approximate vision costs and non-text metadata. A
    rejection based on message length cannot rely on the flat image estimate,
    which may overstate what a small image actually costs.
    """
    content = event.content
    if isinstance(content, CompositeContent):
        return sum(
            estimate_event_tokens(event.model_copy(update={"content": part}), text_only=text_only)
            for part in content.parts
        )
    if isinstance(content, MediaContent) and not is_vision_image(content):
        # Read as its label, never as an image (RFC §6.4).
        return estimate_tokens(media_label(content))
    if isinstance(content, (MediaContent, AudioContent, VideoContent)):
        parts: list[AITextPart | AIImagePart] = []
        text = getattr(content, "caption", None) or getattr(content, "transcript", None)
        if text:
            parts.append(AITextPart(text=text))
        if isinstance(content, MediaContent) and not text_only:
            parts.append(AIImagePart(url=content.url, mime_type=content.mime_type))
        return estimate_message_tokens(AIMessage(role="user", content=parts)) if parts else 0
    if text_only:
        text = _transport_text(event)
        return estimate_tokens(text) if text else 0
    return estimate_tokens(extract_event_text(event))


def estimate_context_tokens(context: AIContext) -> int:
    """Estimate total tokens for an AIContext."""
    total = 0
    if context.system_prompt:
        total += estimate_tokens(context.system_prompt)
    for msg in context.messages:
        total += estimate_message_tokens(msg)
    if context.tools:
        for tool in context.tools:
            total += estimate_tool_tokens(tool)
    return total


def estimate_notes_tokens(notes: list[str]) -> int:
    """What a memory's notes for the turn occupy in the window: they ride the
    turn's input, outside the history, and no trimmer can cut them."""
    return sum(estimate_tokens(note) for note in notes)


def history_budget(
    *,
    max_context_tokens: int,
    safety_margin_ratio: float = 0.15,
    reserved_tokens: int = 0,
    messages: list[AIMessage] | None = None,
    current_event: RoomEvent | None = None,
    reply_tokens: int = 0,
) -> int:
    """Tokens the conversation history may occupy, once the rest of the prompt is paid for.

    A context window holds the system prompt, the tool
    schemas, whatever pre-built messages the memory layer injects, and the
    history plus the current turn. A trimmer measuring only the history
    can return a result that overflows the very window it was given — which is
    what ``max_context_tokens * (1 - safety_margin_ratio)`` computed alone did.

    So the arithmetic is explicit here, in one place:

    - ``max_context_tokens * (1 - safety_margin_ratio)`` is what the *whole*
      prompt may occupy; the margin is headroom for the model's reply. The
      default duplicates the ``0.15`` the provider constructors declare, and
      stays anyway: it shipped in the public signature (0.57.0), so dropping
      it is an API break, not a cleanup.
    - ``reply_tokens`` is the reply budget the turn requests, when known
      (:class:`~roomkit.tools.context.TurnFootprint`). The reply is reserved
      once: the larger of the margin and this budget, never both.
    - ``reserved_tokens`` is the non-history part the caller knows about and the
      trimmer cannot see — system prompt and tool schemas. 0 declares "nothing
      besides what is passed here occupies the window".
    - ``messages`` are the injected blocks the trimmer passes through untouched.
      They are not trimmable, so they are subtracted rather than cut.
    - ``current_event`` is the turn the channel appends after retrieving memory.
      It is not history and cannot be trimmed. Its cost is local to this call,
      never added to the wrapper's shared ``reserved_tokens``.

    What remains is the history's, and nothing else's.
    """
    prompt_budget = min(
        int(max_context_tokens * (1 - safety_margin_ratio)), max_context_tokens - reply_tokens
    )
    injected = sum(estimate_message_tokens(m) for m in messages or ())
    current = estimate_event_tokens(current_event) if current_event is not None else 0
    return max(0, prompt_budget - reserved_tokens - injected - current)
