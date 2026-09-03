"""Reading a message's image part for an AI request.

RoomKit carries an image in an :class:`AIImagePart` as a remote URL or a
``data:<media_type>;base64,<payload>`` URI, and every AI provider turns that
into what its API takes: Anthropic a media type and a base64 payload, Gemini
the bytes, the OpenAI-shaped APIs a data URI, Ollama a bare base64 string.
Each reading the URI itself is how three of them came to read it differently
— a header without a media type sent as ``media_type: ""`` here, defaulted to
``image/jpeg`` there, forwarded as it came elsewhere — and how a corrupt
payload reached the wire. These are the one reading, in the three shapes the
providers need. They live beside the part they read, not with the
image-generation providers that happen to share the URI helper.
"""

from __future__ import annotations

import base64

from roomkit.providers.ai.base import AIImagePart, ProviderError
from roomkit.providers.utils import parse_data_uri, to_data_uri


def image_part_payload(part: AIImagePart, *, provider: str) -> tuple[str, bytes]:
    """The media type and the decoded bytes of a ``data:`` URI image part.

    :func:`~roomkit.providers.utils.parse_data_uri` with the part's own
    ``mime_type`` as the fallback, and its ``ValueError`` surfaced as the
    non-retryable :class:`ProviderError` an AI provider's caller expects: a
    caller error, named as such, raised before the request leaves — never a
    retry or a fallback, since the same URI would fail again. Only for a
    ``data:`` URI; a remote URL is the provider's to forward.
    """
    try:
        return parse_data_uri(part.url, fallback_mime=part.mime_type)
    except ValueError as exc:
        raise ProviderError(
            f"invalid image part: {exc}", retryable=False, provider=provider
        ) from exc


def image_part_base64(part: AIImagePart, *, provider: str) -> tuple[str, str]:
    """The media type and the canonical base64 payload of a ``data:`` URI image part.

    For an API that takes the two apart — Anthropic's ``source`` block,
    Ollama's ``images`` list. Same reading as :func:`image_part_payload`,
    re-encoded from the validated bytes.
    """
    mime_type, data = image_part_payload(part, provider=provider)
    return mime_type, base64.b64encode(data).decode("ascii")


def image_part_uri(part: AIImagePart, *, provider: str) -> str:
    """The URI an OpenAI-shaped request forwards for an image part.

    A remote URL passes through untouched. A ``data:`` URI goes through
    :func:`image_part_payload` and is rebuilt canonically, so a header
    without a media type reaches the vendor with the part's, and a payload an
    encoder wrapped reaches it on one line.
    """
    if not part.url.startswith("data:"):
        return part.url
    mime_type, data = image_part_payload(part, provider=provider)
    return to_data_uri(data, mime_type)
