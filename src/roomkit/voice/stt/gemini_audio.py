"""Audio sources for the Gemini STT provider: inline or uploaded.

The provider in ``gemini.py`` owns the client, the prompt and the
transcription call. This module owns how a recording reaches the request: a
path, a ``data:`` or Files API URL, raw frames or chunks; inlined when small
enough and uploaded through the Files API otherwise, with the mime
normalisation the interactions endpoint demands and the cleanup of what was
uploaded.

It takes the two facts it needs (the inline size bound and the upload's read
budget) and a ``get_client`` accessor rather than a client or the config, so
it neither owns a client nor depends on the provider module: the upload and
the delete ask for one when they run, and the provider decides how it is built.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# The provider's logger, so the upload and cleanup lines keep the name they
# had when they lived there.
logger = logging.getLogger("roomkit.voice.stt.gemini")

SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "audio/wav",
        "audio/mp3",
        "audio/aiff",
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/mpeg",
        "audio/m4a",
        "audio/l16",
        "audio/s16le",
        "audio/opus",
        "audio/alaw",
        "audio/mulaw",
    }
)
"""Mime types the interactions endpoint accepts for audio input.

Taken from the service's own rejection message rather than the prose docs,
which list fewer: raw PCM (``audio/l16``, ``audio/s16le``) and the telephony
codecs are accepted but undocumented. Verified 2026-08-07.
"""

_PCM_MIME_TYPE = "audio/l16"
"""What roomkit's own frames and chunks are: 16-bit little-endian PCM."""

_FILES_API_HOST = "generativelanguage.googleapis.com"

_FILES_DELETE_TIMEOUT = 10.0
"""The Files API delete is cleanup, not the transcript: seconds, not minutes."""

ClientFactory = Callable[[], Any]
"""The provider's lazy client accessor, invoked only by the paths that need one."""


def _files_config(seconds: float) -> dict[str, Any]:
    """Per-call options for the Files API, bounded by *seconds*.

    Those calls go through the SDK's classic request path, which hands
    httpx ``timeout=None`` (no timeout at all) unless ``HttpOptions.timeout``
    is set, so a stalled upload of a long recording never returned. That
    option is one value in milliseconds the SDK spreads over the connect
    as well; the client's request hook caps that connect at
    ``connect_timeout`` (see ``build_genai_client``), so this is the read
    budget.
    """
    return {"http_options": {"timeout": int(seconds * 1000)}}


def _mime_for(path: Path) -> str:
    """Pick a mime type the endpoint accepts for *path*.

    ``mimetypes`` answers ``audio/x-wav`` for ``.wav`` on some platforms and
    the service rejects it, so the guess is normalised rather than trusted.
    """
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed in SUPPORTED_MIME_TYPES:
        return guessed
    by_suffix = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".m4a": "audio/m4a",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".opus": "audio/opus",
        ".flac": "audio/flac",
        ".aiff": "audio/aiff",
        ".aif": "audio/aiff",
    }
    mime = by_suffix.get(path.suffix.lower())
    if mime is None:
        raise ValueError(
            f"Cannot infer a supported audio mime type for {path.name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
        )
    return mime


async def audio_part(
    source: Any, *, max_inline_bytes: int, upload_timeout: float, get_client: ClientFactory
) -> tuple[dict[str, Any], str | None]:
    """Turn *source* into an audio content block.

    Returns the block and, when the recording was uploaded, the file name to
    delete afterwards.
    """
    if isinstance(source, str | Path):
        text = str(source)
        if text.startswith(("data:", "http://", "https://", "file://")):
            return await _part_from_url(
                text,
                max_inline_bytes=max_inline_bytes,
                upload_timeout=upload_timeout,
                get_client=get_client,
            )
        return await _part_from_path(
            Path(source),
            max_inline_bytes=max_inline_bytes,
            upload_timeout=upload_timeout,
            get_client=get_client,
        )

    url = getattr(source, "url", None)
    if url is not None:
        return await _part_from_url(
            str(url),
            max_inline_bytes=max_inline_bytes,
            upload_timeout=upload_timeout,
            get_client=get_client,
        )

    data = getattr(source, "data", None)
    if not isinstance(data, bytes):
        raise TypeError(f"Cannot transcribe {type(source).__name__}: no audio bytes found")
    if not data:
        raise ValueError("Cannot transcribe empty audio")
    return (
        {
            "type": "audio",
            "data": base64.b64encode(data).decode(),
            "mime_type": _PCM_MIME_TYPE,
            "sample_rate": getattr(source, "sample_rate", 16000),
            "channels": getattr(source, "channels", 1),
        },
        None,
    )


async def _part_from_path(
    path: Path, *, max_inline_bytes: int, upload_timeout: float, get_client: ClientFactory
) -> tuple[dict[str, Any], str | None]:
    mime = _mime_for(path)
    try:
        info = await asyncio.to_thread(path.stat)
    except OSError as exc:
        raise FileNotFoundError(f"No such recording: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise FileNotFoundError(f"Not a recording file: {path}")

    size = info.st_size
    if size <= max_inline_bytes:
        data = await asyncio.to_thread(path.read_bytes)
        return (
            {
                "type": "audio",
                "data": base64.b64encode(data).decode(),
                "mime_type": mime,
            },
            None,
        )

    logger.debug("Uploading %s (%d bytes) through the Files API", path.name, size)
    uploaded = await get_client().aio.files.upload(
        file=str(path), config=_files_config(upload_timeout)
    )
    # The upload guesses its own mime and can answer ``audio/x-wav``, which
    # the interactions endpoint rejects — send the normalised one.
    return ({"type": "audio", "uri": uploaded.uri, "mime_type": mime}, uploaded.name)


async def _part_from_url(
    url: str, *, max_inline_bytes: int, upload_timeout: float, get_client: ClientFactory
) -> tuple[dict[str, Any], str | None]:
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        mime = header[5:].split(";", 1)[0] or "audio/wav"
        if mime not in SUPPORTED_MIME_TYPES:
            raise ValueError(f"Unsupported audio mime type in data URL: {mime}")
        return ({"type": "audio", "data": payload, "mime_type": mime}, None)

    parsed = urlparse(url)
    if parsed.scheme in ("", "file"):
        return await _part_from_path(
            Path(parsed.path or url),
            max_inline_bytes=max_inline_bytes,
            upload_timeout=upload_timeout,
            get_client=get_client,
        )
    if parsed.scheme == "https" and parsed.hostname == _FILES_API_HOST:
        return ({"type": "audio", "uri": url, "mime_type": "audio/wav"}, None)

    raise ValueError(
        f"GeminiSTT will not fetch {parsed.scheme}://{parsed.hostname} — the provider "
        "does not dereference arbitrary URLs. Pass a local path, raw audio, or a "
        "Files API URI."
    )


async def delete_upload(name: str, *, get_client: ClientFactory) -> None:
    """Remove an uploaded recording. Failing to is not worth an exception —
    the Files API expires uploads on its own."""
    try:
        # Best-effort cleanup awaited before the transcript is returned: a
        # stalled DELETE must not hold it for the whole read budget.
        await get_client().aio.files.delete(name=name, config=_files_config(_FILES_DELETE_TIMEOUT))
    except Exception:  # pragma: no cover - best effort
        logger.debug("Could not delete uploaded recording %s", name, exc_info=True)
