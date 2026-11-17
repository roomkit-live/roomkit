"""How a media URL is read, before any model exists: a data URI, a link, or a
reference (RFC §5, §6.4).

A ``data:`` URI carries the media; a link (``http`` or ``https``) is fetched by
its reader; a reference, ``<scheme>://...`` in a scheme the application
chooses, names a media the application holds. No import from RoomKit: the
content model's own validator reads URLs with these.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

_LINK_SCHEMES = ("http://", "https://")
_REFERENCE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]+)://(.*)$", re.DOTALL)
# A control character (Unicode Cc: C0, DEL and C1), which a path reader may
# cut a name at or drop.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
# What ends a path segment for a reader: a slash, or the query or fragment.
_SEGMENT_END = re.compile(r"[/?#]")
# The percent-decodings a reference is read through; one still encoded after
# them is refused.
_MOST_DECODINGS = 3
# A percent escape, which a decoding replaces.
_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
# A scheme a reader fetches, opens or runs by itself, without the
# application: files on the machine, in an archive or on a share, the network
# protocols generic readers (curl, ffmpeg, GIO) fetch, the wrappers that read
# through another protocol, a file or a descriptor, code to run, and a link's,
# which a reader that splits a compound scheme on "+" fetches.
_REFUSED_SCHEMES = frozenset(
    {
        # Files, on the machine, in an archive or on a share.
        "admin", "afp", "archive", "cifs", "dav", "davs", "file", "ftp",
        "ftps", "local", "nfs", "phar", "recent", "resource", "scp", "sftp",
        "smb", "smbs", "tar", "tftp", "trash", "webdav", "zip",
        # Network protocols a generic reader fetches.
        "amqp", "dict", "ffrtmpcrypt", "ffrtmphttp", "gopher", "gophers",
        "hls", "http", "httpproxy", "https", "icecast", "imap", "imaps",
        "ipfs", "ipns", "ldap", "ldaps", "mmsh", "mmst", "mqtt", "pop3",
        "pop3s", "rist", "rtmp", "rtmpe", "rtmps", "rtmpt", "rtmpte",
        "rtmpts", "rtp", "rtsp", "rtsps", "sctp", "smtp", "smtps", "srt",
        "srtp", "ssh", "tcp", "telnet", "tls", "udp", "udplite", "unix", "ws",
        "wss", "zmq",
        # Wrappers that read through another protocol, a file or a descriptor.
        "async", "bluray", "cache", "concat", "concatf", "crypto", "fd",
        "pipe", "subfile",
        # Code to run.
        "expect", "jar", "javascript", "php", "vbscript",
    }
)  # fmt: skip
# A drive letter, which a path reader roots a path at.
_DRIVE = re.compile(r"[A-Za-z]:")
_TYPE_ALIASES = {"image/jpg": "image/jpeg", "image/pjpeg": "image/jpeg"}

VISION_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
"""The image types every vision provider takes (RFC §6.4): what RoomKit sends
as an image, a tool's replayed or an author's."""


def is_data_uri(url: str) -> bool:
    """Whether *url* carries its media inline."""
    return url[: len("data:")].lower() == "data:"


def is_link(url: str) -> bool:
    """Whether *url* is a link a reader fetches (``http`` or ``https``)."""
    return url[: len("https://")].lower().startswith(_LINK_SCHEMES)


def is_reference(url: str) -> bool:
    """Whether *url* names a media the application holds and resolves."""
    return not is_data_uri(url) and not is_link(url)


def lowercase_scheme(url: str) -> str:
    """*url* with a link's or a data URI's scheme in lower case, the spelling
    every reader tests: neonize fetches only ``^https?://`` and opens any other
    string as a local file, so ``HTTP://../secret`` would be read from disk. A
    reference keeps its scheme, which is the application's to spell."""
    if not (is_link(url) or is_data_uri(url)):
        return url
    scheme, rest = url.split(":", 1)
    return f"{scheme.lower()}:{rest}"


def is_safe_reference(url: str) -> bool:
    """Whether *url* is a reference no reader could take for something else:
    two letters of scheme at least (one reads as a Windows drive), no part of
    it between ``+`` a scheme a reader fetches, opens or runs (a link's
    included), and nothing a path reader could follow
    once it has percent-decoded the rest, up to three times, the most an
    ordinary reference is encoded: no backslash, no control character, no
    segment that is ``..`` with the whitespace around it trimmed, no escape
    left to decode, and no start a path reader roots a path at."""
    match = _REFERENCE.match(url)
    if match is None or _refused_scheme(match.group(1)):
        return False
    rest = _decoded(match.group(2))
    return (
        rest is not None
        and "\\" not in rest
        and not _holds_control(rest)
        and not _holds_parent(rest)
        and not _rooted(match.group(2), rest)
    )


def _refused_scheme(scheme: str) -> bool:
    """Whether *scheme*, or one of its parts joined by ``+`` (``hls+http``,
    ``crypto+file``), is a scheme a reader fetches, opens or runs."""
    return any(part in _REFUSED_SCHEMES for part in scheme.lower().split("+"))


def _holds_control(text: str) -> bool:
    """Whether *text* holds a control character. An ASCII text holds none
    exactly when it is printable, which Python tests without a regex."""
    if text.isascii():
        return not text.isprintable()
    return _CONTROL.search(text) is not None


def _holds_parent(text: str) -> bool:
    """Whether a segment of *text* is ``..`` once trimmed of whitespace."""
    return ".." in text and any(segment.strip() == ".." for segment in _SEGMENT_END.split(text))


def _rooted(raw: str, rest: str) -> bool:
    """Whether *rest*, decoded from *raw*, starts where a path reader roots a
    path: a home directory (``~``), a drive letter (``C:``), or a slash that
    only decoding writes (``%2F``). A slash written as such (``app:///a``) is
    an empty authority, which the application spells as it chooses."""
    return (
        rest.startswith("~")
        or _DRIVE.match(rest) is not None
        or (rest.startswith("/") and not raw.startswith("/"))
    )


def _decoded(text: str) -> str | None:
    """*text* percent-decoded until it no longer changes, as a reader that
    decodes once or more reads it (``%252e`` reads as ``.`` to one that
    decodes twice); None when it still changes after
    :data:`_MOST_DECODINGS`, a nesting no reference holds and whose cost
    would grow with the square of its length."""
    for _ in range(_MOST_DECODINGS):
        if _ESCAPE.search(text) is None:
            return text
        text = unquote(text)
    return text if _ESCAPE.search(text) is None else None


def media_type(mime_type: str) -> str:
    """*mime_type* as RoomKit compares it: lowercased, parameters aside, aliases folded."""
    bare = mime_type.split(";", 1)[0].strip().lower()
    return _TYPE_ALIASES.get(bare, bare)


def canonical_data_uri(url: str) -> str:
    """A ``data:`` URI spelled as every vendor reads it: the scheme and the
    parameters in lower case, the type folded as :func:`media_type` folds it.
    ``DATA:image/JPG;BASE64,…`` goes as ``data:image/jpeg;base64,…``: Anthropic
    refuses ``image/jpg`` and sends a ``DATA:`` URI as a link to fetch."""
    header, comma, payload = url.partition(",")
    kind, *params = header[len("data:") :].split(";")
    spelled = "".join(f";{param.strip().lower()}" for param in params)
    return f"data:{media_type(kind)}{spelled}{comma}{payload}"


def data_uri_media_type(url: str) -> str | None:
    """The media type a ``data:`` URI declares, as :func:`media_type` reads it;
    None for any other URL."""
    if not is_data_uri(url):
        return None
    return media_type(url[len("data:") :].split(",", 1)[0])


def link_or_type(url: str, mime_type: str | None) -> str:
    """What a label shows of a media: its URL when a link, else its media type."""
    return url if is_link(url) else (mime_type or "media")
