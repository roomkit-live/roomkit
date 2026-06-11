"""Shared types, constants, and helpers for the SIP voice backend."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline.dtmf.base import DTMFEvent

logger = logging.getLogger("roomkit.voice.sip")

# Well-known RTP payload types (RFC 3551)
PT_PCMU = 0  # G.711 µ-law, 8 kHz
PT_PCMA = 8  # G.711 A-law, 8 kHz
PT_G722 = 9  # G.722, 16 kHz audio (8 kHz RTP clock)

# Codec info: payload_type → (name, rtp_clock_rate, audio_sample_rate)
CODEC_INFO: dict[int, tuple[str, int, int]] = {
    PT_PCMU: ("PCMU", 8000, 8000),
    PT_PCMA: ("PCMA", 8000, 8000),
    PT_G722: ("G722", 8000, 16000),  # RTP clock 8000, audio rate 16000
}

# Audio diagnostics interval (seconds)
STATS_INTERVAL = 5.0

# Nonce validity period for inbound auth challenges (seconds)
NONCE_TTL = 30.0

# Callback types
DTMFReceivedCallback = Callable[["VoiceSession", DTMFEvent], Any]
CallCallback = Callable[["VoiceSession"], Any]


class AudioStats:
    """Per-session audio diagnostics counters."""

    __slots__ = (
        "inbound_packets",
        "inbound_bytes",
        "inbound_first_ts",
        "inbound_last_ts",
        "inbound_gaps",
        "inbound_max_gap_ms",
        "concealed_frames",
        "outbound_frames",
        "outbound_bytes",
        "outbound_first_ts",
        "outbound_last_ts",
        "outbound_max_burst",
        "outbound_calls",
        "remote_packets_lost",
        "remote_fraction_lost",
        "remote_jitter_units",
        "has_remote_report",
    )

    def __init__(self) -> None:
        self.inbound_packets = 0
        self.inbound_bytes = 0
        self.inbound_first_ts: float = 0.0
        self.inbound_last_ts: float = 0.0
        self.inbound_gaps = 0
        self.inbound_max_gap_ms: float = 0.0
        self.concealed_frames = 0  # lost packets replaced by PLC (synced from RTP session)
        self.outbound_frames = 0
        self.outbound_bytes = 0
        self.outbound_first_ts: float = 0.0
        self.outbound_last_ts: float = 0.0
        self.outbound_max_burst = 0  # max frames in a single _send_pcm_bytes call
        self.outbound_calls = 0  # number of _send_pcm_bytes calls
        # RTCP Receiver Report view of OUR outbound stream, as seen by the
        # remote endpoint.  Raw RFC 3550 units: fraction_lost is 8-bit fixed
        # point (lost*256/expected, over the last reporting interval) and
        # jitter is in RTP clock units — convert at the log site, where the
        # session clock rate is known.
        self.remote_packets_lost = 0
        self.remote_fraction_lost = 0
        self.remote_jitter_units = 0
        self.has_remote_report = False

    def sync_from_rtp(self, rtp_stats: dict[str, Any]) -> None:
        """Pull RTP-session-sourced counters (no-op on an empty/closed-session dict)."""
        if not rtp_stats:
            return
        self.concealed_frames = rtp_stats.get("concealed_frames", 0)
        # RR keys only appear once the remote has sent a Receiver Report;
        # keep the last-known values across syncs that lack them.
        if "remote_packets_lost" in rtp_stats:
            self.has_remote_report = True
            self.remote_packets_lost = rtp_stats["remote_packets_lost"]
            self.remote_fraction_lost = rtp_stats["remote_fraction_lost"]
            self.remote_jitter_units = rtp_stats["remote_jitter"]


@dataclass
class SIPSessionState:
    """Consolidated per-session state for SIP backend."""

    session: VoiceSession
    call_session: Any = None
    incoming_call: Any = None
    outgoing_call: Any = None
    pending_reinvite_sdp: Any = None
    pending_reinvite_call: Any = None  # deferred call object to accept() later
    codec_rate: int = 8000
    clock_rate: int = 8000
    send_timestamp: int = 0
    send_buffer: bytearray = field(default_factory=bytearray)
    send_frame_count: int = 0
    last_rtp_send_time: float | None = None
    rtp_port: int | None = None
    audio_stats: AudioStats = field(default_factory=AudioStats)
    pacer: Any = None
    playback_task: asyncio.Task[None] | None = None
    is_playing: bool = False


def log_fire_and_forget_exception(task: asyncio.Task[Any]) -> None:
    """Done callback — log exceptions from fire-and-forget async wrappers."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Fire-and-forget task %s failed: %s", task.get_name(), exc)


def wrap_async(callback: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async callback so it can be fired from sync context.

    If *callback* is a coroutine function it is wrapped in
    ``asyncio.create_task``.  Sync callbacks are returned unchanged.
    """
    if asyncio.iscoroutinefunction(callback):
        orig = callback

        def _wrapper(*args: Any, **kwargs: Any) -> None:
            task = asyncio.get_running_loop().create_task(orig(*args, **kwargs))
            task.add_done_callback(log_fire_and_forget_exception)

        return _wrapper
    return callback


def import_aiosipua() -> Any:
    """Import aiosipua, raising a clear error if missing."""
    try:
        import aiosipua

        return aiosipua
    except ImportError as exc:
        raise ImportError(
            "aiosipua is required for SIPVoiceBackend. Install it with: pip install roomkit[sip]"
        ) from exc


def import_rtp_bridge() -> Any:
    """Import aiosipua.rtp_bridge, raising a clear error if missing."""
    try:
        from aiosipua import rtp_bridge

        return rtp_bridge
    except ImportError as exc:
        raise ImportError(
            "aiosipua[rtp] is required for SIPVoiceBackend. "
            "Install it with: pip install roomkit[sip]"
        ) from exc


def compute_digest(
    username: str, realm: str, password: str, method: str, uri: str, nonce: str
) -> str:
    """Compute RFC 2617 MD5 digest response for SIP authentication."""
    ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode()).hexdigest()  # nosec B324  # noqa: S324
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()  # nosec B324  # noqa: S324
    return hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()  # nosec B324  # noqa: S324


# RFC 3326 "Reason" header: ``Reason: Q.850 ;cause=N ;text="…"``.
# Carriers and PBXs attach this to BYE / CANCEL / 4xx-6xx responses to
# explain *why* a call leg dropped — essential for the caller's dashboard
# to distinguish "user rejected" from "no circuits" from "normal hangup".
_BYE_REASON_RE = re.compile(
    r'^\s*Reason\s*:\s*Q\.850\s*;\s*cause\s*=\s*(\d+)(?:\s*;\s*text\s*=\s*"([^"]*)")?',
    re.IGNORECASE | re.MULTILINE,
)

# Q.850 cause codes (ITU-T). Only the ones we regularly see are given a
# short canonical text; anything else falls through to the header's own
# ``text=""`` field if present, or a generic "Q.850 cause N" label.
_Q850_CAUSES: dict[int, str] = {
    16: "Normal call clearing",
    17: "User busy",
    18: "No user responding",
    19: "No answer from user",
    21: "Call rejected",
    22: "Number changed",
    27: "Destination out of order",
    28: "Invalid number format",
    31: "Normal, unspecified",
    34: "No circuit or channel available",
    38: "Network out of order",
    41: "Temporary failure",
    42: "Switching equipment congestion",
    47: "Resource unavailable",
    63: "Service or option unavailable",
    88: "Incompatible destination",
}


def parse_bye_reason(raw: str | bytes | None) -> dict[str, Any] | None:
    """Extract ``(cause, text)`` from a BYE's RFC 3326 ``Reason`` header.

    Accepts either the serialized SIP message (str) or its wire bytes.
    Returns a ``{"cause": int, "text": str}`` dict, or ``None`` if no
    ``Reason: Q.850 ;cause=N`` header is present. When the carrier
    omits the ``text=""`` field, a canonical label is looked up from
    :data:`_Q850_CAUSES`; otherwise ``"Q.850 cause N"`` is used so
    callers always get something human-readable.
    """
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", errors="replace")
        except Exception:
            return None
    if not isinstance(raw, str):
        return None
    m = _BYE_REASON_RE.search(raw)
    if not m:
        return None
    cause = int(m.group(1))
    text = m.group(2)
    return {
        "cause": cause,
        "text": text or _Q850_CAUSES.get(cause) or f"Q.850 cause {cause}",
    }


def resolve_local_ip(local_rtp_ip: str, remote_addr: tuple[str, int]) -> str:
    """Return the local IP to advertise in SDP.

    If *local_rtp_ip* was set to a specific address, use it as-is.
    Otherwise (``0.0.0.0``), probe the OS routing table by opening a
    UDP socket towards the caller to discover the correct local IP.
    """
    if local_rtp_ip and local_rtp_ip != "0.0.0.0":  # nosec B104
        return local_rtp_ip
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((remote_addr[0], remote_addr[1] or 5060))
            result: str = s.getsockname()[0]
            return result
    except OSError:
        logger.warning(
            "Could not auto-detect local IP for remote %s; "
            "falling back to 0.0.0.0 — set local_rtp_ip explicitly",
            remote_addr,
        )
        return local_rtp_ip
