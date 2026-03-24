"""Shared types, constants, and helpers for the SIP voice backend."""

from __future__ import annotations

import asyncio
import hashlib
import logging
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
        "outbound_frames",
        "outbound_bytes",
        "outbound_first_ts",
        "outbound_last_ts",
        "outbound_max_burst",
        "outbound_calls",
    )

    def __init__(self) -> None:
        self.inbound_packets = 0
        self.inbound_bytes = 0
        self.inbound_first_ts: float = 0.0
        self.inbound_last_ts: float = 0.0
        self.inbound_gaps = 0
        self.inbound_max_gap_ms: float = 0.0
        self.outbound_frames = 0
        self.outbound_bytes = 0
        self.outbound_first_ts: float = 0.0
        self.outbound_last_ts: float = 0.0
        self.outbound_max_burst = 0  # max frames in a single _send_pcm_bytes call
        self.outbound_calls = 0  # number of _send_pcm_bytes calls


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
