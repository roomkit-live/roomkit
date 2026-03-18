"""Video transport backends."""

from __future__ import annotations


def get_screen_capture_backend() -> type:
    """Get ScreenCaptureBackend class (requires mss).

    Install with: ``pip install roomkit[screen-capture]``
    """
    from roomkit.video.backends.screen import ScreenCaptureBackend

    return ScreenCaptureBackend


def get_rtp_video_backend() -> type:
    """Get RTPVideoBackend class (requires aiortp).

    Install with: ``pip install roomkit[rtp]``
    """
    from roomkit.video.backends.rtp import RTPVideoBackend

    return RTPVideoBackend


def get_sip_video_backend() -> type:
    """Get SIPVideoBackend class (requires aiosipua[rtp]).

    Install with: ``pip install roomkit[sip]``
    """
    from roomkit.video.backends.sip import SIPVideoBackend

    return SIPVideoBackend


def get_fastrtc_video_backend() -> type:
    """Get FastRTCVideoBackend class (requires fastrtc).

    Install with: ``pip install roomkit[fastrtc]``
    """
    from roomkit.video.backends.fastrtc import FastRTCVideoBackend

    return FastRTCVideoBackend


def get_websocket_video_backend() -> type:
    """Get WebSocketVideoBackend class.

    No extra dependencies required beyond ``fastapi``.
    """
    from roomkit.video.backends.websocket import WebSocketVideoBackend

    return WebSocketVideoBackend
