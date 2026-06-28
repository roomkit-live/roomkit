"""Vendored, gradio-free WebRTC transport extracted from fastrtc 0.0.34.

See ``LICENSE`` (Apache-2.0). Only the headless transport is kept:
``Stream.mount()``, the WebRTC/WebSocket offer handlers, the ``StreamHandler``
base classes and the audio utilities. The Gradio UI builder, the ``gr.WebRTC``
component, the STT/TTS/VAD batteries and ``fastphone`` tunnelling were dropped —
roomkit brings its own STT/TTS/VAD. This is why roomkit ships realtime WebRTC
without depending on gradio (and its transitive CVE/footprint surface).
"""

from .stream import Stream, UIArgs
from .tracks import (
    AsyncAudioVideoStreamHandler,
    AsyncStreamHandler,
    AudioEmitType,
    AudioVideoStreamHandler,
    StreamHandler,
    VideoEmitType,
    VideoStreamHandler,
)
from .utils import (
    AdditionalOutputs,
    CloseStream,
    WebRTCData,
    WebRTCError,
    current_context,
    get_current_context,
)

__all__ = [
    "Stream",
    "UIArgs",
    "StreamHandler",
    "AsyncStreamHandler",
    "AudioVideoStreamHandler",
    "AsyncAudioVideoStreamHandler",
    "VideoStreamHandler",
    "AudioEmitType",
    "VideoEmitType",
    "AdditionalOutputs",
    "CloseStream",
    "WebRTCData",
    "WebRTCError",
    "current_context",
    "get_current_context",
]
