"""PyAV-based video decoder for H.264, VP8, VP9, and AV1."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from roomkit.video.pipeline.decoder.base import VideoDecoderProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.pipeline.decoder")

# Map VideoFrame codec names to PyAV codec names.
_CODEC_MAP: dict[str, str] = {
    "h264": "h264",
    "vp8": "vp8",
    "vp9": "vp9",
    "av1": "av1",
}

# Map output format strings to PyAV pixel format names.
_PIX_FMT_MAP: dict[str, str] = {
    "rgb24": "rgb24",
    "bgr24": "bgr24",
    "yuv420p": "yuv420p",
    "nv12": "nv12",
}


class PyAVVideoDecoder(VideoDecoderProvider):
    """Decode encoded video frames to raw pixels using PyAV.

    Supports H.264, VP8, VP9, and AV1. Handles keyframe gating —
    P-frames are dropped until the first keyframe arrives.

    Args:
        output_format: Target pixel format. One of ``rgb24``, ``bgr24``,
            ``yuv420p``, or ``nv12``.  Defaults to ``rgb24``.

    Example::

        decoder = PyAVVideoDecoder(output_format="rgb24")
        raw_frame = decoder.decode(encoded_frame)
    """

    def __init__(self, output_format: str = "rgb24") -> None:
        if output_format not in _PIX_FMT_MAP:
            raise ValueError(
                f"output_format must be one of {sorted(_PIX_FMT_MAP)}, got {output_format!r}"
            )
        self._output_format = output_format
        self._pix_fmt = _PIX_FMT_MAP[output_format]
        # Per-codec decoder contexts, created lazily.
        self._codecs: dict[str, Any] = {}
        # Track whether we've received a keyframe per codec.
        self._seen_keyframe: dict[str, bool] = {}

    @property
    def name(self) -> str:
        return "PyAVVideoDecoder"

    def _get_codec_context(self, codec_name: str) -> Any:
        """Get or create a PyAV codec context for the given codec."""
        ctx = self._codecs.get(codec_name)
        if ctx is not None:
            return ctx

        import av

        pyav_codec = _CODEC_MAP.get(codec_name)
        if pyav_codec is None:
            raise ValueError(f"Unsupported codec: {codec_name!r}")

        codec = av.codec.Codec(pyav_codec, "r")
        ctx = av.codec.CodecContext.create(codec)
        ctx.open()
        self._codecs[codec_name] = ctx
        self._seen_keyframe[codec_name] = False
        return ctx

    def decode(self, frame: VideoFrame) -> VideoFrame | None:
        """Decode an encoded frame to raw pixels.

        Returns None if the frame cannot be decoded (e.g., waiting
        for a keyframe).
        """
        from roomkit.video.video_frame import RAW_CODECS

        # Already raw — nothing to do.
        if frame.codec in RAW_CODECS:
            return frame

        # Keyframe gating: drop P-frames until the first keyframe.
        if not self._seen_keyframe.get(frame.codec, False):
            if not frame.keyframe:
                logger.debug(
                    "Dropping frame seq=%d (waiting for keyframe, codec=%s)",
                    frame.sequence,
                    frame.codec,
                )
                return None
            self._seen_keyframe[frame.codec] = True

        try:
            return self._decode_with_pyav(frame)
        except Exception:
            logger.exception(
                "Decode error for frame seq=%d codec=%s; resetting decoder",
                frame.sequence,
                frame.codec,
            )
            self._reset_codec(frame.codec)
            return None

    def _decode_with_pyav(self, frame: VideoFrame) -> VideoFrame | None:
        """Perform the actual PyAV decode."""
        import av

        ctx = self._get_codec_context(frame.codec)
        packet = av.Packet(frame.data)
        decoded_frames = ctx.decode(packet)

        for av_frame in decoded_frames:
            # Reformat to the target pixel format.
            reformatted = av_frame.reformat(format=self._pix_fmt)
            raw_bytes = reformatted.to_ndarray().tobytes()
            raw_codec = f"raw_{self._output_format}"
            return replace(
                frame,
                data=raw_bytes,
                codec=raw_codec,
                width=reformatted.width,
                height=reformatted.height,
                metadata={**frame.metadata, "decoder": self.name},
            )

        # No frames decoded (buffering).
        return None

    def _reset_codec(self, codec_name: str) -> None:
        """Flush and remove a codec context to force re-creation."""
        ctx = self._codecs.pop(codec_name, None)
        if ctx is not None:
            ctx.flush_buffers()
        self._seen_keyframe.pop(codec_name, None)

    def reset(self) -> None:
        """Reset all decoder contexts."""
        for ctx in self._codecs.values():
            ctx.flush_buffers()
        self._codecs.clear()
        self._seen_keyframe.clear()

    def close(self) -> None:
        """Release all decoder resources."""
        self.reset()
