"""Anam avatar provider — lip-sync via Anam's audio passthrough mode.

Implements the :class:`AvatarProvider` ABC so Anam can be used as a
drop-in replacement for local avatar providers (MuseTalk, etc.) in
:class:`AudioVideoChannel` and :class:`RealtimeAVBridge`.

Anam renders a photorealistic lip-synced avatar in the cloud.  You
feed TTS audio via ``feed_audio()``, and video frames arrive
asynchronously via the internal consume loop.  Unlike local providers,
``feed_audio()`` does not return frames synchronously — instead,
frames are delivered to ``on_video`` callbacks.

Architecture::

    TTS audio → feed_audio(pcm) → Anam Cloud (passthrough)
                                       ↓
                                  face animation
                                       ↓
                             video frames → on_video callbacks

Requirements:
    pip install roomkit[anam]
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any

from roomkit.core.task_utils import log_task_exception
from roomkit.providers.anam.config import AnamConfig
from roomkit.video.avatar.base import AvatarProvider

logger = logging.getLogger("roomkit.providers.anam.avatar")

# Lazy-loaded optional deps
_anam_mod: Any = None


def _ensure_deps() -> None:
    global _anam_mod  # noqa: PLW0603
    if _anam_mod is None:
        try:
            import anam

            _anam_mod = anam
        except ImportError as exc:
            msg = "anam SDK is required. Install with: pip install anam"
            raise ImportError(msg) from exc


class AnamAvatarProvider(AvatarProvider):
    """Cloud-based avatar using Anam's audio passthrough mode.

    Feed TTS audio via :meth:`feed_audio`, receive lip-synced video
    frames via :meth:`on_video` callbacks.  Implements
    :class:`AvatarProvider` for integration with
    :class:`AudioVideoChannel` and :class:`RealtimeAVBridge`.

    Unlike local avatar providers (MuseTalk), ``feed_audio()`` returns
    an empty list — video frames arrive asynchronously via callbacks
    because Anam processes audio in the cloud.

    Example::

        from roomkit.providers.anam import AnamAvatarProvider, AnamConfig

        avatar = AnamAvatarProvider(AnamConfig(
            api_key="...",
            avatar_id="...",
            enable_audio_passthrough=True,
        ))
        avatar.on_video(handle_video_frame)
        await avatar.start(b"")  # No reference image needed

        # Feed TTS audio (24kHz mono PCM recommended)
        avatar.feed_audio(pcm_chunk, sample_rate=24000)
        avatar.end_turn()  # Signal end of response

    Args:
        config: Anam configuration. ``enable_audio_passthrough``
            is forced to ``True`` regardless of config value.
        audio_sample_rate: Default sample rate for ``feed_audio``
            (default: 24000, recommended by Anam).
        audio_channels: Number of audio channels (default: 1).
        video_fps: Expected output frame rate (default: 25).
    """

    def __init__(
        self,
        config: AnamConfig,
        *,
        audio_sample_rate: int = 24000,
        audio_channels: int = 1,
        video_fps: int = 25,
    ) -> None:
        _ensure_deps()
        self._config = config
        self._audio_sample_rate = audio_sample_rate
        self._audio_channels = audio_channels
        self._video_fps = video_fps

        self._client: Any = None
        self._ctx: Any = None
        self._session: Any = None
        self._audio_stream: Any = None

        self._video_task: asyncio.Task[None] | None = None
        self._video_callbacks: list[Callable[..., Any]] = []
        self._started = False
        self._closed = False

    # -- AvatarProvider ABC implementation -------------------------------------

    @property
    def name(self) -> str:
        return "anam-cloud"

    @property
    def fps(self) -> int:
        return self._video_fps

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_async(self) -> bool:
        return True

    async def start(
        self,
        reference_image: bytes,
        *,
        width: int = 512,
        height: int = 512,
    ) -> None:
        """Connect to Anam in passthrough mode.

        The ``reference_image`` parameter is ignored — Anam uses the
        configured ``avatar_id`` for the face model.
        """
        _ensure_deps()

        pcfg_kwargs: dict[str, Any] = {"enable_audio_passthrough": True}
        if self._config.persona_id:
            pcfg_kwargs["persona_id"] = self._config.persona_id
        if self._config.avatar_id:
            pcfg_kwargs["avatar_id"] = self._config.avatar_id
        if self._config.avatar_model:
            pcfg_kwargs["avatar_model"] = self._config.avatar_model
        if self._config.voice_id:
            pcfg_kwargs["voice_id"] = self._config.voice_id
        if self._config.llm_id:
            pcfg_kwargs["llm_id"] = self._config.llm_id

        pcfg = _anam_mod.PersonaConfig(**pcfg_kwargs)
        self._client = _anam_mod.AnamClient(
            api_key=self._config.api_key,
            persona_config=pcfg,
        )

        self._ctx = self._client.connect()
        self._session = await asyncio.wait_for(
            self._ctx.__aenter__(),
            timeout=self._config.timeout,
        )

        self._audio_stream = self._session.create_agent_audio_input_stream(
            _anam_mod.AgentAudioInputConfig(
                sample_rate=self._audio_sample_rate,
                channels=self._audio_channels,
            )
        )

        self._video_task = asyncio.create_task(
            self._video_consume_loop(),
            name="anam_avatar_video",
        )
        self._started = True
        logger.info("Anam avatar started (passthrough, %dHz)", self._audio_sample_rate)

    def feed_audio(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
    ) -> list[Any]:
        """Feed a TTS audio chunk to Anam for lip-sync.

        Returns an empty list — video frames arrive asynchronously
        via :meth:`on_video` callbacks (cloud processing).

        Args:
            pcm_data: Raw PCM int16 LE audio bytes.
            sample_rate: Ignored (uses constructor's ``audio_sample_rate``).
        """
        if self._audio_stream is not None:
            self._schedule_async(self._audio_stream.send_audio_chunk(pcm_data))
        return []

    def end_turn(self) -> None:
        """Signal end of a TTS response turn.

        Must be called when TTS finishes, otherwise the avatar
        freezes waiting for more audio.
        """
        if self._audio_stream is not None:
            self._schedule_async(self._audio_stream.end_sequence())

    def flush(self) -> list[Any]:
        """Signal end of turn and return empty (frames arrive async)."""
        self.end_turn()
        return []

    async def stop(self) -> None:
        """Disconnect from Anam."""
        self._closed = True
        self._started = False
        if self._video_task is not None:
            self._video_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._video_task
        if self._ctx is not None:
            with contextlib.suppress(Exception):
                await self._ctx.__aexit__(None, None, None)
        self._session = None
        self._audio_stream = None
        self._client = None
        logger.info("Anam avatar stopped")

    # -- Async scheduling ------------------------------------------------------

    @staticmethod
    def _schedule_async(coro: Any) -> None:
        """Schedule an async coroutine from sync context."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                task = loop.create_task(coro)
                task.add_done_callback(log_task_exception)
        except RuntimeError:
            pass

    # -- Video callbacks -------------------------------------------------------

    def on_video(self, callback: Callable[..., Any]) -> None:
        """Register a callback for lip-synced video frames.

        Args:
            callback: Called with a :class:`VideoFrame` for each
                avatar frame.
        """
        self._video_callbacks.append(callback)

    async def _video_consume_loop(self) -> None:
        """Consume video frames from Anam and fire callbacks."""
        if self._session is None:
            return
        try:
            async for av_frame in self._session.video_frames():
                if self._closed:
                    break
                from roomkit.providers.anam._convert import av_video_to_frame

                frame = av_video_to_frame(av_frame)
                for cb in self._video_callbacks:
                    try:
                        result = cb(frame)
                        if hasattr(result, "__await__"):
                            await result
                    except Exception:
                        logger.exception("Video callback error")
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self._closed:
                logger.warning("Anam video stream ended unexpectedly")
