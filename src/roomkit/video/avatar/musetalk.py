"""MuseTalk avatar provider — real-time lip-synced talking head.

Wraps the MuseTalk model for audio-driven lip synchronization.
Requires a local MuseTalk installation (git clone) and GPU.

Prerequisites:
    git clone https://github.com/TMElyralab/MuseTalk.git
    cd MuseTalk && pip install -r requirements.txt

    # Download models (automatic on first run, or manual):
    # https://huggingface.co/TMElyralab/MuseTalk

Usage::

    from roomkit.video.avatar.musetalk import MuseTalkAvatarProvider

    avatar = MuseTalkAvatarProvider(
        musetalk_dir="/path/to/MuseTalk",
        reference_image="./agent_photo.png",
    )
    await avatar.start(image_bytes)
    frames = avatar.feed_audio(pcm_chunk, sample_rate=16000)

GPU: NVIDIA with 4GB+ VRAM (V100/RTX 3060+ for 30fps)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

from roomkit.video.avatar.base import AvatarProvider

logger = logging.getLogger("roomkit.video.avatar.musetalk")


class MuseTalkAvatarProvider(AvatarProvider):
    """Real-time lip-sync avatar using MuseTalk.

    MuseTalk uses a frozen VAE encoder + whisper-tiny audio encoder
    + UNet to generate lip-synced face frames in latent space.
    Preprocessing (face detection, parsing, VAE encode) happens once
    during ``start()``.  Real-time inference only runs UNet + VAE
    decoder (~30fps on V100).

    Args:
        musetalk_dir: Path to the MuseTalk git clone directory.
        model_name: Model variant (default ``"musetalk"``).
        fps: Output video frame rate.
        batch_size: Inference batch size (higher = smoother but more
            VRAM). Default 4.
        device: PyTorch device (default ``"cuda"``).
        bbox_shift: Face bounding box shift for better lip region
            capture (default 0).
    """

    def __init__(
        self,
        musetalk_dir: str,
        *,
        model_name: str = "musetalk",
        fps: int = 30,
        batch_size: int = 4,
        device: str = "cuda",
        bbox_shift: int = 0,
    ) -> None:
        self._musetalk_dir = Path(musetalk_dir).resolve()
        self._model_name = model_name
        self._fps = fps
        self._batch_size = batch_size
        self._device = device
        self._bbox_shift = bbox_shift

        # Lazy-loaded MuseTalk internals
        self._avatar: Any = None  # MuseTalk Avatar instance
        self._audio_processor: Any = None
        self._vae: Any = None
        self._unet: Any = None
        self._pe: Any = None
        self._ref_image: Any = None
        self._coord: Any = None
        self._ref_latent: Any = None
        self._started = False
        self._width = 512
        self._height = 512

        # Audio buffer for accumulating PCM before generating frames
        self._audio_buffer = bytearray()
        self._samples_per_frame: int = 0  # computed from fps + sample_rate

    @property
    def name(self) -> str:
        return "musetalk"

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_started(self) -> bool:
        return self._started

    async def start(
        self,
        reference_image: bytes,
        *,
        width: int = 512,
        height: int = 512,
    ) -> None:
        self._width = width
        self._height = height

        # Add MuseTalk to sys.path so we can import its modules
        musetalk_str = str(self._musetalk_dir)
        if musetalk_str not in sys.path:
            sys.path.insert(0, musetalk_str)

        try:
            self._load_models()
            self._preprocess_face(reference_image)
        except ImportError as exc:
            raise ImportError(
                f"MuseTalk not found at {self._musetalk_dir}. "
                "Clone it: git clone https://github.com/TMElyralab/MuseTalk.git "
                "and install: pip install -r requirements.txt"
            ) from exc

        self._started = True
        logger.info(
            "MuseTalk avatar started: %dx%d @ %dfps (device=%s, batch=%d)",
            width,
            height,
            self._fps,
            self._device,
            self._batch_size,
        )

    def _load_models(self) -> None:
        """Load MuseTalk models (whisper, VAE, UNet)."""
        # Import MuseTalk modules from the cloned directory
        from musetalk.utils.utils import load_all_model

        (
            self._audio_processor,
            self._vae,
            self._unet,
            self._pe,
        ) = load_all_model()
        logger.info("MuseTalk models loaded on %s", self._device)

    def _preprocess_face(self, reference_image: bytes) -> None:
        """Preprocess the reference face image.

        Runs face detection, face parsing, and VAE encoding on the
        reference image.  This is done once — the latent representation
        is reused for all subsequent frames.
        """
        import io

        import numpy as np
        from musetalk.utils.preprocessing import (
            get_landmark_and_bbox,
        )
        from PIL import Image

        img = Image.open(io.BytesIO(reference_image)).convert("RGB")
        img = img.resize((self._width, self._height))
        self._ref_image = np.array(img)

        # Detect face landmarks and compute crop region
        landmark, bbox = get_landmark_and_bbox(
            self._ref_image,
            self._bbox_shift,
        )
        self._coord = bbox
        self._ref_latent = self._encode_face_region(
            self._ref_image,
            bbox,
        )
        logger.info("Reference face preprocessed: bbox=%s", bbox)

    def _encode_face_region(self, image: Any, bbox: Any) -> Any:
        """VAE-encode the face region from the reference image."""

        from musetalk.utils.utils import (
            get_image_prepare_material,
        )

        input_latent = get_image_prepare_material(
            image,
            bbox,
            self._vae,
            self._device,
        )
        return input_latent

    def feed_audio(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
    ) -> list[VideoFrame]:
        """Feed audio and generate lip-synced frames."""
        if not self._started:
            return []

        from roomkit.video.video_frame import VideoFrame

        # Compute how many audio samples per video frame
        if self._samples_per_frame == 0:
            self._samples_per_frame = sample_rate // self._fps

        # Accumulate audio
        self._audio_buffer.extend(pcm_data)

        # Need enough audio for at least one frame
        bytes_per_frame = self._samples_per_frame * 2  # 16-bit PCM
        if len(self._audio_buffer) < bytes_per_frame:
            return []

        frames = []
        while len(self._audio_buffer) >= bytes_per_frame:
            chunk = bytes(self._audio_buffer[:bytes_per_frame])
            del self._audio_buffer[:bytes_per_frame]

            result = self._generate_frame(chunk, sample_rate)
            if result is not None:
                frames.append(
                    VideoFrame(
                        data=result.tobytes(),
                        codec="raw_rgb24",
                        width=self._width,
                        height=self._height,
                        keyframe=True,
                    )
                )

        return frames

    def _generate_frame(self, audio_chunk: bytes, sample_rate: int) -> Any:
        """Generate a single lip-synced frame from an audio chunk."""
        import numpy as np
        import torch
        from musetalk.utils.utils import datagen

        # Extract audio features using whisper
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        whisper_feature = self._audio_processor.audio2feat(audio_np)

        # Generate frame using UNet + VAE decoder
        with torch.no_grad():
            gen = datagen(
                whisper_feature,
                self._ref_latent,
                self._coord,
                self._batch_size,
            )
            for batch in gen:
                pred = self._unet.model(
                    batch["latent"].to(self._device),
                    batch["timesteps"].to(self._device),
                    encoder_hidden_states=batch["whisper"].to(self._device),
                ).sample
                recon = self._vae.decode(pred).sample
                # Convert to numpy RGB
                frame = (recon[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

                # Composite generated face onto reference image
                result = self._ref_image.copy()
                x1, y1, x2, y2 = self._coord
                from PIL import Image

                face = Image.fromarray(frame).resize((x2 - x1, y2 - y1))
                result[y1:y2, x1:x2] = np.array(face)
                return result

        return None

    def flush(self) -> list[VideoFrame]:
        """Flush remaining audio buffer.

        Thread safety: feed_audio/flush are called from a single async
        context (the TTS outbound wrapper in AudioVideoChannel), so
        concurrent access is not expected.
        """
        if not self._started or not self._audio_buffer:
            return []
        # Generate frames from remaining audio (pad if needed)
        remaining = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        if remaining:
            bytes_per_frame = self._samples_per_frame * 2
            if len(remaining) < bytes_per_frame:
                remaining += b"\x00" * (bytes_per_frame - len(remaining))
            return self.feed_audio(remaining)
        return []

    async def stop(self) -> None:
        self._started = False
        self._avatar = None
        self._audio_processor = None
        self._vae = self._unet = self._pe = None
        self._ref_image = self._coord = self._ref_latent = None
        self._audio_buffer.clear()
        logger.info("MuseTalk avatar stopped")
