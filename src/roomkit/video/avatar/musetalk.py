"""MuseTalk avatar provider — real-time lip-synced talking head.

Wraps the MuseTalk v1.5 model for audio-driven lip synchronization.
Requires a local MuseTalk installation (git clone) and GPU.

Usage::

    from roomkit.video.avatar.musetalk import MuseTalkAvatarProvider

    avatar = MuseTalkAvatarProvider(musetalk_dir="/path/to/MuseTalk")
    await avatar.start(image_bytes)
    frames = avatar.feed_audio(pcm_chunk, sample_rate=16000)

GPU: NVIDIA with 4GB+ VRAM (RTX 3060+ for 30fps)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

from roomkit.video.avatar.base import AvatarProvider

logger = logging.getLogger("roomkit.video.avatar.musetalk")


class MuseTalkAvatarProvider(AvatarProvider):
    """Real-time lip-sync avatar using MuseTalk v1.5.

    Args:
        musetalk_dir: Path to the MuseTalk git clone directory.
        fps: Output video frame rate.
        batch_size: Inference batch size (higher = better GPU util,
            more VRAM). Default 8.
        device: PyTorch device (default ``"cuda"``).
        bbox_shift: Face bounding box shift for lip capture (default 0).
        whisper_dir: Whisper model dir inside MuseTalk.
    """

    def __init__(
        self,
        musetalk_dir: str,
        *,
        fps: int = 30,
        batch_size: int = 8,
        device: str = "cuda",
        bbox_shift: int = 0,
        whisper_dir: str = "models/whisper",
    ) -> None:
        self._musetalk_dir = Path(musetalk_dir).resolve()
        self._fps = fps
        self._batch_size = batch_size
        self._device = device
        self._bbox_shift = bbox_shift
        self._whisper_dir = whisper_dir

        # Lazy-loaded MuseTalk internals (v1.5 API)
        self._vae: Any = None
        self._unet: Any = None
        self._pe: Any = None
        self._whisper: Any = None
        self._audio_processor: Any = None
        self._timesteps: Any = None
        self._weight_dtype: Any = None
        self._ref_image: Any = None
        self._coord: Any = None
        self._ref_latent: Any = None
        self._mask: Any = None
        self._mask_crop_box: Any = None
        self._started = False
        self._width = 512
        self._height = 512

        # Audio buffer for accumulating PCM before generating frames
        self._audio_buffer = bytearray()
        self._samples_per_frame: int = 0

        # Pre-computed blending state (populated in _preprocess_face)
        self._ref_bgr: Any = None
        self._crop_box: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._mask_float_3ch: Any = None
        self._body_crop_float: Any = None

    @property
    def name(self) -> str:
        return "musetalk"

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_started(self) -> bool:
        return self._started

    def get_idle_frame(self) -> VideoFrame | None:
        """Return the reference image as an idle frame."""
        if not self._started or self._ref_image is None:
            return None
        from roomkit.video.video_frame import VideoFrame

        return VideoFrame(
            data=self._ref_image.tobytes(),
            codec="raw_rgb24",
            width=self._width,
            height=self._height,
            keyframe=True,
        )

    async def start(
        self,
        reference_image: bytes,
        *,
        width: int = 512,
        height: int = 512,
    ) -> None:
        self._width = width
        self._height = height

        musetalk_str = str(self._musetalk_dir)
        if musetalk_str not in sys.path:
            sys.path.insert(0, musetalk_str)

        # PyTorch 2.6+ defaults weights_only=True which breaks loading
        # older MuseTalk checkpoints saved in legacy format.
        import torch

        _original_load = torch.load
        torch.load = lambda *a, **kw: _original_load(  # type: ignore[assignment]
            *a,
            **{**kw, "weights_only": False},
        )

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
        """Load MuseTalk v1.5 models (VAE, UNet, PE, Whisper)."""
        import torch
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.utils import load_all_model
        from transformers import WhisperModel

        prev_cwd = os.getcwd()
        try:
            os.chdir(self._musetalk_dir)

            self._vae, self._unet, self._pe = load_all_model(
                device=self._device,
            )
            self._timesteps = torch.tensor([0], device=self._device)

            self._pe = self._pe.half().to(self._device)
            self._vae.vae = self._vae.vae.half().to(self._device)
            self._unet.model = self._unet.model.half().to(self._device)
            self._weight_dtype = self._unet.model.dtype

            whisper_path = str(self._musetalk_dir / self._whisper_dir)
            self._audio_processor = AudioProcessor(
                feature_extractor_path=whisper_path,
            )
            self._whisper = WhisperModel.from_pretrained(whisper_path)  # nosec B615
            self._whisper = self._whisper.to(
                device=self._device,
                dtype=self._weight_dtype,
            ).eval()
            self._whisper.requires_grad_(False)
        finally:
            os.chdir(prev_cwd)

        logger.info("MuseTalk v1.5 models loaded on %s", self._device)

    def _preprocess_face(self, reference_image: bytes) -> None:
        """Preprocess face: detect bbox, VAE-encode, build blending mask."""
        import io

        import cv2
        from PIL import Image

        img = Image.open(io.BytesIO(reference_image)).convert("RGB")
        img.thumbnail((self._width, self._height), Image.LANCZOS)
        canvas = Image.new("RGB", (self._width, self._height), (0, 0, 0))
        offset_x = (self._width - img.width) // 2
        offset_y = (self._height - img.height) // 2
        canvas.paste(img, (offset_x, offset_y))
        self._ref_image = np.array(canvas)

        bbox = self._detect_face_bbox(self._ref_image)
        self._coord = bbox

        # VAE-encode the face region (256x256 BGR)
        x1, y1, x2, y2 = bbox
        crop_rgb = self._ref_image[y1:y2, x1:x2]
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        crop_bgr = cv2.resize(crop_bgr, (256, 256))
        self._ref_latent = self._vae.get_latents_for_unet(crop_bgr)

        # Face parsing mask (needs cwd = musetalk_dir for model weights)
        prev_cwd = os.getcwd()
        try:
            os.chdir(self._musetalk_dir)
            from musetalk.utils.blending import get_image_prepare_material
            from musetalk.utils.face_parsing import FaceParsing

            fp = FaceParsing()
            self._mask, self._mask_crop_box = get_image_prepare_material(
                self._ref_image,
                bbox,
                fp=fp,
            )
        finally:
            os.chdir(prev_cwd)

        # Pre-compute blending constants (avoids per-frame PIL overhead)
        self._ref_bgr = self._ref_image[:, :, ::-1].copy()
        self._crop_box = tuple(self._mask_crop_box)
        self._mask_float_3ch = (self._mask.astype(np.float32) / 255.0)[..., np.newaxis]

        # Cache the float32 body crop (same every frame)
        x_s, y_s, x_e, y_e = self._crop_box
        img_h, img_w = self._ref_bgr.shape[:2]
        ry_s, ry_e = max(0, y_s), min(img_h, y_e)
        rx_s, rx_e = max(0, x_s), min(img_w, x_e)
        self._body_crop_float = self._ref_bgr[ry_s:ry_e, rx_s:rx_e].astype(np.float32)

        logger.info("Reference face preprocessed: bbox=%s", bbox)

    def _detect_face_bbox(self, image: Any) -> tuple[int, int, int, int]:
        """Detect face bbox with expansion for MuseTalk quality."""
        import cv2

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        )
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )
        if len(faces) == 0:
            margin = min(h, w) // 8
            logger.warning("No face detected, using center crop")
            return (margin, margin, w - margin, h - margin)

        fx, fy, fw, fh = faces[0]
        cx, cy = fx + fw // 2, fy + fh // 2

        # Expand: 1.5x horizontal, 1.8x vertical (more chin/neck)
        exp_w = int(fw * 1.5) // 2
        exp_h_up = int(fh * 0.8)
        exp_h_down = int(fh * 1.0)

        x1 = max(0, cx - exp_w)
        y1 = max(0, cy - exp_h_up + self._bbox_shift)
        x2 = min(w, cx + exp_w)
        y2 = min(h, cy + exp_h_down)

        logger.info("Face detected: expanded=(%d,%d,%d,%d)", x1, y1, x2, y2)
        return (x1, y1, x2, y2)

    # --- Audio feeding & inference ---

    def feed_audio(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
    ) -> list[VideoFrame]:
        """Feed audio and generate lip-synced frames."""
        if not self._started:
            return []

        from roomkit.video.video_frame import VideoFrame

        if self._samples_per_frame == 0:
            self._samples_per_frame = sample_rate // self._fps

        self._audio_buffer.extend(pcm_data)

        bytes_per_frame = self._samples_per_frame * 2  # 16-bit PCM
        if len(self._audio_buffer) < bytes_per_frame:
            return []

        chunks: list[bytes] = []
        while len(self._audio_buffer) >= bytes_per_frame:
            chunks.append(bytes(self._audio_buffer[:bytes_per_frame]))
            del self._audio_buffer[:bytes_per_frame]

        results = self._generate_frames(chunks, sample_rate)
        return [
            VideoFrame(
                data=r.tobytes(),
                codec="raw_rgb24",
                width=self._width,
                height=self._height,
                keyframe=True,
            )
            for r in results
        ]

    def _generate_frames(
        self,
        audio_chunks: list[bytes],
        sample_rate: int,
    ) -> list[Any]:
        """Generate lip-synced frames directly from PCM — no temp file."""
        pcm_all = b"".join(audio_chunks)
        audio_np = np.frombuffer(pcm_all, dtype=np.int16).astype(np.float32) / 32768.0
        return self._infer_from_audio(audio_np, len(audio_chunks), sample_rate)

    def _extract_features(
        self,
        audio_np: Any,
        sample_rate: int,
    ) -> tuple[list[Any], int]:
        """Extract whisper mel features directly from numpy audio.

        Bypasses librosa.load + temp file — feeds numpy straight to
        the whisper feature extractor.
        """
        segment_length = 30 * sample_rate
        segments = [
            audio_np[i : i + segment_length] for i in range(0, len(audio_np), segment_length)
        ]
        features = []
        for seg in segments:
            feat = self._audio_processor.feature_extractor(
                seg,
                return_tensors="pt",
                sampling_rate=sample_rate,
            ).input_features.to(dtype=self._weight_dtype)
            features.append(feat)
        return features, len(audio_np)

    def _infer_from_audio(
        self,
        audio_np: Any,
        num_frames: int,
        sample_rate: int,
    ) -> list[Any]:
        """Run MuseTalk inference from numpy audio."""
        import torch
        from musetalk.utils.utils import datagen

        features, audio_len = self._extract_features(audio_np, sample_rate)
        if not features:
            return []

        whisper_chunks = self._audio_processor.get_whisper_chunk(
            features,
            self._device,
            self._weight_dtype,
            self._whisper,
            audio_len,
            fps=self._fps,
        )

        latent_list = [self._ref_latent] * len(whisper_chunks)
        gen = datagen(whisper_chunks, latent_list, self._batch_size)

        results: list[Any] = []
        with torch.no_grad():
            for whisper_batch, latent_batch in gen:
                audio_feat = self._pe(whisper_batch.to(self._device))
                latent_batch = latent_batch.to(
                    device=self._device,
                    dtype=self._unet.model.dtype,
                )
                pred = self._unet.model(
                    latent_batch,
                    self._timesteps,
                    encoder_hidden_states=audio_feat,
                ).sample
                pred = pred.to(
                    device=self._device,
                    dtype=self._vae.vae.dtype,
                )
                recon = self._vae.decode_latents(pred)
                for frame in recon:
                    results.append(self._blend_fast(frame))
                    if len(results) >= num_frames:
                        return results
        return results

    def _blend_fast(self, face_bgr: Any) -> Any:
        """Fast numpy blending — replaces per-frame PIL conversion.

        Avoids: PIL.fromarray x2, PIL.crop, PIL.paste, PIL→numpy
        per frame. Uses pre-computed float32 mask and body crop instead.
        """
        import cv2

        x1, y1, x2, y2 = self._coord
        x_s, y_s, x_e, y_e = self._crop_box
        img_h, img_w = self._ref_bgr.shape[:2]

        face_resized = cv2.resize(
            face_bgr.astype(np.uint8),
            (x2 - x1, y2 - y1),
        )

        # Clamp crop region to image bounds
        ry_s, ry_e = max(0, y_s), min(img_h, y_e)
        rx_s, rx_e = max(0, x_s), min(img_w, x_e)
        my_s, mx_s = ry_s - y_s, rx_s - x_s
        my_e = my_s + (ry_e - ry_s)
        mx_e = mx_s + (rx_e - rx_s)

        # Build face_large: crop with generated face pasted in
        result = self._ref_bgr.copy()
        face_large = result[ry_s:ry_e, rx_s:rx_e].copy()

        fy, fx = y1 - ry_s, x1 - rx_s
        fh = min(face_resized.shape[0], ry_e - y1)
        fw = min(face_resized.shape[1], rx_e - x1)
        if fy >= 0 and fx >= 0 and fh > 0 and fw > 0:
            face_large[fy : fy + fh, fx : fx + fw] = face_resized[:fh, :fw]

        # Alpha blend with pre-computed mask (no PIL overhead)
        mask = self._mask_float_3ch[my_s:my_e, mx_s:mx_e]
        blended = face_large.astype(np.float32) * mask + self._body_crop_float * (1.0 - mask)
        result[ry_s:ry_e, rx_s:rx_e] = blended.astype(np.uint8)

        # BGR → RGB for VideoFrame
        return result[:, :, ::-1]

    def flush(self) -> list[VideoFrame]:
        """Flush remaining audio buffer."""
        if not self._started or not self._audio_buffer:
            return []
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
        self._vae = self._unet = self._pe = None
        self._whisper = self._audio_processor = None
        self._ref_image = self._coord = self._ref_latent = None
        self._mask = self._mask_crop_box = None
        self._ref_bgr = self._mask_float_3ch = None
        self._body_crop_float = None
        self._audio_buffer.clear()
        logger.info("MuseTalk avatar stopped")
