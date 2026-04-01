"""MediaPipe face touch detection filter.

Detects when hands touch face zones (cheeks, chin, mouth, forehead) using
MediaPipe Face Landmarker and Hand Landmarker.  Emits
:class:`~roomkit.video.pipeline.filter.base.FilterEvent` instances with
``kind="face_touch"`` when a confirmed touch is detected.

Requires ``mediapipe``.  Install with::

    pip install roomkit[mediapipe]
"""

from __future__ import annotations

import logging
import math
import threading
import urllib.request
from dataclasses import dataclass, field
from enum import StrEnum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roomkit.video.events import VideoDetectionEvent
from roomkit.video.pipeline.filter.base import FilterContext, FilterEvent, VideoFilterProvider

if TYPE_CHECKING:
    from roomkit.video.video_frame import VideoFrame

logger = logging.getLogger("roomkit.video.pipeline.filter")


def _load_mediapipe() -> Any:
    """Lazy-load mediapipe, raising a clear error if missing."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for FaceTouchFilter. "
            "Install with: pip install roomkit[mediapipe]"
        ) from exc
    return mp


# MediaPipe model download URLs (Google Cloud Storage)
_MODEL_URLS: dict[str, str] = {
    "face_landmarker": (
        "https://storage.googleapis.com/mediapipe-models"
        "/face_landmarker/face_landmarker/float16/latest"
        "/face_landmarker.task"
    ),
    "hand_landmarker": (
        "https://storage.googleapis.com/mediapipe-models"
        "/hand_landmarker/hand_landmarker/float16/latest"
        "/hand_landmarker.task"
    ),
}

_CACHE_DIR = Path.home() / ".cache" / "roomkit" / "mediapipe"


def _resolve_model(model_path: str, model_type: str) -> str:
    """Return a valid model file path, downloading if needed.

    If *model_path* already exists on disk (absolute or relative), return
    it as-is.  Otherwise, download the default model from Google Cloud
    Storage into ``~/.cache/roomkit/mediapipe/`` and return that path.
    """
    if Path(model_path).exists():
        return model_path

    # Download to cache
    url = _MODEL_URLS.get(model_type)
    if url is None:
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"No auto-download URL for model type '{model_type}'."
        )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _CACHE_DIR / Path(model_path).name
    if cached.exists():
        return str(cached)

    logger.info("Downloading MediaPipe %s model to %s ...", model_type, cached)
    try:
        urllib.request.urlretrieve(url, cached)  # nosec B310 — fixed Google Storage URL
    except Exception:
        cached.unlink(missing_ok=True)
        raise
    logger.info("Download complete: %s", cached)
    return str(cached)


# ---------------------------------------------------------------------------
# Face zone definitions — MediaPipe 478-landmark mesh indices
# ---------------------------------------------------------------------------


@unique
class FaceZone(StrEnum):
    """Face zones that can be monitored for touch detection."""

    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"
    CHIN = "chin"
    MOUTH = "mouth"
    FOREHEAD = "forehead"


# Landmark indices for each zone — used to compute zone centroids
ZONE_LANDMARKS: dict[FaceZone, list[int]] = {
    FaceZone.LEFT_CHEEK: [36, 50, 101, 116, 117, 118, 123, 147, 187, 205, 206, 207],
    FaceZone.RIGHT_CHEEK: [266, 280, 330, 345, 346, 347, 352, 376, 411, 425, 426, 427],
    FaceZone.CHIN: [152, 175, 199, 200],
    FaceZone.MOUTH: [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 87, 88, 91, 95],
    FaceZone.FOREHEAD: [10, 67, 69, 104, 108, 151, 299, 337, 338],
}

# Hand landmark indices — fingertips
FINGERTIP_INDICES = [4, 8, 12, 16, 20]

DEFAULT_ZONES = frozenset(
    {
        FaceZone.LEFT_CHEEK,
        FaceZone.RIGHT_CHEEK,
        FaceZone.CHIN,
        FaceZone.MOUTH,
    }
)


# ---------------------------------------------------------------------------
# Sensitivity presets
# ---------------------------------------------------------------------------


@unique
class FaceTouchSensitivity(StrEnum):
    """Sensitivity presets controlling detection thresholds."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_SENSITIVITY_PARAMS: dict[FaceTouchSensitivity, dict[str, float | int]] = {
    FaceTouchSensitivity.LOW: {
        "touch_distance_threshold": 0.04,
        "confirmation_frames": 4,
        "cooldown_frames": 30,
        "z_depth_threshold": 0.08,
    },
    FaceTouchSensitivity.MEDIUM: {
        "touch_distance_threshold": 0.06,
        "confirmation_frames": 3,
        "cooldown_frames": 20,
        "z_depth_threshold": 0.08,
    },
    FaceTouchSensitivity.HIGH: {
        "touch_distance_threshold": 0.08,
        "confirmation_frames": 2,
        "cooldown_frames": 12,
        "z_depth_threshold": 0.12,
    },
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FaceTouchConfig:
    """Configuration for face touch detection.

    Args:
        sensitivity: Preset controlling thresholds.  Individual threshold
            overrides take precedence when set.
        zones: Face zones to monitor.  Forehead excluded by default
            (higher false-positive rate).
        every_n_frames: Run detection every N frames.  Higher values
            reduce CPU but slow reaction time.
        face_model: MediaPipe face landmarker model asset path.
        hand_model: MediaPipe hand landmarker model asset path.
        touch_distance_threshold: Normalized distance for touch detection.
            ``None`` uses the sensitivity preset value.
        confirmation_frames: Consecutive positive frames before triggering.
            ``None`` uses the sensitivity preset value.
        cooldown_frames: Frames to suppress after emitting an event.
            ``None`` uses the sensitivity preset value.
        z_depth_threshold: Z-depth difference to reject hovering hands.
            ``None`` uses the sensitivity preset value.
    """

    sensitivity: FaceTouchSensitivity = FaceTouchSensitivity.MEDIUM

    zones: frozenset[FaceZone] = DEFAULT_ZONES

    every_n_frames: int = 3

    face_model: str = "face_landmarker.task"
    hand_model: str = "hand_landmarker.task"

    # Override individual thresholds (None = use sensitivity preset)
    touch_distance_threshold: float | None = None
    confirmation_frames: int | None = None
    cooldown_frames: int | None = None
    z_depth_threshold: float | None = None

    def _resolve(self, param: str) -> float | int:
        """Resolve a parameter value: explicit override or sensitivity preset."""
        override = getattr(self, param)
        if override is not None:
            return override
        return _SENSITIVITY_PARAMS[self.sensitivity][param]

    @property
    def resolved_touch_distance(self) -> float:
        return float(self._resolve("touch_distance_threshold"))

    @property
    def resolved_confirmation(self) -> int:
        return int(self._resolve("confirmation_frames"))

    @property
    def resolved_cooldown(self) -> int:
        return int(self._resolve("cooldown_frames"))

    @property
    def resolved_z_depth(self) -> float:
        return float(self._resolve("z_depth_threshold"))


# ---------------------------------------------------------------------------
# Internal state tracking
# ---------------------------------------------------------------------------


@dataclass
class _ZoneState:
    """Per-zone tracking for confirmation and cooldown."""

    consecutive_frames: int = 0
    cooldown_remaining: int = 0


@dataclass
class _SessionState:
    """Per-session detection state."""

    zones: dict[FaceZone, _ZoneState] = field(default_factory=dict)
    touch_count: int = 0


# ---------------------------------------------------------------------------
# Filter implementation
# ---------------------------------------------------------------------------


class FaceTouchFilter(VideoFilterProvider):
    """Detect hand-to-face contact using MediaPipe landmarks.

    Runs MediaPipe Face Landmarker (478 landmarks) and Hand Landmarker
    (21 landmarks per hand) on video frames.  Computes distance between
    fingertips and face zone centroids in normalized image coordinates.
    Applies layered false-positive filtering before emitting detection
    events.

    The filter never modifies the video frame — it only emits
    :class:`FilterEvent` instances to ``context.events``.

    Args:
        config: Detection configuration.  Defaults to medium sensitivity
            with cheeks, chin, and mouth zones.

    Example::

        from roomkit.video.pipeline.filter.mediapipe_face_touch import (
            FaceTouchConfig, FaceTouchFilter,
        )

        filter = FaceTouchFilter(FaceTouchConfig(
            sensitivity=FaceTouchSensitivity.HIGH,
            zones=frozenset({FaceZone.LEFT_CHEEK, FaceZone.RIGHT_CHEEK}),
        ))
    """

    def __init__(self, config: FaceTouchConfig | None = None) -> None:
        self._config = config or FaceTouchConfig()
        if self._config.every_n_frames < 1:
            raise ValueError(f"every_n_frames must be >= 1, got {self._config.every_n_frames}")

        self._face_detector: Any = None
        self._hand_detector: Any = None
        self._mp: Any = None
        self._init_lock = threading.Lock()

        self._frame_count = 0
        self._sessions: dict[str, _SessionState] = {}
        self._logged_first = False

    @property
    def name(self) -> str:
        return "face_touch"

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        self._frame_count += 1

        # Throttle: only run detection every N frames
        n = self._config.every_n_frames
        if n > 1 and self._frame_count % n != 1:
            return frame

        self._detect(frame, context)
        return frame

    def _detect(self, frame: VideoFrame, context: FilterContext) -> None:
        """Run MediaPipe detection and emit events for confirmed touches."""
        self._ensure_models()

        import numpy as np  # lazy: optional dep (mediapipe requires numpy)

        img = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=img)

        # Run face and hand detection
        face_result = self._face_detector.detect(mp_image)
        hand_result = self._hand_detector.detect(mp_image)

        if not face_result.face_landmarks or not hand_result.hand_landmarks:
            # No face or no hands — tick cooldowns, reset confirmations
            self._tick_no_detection()
            return

        face_landmarks = face_result.face_landmarks[0]  # first face only

        # Compute zone centroids
        zone_centroids = self._compute_zone_centroids(face_landmarks)

        # Compute face bounding box for proximity check
        face_bbox = self._compute_face_bbox(face_landmarks)

        # Check each hand against each zone
        sid = context.session_id or "default"
        session_state = self._sessions.setdefault(sid, _SessionState())
        touches_this_frame: dict[FaceZone, tuple[float, str]] = {}

        for hand_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            hand_label = "left" if hand_idx == 0 else "right"
            if len(hand_result.handedness) > hand_idx:
                hand_label = hand_result.handedness[hand_idx][0].category_name.lower()

            # Proximity check: is the hand near the face?
            if not self._hand_in_face_bbox(hand_landmarks, face_bbox):
                continue

            # Check fingertips against each monitored zone
            for zone in self._config.zones:
                centroid = zone_centroids.get(zone)
                if centroid is None:
                    continue

                distance, z_diff = self._min_fingertip_distance(hand_landmarks, centroid)

                # Distance threshold
                if distance > self._config.resolved_touch_distance:
                    continue

                # Z-depth filter: reject hands hovering in front of face
                if z_diff > self._config.resolved_z_depth:
                    continue

                # Record best touch for this zone this frame
                if zone not in touches_this_frame or distance < touches_this_frame[zone][0]:
                    touches_this_frame[zone] = (distance, hand_label)

        # Update confirmation/cooldown state and emit events
        for zone in self._config.zones:
            zs = session_state.zones.setdefault(zone, _ZoneState())

            if zs.cooldown_remaining > 0:
                zs.cooldown_remaining -= 1

            if zone in touches_this_frame:
                zs.consecutive_frames += 1
                distance, hand_label = touches_this_frame[zone]

                if (
                    zs.consecutive_frames >= self._config.resolved_confirmation
                    and zs.cooldown_remaining == 0
                ):
                    # Confirmed touch — emit event and reset for re-confirmation
                    session_state.touch_count += 1
                    zs.cooldown_remaining = self._config.resolved_cooldown
                    zs.consecutive_frames = 0

                    self._emit_touch_event(
                        frame, context, zone, hand_label, distance, session_state.touch_count
                    )

                    if not self._logged_first:
                        logger.info(
                            "Face touch detected: zone=%s, hand=%s, distance=%.4f",
                            zone.value,
                            hand_label,
                            distance,
                        )
                        self._logged_first = True
            else:
                zs.consecutive_frames = 0

    def _compute_zone_centroids(
        self, face_landmarks: list[Any]
    ) -> dict[FaceZone, tuple[float, float, float]]:
        """Compute centroid (x, y, z) for each monitored zone."""
        centroids: dict[FaceZone, tuple[float, float, float]] = {}
        for zone in self._config.zones:
            indices = ZONE_LANDMARKS[zone]
            xs, ys, zs = [], [], []
            for idx in indices:
                if idx < len(face_landmarks):
                    lm = face_landmarks[idx]
                    xs.append(lm.x)
                    ys.append(lm.y)
                    zs.append(lm.z)
            if xs:
                centroids[zone] = (
                    sum(xs) / len(xs),
                    sum(ys) / len(ys),
                    sum(zs) / len(zs),
                )
        return centroids

    def _compute_face_bbox(self, face_landmarks: list[Any]) -> tuple[float, float, float, float]:
        """Compute normalized bounding box (min_x, min_y, max_x, max_y) of face."""
        xs = [lm.x for lm in face_landmarks]
        ys = [lm.y for lm in face_landmarks]
        margin = 0.05  # expand bbox slightly
        return (
            min(xs) - margin,
            min(ys) - margin,
            max(xs) + margin,
            max(ys) + margin,
        )

    def _hand_in_face_bbox(
        self,
        hand_landmarks: list[Any],
        bbox: tuple[float, float, float, float],
    ) -> bool:
        """Check if any fingertip is within the face bounding box."""
        min_x, min_y, max_x, max_y = bbox
        for idx in FINGERTIP_INDICES:
            if idx < len(hand_landmarks):
                lm = hand_landmarks[idx]
                if min_x <= lm.x <= max_x and min_y <= lm.y <= max_y:
                    return True
        return False

    def _min_fingertip_distance(
        self,
        hand_landmarks: list[Any],
        centroid: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Return (min_2d_distance, z_diff) from closest 3D fingertip to zone centroid.

        Selects the fingertip with the smallest 3D Euclidean distance to the
        centroid, then returns (2D distance, z_diff) for that fingertip.  This
        ensures the z-depth filter is evaluated on the fingertip that is
        actually closest in 3D, not just the one closest in the image plane.
        """
        cx, cy, cz = centroid
        best_dist_3d = float("inf")
        best_dist_2d = float("inf")
        best_z_diff = float("inf")
        for idx in FINGERTIP_INDICES:
            if idx < len(hand_landmarks):
                lm = hand_landmarks[idx]
                dist_2d = math.sqrt((lm.x - cx) ** 2 + (lm.y - cy) ** 2)
                z_diff = abs(lm.z - cz)
                dist_3d = math.sqrt(dist_2d**2 + z_diff**2)
                if dist_3d < best_dist_3d:
                    best_dist_3d = dist_3d
                    best_dist_2d = dist_2d
                    best_z_diff = z_diff
        return best_dist_2d, best_z_diff

    def _emit_touch_event(
        self,
        frame: VideoFrame,
        context: FilterContext,
        zone: FaceZone,
        hand: str,
        distance: float,
        touch_count: int,
    ) -> None:
        """Emit a VideoDetectionEvent as a FilterEvent."""
        event = VideoDetectionEvent(
            kind="face_touch",
            labels=[zone.value],
            confidence=max(0.0, 1.0 - distance / self._config.resolved_touch_distance),
            metadata={
                "zone": zone.value,
                "hand": hand,
                "distance": round(distance, 6),
                "touch_count": touch_count,
            },
            frame_sequence=frame.sequence,
        )
        context.events.append(FilterEvent(kind="face_touch", data=event))

    def _tick_no_detection(self) -> None:
        """Reset confirmation counters and tick cooldowns when no face/hand."""
        for state in self._sessions.values():
            for zs in state.zones.values():
                zs.consecutive_frames = 0
                if zs.cooldown_remaining > 0:
                    zs.cooldown_remaining -= 1

    def _ensure_models(self) -> None:
        """Load MediaPipe models on first use, downloading if needed."""
        if self._face_detector is not None:
            return
        with self._init_lock:
            # Double-check after acquiring lock
            if self._face_detector is not None:
                return

            self._mp = _load_mediapipe()
            mp = self._mp

            face_path = _resolve_model(self._config.face_model, "face_landmarker")
            hand_path = _resolve_model(self._config.hand_model, "hand_landmarker")

            logger.info("Loading MediaPipe models: face=%s, hand=%s", face_path, hand_path)

            base_options_face = mp.tasks.BaseOptions(model_asset_path=face_path)
            face_options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options_face,
                num_faces=1,
            )
            face_detector = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)

            base_options_hand = mp.tasks.BaseOptions(model_asset_path=hand_path)
            hand_options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options_hand,
                num_hands=2,
            )
            self._hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
            # Assign face_detector last — it's the guard variable for the fast path
            self._face_detector = face_detector

    def reset(self) -> None:
        self._frame_count = 0
        self._sessions.clear()
        self._logged_first = False

    def close(self) -> None:
        if self._face_detector is not None:
            self._face_detector.close()
            self._face_detector = None
        if self._hand_detector is not None:
            self._hand_detector.close()
            self._hand_detector = None
        self._mp = None
        self._sessions.clear()
