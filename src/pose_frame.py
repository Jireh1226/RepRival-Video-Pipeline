"""
PoseFrame dataclass and related structures for pose data representation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np


@dataclass
class Point:
    """A 3D point with optional confidence score."""
    x: float
    y: float
    confidence: float = 1.0
    z: Optional[float] = None  # Depth/distance from camera (MediaPipe provides this)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def to_array_3d(self) -> np.ndarray:
        """Return 3D array including Z if available."""
        if self.z is not None:
            return np.array([self.x, self.y, self.z])
        return np.array([self.x, self.y, 0.0])

    def distance_to(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to_3d(self, other: 'Point') -> float:
        """3D distance including Z coordinate."""
        dz = 0.0
        if self.z is not None and other.z is not None:
            dz = (self.z - other.z) ** 2
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + dz)

    def midpoint(self, other: 'Point') -> 'Point':
        z_mid = None
        if self.z is not None and other.z is not None:
            z_mid = (self.z + other.z) / 2
        return Point(
            x=(self.x + other.x) / 2,
            y=(self.y + other.y) / 2,
            confidence=min(self.confidence, other.confidence),
            z=z_mid
        )


@dataclass
class PoseFrame:
    """
    A single frame of pose data with all relevant landmarks.
    Coordinates are normalized to [0, 1] range.
    """
    timestamp: float  # seconds
    frame_index: int

    # Core landmarks (always present if pose detected)
    nose: Optional[Point] = None

    # Shoulders
    left_shoulder: Optional[Point] = None
    right_shoulder: Optional[Point] = None

    # Elbows
    left_elbow: Optional[Point] = None
    right_elbow: Optional[Point] = None

    # Wrists
    left_wrist: Optional[Point] = None
    right_wrist: Optional[Point] = None

    # Fingers (for hand direction tracking - important for archer push-ups)
    left_index: Optional[Point] = None  # Index finger tip
    right_index: Optional[Point] = None
    left_pinky: Optional[Point] = None  # Pinky finger
    right_pinky: Optional[Point] = None

    # Hips
    left_hip: Optional[Point] = None
    right_hip: Optional[Point] = None

    # Knees
    left_knee: Optional[Point] = None
    right_knee: Optional[Point] = None

    # Ankles
    left_ankle: Optional[Point] = None
    right_ankle: Optional[Point] = None

    # Overall frame confidence
    avg_confidence: float = 0.0

    # Individual landmark confidences (for gating)
    left_shoulder_confidence: float = 0.0
    right_shoulder_confidence: float = 0.0
    left_elbow_confidence: float = 0.0
    right_elbow_confidence: float = 0.0
    left_wrist_confidence: float = 0.0
    right_wrist_confidence: float = 0.0
    left_hip_confidence: float = 0.0
    right_hip_confidence: float = 0.0
    left_knee_confidence: float = 0.0
    right_knee_confidence: float = 0.0
    left_ankle_confidence: float = 0.0
    right_ankle_confidence: float = 0.0
    left_index_confidence: float = 0.0
    right_index_confidence: float = 0.0
    left_pinky_confidence: float = 0.0
    right_pinky_confidence: float = 0.0

    # Computed confidence properties (matching iOS)
    @property
    def arms_confidence(self) -> float:
        """Minimum confidence of both elbows - key for push-up tracking."""
        return min(self.left_elbow_confidence, self.right_elbow_confidence)

    @property
    def legs_confidence(self) -> float:
        """Minimum confidence of both knees - key for squat tracking."""
        return min(self.left_knee_confidence, self.right_knee_confidence)

    @property
    def upper_body_confidence(self) -> float:
        """Minimum of shoulders and elbows."""
        return min(
            self.left_shoulder_confidence, self.right_shoulder_confidence,
            self.left_elbow_confidence, self.right_elbow_confidence
        )

    @property
    def lower_body_confidence(self) -> float:
        """Minimum of hips, knees, and ankles."""
        return min(
            self.left_hip_confidence, self.right_hip_confidence,
            self.left_knee_confidence, self.right_knee_confidence,
            self.left_ankle_confidence, self.right_ankle_confidence
        )

    # Computed properties
    @property
    def shoulder_center(self) -> Optional[Point]:
        if self.left_shoulder and self.right_shoulder:
            return self.left_shoulder.midpoint(self.right_shoulder)
        return None

    @property
    def hip_center(self) -> Optional[Point]:
        if self.left_hip and self.right_hip:
            return self.left_hip.midpoint(self.right_hip)
        return None

    @property
    def shoulder_width(self) -> Optional[float]:
        if self.left_shoulder and self.right_shoulder:
            return self.left_shoulder.distance_to(self.right_shoulder)
        return None

    @property
    def hip_width(self) -> Optional[float]:
        if self.left_hip and self.right_hip:
            return self.left_hip.distance_to(self.right_hip)
        return None

    @property
    def wrist_width(self) -> Optional[float]:
        if self.left_wrist and self.right_wrist:
            return self.left_wrist.distance_to(self.right_wrist)
        return None

    @property
    def ankle_width(self) -> Optional[float]:
        if self.left_ankle and self.right_ankle:
            return self.left_ankle.distance_to(self.right_ankle)
        return None

    @property
    def torso_length(self) -> Optional[float]:
        """Distance from shoulder center to hip center - useful for side view normalization."""
        if self.shoulder_center and self.hip_center:
            return self.shoulder_center.distance_to(self.hip_center)
        return None

    @property
    def wrist_y_offset(self) -> Optional[float]:
        """
        Vertical (Y) offset between wrists - useful for side view stagger detection.
        Positive = left wrist lower (further down in image), Negative = right wrist lower.
        """
        if self.left_wrist and self.right_wrist:
            return self.left_wrist.y - self.right_wrist.y
        return None

    @property
    def wrist_x_offset(self) -> Optional[float]:
        """
        Horizontal (X) offset between wrists - useful for side view hand spread.
        From side view, this shows how far apart hands are front-to-back.
        """
        if self.left_wrist and self.right_wrist:
            return abs(self.left_wrist.x - self.right_wrist.x)
        return None

    # Z-based distance metrics (for perspective correction)
    @property
    def torso_z(self) -> Optional[float]:
        """
        Average Z of core torso landmarks (shoulders and hips).
        Represents estimated distance from camera.
        Lower values = closer to camera, higher = further away.
        """
        z_values = []
        for point in [self.left_shoulder, self.right_shoulder, self.left_hip, self.right_hip]:
            if point is not None and point.z is not None:
                z_values.append(point.z)
        return np.mean(z_values) if z_values else None

    @property
    def shoulder_z(self) -> Optional[float]:
        """Average Z of shoulders."""
        z_values = []
        for point in [self.left_shoulder, self.right_shoulder]:
            if point is not None and point.z is not None:
                z_values.append(point.z)
        return np.mean(z_values) if z_values else None

    @property
    def hip_z(self) -> Optional[float]:
        """Average Z of hips."""
        z_values = []
        for point in [self.left_hip, self.right_hip]:
            if point is not None and point.z is not None:
                z_values.append(point.z)
        return np.mean(z_values) if z_values else None

    @property
    def elbow_z(self) -> Optional[float]:
        """Average Z of elbows."""
        z_values = []
        for point in [self.left_elbow, self.right_elbow]:
            if point is not None and point.z is not None:
                z_values.append(point.z)
        return np.mean(z_values) if z_values else None

    @property
    def knee_z(self) -> Optional[float]:
        """Average Z of knees."""
        z_values = []
        for point in [self.left_knee, self.right_knee]:
            if point is not None and point.z is not None:
                z_values.append(point.z)
        return np.mean(z_values) if z_values else None

    def is_valid(self, min_confidence: float = 0.5) -> bool:
        """Check if frame has minimum required landmarks with sufficient confidence."""
        required = [
            self.left_shoulder, self.right_shoulder,
            self.left_hip, self.right_hip
        ]
        return all(p is not None and p.confidence >= min_confidence for p in required)


# MediaPipe landmark indices
class MediaPipeLandmarks:
    """MediaPipe Pose landmark indices."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


def landmarks_to_pose_frame(
    landmarks,
    timestamp: float,
    frame_index: int,
    image_width: int,
    image_height: int
) -> PoseFrame:
    """
    Convert MediaPipe landmarks to PoseFrame.

    Args:
        landmarks: MediaPipe pose landmarks
        timestamp: Frame timestamp in seconds
        frame_index: Frame number
        image_width: Video frame width
        image_height: Video frame height

    Returns:
        PoseFrame with normalized coordinates
    """
    def get_point(idx: int) -> Optional[Point]:
        if landmarks is None or idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        # MediaPipe returns normalized coordinates [0, 1] for x, y
        # Z is relative depth: negative = closer to camera, positive = further
        z_value = lm.z if hasattr(lm, 'z') else None
        return Point(
            x=lm.x,
            y=lm.y,
            confidence=lm.visibility if hasattr(lm, 'visibility') else 1.0,
            z=z_value
        )

    def get_confidence(idx: int) -> float:
        """Extract visibility/confidence for a landmark."""
        if landmarks is None or idx >= len(landmarks):
            return 0.0
        lm = landmarks[idx]
        return lm.visibility if hasattr(lm, 'visibility') else 1.0

    mp = MediaPipeLandmarks

    frame = PoseFrame(
        timestamp=timestamp,
        frame_index=frame_index,
        nose=get_point(mp.NOSE),
        left_shoulder=get_point(mp.LEFT_SHOULDER),
        right_shoulder=get_point(mp.RIGHT_SHOULDER),
        left_elbow=get_point(mp.LEFT_ELBOW),
        right_elbow=get_point(mp.RIGHT_ELBOW),
        left_wrist=get_point(mp.LEFT_WRIST),
        right_wrist=get_point(mp.RIGHT_WRIST),
        left_index=get_point(mp.LEFT_INDEX),
        right_index=get_point(mp.RIGHT_INDEX),
        left_pinky=get_point(mp.LEFT_PINKY),
        right_pinky=get_point(mp.RIGHT_PINKY),
        left_hip=get_point(mp.LEFT_HIP),
        right_hip=get_point(mp.RIGHT_HIP),
        left_knee=get_point(mp.LEFT_KNEE),
        right_knee=get_point(mp.RIGHT_KNEE),
        left_ankle=get_point(mp.LEFT_ANKLE),
        right_ankle=get_point(mp.RIGHT_ANKLE),
        # Individual landmark confidences
        left_shoulder_confidence=get_confidence(mp.LEFT_SHOULDER),
        right_shoulder_confidence=get_confidence(mp.RIGHT_SHOULDER),
        left_elbow_confidence=get_confidence(mp.LEFT_ELBOW),
        right_elbow_confidence=get_confidence(mp.RIGHT_ELBOW),
        left_wrist_confidence=get_confidence(mp.LEFT_WRIST),
        right_wrist_confidence=get_confidence(mp.RIGHT_WRIST),
        left_index_confidence=get_confidence(mp.LEFT_INDEX),
        right_index_confidence=get_confidence(mp.RIGHT_INDEX),
        left_pinky_confidence=get_confidence(mp.LEFT_PINKY),
        right_pinky_confidence=get_confidence(mp.RIGHT_PINKY),
        left_hip_confidence=get_confidence(mp.LEFT_HIP),
        right_hip_confidence=get_confidence(mp.RIGHT_HIP),
        left_knee_confidence=get_confidence(mp.LEFT_KNEE),
        right_knee_confidence=get_confidence(mp.RIGHT_KNEE),
        left_ankle_confidence=get_confidence(mp.LEFT_ANKLE),
        right_ankle_confidence=get_confidence(mp.RIGHT_ANKLE),
    )

    # Calculate average confidence from key points
    key_confidences = [
        frame.left_shoulder_confidence, frame.right_shoulder_confidence,
        frame.left_elbow_confidence, frame.right_elbow_confidence,
        frame.left_hip_confidence, frame.right_hip_confidence,
        frame.left_knee_confidence, frame.right_knee_confidence
    ]
    frame.avg_confidence = np.mean(key_confidences) if key_confidences else 0.0

    return frame
