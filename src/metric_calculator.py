"""
Calculate all possible metrics from pose frames.
This is the heart of the "smart" discovery - we compute everything,
then analyze what's moving vs stable.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from .pose_frame import PoseFrame, Point


@dataclass
class FrameMetrics:
    """All computed metrics for a single frame."""
    timestamp: float
    frame_index: int
    confidence: float

    # Individual landmark confidences (for gating)
    arms_confidence: float = 0.0  # min(left_elbow, right_elbow)
    legs_confidence: float = 0.0  # min(left_knee, right_knee)
    upper_body_confidence: float = 0.0
    lower_body_confidence: float = 0.0

    # Joint angles (in degrees, 0-180 where 180 = fully extended)
    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None
    left_shoulder_angle: Optional[float] = None  # Arm relative to torso
    right_shoulder_angle: Optional[float] = None
    left_hip_angle: Optional[float] = None
    right_hip_angle: Optional[float] = None
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None

    # Averaged angles (for bilateral exercises)
    avg_elbow_angle: Optional[float] = None
    avg_knee_angle: Optional[float] = None
    avg_hip_angle: Optional[float] = None

    # Positions (normalized 0-1, relative to frame)
    shoulder_height: Optional[float] = None
    hip_height: Optional[float] = None
    nose_height: Optional[float] = None

    # Body angles (degrees)
    torso_angle: Optional[float] = None  # Angle from vertical
    upper_arm_angle_left: Optional[float] = None  # Angle from vertical
    upper_arm_angle_right: Optional[float] = None
    forearm_angle_left: Optional[float] = None
    forearm_angle_right: Optional[float] = None
    thigh_angle_left: Optional[float] = None
    thigh_angle_right: Optional[float] = None

    # Ratios
    hand_width_ratio: Optional[float] = None  # wrist_width / shoulder_width (front view)
    hand_spread_ratio: Optional[float] = None  # wrist_x_offset / torso_length (side view - horizontal hand spread)
    stance_width_ratio: Optional[float] = None  # ankle_width / hip_width
    elbow_flare_ratio: Optional[float] = None  # For push-ups: elbow width / wrist width
    hip_elevation_ratio: Optional[float] = None  # For pike push-ups: (shoulderY - hipY) / shoulder_width
    stagger_offset_ratio: Optional[float] = None  # For staggered push-ups (side view): |wrist_y_offset| / torso_length
    # Wide push-up elbow flare detection (perspective-based):
    # Ratio of forearm length to upper arm projected length
    # When elbows flare: ratio ≈ 1.0 (both segments in same 2D plane)
    # When elbows tuck back: ratio > 1.0 (upper arm appears shorter due to perspective)
    forearm_to_upper_arm_ratio_left: Optional[float] = None
    forearm_to_upper_arm_ratio_right: Optional[float] = None
    avg_forearm_to_upper_arm_ratio: Optional[float] = None  # Average of both sides

    # Symmetry metrics (0 = perfect symmetry)
    shoulder_symmetry: Optional[float] = None  # Difference in Y position (raw)
    shoulder_asymmetry_normalized: Optional[float] = None  # Normalized like iOS: |y_diff| / avg_body_dim
    hip_symmetry: Optional[float] = None
    hip_asymmetry_normalized: Optional[float] = None  # Normalized: |y_diff| / hip_width (for hip rotation detection)
    elbow_angle_symmetry: Optional[float] = None
    knee_angle_symmetry: Optional[float] = None

    # Depth metrics (normalized, exercise-agnostic)
    elbow_depth: Optional[float] = None  # (180 - elbow_angle) / 90, clamped to 0-1
    knee_depth: Optional[float] = None  # (180 - knee_angle) / 90, clamped to 0-1

    # Z-based distance metrics (for perspective correction at runtime)
    # MediaPipe Z: negative = closer to camera, positive = further away
    torso_z: Optional[float] = None  # Average Z of shoulders + hips (reference distance)
    shoulder_z: Optional[float] = None  # Average Z of shoulders
    hip_z: Optional[float] = None  # Average Z of hips
    elbow_z: Optional[float] = None  # Average Z of elbows
    knee_z: Optional[float] = None  # Average Z of knees

    # Head trajectory metrics (for pike push-ups - head moves diagonally)
    # These track nose position relative to shoulders for trajectory analysis
    nose_x: Optional[float] = None  # Nose X position (normalized 0-1)
    nose_relative_x: Optional[float] = None  # Nose X relative to shoulder center (for horizontal drift)
    head_trajectory_angle: Optional[float] = None  # Angle of head movement from vertical (0=straight up/down)

    # Hand direction metrics (for archer push-ups - straight arm hand should point outward)
    # These measure the angle between the forearm direction and the hand pointing direction
    # Angle of 0° = fingers pointing along forearm (bad for straight arm in archer)
    # Angle of 90° = fingers pointing perpendicular to forearm (good for straight arm)
    left_hand_forearm_angle: Optional[float] = None  # Angle between wrist→index and elbow→wrist
    right_hand_forearm_angle: Optional[float] = None
    # Hand direction relative to horizontal (useful for checking if fingers point outward)
    # 0° = pointing right, 90° = pointing down, 180° = pointing left
    left_hand_direction_angle: Optional[float] = None  # Angle of wrist→index from horizontal
    right_hand_direction_angle: Optional[float] = None
    # Confidence for hand tracking
    left_hand_confidence: float = 0.0
    right_hand_confidence: float = 0.0

    # Shoulder tap metrics (for shoulder tap push-ups)
    # Distance from wrist to OPPOSITE shoulder, normalized by shoulder width
    # Low value (~0.0-0.2) indicates hand is touching/near opposite shoulder (tap detected)
    left_wrist_to_right_shoulder_dist: Optional[float] = None
    right_wrist_to_left_shoulder_dist: Optional[float] = None
    # Elbow angle difference (useful for detecting single-arm bend during tap)
    # High value indicates one arm is bending while other stays straight
    elbow_angle_difference: Optional[float] = None
    # Which arm is bending more (for tap detection): positive = left bending more, negative = right
    elbow_bend_bias: Optional[float] = None


class MetricCalculator:
    """
    Calculate all possible metrics from pose frames.
    The philosophy: compute everything, let the analyzer figure out what matters.
    """

    def __init__(self, camera_view: str = "front"):
        """
        Args:
            camera_view: "front" or "side" - affects some angle calculations
        """
        self.camera_view = camera_view

    def calculate_angle(self, p1: Point, vertex: Point, p2: Point) -> float:
        """
        Calculate angle at vertex between p1-vertex-p2.
        Returns angle in degrees (0-180).
        """
        v1 = np.array([p1.x - vertex.x, p1.y - vertex.y])
        v2 = np.array([p2.x - vertex.x, p2.y - vertex.y])

        # Handle zero vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 180.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def calculate_angle_from_vertical(self, p1: Point, p2: Point) -> float:
        """
        Calculate angle of line p1->p2 from vertical.
        0 = pointing straight down, 90 = horizontal.
        """
        dx = p2.x - p1.x
        dy = p2.y - p1.y

        # Angle from vertical (down is positive Y in image coordinates)
        angle = np.degrees(np.arctan2(abs(dx), dy))
        return angle

    def calculate_frame_metrics(self, frame: PoseFrame) -> FrameMetrics:
        """Calculate all metrics for a single pose frame."""
        metrics = FrameMetrics(
            timestamp=frame.timestamp,
            frame_index=frame.frame_index,
            confidence=frame.avg_confidence,
            # Populate landmark-specific confidences
            arms_confidence=frame.arms_confidence,
            legs_confidence=frame.legs_confidence,
            upper_body_confidence=frame.upper_body_confidence,
            lower_body_confidence=frame.lower_body_confidence,
        )

        # === JOINT ANGLES ===

        # Elbow angles
        if all([frame.left_shoulder, frame.left_elbow, frame.left_wrist]):
            metrics.left_elbow_angle = self.calculate_angle(
                frame.left_shoulder, frame.left_elbow, frame.left_wrist
            )

        if all([frame.right_shoulder, frame.right_elbow, frame.right_wrist]):
            metrics.right_elbow_angle = self.calculate_angle(
                frame.right_shoulder, frame.right_elbow, frame.right_wrist
            )

        # Average elbow angle
        if metrics.left_elbow_angle is not None and metrics.right_elbow_angle is not None:
            metrics.avg_elbow_angle = (metrics.left_elbow_angle + metrics.right_elbow_angle) / 2
        elif metrics.left_elbow_angle is not None:
            metrics.avg_elbow_angle = metrics.left_elbow_angle
        elif metrics.right_elbow_angle is not None:
            metrics.avg_elbow_angle = metrics.right_elbow_angle

        # Shoulder angles (arm relative to torso)
        if all([frame.left_elbow, frame.left_shoulder, frame.left_hip]):
            metrics.left_shoulder_angle = self.calculate_angle(
                frame.left_elbow, frame.left_shoulder, frame.left_hip
            )

        if all([frame.right_elbow, frame.right_shoulder, frame.right_hip]):
            metrics.right_shoulder_angle = self.calculate_angle(
                frame.right_elbow, frame.right_shoulder, frame.right_hip
            )

        # Hip angles
        if all([frame.left_shoulder, frame.left_hip, frame.left_knee]):
            metrics.left_hip_angle = self.calculate_angle(
                frame.left_shoulder, frame.left_hip, frame.left_knee
            )

        if all([frame.right_shoulder, frame.right_hip, frame.right_knee]):
            metrics.right_hip_angle = self.calculate_angle(
                frame.right_shoulder, frame.right_hip, frame.right_knee
            )

        # Average hip angle
        if metrics.left_hip_angle is not None and metrics.right_hip_angle is not None:
            metrics.avg_hip_angle = (metrics.left_hip_angle + metrics.right_hip_angle) / 2
        elif metrics.left_hip_angle is not None:
            metrics.avg_hip_angle = metrics.left_hip_angle
        elif metrics.right_hip_angle is not None:
            metrics.avg_hip_angle = metrics.right_hip_angle

        # Knee angles
        if all([frame.left_hip, frame.left_knee, frame.left_ankle]):
            metrics.left_knee_angle = self.calculate_angle(
                frame.left_hip, frame.left_knee, frame.left_ankle
            )

        if all([frame.right_hip, frame.right_knee, frame.right_ankle]):
            metrics.right_knee_angle = self.calculate_angle(
                frame.right_hip, frame.right_knee, frame.right_ankle
            )

        # Average knee angle
        if metrics.left_knee_angle is not None and metrics.right_knee_angle is not None:
            metrics.avg_knee_angle = (metrics.left_knee_angle + metrics.right_knee_angle) / 2
        elif metrics.left_knee_angle is not None:
            metrics.avg_knee_angle = metrics.left_knee_angle
        elif metrics.right_knee_angle is not None:
            metrics.avg_knee_angle = metrics.right_knee_angle

        # === POSITIONS ===

        if frame.shoulder_center:
            metrics.shoulder_height = frame.shoulder_center.y

        if frame.hip_center:
            metrics.hip_height = frame.hip_center.y

        if frame.nose:
            metrics.nose_height = frame.nose.y

        # === BODY SEGMENT ANGLES ===

        # Torso angle from vertical
        if frame.shoulder_center and frame.hip_center:
            metrics.torso_angle = self.calculate_angle_from_vertical(
                frame.hip_center, frame.shoulder_center
            )

        # Upper arm angles
        if frame.left_shoulder and frame.left_elbow:
            metrics.upper_arm_angle_left = self.calculate_angle_from_vertical(
                frame.left_shoulder, frame.left_elbow
            )

        if frame.right_shoulder and frame.right_elbow:
            metrics.upper_arm_angle_right = self.calculate_angle_from_vertical(
                frame.right_shoulder, frame.right_elbow
            )

        # Forearm angles
        if frame.left_elbow and frame.left_wrist:
            metrics.forearm_angle_left = self.calculate_angle_from_vertical(
                frame.left_elbow, frame.left_wrist
            )

        if frame.right_elbow and frame.right_wrist:
            metrics.forearm_angle_right = self.calculate_angle_from_vertical(
                frame.right_elbow, frame.right_wrist
            )

        # Thigh angles
        if frame.left_hip and frame.left_knee:
            metrics.thigh_angle_left = self.calculate_angle_from_vertical(
                frame.left_hip, frame.left_knee
            )

        if frame.right_hip and frame.right_knee:
            metrics.thigh_angle_right = self.calculate_angle_from_vertical(
                frame.right_hip, frame.right_knee
            )

        # === RATIOS ===

        if frame.wrist_width is not None and frame.shoulder_width is not None and frame.shoulder_width > 0:
            metrics.hand_width_ratio = frame.wrist_width / frame.shoulder_width

        if frame.ankle_width is not None and frame.hip_width is not None and frame.hip_width > 0:
            metrics.stance_width_ratio = frame.ankle_width / frame.hip_width

        # Elbow flare ratio (for push-ups)
        if frame.left_elbow and frame.right_elbow and frame.wrist_width and frame.wrist_width > 0:
            elbow_width = frame.left_elbow.distance_to(frame.right_elbow)
            metrics.elbow_flare_ratio = elbow_width / frame.wrist_width

        # Hip elevation ratio (for pike push-ups)
        # iOS uses Vision coords: Y=0 at bottom, Y=1 at top
        # MediaPipe uses: Y=0 at top, Y=1 at bottom (opposite!)
        # iOS calculates: (shoulderY - hipY) which is negative when hips elevated
        # To match iOS sign convention, we flip: (hipY - shoulderY)
        # Positive = hips above shoulders (pike position)
        if frame.shoulder_center and frame.hip_center and frame.shoulder_width and frame.shoulder_width > 0.01:
            # In MediaPipe coords, lower Y = higher on screen
            # hipY - shoulderY: positive when hips are higher (have lower Y value)
            elevation = frame.hip_center.y - frame.shoulder_center.y
            metrics.hip_elevation_ratio = elevation / frame.shoulder_width

        # Stagger offset ratio (for staggered push-ups from side view)
        # Measures front-to-back hand offset normalized by torso length
        # From side view, wrist Y positions show the stagger offset
        if frame.wrist_y_offset is not None and frame.torso_length is not None and frame.torso_length > 0.01:
            metrics.stagger_offset_ratio = abs(frame.wrist_y_offset) / frame.torso_length

        # Hand spread ratio (for side view - replaces hand_width_ratio)
        # Measures horizontal distance between wrists normalized by torso length
        # From side view, X positions show how far apart hands are front-to-back
        if frame.wrist_x_offset is not None and frame.torso_length is not None and frame.torso_length > 0.01:
            metrics.hand_spread_ratio = frame.wrist_x_offset / frame.torso_length

        # Forearm to upper arm ratio (for wide push-up elbow flare detection)
        # Uses perspective to detect elbow position:
        # - Forearm (wrist to elbow) length is consistent regardless of elbow position
        # - Upper arm (elbow to shoulder) appears shorter when elbow moves backward (toward camera)
        # - When elbows flare out: both segments in same plane -> ratio ≈ 1.0
        # - When elbows tuck back (good form): upper arm compressed -> ratio > 1.0
        left_ratio = None
        right_ratio = None

        if frame.left_wrist and frame.left_elbow and frame.left_shoulder:
            forearm_len = frame.left_wrist.distance_to(frame.left_elbow)
            upper_arm_len = frame.left_elbow.distance_to(frame.left_shoulder)
            if upper_arm_len > 0.01:
                left_ratio = forearm_len / upper_arm_len
                metrics.forearm_to_upper_arm_ratio_left = left_ratio

        if frame.right_wrist and frame.right_elbow and frame.right_shoulder:
            forearm_len = frame.right_wrist.distance_to(frame.right_elbow)
            upper_arm_len = frame.right_elbow.distance_to(frame.right_shoulder)
            if upper_arm_len > 0.01:
                right_ratio = forearm_len / upper_arm_len
                metrics.forearm_to_upper_arm_ratio_right = right_ratio

        # Average of both sides
        if left_ratio is not None and right_ratio is not None:
            metrics.avg_forearm_to_upper_arm_ratio = (left_ratio + right_ratio) / 2
        elif left_ratio is not None:
            metrics.avg_forearm_to_upper_arm_ratio = left_ratio
        elif right_ratio is not None:
            metrics.avg_forearm_to_upper_arm_ratio = right_ratio

        # === SYMMETRY ===

        if frame.left_shoulder and frame.right_shoulder:
            metrics.shoulder_symmetry = abs(frame.left_shoulder.y - frame.right_shoulder.y)

            # Normalized asymmetry matching iOS calculation:
            # avgBodyDimension = (shoulderWidth + |leftShoulder.y - leftElbow.y| + |rightShoulder.y - rightElbow.y|) / 3.0
            # shoulderAsymmetry = |leftShoulder.y - rightShoulder.y| / avgBodyDimension * shoulderConfWeight
            if frame.left_elbow and frame.right_elbow and frame.shoulder_width and frame.shoulder_width > 0.01:
                left_upper_arm_len = abs(frame.left_shoulder.y - frame.left_elbow.y)
                right_upper_arm_len = abs(frame.right_shoulder.y - frame.right_elbow.y)
                avg_body_dimension = (frame.shoulder_width + left_upper_arm_len + right_upper_arm_len) / 3.0

                if avg_body_dimension > 0.01:
                    # Use min of shoulder confidences (matching iOS: shoulderConfWeight = min(leftShoulderConf, rightShoulderConf))
                    shoulder_conf_weight = min(frame.left_shoulder_confidence, frame.right_shoulder_confidence)
                    metrics.shoulder_asymmetry_normalized = (
                        abs(frame.left_shoulder.y - frame.right_shoulder.y) / avg_body_dimension
                    ) * shoulder_conf_weight

        if frame.left_hip and frame.right_hip:
            metrics.hip_symmetry = abs(frame.left_hip.y - frame.right_hip.y)

            # Normalized hip asymmetry for rotation detection (shoulder tap push-ups)
            # Normalize by hip width to make it scale-invariant
            if frame.hip_width and frame.hip_width > 0.01:
                metrics.hip_asymmetry_normalized = abs(frame.left_hip.y - frame.right_hip.y) / frame.hip_width

        if metrics.left_elbow_angle is not None and metrics.right_elbow_angle is not None:
            metrics.elbow_angle_symmetry = abs(metrics.left_elbow_angle - metrics.right_elbow_angle)

        if metrics.left_knee_angle is not None and metrics.right_knee_angle is not None:
            metrics.knee_angle_symmetry = abs(metrics.left_knee_angle - metrics.right_knee_angle)

        # === DEPTH METRICS (normalized) ===

        if metrics.avg_elbow_angle is not None:
            # 180 = fully extended (depth 0), 90 = bent 90 degrees (depth 1)
            metrics.elbow_depth = np.clip((180 - metrics.avg_elbow_angle) / 90, 0, 1)

        if metrics.avg_knee_angle is not None:
            metrics.knee_depth = np.clip((180 - metrics.avg_knee_angle) / 90, 0, 1)

        # === Z-BASED DISTANCE METRICS ===
        # These help with perspective correction at runtime

        metrics.torso_z = frame.torso_z
        metrics.shoulder_z = frame.shoulder_z
        metrics.hip_z = frame.hip_z
        metrics.elbow_z = frame.elbow_z
        metrics.knee_z = frame.knee_z

        # === HEAD TRAJECTORY METRICS ===
        # For pike push-ups, the head moves diagonally (forward and down), not straight up/down
        # Track nose X position and relative position to shoulder center

        if frame.nose:
            metrics.nose_x = frame.nose.x

            if frame.shoulder_center:
                # Nose X relative to shoulder center
                # Positive = nose is to the right of shoulders, Negative = to the left
                metrics.nose_relative_x = frame.nose.x - frame.shoulder_center.x

        # === HAND DIRECTION METRICS (for archer push-ups) ===
        # Calculate angle between hand pointing direction (wrist→index) and forearm (elbow→wrist)
        # For archer push-ups, the straight arm's hand should point outward (~90° from forearm)

        # Left hand
        if all([frame.left_elbow, frame.left_wrist, frame.left_index]):
            # Forearm vector: elbow → wrist
            forearm_vec = np.array([
                frame.left_wrist.x - frame.left_elbow.x,
                frame.left_wrist.y - frame.left_elbow.y
            ])
            # Hand vector: wrist → index finger
            hand_vec = np.array([
                frame.left_index.x - frame.left_wrist.x,
                frame.left_index.y - frame.left_wrist.y
            ])

            # Calculate angle between forearm and hand direction
            norm_f = np.linalg.norm(forearm_vec)
            norm_h = np.linalg.norm(hand_vec)
            if norm_f > 1e-6 and norm_h > 1e-6:
                cos_angle = np.dot(forearm_vec, hand_vec) / (norm_f * norm_h)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                metrics.left_hand_forearm_angle = np.degrees(np.arccos(cos_angle))

            # Hand direction from horizontal (0° = pointing right)
            if norm_h > 1e-6:
                metrics.left_hand_direction_angle = np.degrees(np.arctan2(hand_vec[1], hand_vec[0]))

            # Confidence based on wrist and index finger visibility
            metrics.left_hand_confidence = min(frame.left_wrist_confidence, frame.left_index_confidence)

        # Right hand
        if all([frame.right_elbow, frame.right_wrist, frame.right_index]):
            # Forearm vector: elbow → wrist
            forearm_vec = np.array([
                frame.right_wrist.x - frame.right_elbow.x,
                frame.right_wrist.y - frame.right_elbow.y
            ])
            # Hand vector: wrist → index finger
            hand_vec = np.array([
                frame.right_index.x - frame.right_wrist.x,
                frame.right_index.y - frame.right_wrist.y
            ])

            # Calculate angle between forearm and hand direction
            norm_f = np.linalg.norm(forearm_vec)
            norm_h = np.linalg.norm(hand_vec)
            if norm_f > 1e-6 and norm_h > 1e-6:
                cos_angle = np.dot(forearm_vec, hand_vec) / (norm_f * norm_h)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                metrics.right_hand_forearm_angle = np.degrees(np.arccos(cos_angle))

            # Hand direction from horizontal (0° = pointing right)
            if norm_h > 1e-6:
                metrics.right_hand_direction_angle = np.degrees(np.arctan2(hand_vec[1], hand_vec[0]))

            # Confidence based on wrist and index finger visibility
            metrics.right_hand_confidence = min(frame.right_wrist_confidence, frame.right_index_confidence)

        # === SHOULDER TAP METRICS ===
        # These help detect when a hand is reaching toward the opposite shoulder (tap motion)

        # Distance from left wrist to right shoulder (normalized by shoulder width)
        # Low value = left hand near right shoulder = left hand tapping
        if frame.left_wrist and frame.right_shoulder and frame.shoulder_width and frame.shoulder_width > 0.01:
            dist = frame.left_wrist.distance_to(frame.right_shoulder)
            metrics.left_wrist_to_right_shoulder_dist = dist / frame.shoulder_width

        # Distance from right wrist to left shoulder (normalized by shoulder width)
        # Low value = right hand near left shoulder = right hand tapping
        if frame.right_wrist and frame.left_shoulder and frame.shoulder_width and frame.shoulder_width > 0.01:
            dist = frame.right_wrist.distance_to(frame.left_shoulder)
            metrics.right_wrist_to_left_shoulder_dist = dist / frame.shoulder_width

        # Elbow angle difference and bias (for detecting single-arm bend during tap)
        if metrics.left_elbow_angle is not None and metrics.right_elbow_angle is not None:
            metrics.elbow_angle_difference = abs(metrics.left_elbow_angle - metrics.right_elbow_angle)
            # Bias: which arm is bending more?
            # Left arm bending = left angle decreases from 180 → (180 - left) is larger
            # Positive bias = left arm bending more, Negative = right arm bending more
            left_bend = 180 - metrics.left_elbow_angle
            right_bend = 180 - metrics.right_elbow_angle
            metrics.elbow_bend_bias = left_bend - right_bend

        return metrics

    def calculate_all_metrics(self, frames: List[PoseFrame]) -> List[FrameMetrics]:
        """Calculate metrics for all frames."""
        return [self.calculate_frame_metrics(frame) for frame in frames]

    def get_metric_time_series(
        self,
        frame_metrics: List[FrameMetrics],
        metric_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a specific metric as a time series.

        Returns:
            Tuple of (timestamps, values) arrays, excluding None values
        """
        timestamps = []
        values = []

        for fm in frame_metrics:
            value = getattr(fm, metric_name, None)
            if value is not None:
                timestamps.append(fm.timestamp)
                values.append(value)

        return np.array(timestamps), np.array(values)

    def get_all_metric_names(self) -> List[str]:
        """Return list of all metric names that can be computed."""
        return [
            # Joint angles
            'left_elbow_angle', 'right_elbow_angle', 'avg_elbow_angle',
            'left_shoulder_angle', 'right_shoulder_angle',
            'left_hip_angle', 'right_hip_angle', 'avg_hip_angle',
            'left_knee_angle', 'right_knee_angle', 'avg_knee_angle',
            # Positions
            'shoulder_height', 'hip_height', 'nose_height',
            # Body angles
            'torso_angle',
            'upper_arm_angle_left', 'upper_arm_angle_right',
            'forearm_angle_left', 'forearm_angle_right',
            'thigh_angle_left', 'thigh_angle_right',
            # Ratios
            'hand_width_ratio', 'hand_spread_ratio', 'stance_width_ratio', 'elbow_flare_ratio', 'hip_elevation_ratio', 'stagger_offset_ratio',
            'forearm_to_upper_arm_ratio_left', 'forearm_to_upper_arm_ratio_right', 'avg_forearm_to_upper_arm_ratio',
            # Symmetry
            'shoulder_symmetry', 'shoulder_asymmetry_normalized', 'hip_symmetry', 'hip_asymmetry_normalized',
            'elbow_angle_symmetry', 'knee_angle_symmetry',
            # Depth
            'elbow_depth', 'knee_depth',
            # Z-based distance (for perspective correction)
            'torso_z', 'shoulder_z', 'hip_z', 'elbow_z', 'knee_z',
            # Head trajectory (for pike push-ups)
            'nose_x', 'nose_relative_x', 'head_trajectory_angle',
            # Hand direction (for archer push-ups)
            'left_hand_forearm_angle', 'right_hand_forearm_angle',
            'left_hand_direction_angle', 'right_hand_direction_angle',
            # Shoulder tap metrics
            'left_wrist_to_right_shoulder_dist', 'right_wrist_to_left_shoulder_dist',
            'elbow_angle_difference', 'elbow_bend_bias',
        ]
