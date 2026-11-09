"""
Rep Segmenter - Segment video into individual repetitions.

Uses the primary signal to find peaks and valleys,
then segments the video into individual reps.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import signal as scipy_signal

from .metric_calculator import FrameMetrics
from .signal_analyzer import SignalCharacteristics


@dataclass
class RepSegment:
    """A single repetition segment."""
    rep_number: int
    start_index: int
    end_index: int
    start_time: float
    end_time: float

    # Phase boundaries (indices)
    descent_start: int = 0
    bottom_index: int = 0
    ascent_end: int = 0

    # Phase timings
    eccentric_duration_ms: float = 0.0  # Descent time
    concentric_duration_ms: float = 0.0  # Ascent time
    total_duration_ms: float = 0.0

    # Primary signal values at key points
    top_value: float = 0.0  # Value at start/end (extended position)
    bottom_value: float = 0.0  # Value at bottom (contracted/deep position)
    amplitude: float = 0.0  # bottom - top (or vice versa depending on signal)

    # Quality indicators
    is_valid: bool = True
    invalidation_reason: str = ""

    # All metrics during this rep (for threshold learning)
    frame_metrics: List[FrameMetrics] = field(default_factory=list)


@dataclass
class SegmentationResult:
    """Result of rep segmentation."""
    reps: List[RepSegment] = field(default_factory=list)
    valid_reps: List[RepSegment] = field(default_factory=list)
    invalid_reps: List[RepSegment] = field(default_factory=list)

    # Statistics
    total_reps: int = 0
    valid_count: int = 0
    invalid_count: int = 0

    # Timing stats
    avg_duration_ms: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float = 0.0

    # Signal used
    primary_signal: str = ""
    signal_direction: str = "up"  # "up" = signal increases going to bottom, "down" = decreases


class RepSegmenter:
    """
    Segment video into individual repetitions using the primary signal.
    """

    def __init__(
        self,
        min_rep_duration_ms: float = 500,
        max_rep_duration_ms: float = 10000,
        min_amplitude_ratio: float = 0.3,  # Min amplitude as ratio of max observed
    ):
        """
        Args:
            min_rep_duration_ms: Minimum time for a valid rep
            max_rep_duration_ms: Maximum time for a valid rep
            min_amplitude_ratio: Minimum amplitude relative to max observed
        """
        self.min_rep_duration_ms = min_rep_duration_ms
        self.max_rep_duration_ms = max_rep_duration_ms
        self.min_amplitude_ratio = min_amplitude_ratio

    def segment_reps(
        self,
        frame_metrics: List[FrameMetrics],
        primary_signal: SignalCharacteristics,
    ) -> SegmentationResult:
        """
        Segment the video into individual reps.

        Args:
            frame_metrics: All frame metrics
            primary_signal: The primary movement signal to use for segmentation

        Returns:
            SegmentationResult with all detected reps
        """
        result = SegmentationResult(primary_signal=primary_signal.name)

        values = primary_signal.values
        timestamps = primary_signal.timestamps

        if len(values) < 20:
            return result

        # Smooth the signal slightly to reduce noise
        smoothed = self._smooth_signal(values)

        # Determine signal direction (does it go up or down at bottom of movement?)
        result.signal_direction = self._determine_direction(smoothed, timestamps, primary_signal.name)

        # Find peaks and valleys
        peaks, valleys = self._find_extrema(smoothed, timestamps)

        if len(peaks) < 1 or len(valleys) < 1:
            return result

        # Determine which are "top" and which are "bottom" based on direction
        if result.signal_direction == "up":
            # Signal increases at bottom (e.g., depth metric)
            bottom_indices = peaks
            top_indices = valleys
        else:
            # Signal decreases at bottom (e.g., joint angle where 180 = extended)
            bottom_indices = valleys
            top_indices = peaks

        # Segment into reps (top -> bottom -> top)
        reps = self._create_rep_segments(
            frame_metrics,
            smoothed,
            timestamps,
            top_indices,
            bottom_indices
        )

        # Validate reps
        max_amplitude = max(r.amplitude for r in reps) if reps else 0
        for rep in reps:
            self._validate_rep(rep, max_amplitude)
            if rep.is_valid:
                result.valid_reps.append(rep)
            else:
                result.invalid_reps.append(rep)

        result.reps = reps
        result.total_reps = len(reps)
        result.valid_count = len(result.valid_reps)
        result.invalid_count = len(result.invalid_reps)

        # Calculate timing statistics from valid reps
        if result.valid_reps:
            durations = [r.total_duration_ms for r in result.valid_reps]
            result.avg_duration_ms = np.mean(durations)
            result.min_duration_ms = np.min(durations)
            result.max_duration_ms = np.max(durations)

        return result

    def _smooth_signal(self, values: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply smoothing to reduce noise."""
        if len(values) < window:
            return values
        kernel = np.ones(window) / window
        # Pad to handle edges
        padded = np.pad(values, (window//2, window//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed[:len(values)]

    def _determine_direction(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        signal_name: str = ""
    ) -> str:
        """
        Determine if signal goes up or down at the bottom of movement.

        For elbow/knee ANGLES:
        - 180° = fully extended = TOP position
        - 90° = bent = BOTTOM position
        - So angle DECREASES at bottom → direction = "down"

        For DEPTH metrics (normalized 0-1):
        - 0 = top position
        - 1 = bottom position
        - So depth INCREASES at bottom → direction = "up"

        Returns:
            "up" if signal increases at bottom, "down" if decreases
        """
        # For angle metrics, we know the physics: lower angle = bottom of movement
        angle_metrics = {
            'avg_elbow_angle', 'left_elbow_angle', 'right_elbow_angle',
            'avg_knee_angle', 'left_knee_angle', 'right_knee_angle',
            'avg_hip_angle', 'left_hip_angle', 'right_hip_angle'
        }

        # For these metrics, signal goes DOWN at bottom (lower angle = more bent)
        if signal_name in angle_metrics:
            return "down"

        # For depth metrics, signal goes UP at bottom (higher depth = deeper)
        depth_metrics = {'elbow_depth', 'knee_depth'}
        if signal_name in depth_metrics:
            return "up"

        # For height metrics (shoulder_height, hip_height), lower = bottom
        height_metrics = {'shoulder_height', 'hip_height', 'nose_height'}
        if signal_name in height_metrics:
            return "down"

        # For unknown metrics, use statistical analysis
        peaks, _ = scipy_signal.find_peaks(values, prominence=np.std(values) * 0.3)
        valleys, _ = scipy_signal.find_peaks(-values, prominence=np.std(values) * 0.3)

        if len(peaks) > 0 and len(valleys) > 0:
            avg_peak_height = np.mean(values[peaks])
            avg_valley_height = np.mean(values[valleys])
            return "up" if avg_peak_height > avg_valley_height else "down"

        return "up"  # Default assumption

    def _find_extrema(
        self,
        values: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks and valleys in the signal."""
        # Calculate prominence threshold based on signal range
        prominence = np.std(values) * 0.4

        # Estimate minimum distance between peaks (based on expected rep duration)
        duration = timestamps[-1] - timestamps[0]
        samples_per_second = len(values) / duration if duration > 0 else 30
        min_distance = max(1, int(samples_per_second * 0.3))  # At least 0.3s between peaks

        peaks, _ = scipy_signal.find_peaks(
            values,
            prominence=prominence,
            distance=min_distance
        )

        valleys, _ = scipy_signal.find_peaks(
            -values,
            prominence=prominence,
            distance=min_distance
        )

        return peaks, valleys

    def _create_rep_segments(
        self,
        frame_metrics: List[FrameMetrics],
        values: np.ndarray,
        timestamps: np.ndarray,
        top_indices: np.ndarray,
        bottom_indices: np.ndarray
    ) -> List[RepSegment]:
        """Create RepSegment objects from detected extrema."""
        reps = []

        # We need to pair tops and bottoms to form reps
        # A rep goes: top -> bottom -> top (or starts from first movement)

        # Merge and sort all extrema with their type
        all_extrema = []
        for idx in top_indices:
            all_extrema.append((idx, 'top', values[idx], timestamps[idx]))
        for idx in bottom_indices:
            all_extrema.append((idx, 'bottom', values[idx], timestamps[idx]))

        all_extrema.sort(key=lambda x: x[0])

        # Find sequences of top-bottom-top
        rep_number = 0
        i = 0
        while i < len(all_extrema) - 1:
            # Look for a complete rep pattern
            # Could start with top or bottom depending on where video starts

            # Find next top
            while i < len(all_extrema) and all_extrema[i][1] != 'top':
                i += 1
            if i >= len(all_extrema) - 1:
                break

            start_top = all_extrema[i]

            # Find next bottom after this top
            j = i + 1
            while j < len(all_extrema) and all_extrema[j][1] != 'bottom':
                j += 1
            if j >= len(all_extrema):
                break

            bottom = all_extrema[j]

            # Find next top after bottom (end of rep)
            k = j + 1
            while k < len(all_extrema) and all_extrema[k][1] != 'top':
                k += 1
            if k >= len(all_extrema):
                # Use last frame as end
                end_idx = len(values) - 1
                end_time = timestamps[-1]
                end_top_value = values[-1]
            else:
                end_top = all_extrema[k]
                end_idx = end_top[0]
                end_time = end_top[3]
                end_top_value = end_top[2]

            rep_number += 1

            # Calculate timings
            start_time = start_top[3]
            bottom_time = bottom[3]
            eccentric_ms = (bottom_time - start_time) * 1000
            concentric_ms = (end_time - bottom_time) * 1000
            total_ms = (end_time - start_time) * 1000

            # Calculate amplitude
            top_value = (start_top[2] + end_top_value) / 2
            bottom_value = bottom[2]
            amplitude = abs(bottom_value - top_value)

            # Get frame metrics for this rep
            start_frame_idx = start_top[0]
            end_frame_idx = end_idx

            # Match by time: signal positions differ from source frame indices
            # after missing poses or per-metric filtering.
            rep_metrics = []
            for fm in frame_metrics:
                if start_time <= fm.timestamp <= end_time:
                    rep_metrics.append(fm)

            rep = RepSegment(
                rep_number=rep_number,
                start_index=start_top[0],
                end_index=end_idx,
                start_time=start_time,
                end_time=end_time,
                descent_start=start_top[0],
                bottom_index=bottom[0],
                ascent_end=end_idx,
                eccentric_duration_ms=eccentric_ms,
                concentric_duration_ms=concentric_ms,
                total_duration_ms=total_ms,
                top_value=top_value,
                bottom_value=bottom_value,
                amplitude=amplitude,
                frame_metrics=rep_metrics
            )

            reps.append(rep)

            # Move to next potential rep start
            i = k

        return reps

    def _validate_rep(self, rep: RepSegment, max_amplitude: float) -> None:
        """Validate a rep and mark invalid ones with reason."""
        # Check duration
        if rep.total_duration_ms < self.min_rep_duration_ms:
            rep.is_valid = False
            rep.invalidation_reason = f"Too short ({rep.total_duration_ms:.0f}ms < {self.min_rep_duration_ms:.0f}ms)"
            return

        if rep.total_duration_ms > self.max_rep_duration_ms:
            rep.is_valid = False
            rep.invalidation_reason = f"Too long ({rep.total_duration_ms:.0f}ms > {self.max_rep_duration_ms:.0f}ms)"
            return

        # Check amplitude
        if max_amplitude > 0:
            amplitude_ratio = rep.amplitude / max_amplitude
            if amplitude_ratio < self.min_amplitude_ratio:
                rep.is_valid = False
                rep.invalidation_reason = f"Amplitude too small ({amplitude_ratio:.2f} < {self.min_amplitude_ratio:.2f})"
                return

        # Check for missing frame metrics
        if len(rep.frame_metrics) < 5:
            rep.is_valid = False
            rep.invalidation_reason = "Insufficient frame data"
            return

        rep.is_valid = True

    def get_segmentation_summary(self, result: SegmentationResult) -> str:
        """Generate human-readable summary of segmentation."""
        lines = []

        lines.append(f"Found {result.total_reps} reps using {result.primary_signal}")
        lines.append(f"Signal direction: {result.signal_direction}")
        lines.append(f"Valid: {result.valid_count}, Invalid: {result.invalid_count}")

        if result.valid_reps:
            lines.append(f"\nRep durations: {result.min_duration_ms:.0f}ms - {result.max_duration_ms:.0f}ms "
                        f"(mean: {result.avg_duration_ms:.0f}ms)")

            lines.append("\nValid reps:")
            for rep in result.valid_reps[:10]:  # Show first 10
                lines.append(
                    f"  Rep {rep.rep_number}: {rep.total_duration_ms:.0f}ms "
                    f"(ecc: {rep.eccentric_duration_ms:.0f}ms, con: {rep.concentric_duration_ms:.0f}ms) "
                    f"amp: {rep.amplitude:.3f}"
                )

        if result.invalid_reps:
            lines.append("\nInvalid reps:")
            for rep in result.invalid_reps:
                lines.append(f"  Rep {rep.rep_number}: {rep.invalidation_reason}")

        return "\n".join(lines)
