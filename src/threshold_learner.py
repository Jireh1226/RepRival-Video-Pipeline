"""
Threshold Learner - Learn exercise thresholds from segmented reps.

Analyzes rep aggregates and derives statistical thresholds for:
- Depth/ROM
- Tempo
- Velocity
- Stability
- Ratios
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from .metric_calculator import FrameMetrics
from .signal_analyzer import SignalAnalysisResult, SignalCharacteristics
from .rep_segmenter import RepSegment, SegmentationResult


@dataclass
class DepthThresholds:
    """Learned depth/ROM thresholds."""
    signal: str  # Which signal measures depth
    depth_good: float  # Target depth for good rep
    depth_acceptable: float  # Minimum acceptable depth
    depth_minimum: float  # Hard floor - below this is invalid
    rom_minimum: float  # Minimum range of motion (bottom - top_threshold)
    top_threshold: float  # Value at "top" position (avoids enforcing lockout)
    observed_range: Tuple[float, float] = (0.0, 0.0)
    observed_top_range: Tuple[float, float] = (0.0, 0.0)  # Top position values


@dataclass
class TempoBuffers:
    """Favorable buffers for tempo thresholds.

    Tempo violation logic:
    - If rep time < min_threshold → "too fast" violation
    - If rep time > max_threshold → "too slow" violation

    These buffers make thresholds more forgiving:
    - Negative buffers on min values = LOWERS the min threshold
      → User can go faster before triggering "too fast"
    - Positive buffers on max values = RAISES the max threshold
      → User can go slower before triggering "too slow"

    Based on standard push-up testing:
    - "Too fast" violations: subtract ~150ms from observed min
    - "Too slow" violations: add ~400-600ms to observed max (more lenient)
    """
    eccentric_min_buffer_ms: int = -150   # Lowers min threshold (allows faster)
    eccentric_max_buffer_ms: int = 400    # Raises max threshold (allows slower)
    concentric_min_buffer_ms: int = -150  # Lowers min threshold (allows faster)
    concentric_max_buffer_ms: int = 400   # Raises max threshold (allows slower)
    total_min_buffer_ms: int = -150       # Lowers min threshold
    total_max_buffer_ms: int = 600        # Raises max threshold (more lenient for slow)


@dataclass
class TempoThresholds:
    """Learned tempo thresholds."""
    # Observed values (raw from video analysis)
    observed_eccentric_min_ms: float
    observed_eccentric_max_ms: float
    observed_concentric_min_ms: float
    observed_concentric_max_ms: float
    observed_total_min_ms: float
    observed_total_max_ms: float

    # Favorable buffers applied
    buffers: TempoBuffers = None

    # Final thresholds (observed + buffers) - these are what's used for enforcement
    eccentric_min_ms: float = 0  # Minimum descent time
    eccentric_max_ms: float = 0  # Maximum descent time
    concentric_min_ms: float = 0  # Minimum ascent time
    concentric_max_ms: float = 0  # Maximum ascent time
    total_min_ms: float = 0  # Minimum total rep time
    total_max_ms: float = 0  # Maximum total rep time

    # For reference/display
    observed_eccentric_range: Tuple[float, float] = (0.0, 0.0)
    observed_concentric_range: Tuple[float, float] = (0.0, 0.0)

    # Ideal range for praise (from observed data)
    ideal_eccentric_min_ms: int = 0
    ideal_eccentric_max_ms: int = 0

    def __post_init__(self):
        """Apply buffers to calculate final thresholds."""
        if self.buffers is None:
            self.buffers = TempoBuffers()

        # Calculate final thresholds = observed + buffer
        # For min values: observed + negative buffer = lower threshold (more lenient)
        # For max values: observed + positive buffer = higher threshold (more lenient)
        self.eccentric_min_ms = max(50, self.observed_eccentric_min_ms + self.buffers.eccentric_min_buffer_ms)
        self.eccentric_max_ms = self.observed_eccentric_max_ms + self.buffers.eccentric_max_buffer_ms
        self.concentric_min_ms = max(50, self.observed_concentric_min_ms + self.buffers.concentric_min_buffer_ms)
        self.concentric_max_ms = self.observed_concentric_max_ms + self.buffers.concentric_max_buffer_ms
        self.total_min_ms = max(100, self.observed_total_min_ms + self.buffers.total_min_buffer_ms)
        self.total_max_ms = self.observed_total_max_ms + self.buffers.total_max_buffer_ms


@dataclass
class VelocityThresholds:
    """Learned velocity thresholds for state machine."""
    v_enter: float  # Velocity to start movement detection
    v_exit: float  # Velocity to end movement detection
    depth_ema: float  # EMA smoothing factor for depth
    velocity_ema: float  # EMA smoothing factor for velocity


@dataclass
class StabilityThreshold:
    """Threshold for a single stability metric."""
    metric: str
    baseline: float  # Expected/mean value
    max_deviation: float  # Maximum acceptable deviation
    observed_std: float  # Observed standard deviation
    description: str = ""


@dataclass
class RatioBuffers:
    """Favorable buffers for ratio thresholds (percentage-based).

    Ratio violation logic:
    - If ratio < min_threshold → "too narrow" violation
    - If ratio > max_threshold → "too wide" violation

    Buffers are percentage-based (e.g., 0.10 = 10%):
    - Negative buffer on min = LOWERS min threshold (allows narrower)
    - Positive buffer on max = RAISES max threshold (allows wider)

    Based on standard push-up testing with hand_width_ratio:
    - Observed range might be ~1.0-1.4 for standard push-ups
    - Buffer of -0.10 on min → min becomes 0.90 (allows 10% narrower)
    - Buffer of +0.15 on max → max becomes 1.55 (allows 15% wider)
    """
    min_buffer_percent: float = -0.10  # Lowers min by 10% of baseline
    max_buffer_percent: float = 0.15   # Raises max by 15% of baseline


@dataclass
class RatioThreshold:
    """Threshold for a ratio metric with observed, buffers, and final values."""
    metric: str
    baseline: float  # Mean observed value

    # Observed range (raw from video analysis)
    observed_min: float = 0.0
    observed_max: float = 0.0

    # Favorable buffers (percentage-based)
    buffers: RatioBuffers = None

    # Final acceptable range (observed + buffers) - used for enforcement
    acceptable_range: Tuple[float, float] = (0.0, 0.0)

    # Ideal range for praise
    ideal_range: Tuple[float, float] = (0.0, 0.0)

    description: str = ""

    def __post_init__(self):
        """Apply percentage-based buffers to calculate final thresholds."""
        if self.buffers is None:
            self.buffers = RatioBuffers()

        # Calculate final range using percentage buffers of the baseline
        # min_final = observed_min + (baseline * min_buffer_percent)
        # max_final = observed_max + (baseline * max_buffer_percent)
        min_final = self.observed_min + (self.baseline * self.buffers.min_buffer_percent)
        max_final = self.observed_max + (self.baseline * self.buffers.max_buffer_percent)

        self.acceptable_range = (min_final, max_final)

        # Ideal range is the middle 50% of observed (for praise)
        # This will be set externally if needed


@dataclass
class DistanceReference:
    """Reference Z values for perspective correction at runtime."""
    torso_z_mean: float  # Average torso Z during training
    torso_z_range: Tuple[float, float]  # (min, max) Z during training
    shoulder_z_mean: float
    hip_z_mean: float
    elbow_z_mean: Optional[float] = None  # For upper body exercises
    knee_z_mean: Optional[float] = None  # For lower body exercises


@dataclass
class LearnedThresholds:
    """Complete set of learned thresholds."""
    exercise_id: str
    exercise_name: str
    camera_view: str
    limb_mode: str

    primary_signal: str
    signal_direction: str  # "up" or "down"

    depth: DepthThresholds = None
    tempo: TempoThresholds = None
    velocity: VelocityThresholds = None
    stability: List[StabilityThreshold] = field(default_factory=list)
    ratios: List[RatioThreshold] = field(default_factory=list)
    distance_reference: Optional[DistanceReference] = None  # Z-based distance for perspective correction

    # Metadata
    reps_analyzed: int = 0
    avg_confidence: float = 0.0
    video_source: str = ""


class ThresholdLearner:
    """
    Learn exercise thresholds from segmented reps.
    """

    # Buffer percentages for safety margins
    DEPTH_BUFFER = 0.05  # 5% buffer below observed minimum
    TEMPO_BUFFER = 0.15  # 15% buffer on tempo
    STABILITY_BUFFER = 1.5  # Multiply observed std by this for threshold

    # Percentiles for threshold derivation
    DEPTH_GOOD_PERCENTILE = 25  # p25 = most reps hit this
    DEPTH_ACCEPTABLE_PERCENTILE = 10  # p10 = buffer for acceptable
    TEMPO_MIN_PERCENTILE = 10  # p10 for minimum tempo
    TEMPO_MAX_PERCENTILE = 90  # p90 for maximum tempo

    def __init__(self):
        pass

    def learn_thresholds(
        self,
        segmentation: SegmentationResult,
        signal_analysis: SignalAnalysisResult,
        exercise_id: str,
        exercise_name: str,
        camera_view: str,
        limb_mode: str,
        video_source: str = ""
    ) -> LearnedThresholds:
        """
        Learn all thresholds from the segmented reps.

        Args:
            segmentation: Rep segmentation result
            signal_analysis: Signal analysis result
            exercise_id: Unique ID for the exercise
            exercise_name: Human-readable name
            camera_view: "front" or "side"
            limb_mode: "bilateral", "unilateral_alternating", etc.
            video_source: Source video filename

        Returns:
            Complete learned thresholds
        """
        valid_reps = segmentation.valid_reps

        if not valid_reps:
            raise ValueError("No valid reps to learn from")

        result = LearnedThresholds(
            exercise_id=exercise_id,
            exercise_name=exercise_name,
            camera_view=camera_view,
            limb_mode=limb_mode,
            primary_signal=segmentation.primary_signal,
            signal_direction=segmentation.signal_direction,
            reps_analyzed=len(valid_reps),
            video_source=video_source
        )

        # Learn depth thresholds
        result.depth = self._learn_depth_thresholds(
            valid_reps,
            segmentation.primary_signal,
            segmentation.signal_direction
        )

        # Learn tempo thresholds
        result.tempo = self._learn_tempo_thresholds(valid_reps)

        # Learn velocity thresholds
        result.velocity = self._learn_velocity_thresholds(valid_reps)

        # Learn stability thresholds
        result.stability = self._learn_stability_thresholds(
            valid_reps,
            signal_analysis.stability_signals
        )

        # Learn ratio thresholds
        result.ratios = self._learn_ratio_thresholds(
            valid_reps,
            signal_analysis.ratio_signals
        )

        # Learn distance reference (Z-based for perspective correction)
        result.distance_reference = self._learn_distance_reference(valid_reps)

        # Calculate average confidence
        all_confidences = []
        for rep in valid_reps:
            for fm in rep.frame_metrics:
                all_confidences.append(fm.confidence)
        result.avg_confidence = np.mean(all_confidences) if all_confidences else 0.0

        return result

    def _learn_depth_thresholds(
        self,
        reps: List[RepSegment],
        primary_signal: str,
        signal_direction: str
    ) -> DepthThresholds:
        """
        Learn depth/ROM thresholds from rep bottoms and tops.

        Key insight: We don't want to enforce joint lockout at the top.
        Instead, we define "top" as ~30% of normalized depth (0.3), so users
        don't need to fully extend to count a rep. ROM is measured from this
        top threshold to the bottom depth achieved.
        """
        # Collect bottom and top values from each rep
        bottom_values = [rep.bottom_value for rep in reps]
        top_values = [rep.top_value for rep in reps]
        amplitudes = [rep.amplitude for rep in reps]

        # For "up" direction, higher value = deeper (e.g., elbow angle increases at bottom)
        # For "down" direction, lower value = deeper (e.g., shoulder height decreases)

        if signal_direction == "up":
            # Higher is better for depth
            depth_good = np.percentile(bottom_values, self.DEPTH_GOOD_PERCENTILE)
            depth_acceptable = np.percentile(bottom_values, self.DEPTH_ACCEPTABLE_PERCENTILE)
            depth_minimum = depth_acceptable - (max(bottom_values) - min(bottom_values)) * self.DEPTH_BUFFER

            # Top threshold: We use the observed top values but set it generously
            # to avoid enforcing lockout. Use p75 of top values (higher = more lenient)
            observed_top_min = min(top_values)
            observed_top_max = max(top_values)
            # Set top threshold slightly above observed mean to allow for natural variation
            top_threshold = np.percentile(top_values, 75)
        else:
            # Lower is better (e.g., elbow angle where lower = more bent)
            depth_good = np.percentile(bottom_values, 100 - self.DEPTH_GOOD_PERCENTILE)
            depth_acceptable = np.percentile(bottom_values, 100 - self.DEPTH_ACCEPTABLE_PERCENTILE)
            depth_minimum = depth_acceptable + (max(bottom_values) - min(bottom_values)) * self.DEPTH_BUFFER

            # For down direction, top is higher value, use p25 (lower = more lenient)
            observed_top_min = min(top_values)
            observed_top_max = max(top_values)
            top_threshold = np.percentile(top_values, 25)

        # ROM minimum: The range from top_threshold to acceptable bottom
        # This ensures the user travels enough distance, not just hits bottom depth
        rom_minimum = np.percentile(amplitudes, self.DEPTH_ACCEPTABLE_PERCENTILE) * (1 - self.DEPTH_BUFFER)

        return DepthThresholds(
            signal=primary_signal,
            depth_good=depth_good,
            depth_acceptable=depth_acceptable,
            depth_minimum=depth_minimum,
            rom_minimum=rom_minimum,
            top_threshold=top_threshold,
            observed_range=(min(bottom_values), max(bottom_values)),
            observed_top_range=(observed_top_min, observed_top_max)
        )

    def _learn_tempo_thresholds(self, reps: List[RepSegment]) -> TempoThresholds:
        """Learn tempo thresholds from rep timings.

        Returns observed thresholds with favorable buffers that will be
        applied to create the final enforcement thresholds.
        """
        eccentric = [rep.eccentric_duration_ms for rep in reps]
        concentric = [rep.concentric_duration_ms for rep in reps]
        total = [rep.total_duration_ms for rep in reps]

        # Get observed values using percentiles
        # p10 for minimums, p90 for maximums (covers 80% of observed reps)
        observed_ecc_min = np.percentile(eccentric, self.TEMPO_MIN_PERCENTILE)
        observed_ecc_max = np.percentile(eccentric, self.TEMPO_MAX_PERCENTILE)
        observed_con_min = np.percentile(concentric, self.TEMPO_MIN_PERCENTILE)
        observed_con_max = np.percentile(concentric, self.TEMPO_MAX_PERCENTILE)
        observed_total_min = np.percentile(total, self.TEMPO_MIN_PERCENTILE)
        observed_total_max = np.percentile(total, self.TEMPO_MAX_PERCENTILE)

        # Ideal range for praise (middle 50% of observed)
        ideal_ecc_min = int(np.percentile(eccentric, 25))
        ideal_ecc_max = int(np.percentile(eccentric, 75))

        # Create with default favorable buffers
        # The __post_init__ will automatically calculate final thresholds
        return TempoThresholds(
            observed_eccentric_min_ms=observed_ecc_min,
            observed_eccentric_max_ms=observed_ecc_max,
            observed_concentric_min_ms=observed_con_min,
            observed_concentric_max_ms=observed_con_max,
            observed_total_min_ms=observed_total_min,
            observed_total_max_ms=observed_total_max,
            buffers=TempoBuffers(),  # Use default favorable buffers
            observed_eccentric_range=(min(eccentric), max(eccentric)),
            observed_concentric_range=(min(concentric), max(concentric)),
            ideal_eccentric_min_ms=ideal_ecc_min,
            ideal_eccentric_max_ms=ideal_ecc_max
        )

    def _learn_velocity_thresholds(self, reps: List[RepSegment]) -> VelocityThresholds:
        """
        Learn velocity thresholds for state machine transitions.

        These are used to detect movement start/stop.
        """
        # Collect velocities at movement start/stop
        # We estimate these from the rep boundaries

        enter_velocities = []
        exit_velocities = []

        for rep in reps:
            if len(rep.frame_metrics) < 5:
                continue

            # Estimate velocity at start of descent (first few frames)
            start_frames = rep.frame_metrics[:5]
            if len(start_frames) >= 2:
                # Calculate approximate velocity from timestamps and primary signal change
                # This is a simplified estimate
                enter_velocities.append(0.02)  # Default reasonable value

            # Estimate velocity at end of ascent
            exit_velocities.append(0.01)  # Default with hysteresis

        # Use reasonable defaults if we couldn't estimate
        v_enter = 0.02 if not enter_velocities else np.mean(enter_velocities)
        v_exit = v_enter * 0.5  # Exit threshold with hysteresis

        return VelocityThresholds(
            v_enter=v_enter,
            v_exit=v_exit,
            depth_ema=0.25,  # Standard smoothing
            velocity_ema=0.25
        )

    def _learn_stability_thresholds(
        self,
        reps: List[RepSegment],
        stability_signals: List[SignalCharacteristics]
    ) -> List[StabilityThreshold]:
        """Learn stability thresholds from low-variance metrics."""
        thresholds = []

        for sig in stability_signals:
            # Get values during all reps
            all_values = []
            for rep in reps:
                for fm in rep.frame_metrics:
                    val = getattr(fm, sig.name, None)
                    if val is not None:
                        all_values.append(val)

            if not all_values:
                continue

            baseline = np.mean(all_values)
            observed_std = np.std(all_values)

            # Max deviation = observed std * buffer
            # This allows for natural variation but catches form breaks
            max_deviation = observed_std * self.STABILITY_BUFFER

            # Ensure minimum threshold to avoid over-sensitivity
            if 'angle' in sig.name:
                max_deviation = max(max_deviation, 5.0)  # At least 5 degrees
            elif 'symmetry' in sig.name:
                max_deviation = max(max_deviation, 0.02)  # At least 0.02 normalized

            description = self._get_metric_description(sig.name)

            thresholds.append(StabilityThreshold(
                metric=sig.name,
                baseline=baseline,
                max_deviation=max_deviation,
                observed_std=observed_std,
                description=description
            ))

        return thresholds

    def _learn_ratio_thresholds(
        self,
        reps: List[RepSegment],
        ratio_signals: List[SignalCharacteristics]
    ) -> List[RatioThreshold]:
        """Learn ratio thresholds from ratio metrics.

        Returns observed thresholds with favorable buffers that will be
        applied to create the final enforcement thresholds.
        """
        thresholds = []

        for sig in ratio_signals:
            # Get values during all reps
            all_values = []
            for rep in reps:
                for fm in rep.frame_metrics:
                    val = getattr(fm, sig.name, None)
                    if val is not None:
                        all_values.append(val)

            if not all_values:
                continue

            baseline = np.mean(all_values)

            # Observed range using percentiles (p5 and p95 for robustness)
            observed_min = np.percentile(all_values, 5)
            observed_max = np.percentile(all_values, 95)

            # Ideal range for praise (middle 50%)
            ideal_min = np.percentile(all_values, 25)
            ideal_max = np.percentile(all_values, 75)

            description = self._get_metric_description(sig.name)

            # Create with default favorable buffers
            # The __post_init__ will automatically calculate final thresholds
            thresholds.append(RatioThreshold(
                metric=sig.name,
                baseline=baseline,
                observed_min=observed_min,
                observed_max=observed_max,
                buffers=RatioBuffers(),  # Use default percentage-based buffers
                ideal_range=(ideal_min, ideal_max),
                description=description
            ))

        return thresholds

    def _learn_distance_reference(
        self,
        reps: List[RepSegment]
    ) -> Optional[DistanceReference]:
        """
        Learn reference Z values for perspective correction.

        These values represent the distance from the camera during training.
        At runtime, if the user is further away (higher Z), angles will appear
        smaller and movements faster due to perspective distortion. The iOS app
        can use these reference values to apply correction factors.
        """
        torso_z_values = []
        shoulder_z_values = []
        hip_z_values = []
        elbow_z_values = []
        knee_z_values = []

        for rep in reps:
            for fm in rep.frame_metrics:
                if fm.torso_z is not None:
                    torso_z_values.append(fm.torso_z)
                if fm.shoulder_z is not None:
                    shoulder_z_values.append(fm.shoulder_z)
                if fm.hip_z is not None:
                    hip_z_values.append(fm.hip_z)
                if fm.elbow_z is not None:
                    elbow_z_values.append(fm.elbow_z)
                if fm.knee_z is not None:
                    knee_z_values.append(fm.knee_z)

        # Need at least some Z data to create reference
        if not torso_z_values:
            return None

        return DistanceReference(
            torso_z_mean=float(np.mean(torso_z_values)),
            torso_z_range=(float(np.min(torso_z_values)), float(np.max(torso_z_values))),
            shoulder_z_mean=float(np.mean(shoulder_z_values)) if shoulder_z_values else 0.0,
            hip_z_mean=float(np.mean(hip_z_values)) if hip_z_values else 0.0,
            elbow_z_mean=float(np.mean(elbow_z_values)) if elbow_z_values else None,
            knee_z_mean=float(np.mean(knee_z_values)) if knee_z_values else None
        )

    def _get_metric_description(self, metric_name: str) -> str:
        """Get human-readable description for a metric."""
        descriptions = {
            'torso_angle': 'Forward/backward lean angle',
            'shoulder_symmetry': 'Left/right shoulder level difference',
            'hip_symmetry': 'Left/right hip level difference',
            'hip_asymmetry_normalized': 'Hip rotation/tilt (normalized by hip width, for shoulder tap)',
            'elbow_angle_symmetry': 'Left/right elbow angle difference',
            'knee_angle_symmetry': 'Left/right knee angle difference',
            'hand_width_ratio': 'Hand spacing relative to shoulders (front view)',
            'hand_spread_ratio': 'Hand spacing relative to torso (side view)',
            'stance_width_ratio': 'Foot spacing relative to hips',
            'elbow_flare_ratio': 'Elbow width relative to hands',
            'stagger_offset_ratio': 'Front-to-back hand offset relative to torso',
            'forearm_to_upper_arm_ratio_left': 'Left forearm/upper arm length ratio (perspective-based elbow position)',
            'forearm_to_upper_arm_ratio_right': 'Right forearm/upper arm length ratio (perspective-based elbow position)',
            'avg_forearm_to_upper_arm_ratio': 'Avg forearm/upper arm ratio (>1.0 = elbows back, ~1.0 = elbows flare)',
        }
        return descriptions.get(metric_name, metric_name.replace('_', ' ').title())

    def get_thresholds_summary(self, thresholds: LearnedThresholds) -> str:
        """Generate human-readable summary of learned thresholds."""
        lines = []

        lines.append(f"Exercise: {thresholds.exercise_name} ({thresholds.exercise_id})")
        lines.append(f"View: {thresholds.camera_view}, Mode: {thresholds.limb_mode}")
        lines.append(f"Primary signal: {thresholds.primary_signal} ({thresholds.signal_direction})")
        lines.append(f"Reps analyzed: {thresholds.reps_analyzed}")
        lines.append(f"Avg confidence: {thresholds.avg_confidence:.1%}")

        if thresholds.depth:
            lines.append("\nDEPTH/ROM THRESHOLDS:")
            d = thresholds.depth
            lines.append(f"  signal: {d.signal}")
            lines.append(f"  observed bottom range: {d.observed_range[0]:.3f} - {d.observed_range[1]:.3f}")
            lines.append(f"  observed top range: {d.observed_top_range[0]:.3f} - {d.observed_top_range[1]:.3f}")
            lines.append(f"  → depth_good: {d.depth_good:.3f}")
            lines.append(f"  → depth_acceptable: {d.depth_acceptable:.3f}")
            lines.append(f"  → depth_minimum: {d.depth_minimum:.3f}")
            lines.append(f"  → top_threshold: {d.top_threshold:.3f} (avoids enforcing lockout)")
            lines.append(f"  → rom_minimum: {d.rom_minimum:.3f}")

        if thresholds.tempo:
            lines.append("\nTEMPO THRESHOLDS:")
            t = thresholds.tempo
            b = t.buffers
            lines.append(f"  OBSERVED (from video):")
            lines.append(f"    eccentric range: {t.observed_eccentric_range[0]:.0f}ms - {t.observed_eccentric_range[1]:.0f}ms")
            lines.append(f"    eccentric p10/p90: {t.observed_eccentric_min_ms:.0f}ms - {t.observed_eccentric_max_ms:.0f}ms")
            lines.append(f"    concentric range: {t.observed_concentric_range[0]:.0f}ms - {t.observed_concentric_range[1]:.0f}ms")
            lines.append(f"    concentric p10/p90: {t.observed_concentric_min_ms:.0f}ms - {t.observed_concentric_max_ms:.0f}ms")
            lines.append(f"    total p10/p90: {t.observed_total_min_ms:.0f}ms - {t.observed_total_max_ms:.0f}ms")
            lines.append(f"  FAVORABLE BUFFERS:")
            lines.append(f"    eccentric_min: {b.eccentric_min_buffer_ms:+d}ms (more lenient for 'too fast')")
            lines.append(f"    eccentric_max: {b.eccentric_max_buffer_ms:+d}ms (more lenient for 'too slow')")
            lines.append(f"    concentric_min: {b.concentric_min_buffer_ms:+d}ms")
            lines.append(f"    concentric_max: {b.concentric_max_buffer_ms:+d}ms")
            lines.append(f"    total_min: {b.total_min_buffer_ms:+d}ms")
            lines.append(f"    total_max: {b.total_max_buffer_ms:+d}ms")
            lines.append(f"  FINAL THRESHOLDS (observed + buffers):")
            lines.append(f"    → eccentric_min: {t.eccentric_min_ms:.0f}ms")
            lines.append(f"    → eccentric_max: {t.eccentric_max_ms:.0f}ms")
            lines.append(f"    → concentric_min: {t.concentric_min_ms:.0f}ms")
            lines.append(f"    → concentric_max: {t.concentric_max_ms:.0f}ms")
            lines.append(f"    → total_min: {t.total_min_ms:.0f}ms")
            lines.append(f"    → total_max: {t.total_max_ms:.0f}ms")
            lines.append(f"  IDEAL (for praise): {t.ideal_eccentric_min_ms}ms - {t.ideal_eccentric_max_ms}ms eccentric")

        if thresholds.velocity:
            lines.append("\nVELOCITY THRESHOLDS:")
            v = thresholds.velocity
            lines.append(f"  → v_enter: {v.v_enter:.4f}")
            lines.append(f"  → v_exit: {v.v_exit:.4f}")
            lines.append(f"  → depth_ema: {v.depth_ema:.2f}")
            lines.append(f"  → velocity_ema: {v.velocity_ema:.2f}")

        if thresholds.stability:
            lines.append("\nSTABILITY THRESHOLDS:")
            for s in thresholds.stability:
                lines.append(f"  {s.metric}:")
                lines.append(f"    baseline: {s.baseline:.3f}")
                lines.append(f"    max_deviation: {s.max_deviation:.3f}")
                lines.append(f"    description: {s.description}")

        if thresholds.ratios:
            lines.append("\nRATIO THRESHOLDS:")
            for r in thresholds.ratios:
                lines.append(f"  {r.metric}:")
                lines.append(f"    baseline: {r.baseline:.3f}")
                lines.append(f"    acceptable_range: [{r.acceptable_range[0]:.3f}, {r.acceptable_range[1]:.3f}]")
                lines.append(f"    description: {r.description}")

        if thresholds.distance_reference:
            lines.append("\nDISTANCE REFERENCE (for perspective correction):")
            dr = thresholds.distance_reference
            lines.append(f"  torso_z_mean: {dr.torso_z_mean:.4f}")
            lines.append(f"  torso_z_range: [{dr.torso_z_range[0]:.4f}, {dr.torso_z_range[1]:.4f}]")
            lines.append(f"  shoulder_z_mean: {dr.shoulder_z_mean:.4f}")
            lines.append(f"  hip_z_mean: {dr.hip_z_mean:.4f}")
            if dr.elbow_z_mean is not None:
                lines.append(f"  elbow_z_mean: {dr.elbow_z_mean:.4f}")
            if dr.knee_z_mean is not None:
                lines.append(f"  knee_z_mean: {dr.knee_z_mean:.4f}")

        return "\n".join(lines)
