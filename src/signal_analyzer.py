"""
Signal Analyzer - The smart discovery component.

Analyzes all computed metrics to discover:
- Primary movement signals (what's driving the rep cycle)
- Stability signals (what should remain constant)
- Correlation patterns (bilateral vs alternating movement)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import pearsonr

from .metric_calculator import FrameMetrics, MetricCalculator


@dataclass
class SignalCharacteristics:
    """Characteristics of a single metric signal."""
    name: str
    values: np.ndarray
    timestamps: np.ndarray

    # Variance analysis
    variance: float = 0.0
    std_dev: float = 0.0
    coefficient_of_variation: float = 0.0  # std/mean - normalized variance

    # Cyclical analysis
    is_cyclical: bool = False
    num_cycles: int = 0
    cycle_regularity: float = 0.0  # How regular are the cycles (0-1)
    dominant_frequency: float = 0.0  # Hz

    # Range
    min_value: float = 0.0
    max_value: float = 0.0
    range_value: float = 0.0
    mean_value: float = 0.0

    # Classification
    signal_type: str = "unknown"  # "primary", "stability", "ratio", "symmetry"
    confidence: float = 0.0  # How confident are we in this classification


@dataclass
class SignalCorrelation:
    """Correlation between two signals."""
    signal_a: str
    signal_b: str
    correlation: float  # -1 to 1
    is_synchronized: bool  # Move together (correlation > 0.7)
    is_alternating: bool  # Move opposite (correlation < -0.7)


@dataclass
class SignalAnalysisResult:
    """Complete analysis of all signals."""
    # Classified signals
    primary_signals: List[SignalCharacteristics] = field(default_factory=list)
    stability_signals: List[SignalCharacteristics] = field(default_factory=list)
    ratio_signals: List[SignalCharacteristics] = field(default_factory=list)
    symmetry_signals: List[SignalCharacteristics] = field(default_factory=list)

    # All signals for reference
    all_signals: Dict[str, SignalCharacteristics] = field(default_factory=dict)

    # Correlations
    correlations: List[SignalCorrelation] = field(default_factory=list)

    # Recommended primary signal
    recommended_primary: Optional[str] = None
    recommendation_confidence: float = 0.0


class SignalAnalyzer:
    """
    Analyzes metric signals to discover movement patterns.
    """

    # Metrics that are inherently about ratios
    RATIO_METRICS = {
        'hand_width_ratio', 'hand_spread_ratio', 'stance_width_ratio', 'elbow_flare_ratio', 'hip_elevation_ratio', 'stagger_offset_ratio',
        'forearm_to_upper_arm_ratio_left', 'forearm_to_upper_arm_ratio_right', 'avg_forearm_to_upper_arm_ratio'
    }

    # Metrics that measure symmetry
    SYMMETRY_METRICS = {
        'shoulder_symmetry', 'hip_symmetry',
        'elbow_angle_symmetry', 'knee_angle_symmetry'
    }

    # Metrics that are good primary signal candidates
    PRIMARY_CANDIDATES = {
        'avg_elbow_angle', 'avg_knee_angle', 'avg_hip_angle',
        'left_elbow_angle', 'right_elbow_angle',
        'left_knee_angle', 'right_knee_angle',
        'elbow_depth', 'knee_depth',
        'shoulder_height', 'hip_height', 'nose_height'
    }

    # Preferred primary signals for bilateral exercises (average of both sides)
    # These should be prioritized over single-sided measurements
    BILATERAL_PREFERRED = {
        'avg_elbow_angle',  # For push-ups, dips, etc.
        'avg_knee_angle',   # For squats, lunges, etc.
        'avg_hip_angle',    # For hip hinge movements
        'elbow_depth',      # Normalized elbow depth
        'knee_depth',       # Normalized knee depth
    }

    # Exercise-specific signal preferences
    # For some exercises, we should prefer certain signals regardless of variance
    # Key = exercise type pattern (matched via "in" check), Value = list of preferred signals in order
    EXERCISE_SIGNAL_PREFERENCES = {
        'pushup': ['elbow_depth', 'avg_elbow_angle', 'left_elbow_angle', 'right_elbow_angle'],
        'squat': ['knee_depth', 'avg_knee_angle', 'left_knee_angle', 'right_knee_angle'],
        'lunge': ['knee_depth', 'avg_knee_angle'],
    }

    # Exercise-specific stability metrics
    # These metrics should be treated as stability signals (should remain constant) for specific exercises
    # even if they show variance. For example, knee angle should stay straight during pike push-ups.
    EXERCISE_STABILITY_METRICS = {
        'pike': ['avg_knee_angle', 'knee_depth', 'left_knee_angle', 'right_knee_angle'],
        # For pike push-ups, legs should stay mostly straight
    }

    # Minimum cycles to consider a signal as primary
    MIN_CYCLES_FOR_PRIMARY = 3

    # Thresholds for classification
    CYCLICAL_THRESHOLD = 0.3  # Minimum regularity to be considered cyclical
    STABILITY_CV_THRESHOLD = 0.15  # Max coefficient of variation for stability
    PRIMARY_CV_THRESHOLD = 0.20  # Min coefficient of variation for primary

    def __init__(self, expected_rep_duration: float = 2.0):
        """
        Args:
            expected_rep_duration: Expected duration of one rep in seconds
                                   Used to guide frequency analysis
        """
        self.expected_rep_duration = expected_rep_duration

    def analyze_signal(
        self,
        name: str,
        values: np.ndarray,
        timestamps: np.ndarray
    ) -> SignalCharacteristics:
        """Analyze a single signal's characteristics."""
        if len(values) < 10:
            return SignalCharacteristics(
                name=name,
                values=values,
                timestamps=timestamps,
                signal_type="insufficient_data"
            )

        # Basic statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        var_val = np.var(values)
        min_val = np.min(values)
        max_val = np.max(values)
        range_val = max_val - min_val

        # Coefficient of variation (normalized variance)
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-6 else 0

        # Cyclical analysis using FFT
        is_cyclical, num_cycles, regularity, dom_freq = self._analyze_cyclical_pattern(
            values, timestamps
        )

        char = SignalCharacteristics(
            name=name,
            values=values,
            timestamps=timestamps,
            variance=var_val,
            std_dev=std_val,
            coefficient_of_variation=cv,
            is_cyclical=is_cyclical,
            num_cycles=num_cycles,
            cycle_regularity=regularity,
            dominant_frequency=dom_freq,
            min_value=min_val,
            max_value=max_val,
            range_value=range_val,
            mean_value=mean_val
        )

        # Classify signal type
        char.signal_type, char.confidence = self._classify_signal(name, char)

        return char

    def _analyze_cyclical_pattern(
        self,
        values: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[bool, int, float, float]:
        """
        Analyze if signal has cyclical pattern.

        Returns:
            Tuple of (is_cyclical, num_cycles, regularity, dominant_frequency)
        """
        if len(values) < 20:
            return False, 0, 0.0, 0.0

        # Detrend the signal
        detrended = scipy_signal.detrend(values)

        # Estimate sample rate
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return False, 0, 0.0, 0.0
        sample_rate = len(values) / duration

        # Find peaks and valleys
        # Use prominence to filter noise
        prominence = np.std(detrended) * 0.3
        peaks, peak_props = scipy_signal.find_peaks(
            detrended,
            prominence=prominence,
            distance=max(1, int(sample_rate * 0.3))  # Min 0.3s between peaks
        )
        valleys, valley_props = scipy_signal.find_peaks(
            -detrended,
            prominence=prominence,
            distance=max(1, int(sample_rate * 0.3))
        )

        # Count cycles (each peak-valley pair is roughly a cycle)
        num_cycles = min(len(peaks), len(valleys))

        if num_cycles < 2:
            return False, num_cycles, 0.0, 0.0

        # Calculate cycle regularity from peak-to-peak intervals
        if len(peaks) >= 2:
            peak_intervals = np.diff(timestamps[peaks])
            interval_cv = np.std(peak_intervals) / np.mean(peak_intervals) if np.mean(peak_intervals) > 0 else 1
            regularity = max(0, 1 - interval_cv)  # Higher = more regular
        else:
            regularity = 0.0

        # Dominant frequency from FFT
        fft = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(len(detrended), d=1/sample_rate)
        dominant_idx = np.argmax(np.abs(fft[1:])) + 1  # Skip DC component
        dominant_freq = freqs[dominant_idx] if dominant_idx < len(freqs) else 0

        is_cyclical = (
            num_cycles >= self.MIN_CYCLES_FOR_PRIMARY and
            regularity >= self.CYCLICAL_THRESHOLD
        )

        return is_cyclical, num_cycles, regularity, dominant_freq

    def _classify_signal(
        self,
        name: str,
        char: SignalCharacteristics
    ) -> Tuple[str, float]:
        """
        Classify signal type based on characteristics.

        Returns:
            Tuple of (signal_type, confidence)
        """
        # Check if it's a known ratio or symmetry metric
        if name in self.RATIO_METRICS:
            return "ratio", 0.95

        if name in self.SYMMETRY_METRICS:
            return "symmetry", 0.95

        # Primary signal: high variance, cyclical, regular
        if (name in self.PRIMARY_CANDIDATES and
            char.is_cyclical and
            char.coefficient_of_variation >= self.PRIMARY_CV_THRESHOLD):

            confidence = min(1.0, char.cycle_regularity * 1.2)
            return "primary", confidence

        # Stability signal: low variance, not cyclical
        if char.coefficient_of_variation < self.STABILITY_CV_THRESHOLD:
            confidence = 1.0 - (char.coefficient_of_variation / self.STABILITY_CV_THRESHOLD)
            return "stability", confidence

        # Could be either - moderate variance
        if char.is_cyclical:
            return "primary_candidate", char.cycle_regularity
        else:
            return "stability_candidate", 0.5

    def analyze_all_signals(
        self,
        frame_metrics: List[FrameMetrics],
        metric_calculator: MetricCalculator,
        limb_mode: str = "bilateral",
        exercise_id: str = ""
    ) -> SignalAnalysisResult:
        """
        Analyze all metrics and classify them.

        Args:
            frame_metrics: List of computed frame metrics
            metric_calculator: MetricCalculator instance for getting metric names
            limb_mode: "bilateral", "unilateral_alternating", or "unilateral_single"
            exercise_id: Exercise identifier (e.g., "pike_pushup") for exercise-specific
                        signal preferences. Push-ups will prefer elbow signals, squats knee signals.

        Returns:
            Complete signal analysis result
        """
        result = SignalAnalysisResult()

        metric_names = metric_calculator.get_all_metric_names()

        for name in metric_names:
            timestamps, values = metric_calculator.get_metric_time_series(
                frame_metrics, name
            )

            if len(values) < 10:
                continue

            char = self.analyze_signal(name, values, timestamps)
            result.all_signals[name] = char

            # Check if this metric should be forced to stability for this exercise
            # (e.g., knee angle should stay straight for pike push-ups)
            force_stability = self._is_forced_stability_metric(name, exercise_id)

            # Classify into categories
            if force_stability:
                # Force to stability regardless of detected signal type
                char.signal_type = "stability"
                char.confidence = 0.9  # High confidence since we're explicitly forcing it
                result.stability_signals.append(char)
            elif char.signal_type == "primary":
                result.primary_signals.append(char)
            elif char.signal_type == "primary_candidate":
                result.primary_signals.append(char)
            elif char.signal_type == "stability":
                result.stability_signals.append(char)
            elif char.signal_type == "stability_candidate":
                result.stability_signals.append(char)
            elif char.signal_type == "ratio":
                result.ratio_signals.append(char)
            elif char.signal_type == "symmetry":
                result.symmetry_signals.append(char)

        # Determine exercise-specific signal preferences
        exercise_preferred_signals = self._get_exercise_preferred_signals(exercise_id)

        # Sort primary signals with exercise preference > bilateral preference > confidence
        if limb_mode == "bilateral":
            def sort_key(sig):
                # Exercise-specific preference gets highest priority
                if exercise_preferred_signals:
                    try:
                        exercise_rank = exercise_preferred_signals.index(sig.name)
                        # Higher rank = lower index = higher priority
                        exercise_bonus = 100.0 - exercise_rank
                    except ValueError:
                        exercise_bonus = 0.0
                else:
                    exercise_bonus = 0.0

                # Bilateral preference gets second priority
                is_bilateral_preferred = sig.name in self.BILATERAL_PREFERRED
                bilateral_bonus = 2.0 if is_bilateral_preferred else 1.0

                return (exercise_bonus, bilateral_bonus * sig.confidence, sig.cycle_regularity)

            result.primary_signals.sort(key=sort_key, reverse=True)
        else:
            # For unilateral, still apply exercise preference
            def sort_key(sig):
                if exercise_preferred_signals:
                    try:
                        exercise_rank = exercise_preferred_signals.index(sig.name)
                        exercise_bonus = 100.0 - exercise_rank
                    except ValueError:
                        exercise_bonus = 0.0
                else:
                    exercise_bonus = 0.0
                return (exercise_bonus, sig.confidence, sig.cycle_regularity)

            result.primary_signals.sort(key=sort_key, reverse=True)

        # Sort stability signals by how stable they are (lower CV = more stable)
        result.stability_signals.sort(
            key=lambda x: x.coefficient_of_variation
        )

        # Calculate correlations between primary candidates
        result.correlations = self._calculate_correlations(result.primary_signals)

        # Recommend primary signal
        if result.primary_signals:
            best = result.primary_signals[0]
            result.recommended_primary = best.name
            result.recommendation_confidence = best.confidence

        return result

    def _is_forced_stability_metric(self, metric_name: str, exercise_id: str) -> bool:
        """
        Check if a metric should be forced to stability classification for this exercise.

        For example, knee angle should stay mostly constant (straight legs) during pike push-ups,
        so we want to track it as a stability metric even if it shows some variance.

        Args:
            metric_name: Name of the metric to check
            exercise_id: Exercise identifier (e.g., "pike_pushup")

        Returns:
            True if this metric should be forced to stability for this exercise
        """
        if not exercise_id:
            return False

        exercise_id_lower = exercise_id.lower()

        for exercise_pattern, stability_metrics in self.EXERCISE_STABILITY_METRICS.items():
            if exercise_pattern in exercise_id_lower:
                return metric_name in stability_metrics

        return False

    def _get_exercise_preferred_signals(self, exercise_id: str) -> List[str]:
        """
        Get preferred signals for a specific exercise type.

        For push-ups, we always want elbow-based depth regardless of camera view.
        For squats, we always want knee-based depth.

        Args:
            exercise_id: Exercise identifier (e.g., "pike_pushup", "standard_pushup")

        Returns:
            List of preferred signal names in priority order, or empty list if no preference
        """
        if not exercise_id:
            return []

        exercise_id_lower = exercise_id.lower()

        for exercise_pattern, preferred_signals in self.EXERCISE_SIGNAL_PREFERENCES.items():
            if exercise_pattern in exercise_id_lower:
                return preferred_signals

        return []

    def _calculate_correlations(
        self,
        signals: List[SignalCharacteristics]
    ) -> List[SignalCorrelation]:
        """Calculate correlations between signals."""
        correlations = []

        for i, sig_a in enumerate(signals):
            for sig_b in signals[i+1:]:
                # Need same length for correlation
                min_len = min(len(sig_a.values), len(sig_b.values))
                if min_len < 10:
                    continue

                corr, _ = pearsonr(
                    sig_a.values[:min_len],
                    sig_b.values[:min_len]
                )

                correlations.append(SignalCorrelation(
                    signal_a=sig_a.name,
                    signal_b=sig_b.name,
                    correlation=corr,
                    is_synchronized=corr > 0.7,
                    is_alternating=corr < -0.7
                ))

        return correlations

    def detect_limb_mode(
        self,
        result: SignalAnalysisResult
    ) -> Tuple[str, float]:
        """
        Detect if exercise is bilateral, unilateral, or alternating.

        Returns:
            Tuple of (limb_mode, confidence)
        """
        # Look for left/right correlations
        left_right_pairs = [
            ('left_elbow_angle', 'right_elbow_angle'),
            ('left_knee_angle', 'right_knee_angle'),
            ('left_hip_angle', 'right_hip_angle'),
        ]

        sync_scores = []
        alt_scores = []

        for left, right in left_right_pairs:
            if left in result.all_signals and right in result.all_signals:
                left_sig = result.all_signals[left]
                right_sig = result.all_signals[right]

                min_len = min(len(left_sig.values), len(right_sig.values))
                if min_len >= 10:
                    corr, _ = pearsonr(
                        left_sig.values[:min_len],
                        right_sig.values[:min_len]
                    )

                    if corr > 0.5:
                        sync_scores.append(corr)
                    elif corr < -0.5:
                        alt_scores.append(abs(corr))

        if sync_scores and np.mean(sync_scores) > 0.7:
            return "bilateral", np.mean(sync_scores)
        elif alt_scores and np.mean(alt_scores) > 0.7:
            return "unilateral_alternating", np.mean(alt_scores)
        else:
            return "unknown", 0.5

    def get_analysis_summary(self, result: SignalAnalysisResult) -> str:
        """Generate human-readable summary of analysis."""
        lines = []

        lines.append("PRIMARY SIGNALS (high cyclic variance):")
        for sig in result.primary_signals[:5]:
            marker = "★ RECOMMENDED" if sig.name == result.recommended_primary else ""
            lines.append(
                f"  → {sig.name}: variance={sig.coefficient_of_variation:.3f}  "
                f"cycles={sig.num_cycles}  regularity={sig.cycle_regularity:.2f}  {marker}"
            )

        lines.append("\nSTABILITY SIGNALS (low variance - form metrics):")
        for sig in result.stability_signals[:5]:
            lines.append(
                f"  → {sig.name}: CV={sig.coefficient_of_variation:.3f}  "
                f"mean={sig.mean_value:.2f}  range={sig.range_value:.2f}"
            )

        lines.append("\nRATIO SIGNALS:")
        for sig in result.ratio_signals:
            lines.append(
                f"  → {sig.name}: mean={sig.mean_value:.3f}  "
                f"range=[{sig.min_value:.3f}, {sig.max_value:.3f}]"
            )

        lines.append("\nSYMMETRY SIGNALS:")
        for sig in result.symmetry_signals:
            lines.append(
                f"  → {sig.name}: mean={sig.mean_value:.4f}  "
                f"max={sig.max_value:.4f}"
            )

        return "\n".join(lines)
