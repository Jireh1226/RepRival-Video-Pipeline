"""
JSON Exporter - Export learned thresholds and feedback to JSON files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .threshold_learner import (
    LearnedThresholds,
    DepthThresholds,
    TempoThresholds,
    TempoBuffers,
    VelocityThresholds,
    StabilityThreshold,
    RatioThreshold,
    RatioBuffers,
    DistanceReference
)
from .feedback_generator import GeneratedFeedback, FeedbackGenerator


class JSONExporter:
    """
    Export learned thresholds to JSON format.
    """

    VERSION = "1.0.0"

    def __init__(self, output_dir: str = "./learned_thresholds"):
        """
        Args:
            output_dir: Directory to save JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        thresholds: LearnedThresholds,
        feedback: GeneratedFeedback,
        filename: Optional[str] = None
    ) -> Path:
        """
        Export thresholds and feedback to JSON.

        Args:
            thresholds: Learned thresholds
            feedback: Generated feedback
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to the saved JSON file
        """
        # Build the complete export structure
        export_data = self._build_export_data(thresholds, feedback)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"{thresholds.exercise_id}_{thresholds.camera_view}_{timestamp}.json"

        filepath = self.output_dir / filename

        # Write JSON with nice formatting
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        return filepath

    def _build_export_data(
        self,
        thresholds: LearnedThresholds,
        feedback: GeneratedFeedback
    ) -> Dict[str, Any]:
        """Build the complete export dictionary."""
        feedback_gen = FeedbackGenerator()

        return {
            "exercise_id": thresholds.exercise_id,
            "exercise_name": thresholds.exercise_name,
            "camera_view": thresholds.camera_view,
            "limb_mode": thresholds.limb_mode,

            "primary_signal": {
                "metric": thresholds.primary_signal,
                "direction": thresholds.signal_direction,
                "description": self._get_signal_description(
                    thresholds.primary_signal,
                    thresholds.signal_direction
                )
            },

            "depth_thresholds": self._export_depth(thresholds.depth) if thresholds.depth else None,
            "tempo_thresholds": self._export_tempo(thresholds.tempo) if thresholds.tempo else None,
            "velocity_thresholds": self._export_velocity(thresholds.velocity) if thresholds.velocity else None,
            "stability_thresholds": self._export_stability(thresholds.stability, feedback),
            "ratio_thresholds": self._export_ratios(thresholds.ratios, feedback),

            # Distance reference for perspective correction
            "distance_reference": self._export_distance_reference(thresholds.distance_reference),

            # Confidence gating thresholds (matching iOS config)
            "confidence_thresholds": {
                "min_arms_confidence": 0.4,
                "min_legs_confidence": 0.4,
                "min_elbow_confidence": 0.5,
                "min_knee_confidence": 0.5,
                "min_frame_confidence": 0.5,
                "description": "Minimum landmark visibility to trust frame data"
            },

            "feedback": feedback_gen.feedback_to_dict(feedback),

            "metadata": {
                "learned_from_video": thresholds.video_source,
                "reps_analyzed": thresholds.reps_analyzed,
                "avg_confidence": round(thresholds.avg_confidence, 3),
                "created_at": datetime.now().isoformat(),
                "pipeline_version": self.VERSION
            }
        }

    def _get_signal_description(self, signal: str, direction: str) -> str:
        """Get description of how the primary signal works."""
        desc = {
            'avg_elbow_angle': 'Average elbow flexion angle',
            'avg_knee_angle': 'Average knee flexion angle',
            'avg_hip_angle': 'Average hip flexion angle',
            'elbow_depth': 'Normalized elbow bend depth (0=extended, 1=fully bent)',
            'knee_depth': 'Normalized knee bend depth (0=extended, 1=fully bent)',
            'shoulder_height': 'Vertical position of shoulders',
            'hip_height': 'Vertical position of hips',
        }

        base = desc.get(signal, signal.replace('_', ' ').title())

        if direction == 'up':
            return f"{base} - increases at bottom of movement"
        else:
            return f"{base} - decreases at bottom of movement"

    def _angle_to_depth(self, angle: float) -> float:
        """
        Convert elbow/knee angle to normalized depth (matching iOS).

        iOS formula: depth = (170 - angle) / 80
        - 170° = 0.0 (top, arms nearly extended)
        - 90° = 1.0 (bottom, arms bent 90°)
        """
        return max(0.0, min(1.0, (170 - angle) / 80))

    def _export_depth(self, depth: DepthThresholds) -> Dict[str, Any]:
        """Export depth thresholds."""
        # Check if this is an angle-based signal
        is_angle_signal = 'angle' in depth.signal.lower()

        result = {
            "signal": depth.signal,
            "depth_good": round(depth.depth_good, 4),
            "depth_acceptable": round(depth.depth_acceptable, 4),
            "depth_minimum": round(depth.depth_minimum, 4),
            "rom_minimum": round(depth.rom_minimum, 4),
            "top_threshold": round(depth.top_threshold, 4),
            "observed_range": [
                round(depth.observed_range[0], 4),
                round(depth.observed_range[1], 4)
            ],
            "observed_top_range": [
                round(depth.observed_top_range[0], 4),
                round(depth.observed_top_range[1], 4)
            ],
            "description": "top_threshold defines 'returned to top' position (avoids enforcing joint lockout). ROM is measured from top_threshold to bottom depth."
        }

        # Add iOS-compatible normalized depth values for angle signals
        if is_angle_signal:
            result["normalized_depth"] = {
                "description": "iOS-compatible depth values: depth = (170 - angle) / 80. 0.0 = top (170°), 1.0 = bottom (90°)",
                "depth_good": round(self._angle_to_depth(depth.depth_good), 4),
                "depth_acceptable": round(self._angle_to_depth(depth.depth_acceptable), 4),
                "depth_minimum": round(self._angle_to_depth(depth.depth_minimum), 4),
                "top_threshold": round(self._angle_to_depth(depth.top_threshold), 4),
                "observed_bottom_depth": [
                    round(self._angle_to_depth(depth.observed_range[1]), 4),  # Min angle = max depth
                    round(self._angle_to_depth(depth.observed_range[0]), 4)   # Max angle = min depth
                ],
                "observed_top_depth": [
                    round(self._angle_to_depth(depth.observed_top_range[1]), 4),
                    round(self._angle_to_depth(depth.observed_top_range[0]), 4)
                ]
            }

        return result

    def _export_tempo(self, tempo: TempoThresholds) -> Dict[str, Any]:
        """Export tempo thresholds with observed, buffers, and final values."""
        b = tempo.buffers
        return {
            # Observed thresholds (raw from video analysis)
            "observed": {
                "eccentric_min_ms": round(tempo.observed_eccentric_min_ms),
                "eccentric_max_ms": round(tempo.observed_eccentric_max_ms),
                "concentric_min_ms": round(tempo.observed_concentric_min_ms),
                "concentric_max_ms": round(tempo.observed_concentric_max_ms),
                "total_min_ms": round(tempo.observed_total_min_ms),
                "total_max_ms": round(tempo.observed_total_max_ms),
                "eccentric_range_ms": [
                    round(tempo.observed_eccentric_range[0]),
                    round(tempo.observed_eccentric_range[1])
                ],
                "concentric_range_ms": [
                    round(tempo.observed_concentric_range[0]),
                    round(tempo.observed_concentric_range[1])
                ],
                "description": "Raw threshold values extracted from video analysis (p10/p90 percentiles)"
            },

            # Favorable buffers (adjustments for real-world tolerance)
            "favorable_buffers": {
                "eccentric_min_buffer_ms": b.eccentric_min_buffer_ms,
                "eccentric_max_buffer_ms": b.eccentric_max_buffer_ms,
                "concentric_min_buffer_ms": b.concentric_min_buffer_ms,
                "concentric_max_buffer_ms": b.concentric_max_buffer_ms,
                "total_min_buffer_ms": b.total_min_buffer_ms,
                "total_max_buffer_ms": b.total_max_buffer_ms,
                "description": "Buffers applied to observed thresholds. Negative on min = more lenient for 'too fast'. Positive on max = more lenient for 'too slow'."
            },

            # Final thresholds (observed + buffers) - used for enforcement
            "final": {
                "eccentric_min_ms": round(tempo.eccentric_min_ms),
                "eccentric_max_ms": round(tempo.eccentric_max_ms),
                "concentric_min_ms": round(tempo.concentric_min_ms),
                "concentric_max_ms": round(tempo.concentric_max_ms),
                "total_min_ms": round(tempo.total_min_ms),
                "total_max_ms": round(tempo.total_max_ms),
                "description": "Final enforcement thresholds = observed + favorable_buffers"
            },

            # Ideal range for praise
            "ideal": {
                "eccentric_min_ms": tempo.ideal_eccentric_min_ms,
                "eccentric_max_ms": tempo.ideal_eccentric_max_ms,
                "description": "Ideal tempo range for praise feedback (middle 50% of observed)"
            }
        }

    def _export_velocity(self, velocity: VelocityThresholds) -> Dict[str, Any]:
        """Export velocity thresholds."""
        return {
            "v_enter": round(velocity.v_enter, 4),
            "v_exit": round(velocity.v_exit, 4),
            "depth_ema": round(velocity.depth_ema, 2),
            "velocity_ema": round(velocity.velocity_ema, 2)
        }

    def _export_stability(
        self,
        stability: list,
        feedback: GeneratedFeedback
    ) -> Dict[str, Any]:
        """Export stability thresholds with feedback."""
        result = {}

        for s in stability:
            entry = {
                "baseline": round(s.baseline, 4),
                "max_deviation": round(s.max_deviation, 4),
                "observed_std": round(s.observed_std, 4),
                "description": s.description
            }

            # Add feedback if available
            if s.metric in feedback.stability_feedback:
                fb = feedback.stability_feedback[s.metric]
                entry["feedback"] = {
                    "cue_short": fb.cue_short,
                    "cue_detailed": fb.cue_detailed,
                    "severity": fb.severity
                }

            result[s.metric] = entry

        return result

    def _export_distance_reference(
        self,
        distance_ref: Optional[DistanceReference]
    ) -> Optional[Dict[str, Any]]:
        """Export distance reference for perspective correction."""
        if distance_ref is None:
            return None

        result = {
            "torso_z_mean": round(distance_ref.torso_z_mean, 4),
            "torso_z_range": [
                round(distance_ref.torso_z_range[0], 4),
                round(distance_ref.torso_z_range[1], 4)
            ],
            "shoulder_z_mean": round(distance_ref.shoulder_z_mean, 4),
            "hip_z_mean": round(distance_ref.hip_z_mean, 4),
            "description": "Reference Z values from training video. MediaPipe Z: negative = closer to camera, positive = further. Use to scale thresholds based on user distance."
        }

        if distance_ref.elbow_z_mean is not None:
            result["elbow_z_mean"] = round(distance_ref.elbow_z_mean, 4)

        if distance_ref.knee_z_mean is not None:
            result["knee_z_mean"] = round(distance_ref.knee_z_mean, 4)

        return result

    def _export_ratios(
        self,
        ratios: list,
        feedback: GeneratedFeedback
    ) -> Dict[str, Any]:
        """Export ratio thresholds with observed, buffers, and final values."""
        result = {}

        for r in ratios:
            b = r.buffers
            entry = {
                # Observed values (raw from video analysis)
                "observed": {
                    "baseline": round(r.baseline, 4),
                    "min": round(r.observed_min, 4),
                    "max": round(r.observed_max, 4),
                    "description": "Raw values extracted from video analysis"
                },

                # Favorable buffers (percentage-based adjustments)
                "favorable_buffers": {
                    "min_buffer_percent": b.min_buffer_percent,
                    "max_buffer_percent": b.max_buffer_percent,
                    "description": "Percentage-based buffers applied to baseline. Negative on min = allows narrower. Positive on max = allows wider."
                },

                # Final thresholds (observed + buffers) - used for enforcement
                "final": {
                    "acceptable_range": [
                        round(r.acceptable_range[0], 4),
                        round(r.acceptable_range[1], 4)
                    ],
                    "description": "Final enforcement thresholds = observed ± (baseline × buffer_percent)"
                },

                # Ideal range for praise
                "ideal": {
                    "ideal_range": [
                        round(r.ideal_range[0], 4),
                        round(r.ideal_range[1], 4)
                    ],
                    "description": "Ideal range for praise feedback (middle 50% of observed)"
                },

                "metric_description": r.description
            }

            # Add feedback if available
            if r.metric in feedback.ratio_feedback:
                fb_dict = feedback.ratio_feedback[r.metric]
                entry["feedback"] = {
                    "too_wide": {
                        "cue_short": fb_dict['too_wide'].cue_short,
                        "cue_detailed": fb_dict['too_wide'].cue_detailed,
                        "severity": fb_dict['too_wide'].severity
                    },
                    "too_narrow": {
                        "cue_short": fb_dict['too_narrow'].cue_short,
                        "cue_detailed": fb_dict['too_narrow'].cue_detailed,
                        "severity": fb_dict['too_narrow'].severity
                    }
                }

            result[r.metric] = entry

        return result

    def load(self, filepath: str) -> Dict[str, Any]:
        """Load a previously exported JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
