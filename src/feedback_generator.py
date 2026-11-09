"""
Feedback Generator - Generate potential coaching feedback from learned thresholds.

Maps discovered thresholds to actionable coaching cues.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .threshold_learner import (
    LearnedThresholds,
    DepthThresholds,
    TempoThresholds,
    StabilityThreshold,
    RatioThreshold
)


@dataclass
class FeedbackCue:
    """A single feedback cue."""
    violation_type: str  # "depth", "tempo", "stability", "ratio"
    trigger: str  # What triggers this feedback (e.g., "insufficient_depth")
    cue_short: str  # Short cue for live display (2-3 words)
    cue_detailed: str  # Detailed explanation
    severity: str  # "minor", "moderate", "major"


@dataclass
class GeneratedFeedback:
    """All generated feedback for an exercise."""
    exercise_id: str

    # Depth feedback
    depth_feedback: Dict[str, FeedbackCue] = field(default_factory=dict)

    # Tempo feedback
    tempo_feedback: Dict[str, FeedbackCue] = field(default_factory=dict)

    # Stability feedback (keyed by metric name)
    stability_feedback: Dict[str, FeedbackCue] = field(default_factory=dict)

    # Ratio feedback (keyed by metric name)
    ratio_feedback: Dict[str, FeedbackCue] = field(default_factory=dict)


class FeedbackGenerator:
    """
    Generate potential feedback cues from learned thresholds.
    """

    # Templates for different metric types
    # These provide reasonable defaults that can be customized

    STABILITY_TEMPLATES = {
        'torso_angle': {
            'cue_short': 'Back straight',
            'cue_detailed': 'Keep your spine neutral - avoid excessive forward lean',
            'severity': 'moderate'
        },
        'shoulder_symmetry': {
            'cue_short': 'Stay level',
            'cue_detailed': 'Keep your shoulders even - one side is dropping',
            'severity': 'minor'
        },
        'hip_symmetry': {
            'cue_short': 'Hips level',
            'cue_detailed': 'Your hips are tilting - maintain a level pelvis',
            'severity': 'minor'
        },
        'elbow_angle_symmetry': {
            'cue_short': 'Even arms',
            'cue_detailed': 'Both arms should bend equally - check your form',
            'severity': 'minor'
        },
        'knee_angle_symmetry': {
            'cue_short': 'Even legs',
            'cue_detailed': 'Both legs should bend equally - distribute weight evenly',
            'severity': 'minor'
        },
        # Knee angle - for exercises requiring straight legs (pike push-ups)
        'avg_knee_angle': {
            'cue_short': 'Straighten legs',
            'cue_detailed': 'Keep your legs straight - avoid bending at the knees',
            'severity': 'moderate'
        },
        'left_knee_angle': {
            'cue_short': 'Straighten left leg',
            'cue_detailed': 'Keep your left leg straight',
            'severity': 'minor'
        },
        'right_knee_angle': {
            'cue_short': 'Straighten right leg',
            'cue_detailed': 'Keep your right leg straight',
            'severity': 'minor'
        },
        'knee_depth': {
            'cue_short': 'Legs straight',
            'cue_detailed': 'Keep your legs extended - avoid bending your knees',
            'severity': 'moderate'
        },
        # Default for unknown metrics
        '_default': {
            'cue_short': 'Hold steady',
            'cue_detailed': 'Maintain stability in your {body_part}',
            'severity': 'minor'
        }
    }

    RATIO_TEMPLATES = {
        'hand_width_ratio': {
            'too_wide': {
                'cue_short': 'Hands closer',
                'cue_detailed': 'Bring your hands in slightly',
                'severity': 'minor'
            },
            'too_narrow': {
                'cue_short': 'Hands wider',
                'cue_detailed': 'Widen your hand placement',
                'severity': 'minor'
            }
        },
        'hand_spread_ratio': {
            'too_wide': {
                'cue_short': 'Hands closer',
                'cue_detailed': 'Your hands are too far apart - bring them closer',
                'severity': 'minor'
            },
            'too_narrow': {
                'cue_short': 'Hands wider',
                'cue_detailed': 'Spread your hands further apart',
                'severity': 'minor'
            }
        },
        'stance_width_ratio': {
            'too_wide': {
                'cue_short': 'Feet closer',
                'cue_detailed': 'Your stance is too wide - bring feet in',
                'severity': 'minor'
            },
            'too_narrow': {
                'cue_short': 'Feet wider',
                'cue_detailed': 'Widen your stance for better stability',
                'severity': 'minor'
            }
        },
        'elbow_flare_ratio': {
            'too_wide': {
                'cue_short': 'Elbows in',
                'cue_detailed': 'Tuck your elbows closer to your body',
                'severity': 'moderate'
            },
            'too_narrow': {
                'cue_short': 'Elbows out',
                'cue_detailed': 'Allow your elbows to flare slightly',
                'severity': 'minor'
            }
        },
        'stagger_offset_ratio': {
            'too_wide': {
                'cue_short': 'Less stagger',
                'cue_detailed': 'Your hands are too far apart front-to-back',
                'severity': 'minor'
            },
            'too_narrow': {
                'cue_short': 'More stagger',
                'cue_detailed': 'Offset your hands more front-to-back',
                'severity': 'minor'
            }
        },
        '_default': {
            'too_wide': {
                'cue_short': 'Adjust position',
                'cue_detailed': 'Your {body_part} position is too wide',
                'severity': 'minor'
            },
            'too_narrow': {
                'cue_short': 'Adjust position',
                'cue_detailed': 'Your {body_part} position is too narrow',
                'severity': 'minor'
            }
        }
    }

    DEPTH_TEMPLATES = {
        'insufficient_depth': {
            'cue_short': 'Go deeper',
            'cue_detailed': 'Increase your range of motion - go deeper',
            'severity': 'major'
        },
        'insufficient_rom': {
            'cue_short': 'Full range',
            'cue_detailed': 'Complete the full range of motion',
            'severity': 'moderate'
        },
        'excessive_depth': {
            'cue_short': 'Not so deep',
            'cue_detailed': 'You may be going too deep - maintain control',
            'severity': 'minor'
        }
    }

    TEMPO_TEMPLATES = {
        'eccentric_too_fast': {
            'cue_short': 'Slow down',
            'cue_detailed': 'Control the lowering phase - don\'t drop',
            'severity': 'moderate'
        },
        'eccentric_too_slow': {
            'cue_short': 'Keep moving',
            'cue_detailed': 'Maintain smooth continuous movement down',
            'severity': 'minor'
        },
        'concentric_too_fast': {
            'cue_short': 'Control it',
            'cue_detailed': 'Don\'t rush - maintain control throughout',
            'severity': 'minor'
        },
        'concentric_too_slow': {
            'cue_short': 'Drive up',
            'cue_detailed': 'Push through - you\'re stalling',
            'severity': 'minor'
        },
        'rep_too_fast': {
            'cue_short': 'Slow down',
            'cue_detailed': 'Take your time - quality over speed',
            'severity': 'moderate'
        },
        'rep_too_slow': {
            'cue_short': 'Pick up pace',
            'cue_detailed': 'Maintain a steady rhythm',
            'severity': 'minor'
        }
    }

    def __init__(self):
        pass

    def generate_feedback(self, thresholds: LearnedThresholds) -> GeneratedFeedback:
        """
        Generate all potential feedback from learned thresholds.

        Args:
            thresholds: Learned thresholds from the pipeline

        Returns:
            GeneratedFeedback with all cues
        """
        feedback = GeneratedFeedback(exercise_id=thresholds.exercise_id)

        # Generate depth feedback
        if thresholds.depth:
            feedback.depth_feedback = self._generate_depth_feedback(thresholds.depth)

        # Generate tempo feedback
        if thresholds.tempo:
            feedback.tempo_feedback = self._generate_tempo_feedback(thresholds.tempo)

        # Generate stability feedback
        for stability in thresholds.stability:
            cue = self._generate_stability_feedback(stability)
            if cue:
                feedback.stability_feedback[stability.metric] = cue

        # Generate ratio feedback
        for ratio in thresholds.ratios:
            cues = self._generate_ratio_feedback(ratio)
            if cues:
                feedback.ratio_feedback[ratio.metric] = cues

        return feedback

    def _generate_depth_feedback(self, depth: DepthThresholds) -> Dict[str, FeedbackCue]:
        """Generate depth-related feedback cues."""
        cues = {}

        for key, template in self.DEPTH_TEMPLATES.items():
            cues[key] = FeedbackCue(
                violation_type='depth',
                trigger=key,
                cue_short=template['cue_short'],
                cue_detailed=template['cue_detailed'],
                severity=template['severity']
            )

        return cues

    def _generate_tempo_feedback(self, tempo: TempoThresholds) -> Dict[str, FeedbackCue]:
        """Generate tempo-related feedback cues."""
        cues = {}

        for key, template in self.TEMPO_TEMPLATES.items():
            cues[key] = FeedbackCue(
                violation_type='tempo',
                trigger=key,
                cue_short=template['cue_short'],
                cue_detailed=template['cue_detailed'],
                severity=template['severity']
            )

        return cues

    def _generate_stability_feedback(self, stability: StabilityThreshold) -> FeedbackCue:
        """Generate feedback for a stability metric."""
        template = self.STABILITY_TEMPLATES.get(
            stability.metric,
            self.STABILITY_TEMPLATES['_default']
        )

        # Fill in body part for default template
        body_part = stability.metric.replace('_', ' ').replace('angle', '').replace('symmetry', '').strip()

        cue_detailed = template['cue_detailed']
        if '{body_part}' in cue_detailed:
            cue_detailed = cue_detailed.format(body_part=body_part)

        return FeedbackCue(
            violation_type='stability',
            trigger=f'{stability.metric}_deviation',
            cue_short=template['cue_short'],
            cue_detailed=cue_detailed,
            severity=template['severity']
        )

    def _generate_ratio_feedback(self, ratio: RatioThreshold) -> Dict[str, FeedbackCue]:
        """Generate feedback for a ratio metric (too wide and too narrow)."""
        templates = self.RATIO_TEMPLATES.get(
            ratio.metric,
            self.RATIO_TEMPLATES['_default']
        )

        body_part = ratio.metric.replace('_ratio', '').replace('_', ' ')

        cues = {}
        for direction in ['too_wide', 'too_narrow']:
            template = templates[direction]
            cue_detailed = template['cue_detailed']
            if '{body_part}' in cue_detailed:
                cue_detailed = cue_detailed.format(body_part=body_part)

            cues[direction] = FeedbackCue(
                violation_type='ratio',
                trigger=f'{ratio.metric}_{direction}',
                cue_short=template['cue_short'],
                cue_detailed=cue_detailed,
                severity=template['severity']
            )

        return cues

    def feedback_to_dict(self, feedback: GeneratedFeedback) -> Dict:
        """Convert GeneratedFeedback to dictionary for JSON export."""
        def cue_to_dict(cue: FeedbackCue) -> Dict:
            return {
                'violation_type': cue.violation_type,
                'trigger': cue.trigger,
                'cue_short': cue.cue_short,
                'cue_detailed': cue.cue_detailed,
                'severity': cue.severity
            }

        result = {
            'exercise_id': feedback.exercise_id,
            'depth_feedback': {k: cue_to_dict(v) for k, v in feedback.depth_feedback.items()},
            'tempo_feedback': {k: cue_to_dict(v) for k, v in feedback.tempo_feedback.items()},
            'stability_feedback': {k: cue_to_dict(v) for k, v in feedback.stability_feedback.items()},
            'ratio_feedback': {}
        }

        # Ratio feedback is nested (too_wide, too_narrow)
        for metric, cues in feedback.ratio_feedback.items():
            result['ratio_feedback'][metric] = {
                direction: cue_to_dict(cue)
                for direction, cue in cues.items()
            }

        return result

    def get_feedback_summary(self, feedback: GeneratedFeedback) -> str:
        """Generate human-readable summary of feedback."""
        lines = []

        lines.append(f"Generated Feedback for: {feedback.exercise_id}")

        if feedback.depth_feedback:
            lines.append("\nDEPTH FEEDBACK:")
            for trigger, cue in feedback.depth_feedback.items():
                lines.append(f"  {trigger}:")
                lines.append(f"    short: \"{cue.cue_short}\"")
                lines.append(f"    detailed: \"{cue.cue_detailed}\"")
                lines.append(f"    severity: {cue.severity}")

        if feedback.tempo_feedback:
            lines.append("\nTEMPO FEEDBACK:")
            for trigger, cue in feedback.tempo_feedback.items():
                lines.append(f"  {trigger}:")
                lines.append(f"    short: \"{cue.cue_short}\"")
                lines.append(f"    detailed: \"{cue.cue_detailed}\"")

        if feedback.stability_feedback:
            lines.append("\nSTABILITY FEEDBACK:")
            for metric, cue in feedback.stability_feedback.items():
                lines.append(f"  {metric}:")
                lines.append(f"    short: \"{cue.cue_short}\"")
                lines.append(f"    detailed: \"{cue.cue_detailed}\"")

        if feedback.ratio_feedback:
            lines.append("\nRATIO FEEDBACK:")
            for metric, cues in feedback.ratio_feedback.items():
                lines.append(f"  {metric}:")
                for direction, cue in cues.items():
                    lines.append(f"    {direction}: \"{cue.cue_short}\" / \"{cue.cue_detailed}\"")

        return "\n".join(lines)
