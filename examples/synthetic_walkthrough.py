"""Deterministic software walkthrough, not a real-video accuracy experiment."""
from pathlib import Path
import argparse, json, math, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pose_frame import Point, PoseFrame
from src.metric_calculator import MetricCalculator
from src.signal_analyzer import SignalAnalyzer
from src.rep_segmenter import RepSegmenter
from src.threshold_learner import ThresholdLearner
from src.feedback_generator import FeedbackGenerator
from src.json_exporter import JSONExporter


def synthetic_frames():
    """A geometric elbow-motion fixture; no person or video is represented."""
    frames = []
    for i in range(601):
        t = i / 30
        theta = math.radians(130 + 40 * math.cos(math.pi * t))
        positions = dict(
            nose=(.5, .15), left_shoulder=(.35, .3), right_shoulder=(.65, .3),
            left_elbow=(.35, .5), right_elbow=(.65, .5),
            left_wrist=(.35+.15*math.sin(theta), .5-.15*math.cos(theta)),
            right_wrist=(.65-.15*math.sin(theta), .5-.15*math.cos(theta)),
            left_hip=(.4, .65), right_hip=(.6, .65),
            left_knee=(.4, .8), right_knee=(.6, .8),
            left_ankle=(.4, .95), right_ankle=(.6, .95))
        frame = PoseFrame(timestamp=t, frame_index=i, avg_confidence=.95,
                          **{k: Point(x, y, .95, z=0.0) for k,(x,y) in positions.items()})
        for name in positions:
            attr = name + '_confidence'
            if hasattr(frame, attr):
                setattr(frame, attr, .95)
        frames.append(frame)
    return frames


def run(output_dir):
    frames = synthetic_frames()
    calculator = MetricCalculator(camera_view='front')
    metrics = calculator.calculate_all_metrics(frames)
    analysis = SignalAnalyzer().analyze_all_signals(
        metrics, calculator, limb_mode='bilateral', exercise_id='synthetic_pushup')
    if not analysis.recommended_primary:
        raise RuntimeError('Synthetic fixture did not produce a primary signal')
    segmentation = RepSegmenter().segment_reps(
        metrics, analysis.all_signals[analysis.recommended_primary])
    thresholds = ThresholdLearner().learn_thresholds(
        segmentation, analysis, exercise_id='synthetic_pushup',
        exercise_name='SYNTHETIC geometry fixture', camera_view='front',
        limb_mode='bilateral', video_source='SYNTHETIC_NO_VIDEO')
    feedback = FeedbackGenerator().generate_feedback(thresholds)
    path = JSONExporter(str(output_dir)).export(thresholds, feedback, filename='synthetic_thresholds.json')
    data = json.loads(path.read_text())
    data['metadata']['synthetic_fixture'] = True
    data['metadata']['purpose'] = 'Software demonstration; not an accuracy or form-safety benchmark'
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    summary = dict(synthetic=True, input_frames=len(frames),
                   primary_signal=analysis.recommended_primary,
                   detected_segments=segmentation.total_reps,
                   valid_segments=segmentation.valid_count,
                   output_file=path.name, neural_inference_run=False)
    (Path(output_dir)/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/synthetic'))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
