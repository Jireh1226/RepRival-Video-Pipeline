#!/usr/bin/env python3
"""
Exercise Threshold Learner - CLI Entry Point

Learn exercise thresholds from reference videos using pose estimation.

Usage:
    python learn_exercise.py --video path/to/video.mp4
    python learn_exercise.py --video path/to/video.mp4 --exercise-id squat --name "Bodyweight Squat"
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.video_processor import VideoProcessor
from src.metric_calculator import MetricCalculator
from src.signal_analyzer import SignalAnalyzer
from src.rep_segmenter import RepSegmenter
from src.threshold_learner import ThresholdLearner
from src.feedback_generator import FeedbackGenerator
from src.json_exporter import JSONExporter


def print_header():
    """Print CLI header."""
    print("=" * 65)
    print("                 EXERCISE THRESHOLD LEARNER")
    print("=" * 65)
    print()


def print_step(step_num: int, title: str):
    """Print step header."""
    print(f"\nStep {step_num}: {title}")
    print("-" * 65)


def get_user_input(prompt: str, default: str = None, options: list = None) -> str:
    """Get input from user with optional default and validation."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    while True:
        value = input(prompt).strip()

        if not value and default:
            return default

        if not value:
            print("  Please enter a value.")
            continue

        if options and value.lower() not in [o.lower() for o in options]:
            print(f"  Please choose from: {', '.join(options)}")
            continue

        return value


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user."""
    default_str = "Y/n" if default else "y/N"
    value = input(f"{prompt} [{default_str}]: ").strip().lower()

    if not value:
        return default

    return value in ('y', 'yes', 'true', '1')


def progress_callback(current: int, total: int):
    """Print progress bar."""
    percent = current / total * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r  Processing: {bar} {percent:.1f}%", end='', flush=True)
    if current == total:
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Learn exercise thresholds from reference video"
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to reference video file"
    )
    parser.add_argument(
        "--exercise-id", "-i",
        help="Exercise ID (e.g., 'bodyweight_squat')"
    )
    parser.add_argument(
        "--name", "-n",
        help="Exercise name (e.g., 'Bodyweight Squat')"
    )
    parser.add_argument(
        "--view",
        choices=["front", "side"],
        help="Camera view (front or side)"
    )
    parser.add_argument(
        "--limb-mode",
        choices=["bilateral", "unilateral_alternating", "unilateral_single"],
        help="Limb mode"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./learned_thresholds",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without prompts (requires all options)"
    )

    args = parser.parse_args()

    # Validate video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print_header()

    # =========================================================================
    # Step 1: Exercise Setup
    # =========================================================================
    print_step(1, "Exercise Setup")

    if args.non_interactive:
        if not all([args.exercise_id, args.name, args.view, args.limb_mode]):
            print("Error: Non-interactive mode requires --exercise-id, --name, --view, and --limb-mode")
            sys.exit(1)
        exercise_id = args.exercise_id
        exercise_name = args.name
        camera_view = args.view
        limb_mode = args.limb_mode
    else:
        exercise_id = args.exercise_id or get_user_input(
            "Exercise ID (e.g., bodyweight_squat)",
            default=video_path.stem.lower().replace(" ", "_").replace("-", "_")
        )

        default_name = exercise_id.replace("_", " ").title()
        exercise_name = args.name or get_user_input(
            "Exercise name",
            default=default_name
        )

        camera_view = args.view or get_user_input(
            "Camera view",
            options=["front", "side"],
            default="front"
        ).lower()

        limb_mode = args.limb_mode or get_user_input(
            "Limb mode",
            options=["bilateral", "unilateral_alternating", "unilateral_single"],
            default="bilateral"
        ).lower()

    print(f"\n  Exercise ID: {exercise_id}")
    print(f"  Exercise Name: {exercise_name}")
    print(f"  Camera View: {camera_view}")
    print(f"  Limb Mode: {limb_mode}")

    # =========================================================================
    # Step 2: Processing Video
    # =========================================================================
    print_step(2, "Processing Video")

    processor = VideoProcessor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    video_info = processor.get_video_info(str(video_path))
    print(f"  Video: {video_path.name}")
    print(f"  Resolution: {video_info.width}x{video_info.height}")
    print(f"  Duration: {video_info.duration_seconds:.1f}s @ {video_info.fps:.1f}fps")
    print(f"  Frames: {video_info.frame_count}")
    print()

    pose_frames, _ = processor.process_video(
        str(video_path),
        progress_callback=progress_callback
    )

    if not pose_frames:
        print("No poses detected; no thresholds were generated.")
        return 1

    conf_stats = processor.get_confidence_stats(pose_frames)
    print(f"\n  Poses extracted: {len(pose_frames)} frames")
    print(f"  Confidence: {conf_stats['mean']:.1%} mean, {conf_stats['above_50']:.1%} above 50%")

    if conf_stats['above_50'] < 0.7:
        print("\n  ⚠️  Warning: Low confidence scores. Video quality may affect results.")
        if not args.non_interactive and not get_yes_no("  Continue anyway?"):
            sys.exit(0)

    # =========================================================================
    # Step 3: Calculating Metrics
    # =========================================================================
    print_step(3, "Calculating Metrics")

    metric_calc = MetricCalculator(camera_view=camera_view)
    frame_metrics = metric_calc.calculate_all_metrics(pose_frames)
    print(f"  Calculated {len(metric_calc.get_all_metric_names())} metrics per frame")
    print(f"  Total data points: {len(frame_metrics) * len(metric_calc.get_all_metric_names())}")

    # =========================================================================
    # Step 4: Signal Analysis
    # =========================================================================
    print_step(4, "Signal Analysis")
    print("  Analyzing metrics for movement patterns...")

    analyzer = SignalAnalyzer()
    # Pass limb_mode and exercise_id to prioritize correct signals for this exercise type
    # For push-ups, this ensures elbow_depth is preferred over knee_depth
    signal_analysis = analyzer.analyze_all_signals(
        frame_metrics, metric_calc, limb_mode=limb_mode, exercise_id=exercise_id
    )

    print()
    print(analyzer.get_analysis_summary(signal_analysis))

    # Allow user to override primary signal
    recommended = signal_analysis.recommended_primary
    if not args.non_interactive and signal_analysis.primary_signals:
        print(f"\n  Recommended primary signal: {recommended}")
        options = [s.name for s in signal_analysis.primary_signals[:5]]

        override = get_user_input(
            f"  Select primary signal",
            default=recommended
        )
        if override != recommended and override in options:
            recommended = override
            print(f"  Using: {recommended}")

    if not recommended:
        print("\n  ❌ Error: Could not detect primary movement signal.")
        print("     The video may not contain clear repetitive movement.")
        sys.exit(1)

    primary_signal = signal_analysis.all_signals[recommended]

    # =========================================================================
    # Step 5: Rep Segmentation
    # =========================================================================
    print_step(5, "Rep Segmentation")

    segmenter = RepSegmenter(
        min_rep_duration_ms=500,
        max_rep_duration_ms=10000
    )

    segmentation = segmenter.segment_reps(frame_metrics, primary_signal)

    print(segmenter.get_segmentation_summary(segmentation))

    if segmentation.valid_count < 3:
        print(f"\n  ⚠️  Warning: Only {segmentation.valid_count} valid reps detected.")
        print("     At least 5-10 reps recommended for reliable thresholds.")
        if not args.non_interactive and not get_yes_no("  Continue anyway?"):
            sys.exit(0)

    # =========================================================================
    # Step 6: Learning Thresholds
    # =========================================================================
    print_step(6, "Learning Thresholds")
    print(f"  Learning from {segmentation.valid_count} valid reps...")

    learner = ThresholdLearner()
    thresholds = learner.learn_thresholds(
        segmentation=segmentation,
        signal_analysis=signal_analysis,
        exercise_id=exercise_id,
        exercise_name=exercise_name,
        camera_view=camera_view,
        limb_mode=limb_mode,
        video_source=video_path.name
    )

    print()
    print(learner.get_thresholds_summary(thresholds))

    # =========================================================================
    # Step 7: Generating Feedback
    # =========================================================================
    print_step(7, "Generating Feedback")

    feedback_gen = FeedbackGenerator()
    feedback = feedback_gen.generate_feedback(thresholds)

    print(feedback_gen.get_feedback_summary(feedback))

    # =========================================================================
    # Step 8: Export
    # =========================================================================
    print_step(8, "Export")

    exporter = JSONExporter(output_dir=args.output_dir)
    output_path = exporter.export(thresholds, feedback)

    print(f"  ✅ Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 65)
    print("                         SUMMARY")
    print("=" * 65)
    print(f"  Exercise: {exercise_name}")
    print(f"  Primary Signal: {thresholds.primary_signal}")
    print(f"  Reps Analyzed: {thresholds.reps_analyzed}")
    print(f"  Confidence: {thresholds.avg_confidence:.1%}")
    print(f"  Stability Metrics: {len(thresholds.stability)}")
    print(f"  Ratio Metrics: {len(thresholds.ratios)}")
    print(f"  Output: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    sys.exit(main())
