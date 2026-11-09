# Software requirements

## Problem and purpose

RepRival provides guided workouts at home for people who may struggle to afford a trainer or fit exercise into a busy schedule. Developing useful feedback requires comparing correct exercise form with common mistakes and checking whether the coaching logic counts repetitions and identifies those mistakes.

The Python tools turn recorded exercise video into measurements for that development process. One workflow exports body points for replay through the app's Swift tests. Another estimates candidate movement thresholds for the developer to inspect and use when configuring exercise coaches. This document describes the supplied software and its expected behavior.

## Functional requirements

| Requirement | Implementation | Available check |
| --- | --- | --- |
| Extract selected body points, visibility, and source timestamps from a supplied video; record frames with no detected pose | `extract_poses.py` exports 21 named MediaPipe landmarks and sampling metadata | Tests cover landmark conversion, missing detections, and integer-stride sampling with fake video/model interfaces |
| Reject invalid sampling rates and release video/model resources | Input validation and cleanup in `extract_poses.py` | Tests cover invalid rates and resource release |
| Convert detected poses into joint angles and other movement measurements | `src/pose_frame.py`, `src/video_processor.py`, and `src/metric_calculator.py` | Empty-detection checks and a synthetic geometry walkthrough |
| Select a movement signal and segment repetitions while keeping measurements aligned with timestamps | `src/signal_analyzer.py` and `src/rep_segmenter.py` | Tests cover sparse sampling and gaps in the original frame sequence |
| Export candidate thresholds and feedback templates for developer review | `src/threshold_learner.py`, `src/feedback_generator.py`, and `src/json_exporter.py`, called by `learn_exercise.py` | Synthetic walkthrough checks the path from geometry to finite JSON output |
| Summarize labeled repetition metrics from the historical JSONL format | `analyze_rep_calibration.py` | Previously processed 37 saved records; raw records are not included |

## Operating requirements and boundaries

- Python 3.11 and the dependency versions in `requirements.txt`.
- User-supplied video for pose extraction. The synthetic example runs without a recording or pose-model inference.
- A writable output directory. Exported thresholds require developer review before use in a coach.
- Swift replay requires the app's test harness, which is documented in [INTEGRATION.md](INTEGRATION.md) and is not included here.
- The threshold export is not a verified drop-in replacement for every Swift configuration. The historical calibration command does not accept the later Firestore schema without an adapter.
- Recordings, personal motion data, app backend configuration, and model weights are not part of this package.

## Evaluation

Run the commands in the [README](../README.md) to execute the seven software checks and synthetic walkthrough. These exercise conversion, timing, segmentation, and export behavior. Their results do not measure coaching accuracy on people. [VALIDATION.md](VALIDATION.md) records execution scope, and [HISTORICAL_TESTING.md](HISTORICAL_TESTING.md) describes the separate physical and labeled-video testing.
