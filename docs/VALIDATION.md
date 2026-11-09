# Validation scope

The review checks run on Python 3.11.14 using the original project's existing environment, with versions pinned in `requirements.txt`. A new environment installation and native iOS build are not part of this validation.

`tests/test_pipeline.py` uses synthetic landmarks, fake video/model interfaces, generated geometric motion, and temporary outputs. It checks:

- landmark-name, coordinate, visibility, and timestamp conversion;
- integer-stride sampling metadata and explicit no-pose frames;
- invalid frame-rate rejection and resource release;
- empty-detection confidence statistics;
- repetition membership after source-frame gaps;
- peak analysis at a low sampling rate;
- geometry through metric calculation, signal selection, segmentation, threshold estimation, and finite JSON export.

Result on November 5: **7/7 checks passed**. Both CLI help commands succeeded. The saved synthetic walkthrough processes 601 generated pose frames, selects elbow_depth, and exports nine valid segments. See [VALIDATION.json](../VALIDATION.json) and [generated example](../examples/generated/summary.json). These counts describe a synthetic fixture, not a real-video benchmark. The sample output is labeled synthetic. A passing synthetic walkthrough does not measure pose detection or repetition accuracy on people. No human video, pose-estimator inference, model training, or Swift coaching test is part of these checks.

On November 8, the seven checks and the synthetic walkthrough passed again while preparing the standalone GitHub package. The walkthrough again processed 601 generated frames and exported nine valid segments. This recheck used the existing project environment; it did not test a fresh dependency installation or rerun the historical human-data analysis.

Historical evidence also includes physical-test exports, labeled video cases and device calibration, as detailed in [historical testing](HISTORICAL_TESTING.md). The main pose directory has 34 JSON fixtures, including one explicitly synthetic sample; a November 27, 2025 threshold export records ten analyzed repetitions. The 51 video files, 68 video-harness methods and recorded calibration samples establish an implemented testing workflow, not a verified native pass total or held-out benchmark. Human-derived coordinate sequences and videos are not bundled.

The unchanged historical `analyze_rep_calibration.py` was also executed successfully on the 37 saved device records in the same Python environment. It produced per-label distributions with no stderr output. This is numerical reanalysis of saved data, not new predictions or a new human trial. Its earlier JSONL input schema differs from the later Firestore schema.

Further evaluation should use separate labeled sessions across viewpoints/occlusion, save predictions and the exact source version, and verify JSON-to-Swift replay in the actual app test target. Existing calibration labels and intended assertions are not substitutes for that held-out evaluation.
