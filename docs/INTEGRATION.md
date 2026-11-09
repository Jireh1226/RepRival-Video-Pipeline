# How the Python pipeline supports RepRival

## Recorded-video replay

The checked private-app flow is:

1. `extract_poses.py` processes frames with MediaPipe and writes JSON.
2. `PushUpTrackerTests/VideoTestHarness/PoseDataLoader.swift` decodes `PoseDataFile`, then maps each `JSONPoseFrame` into the app's `PoseFrame` type. Frames explicitly marked `poseDetected: false`, or missing required shoulder/elbow/wrist points, are skipped. Original timestamps are retained rather than renumbered.
3. `PushUpTrackerTests/VideoTestHarness/PushUpCoachingTests.swift` defines `VideoCoachingTestHarness`. It chooses a production coach for the exercise and camera view and calls `coach.analyzeFrame(frame)` with the decoded pose frames. It accumulates repetition counts/qualities, violations, feedback, and live cues.
4. These outputs let the developer inspect changes to the same coaching logic used by the app. Test source demonstrates the integration design; the Swift suite was not executed during this portfolio packaging session.

The exporter includes 21 landmarks, while the checked decoder consumes a subset. Extra heel/foot fields are not evidence that every field is used by every coach. Missing-confidence defaults in the historical decoder also need consideration when comparing exports from other sources. Other harnesses, such as the plank path, explicitly enable frame-timestamp timing; the basic push-up harness does not make that same call in its inspected loop. Timing parity should therefore be checked per coach rather than assumed from the README.

The historical Swift metadata decoder declares `fps` as `Int`. The release exporter reports the effective sampling rate, which can be fractional (for example 12.5). That case requires updating the Swift metadata type before replay; the existing decoder is not claimed to accept every release export. No Swift source is modified or distributed by this package.

## Threshold calibration

The `ThresholdLearner` README describes a developer-mediated loop: record a reference, run Python, inspect thresholds, implement or update the Swift form analyzer/tracker/coach, then test and tune. The current Swift `PushUpConfig.swift` can load custom JSON configurations, but the full Python export is not asserted to match every field of that decoder automatically. A schema adapter and a real Swift round-trip test would be needed for that stronger claim.

Signal selection is not entirely unsupervised discovery. It combines observed signal statistics with explicit exercise-specific priorities and stability rules. This makes domain assumptions visible and editable. Exported thresholds and feedback require review; they are not population-level or clinical validation.

## Physical testing and device feedback

The development loop also records real-runtime repetition metrics, compares them with human labels, and revises rules that behave differently from controlled videos. `analyze_rep_calibration.py` analyzes the historical JSONL exports. The original Swift app later added a per-repetition labeling UI and a Firestore logger for accepted/rejected candidates, observed violations and user corrections. This extends the offline video workflow; it does not imply that the standalone Python release contains those Swift/cloud components. See [the recovered testing account](HISTORICAL_TESTING.md).

## Product scope and authorship

Jireh Katebe designed the full system, architecture, and workflow and used AI assistance for implementation, confirmed November 5, 2026. MediaPipe is the external pose-estimation component. The design contribution includes the data interfaces, geometry/signal pipeline, calibration workflow, and connection to product testing.

[RepRival – Fitness Trainer](https://apps.apple.com/us/app/reprival-fitness-trainer/id6749421348) has a public US App Store listing naming Jireh Katebe as developer. The published product supplies deployment context. This repository supplies inspectable development software. The App Store binary has not been matched to a particular source commit, and development support for an exercise is not proof that it has shipped.

Only the Python pipeline is included in this proposed source release. Swift types, coach implementation, videos, extracted human landmark sequences, cloud configuration, app assets, and user records remain in the original private project. The integration account above comes from inspected source, not a claim that this standalone Python package builds the iOS app.
