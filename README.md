# RepRival video analysis pipeline

RepRival provides guided workouts at home for beginners who may struggle to afford a trainer or fit exercise into a busy schedule. The [published iPhone app](https://apps.apple.com/us/app/reprival-fitness-trainer/id6749421348) uses the phone's camera and MediaPipe's pose model to support repetition counting and form feedback.

This repository contains the Python tools I developed to understand exercise form and test that feedback using recorded videos. The tools extract body points, measure movement, identify repetition phases, and suggest thresholds for the app's exercise coaches. I designed the system and workflow, with AI assistance for implementation.

## Documentation

| Document | What it explains |
| --- | --- |
| [Requirements](docs/REQUIREMENTS.md) | Problem, expected behavior, inputs, outputs, and checks |
| [Design](docs/DESIGN.md) | Processing stages, data structures, and module responsibilities |
| [App integration](docs/INTEGRATION.md) | How the JSON exports support Swift replay and coach configuration |
| [Testing](docs/VALIDATION.md) | Executed software checks and commands |
| [Physical testing](docs/HISTORICAL_TESTING.md) | Recorded exercises, good-form/bad-form labels, and calibration findings |
| [Release changes](docs/RELEASE_CHANGES.md) | Changes made when preparing this standalone package |

## Setup and quick check

Use Python 3.11. Dependencies are pinned in `requirements.txt`, including MediaPipe 0.10.21 for its `mp.solutions.pose` interface.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/synthetic_walkthrough.py --output-dir outputs/synthetic
```

The seven checks cover pose conversion, missing detections, sampling, timing, and the path from geometric measurements to JSON export. The example processes 601 synthetic pose frames and produces nine valid repetition segments. These are software checks using generated inputs.

## Extract body points from a recording

```sh
python extract_poses.py /path/to/video.mp4 --output outputs/poses.json --fps 30
```

The exporter saves 21 of MediaPipe's 33 landmarks, visibility, source timestamps, and records of frames without a detected pose. The app's Swift test loader can replay these points through the exercise coaches. Coordinates use the image frame, with y increasing downward; relative z is not a measurement in metres.

## Estimate movement thresholds

```sh
python learn_exercise.py --video /path/to/reference.mp4 --exercise-id standard_pushup --name "Standard push-up" --view front --limb-mode bilateral --non-interactive --output-dir outputs/thresholds
```

This command extracts poses, calculates angles and ratios, selects a movement signal, and segments repetitions. It then uses percentiles and adjustable margins to suggest settings for depth, tempo, stability, and feedback. I review these settings when developing the Swift coaches. The Python tools run offline, outside the app.

## Analyze labeled repetitions

```sh
python analyze_rep_calibration.py /path/to/rep_calibration.jsonl
```

The analyzer summarizes the original JSONL format, which uses `label`, `exercise`, `view`, `summaryStats`, and `descentSnapshots`. During development, I compared 37 pike and knee-pike repetitions: 22 labeled as good form and 15 labeled as bad form. These examples helped revise feedback rules based on head movement and hip angle. The recordings were of me; friends also tried the app informally to explore different body proportions.

## Files and reuse

`src/` contains the processing modules. `tests/` and `examples/` provide the checks and synthetic example. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records the original repository paths, source hashes, and changes to the selected implementation files.

This package includes the Python pipeline and documentation. Recordings, personal motion records, model weights, backend configuration, and the full iOS app are maintained separately. See [validation notes](docs/VALIDATION.md) for operating limits. No open-source license has been selected for this code; dependency licenses remain with their respective projects.
