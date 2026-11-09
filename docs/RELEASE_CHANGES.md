# Changes made for the November 5, 2026 review package

The original project files are preserved unchanged in the portfolio evidence folder. The standalone copy contains the original core modules and two command-line entry points. It adds documentation, dependency pins, a synthetic walkthrough, and focused tests. These additions are new review work with AI assistance, not retroactively attributed to the original graduate implementation.

Bounded maintenance fixes:

- Reject non-finite/non-positive requested or source frame rates before timestamp division.
- Ensure the replay extractor releases video/model resources on success and exceptions; create output directories as needed.
- Preserve the integer-stride sampling algorithm but report its effective rate separately from the requested rate.
- Return consistent confidence-statistic keys for an empty detection set and exit the threshold CLI with a clear no-detections result.
- Match repetition metrics by timestamps rather than treating compacted signal indices as original video frame numbers. Missing poses and per-metric filtering can make those index spaces differ.
- Clamp the SciPy peak-spacing argument to at least one sample, including sparse sampling rates.

No threshold constants were tuned to improve a reported real-video score. No new pose model was trained. This package does not silently retrofit the fixes into the app repository or claim that they shipped in the App Store binary.

The later testing-evidence review adds the historical `analyze_rep_calibration.py` unchanged and documents the recovered physical/video/device-testing process. The analyzer successfully summarized 37 saved records; no raw records are in the package. The original seven synthetic checks remain distinct from that historical-data reanalysis.

## November 8, 2026: GitHub package preparation

Added the software requirements document and its README link. Clarified the repository scope and existing absence of an open-source license. Extended `.gitignore` to exclude local environment settings, recordings, data folders, and raw JSONL calibration exports. Initialized a separate local Git repository on `main`; no remote was configured or published. The Python implementation is unchanged from the earlier prepared package.

The seven software checks and the synthetic walkthrough passed again in the existing Python environment. No video inference, Swift tests, or new human-data evaluation was performed.

## Standalone repository documentation, November 8, 2026

Added a separate design document and simplified the README around the exercise-feedback problem, commands, and evidence. Historical-testing documentation uses good-form and bad-form labels and explains their correspondence to the original data schema. The implementation and saved synthetic examples are unchanged.
