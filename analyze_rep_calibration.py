#!/usr/bin/env python3
"""
Analyze labeled per-rep calibration data dumped by the iOS
RepCalibrationLogger. Splits reps by their `label` field and
prints distributions of every numerical metric, suggesting
threshold cuts that separate the labels.

Input:  rep_calibration.jsonl (one rep per line) — pulled from
        the device via Xcode → Devices & Simulators → app →
        Container → Documents/

Usage:  python analyze_rep_calibration.py path/to/rep_calibration.jsonl

Output:
- per-label sample count
- per-metric distribution table (median, min, max, p25, p75) per label
- threshold suggestion (midpoint between label medians) per metric
- per-landmark trajectory shape (thinness, pathChord) computed from
  the descentSnapshots — same metrics the iOS rule uses, plus the
  same shape on ear/eye paths so we can see whether they're more
  discriminative than the nose alone

Note: only knows about labels that appear in the data. Ignore
"unlabeled" reps when comparing — those are usually warmup/setup.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


LANDMARKS_TO_ANALYZE = [
    "nose",
    "leftEar", "rightEar",
    "leftEye", "rightEye",
    "leftShoulder", "rightShoulder",
]

# MediaPipe Pose Landmarker index → human-readable name. Used when
# parsing the rawLandmarks array on each snapshot. Kept here so the
# analyzer can derive shape stats from any raw landmark, not just
# the ones the curated DescentFrameSnapshot fields expose.
RAW_LANDMARK_INDEX_TO_NAME = {
    0: "raw.nose",
    1: "raw.leftEyeInner", 2: "raw.leftEye", 3: "raw.leftEyeOuter",
    4: "raw.rightEyeInner", 5: "raw.rightEye", 6: "raw.rightEyeOuter",
    7: "raw.leftEar", 8: "raw.rightEar",
    9: "raw.mouthLeft", 10: "raw.mouthRight",
    11: "raw.leftShoulder", 12: "raw.rightShoulder",
    13: "raw.leftElbow", 14: "raw.rightElbow",
    15: "raw.leftWrist", 16: "raw.rightWrist",
    17: "raw.leftPinky", 18: "raw.rightPinky",
    19: "raw.leftIndex", 20: "raw.rightIndex",
    21: "raw.leftThumb", 22: "raw.rightThumb",
    23: "raw.leftHip", 24: "raw.rightHip",
    25: "raw.leftKnee", 26: "raw.rightKnee",
    27: "raw.leftAnkle", 28: "raw.rightAnkle",
    29: "raw.leftHeel", 30: "raw.rightHeel",
    31: "raw.leftFootIndex", 32: "raw.rightFootIndex",
}


def trajectory_shape(points: np.ndarray) -> dict:
    """PCA-based shape stats. Same definition as the iOS code +
    analyze_pike_head_arc.py. Returns None if too few/degenerate."""
    if len(points) < 4:
        return None
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    lam2, lam1 = float(eigvals[0]), float(eigvals[1])
    if lam1 <= 1e-10:
        return None
    thinness = float(np.sqrt(lam2 / lam1))

    diffs = np.diff(points, axis=0)
    path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    chord = float(np.linalg.norm(points[-1] - points[0]))
    path_chord = path_len / chord if chord > 1e-6 else float("inf")

    bbox = points.max(axis=0) - points.min(axis=0)
    movement = float(np.linalg.norm(bbox))

    return {
        "thinness": thinness,
        "pathChord": path_chord,
        "movement": movement,
    }


def extract_landmark_path(snapshots: list, landmark: str) -> np.ndarray:
    """Pull a single landmark's [x,y] points across all snapshots.
    Returns empty array if the landmark wasn't recorded."""
    pts = []
    for snap in snapshots:
        v = snap.get(landmark)
        if v is None or len(v) < 2:
            continue
        pts.append([v[0], v[1]])
    return np.array(pts, dtype=float) if pts else np.empty((0, 2))


def extract_raw_landmark_path(snapshots: list, idx: int,
                               min_visibility: float = 0.0) -> np.ndarray:
    """Pull a raw landmark's [x,y] across all snapshots, filtering
    by visibility (entry [3] in each landmark vec). Returns empty
    array if rawLandmarks wasn't recorded or no qualifying samples."""
    pts = []
    for snap in snapshots:
        raw = snap.get("rawLandmarks")
        if raw is None or idx >= len(raw):
            continue
        lm = raw[idx]
        if len(lm) < 4:
            continue
        if lm[3] < min_visibility:
            continue
        pts.append([lm[0], lm[1]])
    return np.array(pts, dtype=float) if pts else np.empty((0, 2))


def collect_per_rep_metrics(reps: list) -> list:
    """For each rep, derive per-landmark shape stats from the
    descentSnapshots. Returns a list of flat dicts, one per rep,
    with keys like nose.thinness, leftEar.pathChord, etc."""
    rows = []
    for rep in reps:
        snapshots = rep.get("descentSnapshots", [])
        row = {
            "label": rep.get("label", "unlabeled"),
            "exercise": rep.get("exercise", "?"),
            "view": rep.get("view", "?"),
            "repIndex": rep.get("repIndex", -1),
            "ruleFlagged": bool(rep.get("ruleFlagged", False)),
            "snapshotCount": rep.get("snapshotCount", len(snapshots)),
        }
        # Add summaryStats fields with prefix
        for k, v in (rep.get("summaryStats") or {}).items():
            if isinstance(v, (int, float)):
                row[f"summary.{k}"] = float(v)

        # Per-landmark shape stats from the curated DescentFrameSnapshot
        # fields. Smaller, named, used by the iOS rule today.
        for lm in LANDMARKS_TO_ANALYZE:
            pts = extract_landmark_path(snapshots, lm)
            shape = trajectory_shape(pts)
            if shape is None:
                continue
            for k, v in shape.items():
                row[f"{lm}.{k}"] = v

        # Per-landmark shape stats from the raw MediaPipe array.
        # Catches landmarks the curated fields don't expose (mouth,
        # individual body points, etc.) and lets us see noise patterns
        # across the whole pose. Filter by visibility >= 0.5 so we
        # don't include occluded-frame jitter.
        for idx, name in RAW_LANDMARK_INDEX_TO_NAME.items():
            pts = extract_raw_landmark_path(snapshots, idx, min_visibility=0.5)
            shape = trajectory_shape(pts)
            if shape is None:
                continue
            for k, v in shape.items():
                row[f"{name}.{k}"] = v

        rows.append(row)
    return rows


def percentile(values, p):
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def print_distribution_by_label(rows: list, metric: str):
    """Per-label stats for one metric: count, median, p25, p75,
    min, max. Skips labels with no values for this metric."""
    by_label = defaultdict(list)
    for row in rows:
        v = row.get(metric)
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            continue
        by_label[row["label"]].append(v)

    if not by_label:
        return  # no data for this metric

    print(f"\n{metric}")
    print("  " + "-" * 70)
    print(f"  {'label':<14} {'n':>4} {'median':>10} {'p25':>10} {'p75':>10} {'min':>10} {'max':>10}")
    medians = {}
    for label in sorted(by_label.keys()):
        vs = by_label[label]
        med = float(np.median(vs))
        medians[label] = med
        print(
            f"  {label:<14} {len(vs):>4} {med:>10.4f} "
            f"{percentile(vs, 25):>10.4f} {percentile(vs, 75):>10.4f} "
            f"{min(vs):>10.4f} {max(vs):>10.4f}"
        )

    # Threshold suggestion: midpoint between any two label medians
    labels = list(medians.keys())
    if len(labels) >= 2:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                mid = (medians[a] + medians[b]) / 2
                direction = ">" if medians[a] > medians[b] else "<"
                print(
                    f"    suggest: {a} {direction} {mid:.4f} {direction} {b}"
                )


def main():
    if len(sys.argv) != 2:
        print("Usage: analyze_rep_calibration.py <rep_calibration.jsonl>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    reps = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                reps.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Line {i} skipped (parse error): {e}")
    if not reps:
        print("No reps found.")
        sys.exit(1)

    print(f"Loaded {len(reps)} reps from {path}")

    # Per-label rep counts
    label_counts = defaultdict(int)
    for r in reps:
        label_counts[r.get("label", "unlabeled")] += 1
    print("\nReps by label:")
    for label, n in sorted(label_counts.items()):
        print(f"  {label}: {n}")

    rows = collect_per_rep_metrics(reps)
    if not rows:
        print("No analyzable reps after metric derivation.")
        sys.exit(1)

    # All numerical metrics that appear in any row
    metric_keys = set()
    for row in rows:
        for k, v in row.items():
            if isinstance(v, float) and not np.isnan(v) and not np.isinf(v):
                metric_keys.add(k)
    metric_keys.discard("repIndex")
    metric_keys.discard("snapshotCount")

    # Group by prefix for readability
    grouped = defaultdict(list)
    for k in sorted(metric_keys):
        prefix = k.split(".", 1)[0]
        grouped[prefix].append(k)

    # Print order: summary stats first, then curated landmarks,
    # then raw landmarks (one section per raw index that had data).
    raw_prefixes = sorted({k for k in grouped if k == "raw"})
    section_order = ["summary"] + LANDMARKS_TO_ANALYZE + raw_prefixes
    for prefix in section_order:
        if prefix not in grouped:
            continue
        print(f"\n=== {prefix} ===")
        for metric in grouped[prefix]:
            print_distribution_by_label(rows, metric)


if __name__ == "__main__":
    main()
