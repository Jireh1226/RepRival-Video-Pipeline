#!/usr/bin/env python3
"""
MediaPipe Pose Extraction Script for PushUpTracker Testing

This script extracts pose landmarks from video files using MediaPipe
and exports them as JSON files that can be loaded by the Swift test harness.

Usage:
    python3 extract_poses.py <video_file> [--output <output_file>] [--fps <frame_rate>]

Example:
    python3 extract_poses.py archer_pushup_test.mp4 --output archer_test_poses.json --fps 30

The output JSON format matches the PoseFrame structure in Swift:
{
    "metadata": {
        "video_file": "archer_pushup_test.mp4",
        "total_frames": 450,
        "fps": 30,
        "duration_seconds": 15.0,
        "extraction_date": "2025-12-25T10:30:00Z"
    },
    "frames": [
        {
            "timestamp": 0.0,
            "confidence": 0.95,
            "leftShoulder": {"x": 0.45, "y": 0.35},
            "rightShoulder": {"x": 0.55, "y": 0.35},
            "leftElbow": {"x": 0.40, "y": 0.50},
            "rightElbow": {"x": 0.60, "y": 0.50},
            "leftWrist": {"x": 0.35, "y": 0.65},
            "rightWrist": {"x": 0.65, "y": 0.65},
            "nose": {"x": 0.50, "y": 0.25},
            "leftHip": {"x": 0.47, "y": 0.55},
            "rightHip": {"x": 0.53, "y": 0.55},
            "leftKnee": {"x": 0.46, "y": 0.75},
            "rightKnee": {"x": 0.54, "y": 0.75},
            "leftAnkle": {"x": 0.45, "y": 0.90},
            "rightAnkle": {"x": 0.55, "y": 0.90},
            "leftIndex": {"x": 0.33, "y": 0.68},
            "rightIndex": {"x": 0.67, "y": 0.68},
            "leftPinky": {"x": 0.36, "y": 0.67},
            "rightPinky": {"x": 0.64, "y": 0.67},
            "landmarkConfidences": {
                "leftShoulder": 0.98,
                "rightShoulder": 0.97,
                "leftElbow": 0.95,
                "rightElbow": 0.96,
                ...
            }
        },
        ...
    ]
}

Requirements:
    pip install mediapipe opencv-python numpy
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package. Please install with:")
    print("  pip install mediapipe opencv-python numpy")
    sys.exit(1)


# MediaPipe landmark indices (matching the order in MediaPipe)
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LANDMARK_NAMES = [
    "nose",              # 0
    "left_eye_inner",    # 1
    "left_eye",          # 2
    "left_eye_outer",    # 3
    "right_eye_inner",   # 4
    "right_eye",         # 5
    "right_eye_outer",   # 6
    "left_ear",          # 7
    "right_ear",         # 8
    "mouth_left",        # 9
    "mouth_right",       # 10
    "left_shoulder",     # 11
    "right_shoulder",    # 12
    "left_elbow",        # 13
    "right_elbow",       # 14
    "left_wrist",        # 15
    "right_wrist",       # 16
    "left_pinky",        # 17
    "right_pinky",       # 18
    "left_index",        # 19
    "right_index",       # 20
    "left_thumb",        # 21
    "right_thumb",       # 22
    "left_hip",          # 23
    "right_hip",         # 24
    "left_knee",         # 25
    "right_knee",        # 26
    "left_ankle",        # 27
    "right_ankle",       # 28
    "left_heel",         # 29
    "right_heel",        # 30
    "left_foot_index",   # 31
    "right_foot_index",  # 32
]

# Mapping from MediaPipe index to Swift PoseFrame property names
SWIFT_LANDMARK_MAP = {
    0: "nose",
    11: "leftShoulder",
    12: "rightShoulder",
    13: "leftElbow",
    14: "rightElbow",
    15: "leftWrist",
    16: "rightWrist",
    17: "leftPinky",
    18: "rightPinky",
    19: "leftIndex",
    20: "rightIndex",
    23: "leftHip",
    24: "rightHip",
    25: "leftKnee",
    26: "rightKnee",
    27: "leftAnkle",
    28: "rightAnkle",
    29: "leftHeel",
    30: "rightHeel",
    31: "leftFootIndex",
    32: "rightFootIndex",
}


def extract_poses(video_path: str, target_fps: int = 30) -> dict:
    """Export sampled frames using original timestamps and an explicit stride.

    The legacy integer-stride policy is retained. Metadata records the actual
    nominal sampling rate, which can differ from the requested target rate.
    """
    import math
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be finite and positive")
    cap = cv2.VideoCapture(video_path)
    pose = None
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(video_fps) or video_fps <= 0:
            raise ValueError("Video has no valid positive frame rate")
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_video_frames / video_fps
        frame_skip = max(1, int(video_fps / target_fps))
        effective_fps = video_fps / frame_skip
        pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=2,
            enable_segmentation=False, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        frames = []
        frame_idx = 0
        detected = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_skip == 0:
                timestamp = frame_idx / video_fps
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.pose_landmarks:
                    frames.append(extract_frame_data(result.pose_landmarks, timestamp))
                    detected += 1
                else:
                    frames.append(dict(timestamp=timestamp, confidence=0.0,
                                       poseDetected=False))
            frame_idx += 1
        return dict(metadata=dict(
            video_file=Path(video_path).name, total_frames=len(frames),
            fps=effective_fps, requested_fps=target_fps, original_fps=video_fps,
            frame_stride=frame_skip, duration_seconds=duration,
            frames_with_pose=detected,
            extraction_date=datetime.utcnow().isoformat() + "Z"), frames=frames)
    finally:
        cap.release()
        if pose is not None:
            pose.close()


def extract_frame_data(landmarks, timestamp: float) -> dict:
    """
    Extract landmark data from MediaPipe pose landmarks.

    Args:
        landmarks: MediaPipe pose landmarks
        timestamp: Frame timestamp in seconds

    Returns:
        Dictionary with landmark positions and confidences
    """
    frame_data = {
        "timestamp": timestamp,
        "poseDetected": True,
        "landmarkConfidences": {}
    }

    # Calculate overall confidence as average of key landmarks
    key_landmark_indices = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
    confidences = [landmarks.landmark[i].visibility for i in key_landmark_indices]
    frame_data["confidence"] = float(np.mean(confidences))

    # Extract each landmark we care about
    for mp_idx, swift_name in SWIFT_LANDMARK_MAP.items():
        landmark = landmarks.landmark[mp_idx]

        # MediaPipe uses normalized coordinates (0-1)
        # x: left edge = 0, right edge = 1
        # y: top edge = 0, bottom edge = 1
        # z: depth relative to hips (negative = closer to camera)
        frame_data[swift_name] = {
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z)  # Include Z for depth/leg identification
        }

        # Store confidence (visibility) for each landmark
        # Convert to camelCase for Swift consistency
        confidence_key = swift_name[0].lower() + swift_name[1:] + "Confidence"
        frame_data["landmarkConfidences"][swift_name] = float(landmark.visibility)

    return frame_data


def main():
    parser = argparse.ArgumentParser(
        description="Extract pose landmarks from video using MediaPipe"
    )
    parser.add_argument(
        "video_file",
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: <video_name>_poses.json)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate for extraction (default: 30)"
    )

    args = parser.parse_args()

    # Validate input file
    video_path = Path(args.video_file)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Determine output file
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.with_suffix("").with_name(f"{video_path.stem}_poses.json")

    # Extract poses
    try:
        data = extract_poses(str(video_path), args.fps)
    except Exception as e:
        print(f"Error extracting poses: {e}")
        sys.exit(1)

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nOutput saved to: {output_path}")
    print(f"Total frames: {data['metadata']['total_frames']}")
    print(f"Frames with pose: {data['metadata']['frames_with_pose']}")


if __name__ == "__main__":
    main()
