"""
Video processing with MediaPipe pose estimation.
Extracts pose landmarks from video frames.
"""

import math
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple, Generator
from dataclasses import dataclass
from pathlib import Path

from .pose_frame import PoseFrame, landmarks_to_pose_frame


@dataclass
class VideoInfo:
    """Basic video metadata."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


class VideoProcessor:
    """
    Process video files and extract pose data using MediaPipe.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Initialize the video processor.

        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            model_complexity: 0, 1, or 2. Higher = more accurate but slower
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        # MediaPipe pose
        self.mp_pose = mp.solutions.pose

    def get_video_info(self, video_path: str) -> VideoInfo:
        """Get basic information about a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid positive frame rate")
        duration = frame_count / fps

        return VideoInfo(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration
        )

    def process_video(
        self,
        video_path: str,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[PoseFrame], VideoInfo]:
        """
        Process entire video and extract pose frames.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(current_frame, total_frames)

        Returns:
            Tuple of (list of PoseFrames, VideoInfo)
        """
        video_info = self.get_video_info(video_path)
        pose_frames: List[PoseFrame] = []

        cap = cv2.VideoCapture(video_path)

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        ) as pose:

            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame
                results = pose.process(rgb_frame)

                # Calculate timestamp
                timestamp = frame_index / video_info.fps

                if results.pose_landmarks:
                    pose_frame = landmarks_to_pose_frame(
                        landmarks=results.pose_landmarks.landmark,
                        timestamp=timestamp,
                        frame_index=frame_index,
                        image_width=video_info.width,
                        image_height=video_info.height
                    )
                    pose_frames.append(pose_frame)

                frame_index += 1

                if progress_callback:
                    progress_callback(frame_index, video_info.frame_count)

        cap.release()

        return pose_frames, video_info

    def process_video_generator(
        self,
        video_path: str
    ) -> Generator[Tuple[PoseFrame, int, int], None, None]:
        """
        Generator that yields pose frames one at a time.
        Useful for memory-efficient processing of long videos.

        Yields:
            Tuple of (PoseFrame, current_frame_index, total_frames)
        """
        video_info = self.get_video_info(video_path)
        cap = cv2.VideoCapture(video_path)

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        ) as pose:

            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                timestamp = frame_index / video_info.fps

                if results.pose_landmarks:
                    pose_frame = landmarks_to_pose_frame(
                        landmarks=results.pose_landmarks.landmark,
                        timestamp=timestamp,
                        frame_index=frame_index,
                        image_width=video_info.width,
                        image_height=video_info.height
                    )
                    yield pose_frame, frame_index, video_info.frame_count

                frame_index += 1

        cap.release()

    def filter_low_confidence_frames(
        self,
        frames: List[PoseFrame],
        min_confidence: float = 0.5
    ) -> List[PoseFrame]:
        """Filter out frames with low confidence scores."""
        return [f for f in frames if f.avg_confidence >= min_confidence]

    def get_confidence_stats(self, frames: List[PoseFrame]) -> dict:
        """Get statistics about frame confidence scores."""
        if not frames:
            return {key: 0 for key in ("count", "mean", "min", "max", "std",
                                      "above_50", "above_70", "arms_mean",
                                      "arms_above_40", "legs_mean", "legs_above_40")}

        avg_confidences = [f.avg_confidence for f in frames]
        arms_confidences = [f.arms_confidence for f in frames]
        legs_confidences = [f.legs_confidence for f in frames]

        return {
            "count": len(frames),
            "mean": np.mean(avg_confidences),
            "min": np.min(avg_confidences),
            "max": np.max(avg_confidences),
            "std": np.std(avg_confidences),
            "above_50": sum(1 for c in avg_confidences if c >= 0.5) / len(frames),
            "above_70": sum(1 for c in avg_confidences if c >= 0.7) / len(frames),
            # Landmark-specific confidence stats
            "arms_mean": np.mean(arms_confidences),
            "arms_above_40": sum(1 for c in arms_confidences if c >= 0.4) / len(frames),
            "legs_mean": np.mean(legs_confidences),
            "legs_above_40": sum(1 for c in legs_confidences if c >= 0.4) / len(frames),
        }
