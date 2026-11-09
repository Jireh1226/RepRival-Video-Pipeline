"""Boundary and integration checks using synthetic inputs, without video inference."""
import json, math, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import extract_poses as extraction
from src.video_processor import VideoProcessor
from src.metric_calculator import FrameMetrics
from src.signal_analyzer import SignalAnalyzer, SignalCharacteristics
from src.rep_segmenter import RepSegmenter
from examples.synthetic_walkthrough import run


class FakeCapture:
    def __init__(self, fps=25, count=10):
        self.fps, self.count, self.index, self.released = fps, count, 0, False
    def isOpened(self): return True
    def get(self, key):
        return self.fps if key == extraction.cv2.CAP_PROP_FPS else self.count
    def read(self):
        if self.index >= self.count: return False, None
        self.index += 1
        return True, np.zeros((8,8,3), dtype=np.uint8)
    def release(self): self.released = True


class FakePose:
    def __init__(self): self.closed = False
    def process(self, image): return SimpleNamespace(pose_landmarks=None)
    def close(self): self.closed = True


class PipelineTests(unittest.TestCase):
    def test_landmark_mapping_preserves_coordinates_visibility_and_time(self):
        points = [SimpleNamespace(x=i/100, y=1-i/100, z=-i/100, visibility=.8) for i in range(33)]
        frame = extraction.extract_frame_data(SimpleNamespace(landmark=points), 2.5)
        self.assertEqual(len(extraction.SWIFT_LANDMARK_MAP), 21)
        self.assertEqual(frame['leftShoulder'], {'x':.11,'y':.89,'z':-.11})
        self.assertEqual(frame['timestamp'], 2.5)
        self.assertEqual(frame['landmarkConfidences']['rightWrist'], .8)
        self.assertAlmostEqual(frame['confidence'], .8)

    def test_sampling_metadata_and_missing_pose_frames(self):
        cap, pose = FakeCapture(), FakePose()
        with patch.object(extraction.cv2,'VideoCapture',return_value=cap), \
             patch.object(extraction.mp.solutions.pose,'Pose',return_value=pose):
            result = extraction.extract_poses('synthetic-not-a-video',10)
        self.assertEqual(result['metadata']['requested_fps'],10)
        self.assertEqual(result['metadata']['fps'],12.5)
        self.assertEqual([x['timestamp'] for x in result['frames']],[0,.08,.16,.24,.32])
        self.assertTrue(all(x['poseDetected'] is False for x in result['frames']))
        self.assertEqual(result['metadata']['frames_with_pose'],0)
        self.assertTrue(cap.released and pose.closed)

    def test_bad_rates_rejected_and_capture_released(self):
        for rate in [0,-1,float('nan')]:
            with self.assertRaises(ValueError): extraction.extract_poses('unused',rate)
        cap=FakeCapture(fps=0)
        with patch.object(extraction.cv2,'VideoCapture',return_value=cap):
            with self.assertRaises(ValueError): extraction.extract_poses('unused',30)
        self.assertTrue(cap.released)

    def test_empty_pose_statistics_keep_consistent_fields(self):
        stats=VideoProcessor().get_confidence_stats([])
        self.assertEqual(stats['above_50'],0)
        self.assertEqual(stats['count'],0)

    def test_segment_metrics_follow_timestamps_after_dropped_frames(self):
        metrics=[FrameMetrics(timestamp=t,frame_index=i,confidence=.9)
                 for t,i in [(10,300),(11,330),(12,360),(13,390)]]
        reps=RepSegmenter()._create_rep_segments(metrics,np.array([0.,1.,0.]),
              np.array([10.,11.,12.]),np.array([0,2]),np.array([1]))
        self.assertEqual(len(reps),1)
        self.assertEqual([x.frame_index for x in reps[0].frame_metrics],[300,330,360])

    def test_sparse_sample_rate_does_not_create_zero_peak_distance(self):
        times=np.arange(40,dtype=float)
        values=1+np.sin(2*np.pi*times/8)
        char=SignalAnalyzer().analyze_signal('elbow_depth',values,times)
        self.assertEqual(len(char.values),40)

    def test_synthetic_geometry_to_finite_threshold_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary=run(Path(tmp))
            payload=json.loads((Path(tmp)/'synthetic_thresholds.json').read_text())
        self.assertGreater(summary['valid_segments'],0)
        self.assertIn('elbow',summary['primary_signal'])
        self.assertTrue(payload['metadata']['synthetic_fixture'])
        self.assertEqual(payload['metadata']['reps_analyzed'],summary['valid_segments'])
        self.assertIsNotNone(payload['depth_thresholds'])
        json.dumps(payload,allow_nan=False)


if __name__=='__main__': unittest.main()
