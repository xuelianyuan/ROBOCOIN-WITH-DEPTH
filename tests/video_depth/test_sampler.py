"""验证关键帧首尾、尾段和场景切换边界。"""

from __future__ import annotations

import unittest

import numpy as np

from src.video_depth.sampler import detect_scene_changes, select_keyframes


class SamplerTest(unittest.TestCase):
    def test_keyframes_are_unique_and_include_both_ends(self) -> None:
        self.assertEqual(select_keyframes(1, stride=5), [0])
        self.assertEqual(select_keyframes(8, stride=3), [0, 3, 6, 7])
        self.assertEqual(select_keyframes(3, stride=10), [0, 2])

    def test_scene_boundary_is_promoted(self) -> None:
        indices = select_keyframes(10, stride=5, scene_boundaries=[4])
        self.assertEqual(indices, [0, 3, 4, 5, 9])

    def test_detects_abrupt_scene_change(self) -> None:
        frames = [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8, 3), 255, dtype=np.uint8),
        ]
        self.assertEqual(detect_scene_changes(frames, threshold=0.3), [2])

    def test_rejects_non_positive_stride(self) -> None:
        with self.assertRaises(ValueError):
            select_keyframes(3, stride=0)


if __name__ == "__main__":
    unittest.main()
