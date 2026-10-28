"""验证双向深度 warp、无效像素融合和关键帧保持。"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.video_depth.rife_depth import RIFEDepthInterpolator


class _ZeroMotion:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _rgb0, _rgb1, _timestep):
        self.calls += 1
        flow = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        return flow, mask


class RIFEDepthInterpolatorTest(unittest.TestCase):
    def test_interpolates_valid_depth_and_preserves_keyframes(self) -> None:
        motion = _ZeroMotion()
        interpolator = RIFEDepthInterpolator(
            device="cpu", motion_predictor=motion, pad_multiple=1
        )
        rgb = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
        first = np.full((4, 4), 1.0, dtype=np.float32)
        last = np.full((4, 4), 3.0, dtype=np.float32)

        depth, confidence = interpolator.interpolate(
            rgb, [0, 2], np.stack([first, last]), np.ones((2, 4, 4), np.float32)
        )

        self.assertTrue(np.array_equal(depth[0], first))
        self.assertTrue(np.array_equal(depth[2], last))
        np.testing.assert_allclose(depth[1], 2.0, atol=1e-6)
        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(confidence.shape, depth.shape)
        self.assertEqual(motion.calls, 1)

    def test_invalid_zero_depth_is_not_averaged_as_near_surface(self) -> None:
        interpolator = RIFEDepthInterpolator(
            device="cpu", motion_predictor=_ZeroMotion(), pad_multiple=1
        )
        rgb = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
        first = np.zeros((4, 4), dtype=np.float32)
        last = np.full((4, 4), 4.0, dtype=np.float32)
        depth, _ = interpolator.interpolate(
            rgb, [0, 2], np.stack([first, last]), np.ones((2, 4, 4), np.float32)
        )
        np.testing.assert_allclose(depth[1], 4.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
