"""使用 fake 模型验证总流水线的数据契约和资源分支。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.video_depth.config import VideoDepthConfig
from src.video_depth.da3_nested import DA3NestedEstimator
from src.video_depth.pipeline import VideoDepthPipeline
from src.video_depth.types import KeyframeDepth


class _FakeDA3:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, frames):
        self.calls += 1
        h, w = frames[0].shape[:2]
        values = [float(frame[0, 0, 0]) / 10.0 + 1.0 for frame in frames]
        depth = np.stack([np.full((h, w), v, np.float32) for v in values])
        return KeyframeDepth(depth_m=depth, confidence=np.ones_like(depth))


class _FakeRIFE:
    def __init__(self) -> None:
        self.calls = 0

    def interpolate(self, frames, key_indices, key_depth_m, key_confidence):
        self.calls += 1
        n = len(frames)
        result = np.empty((n, *key_depth_m.shape[1:]), np.float32)
        confidence = np.empty_like(result)
        for pair in range(len(key_indices) - 1):
            i0, i1 = key_indices[pair : pair + 2]
            for index in range(i0, i1 + 1):
                t = (index - i0) / (i1 - i0)
                result[index] = key_depth_m[pair] * (1 - t) + key_depth_m[pair + 1] * t
                confidence[index] = 1.0
        return result, confidence


class _FakeNestedModel:
    def __init__(self) -> None:
        self.inference_calls = 0
        self.eval_calls = 0

    def to(self, _device):
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def inference(self, frames, **_kwargs):
        self.inference_calls += 1
        values = np.asarray([frame[0, 0, 0] + 1.0 for frame in frames], np.float32)
        depth = np.repeat(values[:, None, None], 4 * 6, axis=1).reshape(-1, 4, 6)
        return SimpleNamespace(depth=depth, conf=np.ones_like(depth), is_metric=1)


def _frames(count: int) -> list[np.ndarray]:
    return [np.full((5, 7, 3), i, dtype=np.uint8) for i in range(count)]


class VideoDepthPipelineTest(unittest.TestCase):
    def test_empty_video_is_rejected_and_single_frame_bypasses_rife(self) -> None:
        config = VideoDepthConfig.from_dict(
            {"device": "cpu", "sampling": {"stride": 5}}
        )
        da3, rife = _FakeDA3(), _FakeRIFE()
        pipeline = VideoDepthPipeline(config, da3=da3, rife=rife)
        with self.assertRaisesRegex(ValueError, "空视频"):
            pipeline.estimate_frames([], fps=30.0)
        result = pipeline.estimate_frames(_frames(1), fps=30.0)
        self.assertEqual(result.keyframe_indices, [0])
        self.assertEqual(rife.calls, 0)

    def test_da3_windows_keep_temporal_order_and_load_model_once(self) -> None:
        config = VideoDepthConfig.from_dict(
            {
                "device": "cpu",
                "da3": {
                    "window_size": 4,
                    "overlap": 2,
                    "align_scale": False,
                },
            }
        )
        model = _FakeNestedModel()
        estimator = DA3NestedEstimator(
            config.da3, "cpu", model_factory=lambda _model_id: model
        )
        frames = [np.full((5, 7, 3), i, np.uint8) for i in range(7)]
        result = estimator.infer(frames)
        self.assertEqual(result.depth_m.shape, (7, 5, 7))
        np.testing.assert_allclose(result.depth_m[:, 0, 0], np.arange(1, 8))
        self.assertEqual(model.inference_calls, 3)
        self.assertEqual(model.eval_calls, 1)

    def test_stride_one_bypasses_rife(self) -> None:
        config = VideoDepthConfig.from_dict(
            {"device": "cpu", "sampling": {"stride": 1}}
        )
        da3, rife = _FakeDA3(), _FakeRIFE()
        result = VideoDepthPipeline(config, da3=da3, rife=rife).estimate_frames(
            _frames(4), fps=24.0
        )
        self.assertEqual(result.depth_m.shape, (4, 5, 7))
        self.assertEqual(result.unit, "m")
        self.assertEqual(rife.calls, 0)
        self.assertEqual(result.keyframe_indices, [0, 1, 2, 3])

    def test_sampled_path_preserves_da3_keyframes(self) -> None:
        config = VideoDepthConfig.from_dict(
            {
                "device": "cpu",
                "sampling": {"stride": 3, "scene_change_threshold": 0.0},
            }
        )
        da3, rife = _FakeDA3(), _FakeRIFE()
        result = VideoDepthPipeline(config, da3=da3, rife=rife).estimate_frames(
            _frames(8), fps=30.0
        )
        self.assertEqual(result.keyframe_indices, [0, 3, 6, 7])
        for local, index in enumerate(result.keyframe_indices):
            expected = np.full((5, 7), float(index) / 10.0 + 1.0, np.float32)
            np.testing.assert_array_equal(result.depth_m[index], expected)
        self.assertEqual(rife.calls, 1)
        self.assertEqual(result.depth_m.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
