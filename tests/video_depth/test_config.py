"""验证视频深度配置加载和跨字段约束。"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.video_depth.config import VideoDepthConfig


class VideoDepthConfigTest(unittest.TestCase):
    def test_default_yaml_uses_refreshed_nested_checkpoint(self) -> None:
        config = VideoDepthConfig.from_yaml(Path("configs/da3_nested_rife.yaml"))
        self.assertEqual(
            config.da3.model_id,
            "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        )
        self.assertEqual(config.strategy, "da3_nested_rife")

    def test_rejects_invalid_stride_and_overlap(self) -> None:
        config = VideoDepthConfig.from_dict({"sampling": {"stride": 0}})
        with self.assertRaisesRegex(ValueError, "stride"):
            config.validate()

        config = VideoDepthConfig.from_dict(
            {"da3": {"window_size": 4, "overlap": 4}}
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
