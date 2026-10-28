"""验证视频 I/O 写出前的数据检查与覆盖保护。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.video_depth.config import OutputConfig
from src.video_depth.types import VideoMetadata
from src.video_depth.video_io import probe_video, validate_depth_frames, write_depth_video


class VideoIOTest(unittest.TestCase):
    def test_validates_shape_dtype_and_values(self) -> None:
        frames = np.ones((2, 3, 5), dtype=np.float32)
        validate_depth_frames(frames, expected_frames=2, height=3, width=5)

        with self.assertRaisesRegex(ValueError, "float32"):
            validate_depth_frames(frames.astype(np.float64), 2, 3, 5)
        broken = frames.copy()
        broken[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            validate_depth_frames(broken, 2, 3, 5)

    def test_output_fixture_uses_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(Path(directory).is_dir())

    def test_refuses_overwrite_before_starting_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "depth.mp4"
            output.write_bytes(b"existing")
            frames = np.ones((1, 3, 5), dtype=np.float32)
            metadata = VideoMetadata(5, 3, 1, 30.0, [0.0])
            with self.assertRaises(FileExistsError):
                write_depth_video(frames, output, metadata, OutputConfig())

    def test_corrupt_video_has_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.mp4"
            broken.write_bytes(b"not a video")
            with self.assertRaises(RuntimeError):
                probe_video(broken)


if __name__ == "__main__":
    unittest.main()
