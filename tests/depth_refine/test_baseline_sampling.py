"""Verify temporal sampling and source-frame metadata without loading DA3."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.depth_refine import build_baseline, run_batch


class BaselineSamplingTests(unittest.TestCase):
    def test_stride_boundaries(self):
        for length, expected in [(0, []), (1, [0]), (4, [0]), (5, [0]),
                                 (6, [0, 5]), (12, [0, 5, 10])]:
            with self.subTest(length=length):
                self.assertEqual(build_baseline._sample_indices(length), expected)
        indices = build_baseline._sample_indices(1003)
        self.assertEqual(len(indices), 201)
        self.assertEqual(indices[-1], 1000)
        self.assertEqual(build_baseline._sample_indices(8, stride=3), [0, 3, 6])
        self.assertEqual(build_baseline._sample_indices(3, stride=1), [0, 1, 2])
        for stride in [0, -1]:
            with self.assertRaises(ValueError):
                build_baseline._sample_indices(0, stride=stride)

    def test_explicit_count_preserves_uniform_sampling(self):
        self.assertEqual(build_baseline._sample_indices(12, 3), [0, 5, 11])
        self.assertEqual(build_baseline._sample_indices(12, 1), [0])
        for count in [0, -1, 3, 20]:
            self.assertEqual(build_baseline._sample_indices(3, count), [0, 1, 2])

    def test_pipeline_passes_sampled_rgb_and_records_original_indices(self):
        cases = [([], [0, 5, 10], "stride", 5),
                 (["--sample-stride", "3"], [0, 3, 6, 9], "stride", 3),
                 (["--sample-count", "3"], [0, 5, 11], "count", None)]
        for options, expected, mode, stride in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                video = root / "videos/chunk-000/observation.images.head_rgb/episode_000000.mp4"
                video.parent.mkdir(parents=True)
                video.touch()
                frames = [np.full((4, 6, 3), i, dtype=np.uint8) for i in range(12)]
                with patch.object(build_baseline, "_decode_rgb_video", return_value=frames), \
                     patch.object(build_baseline, "create_depth_repair") as factory, \
                     patch.object(build_baseline, "DepthVideoEncoder") as encoder, \
                     patch("sys.argv", ["build_baseline", "--dataset-dir", str(root), "--save-video", *options]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    repair = factory.return_value
                    repair.repair_frames.side_effect = lambda rgb, depth: [
                        np.full((4, 6), 100 + int(frame[0, 0, 0]), dtype=np.float32)
                        for frame in rgb
                    ]
                    encoder.return_value.encode_from_arrays.return_value = 0
                    build_baseline.main()
                    selected, constraints = repair.repair_frames.call_args.args
                    self.assertEqual([int(frame[0, 0, 0]) for frame in selected], expected)
                    self.assertEqual(constraints, [])
                    output = root / "depth_results/baseline/statistics/head/episode_000000"
                    for filename, index_key in [("sample_indices.json", "sample_frame_indices"),
                                                ("manifest.json", "sample_frame_indices"),
                                                ("statistics.json", "sample_indices")]:
                        metadata = json.loads((output / filename).read_text())
                        self.assertEqual(metadata[index_key], expected)
                        self.assertEqual(metadata["sampling_mode"], mode)
                        self.assertEqual(metadata["sample_stride"], stride)
                    self.assertEqual(sorted(p.name for p in output.glob("frame_*.npy")),
                                     [f"frame_{index:09d}.npy" for index in expected])
                    for index in expected:
                        np.testing.assert_array_equal(np.load(output / f"frame_{index:09d}.npy"),
                                                      np.full((4, 6), 100 + index))
                    timing_path = root / "depth_results/timings/da3_nested/chunk-000/head/episode_000000.json"
                    timing = json.loads(timing_path.read_text())
                    self.assertEqual(timing["sample_num_frames"], len(expected))
                    self.assertEqual(timing["sample_stride"], stride)
                    # Existing results are reused until explicitly regenerated.
                    repair.repair_frames.reset_mock()
                    build_baseline.main()
                    repair.repair_frames.assert_not_called()
                    with patch("sys.argv", ["build_baseline", "--dataset-dir", str(root), "--force", *options]):
                        build_baseline.main()
                    repair.repair_frames.assert_called_once()

    def test_sampling_cli_rejects_invalid_stride_and_conflicting_modes(self):
        for module, required in [(build_baseline, ["--dataset-dir", "/unused"]),
                                 (run_batch, ["--root", "/unused"])]:
            for options in [["--sample-stride", "0"], ["--sample-stride", "-2"],
                            ["--sample-stride", "5", "--sample-count", "20"]]:
                with self.subTest(module=module.__name__, options=options), \
                     patch("sys.argv", ["script", *required, *options]), \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        module.main()
                    self.assertEqual(error.exception.code, 2)

    def test_batch_forwards_sampling_mode_and_force_only_to_baseline(self):
        for options, flag, value in [([], "--sample-stride", "5"),
                                     (["--sample-stride", "3"], "--sample-stride", "3"),
                                     (["--sample-count", "20"], "--sample-count", "20")]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "videos/chunk-000/observation.images.head_rgb").mkdir(parents=True)
                with patch("sys.argv", ["run_batch", "--root", str(root), "--cameras", "head",
                                        "--stage", "all", "--force", *options]), \
                     patch.object(run_batch.subprocess, "run") as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    run_batch.main()
                baseline, refine = [call.args[0] for call in run.call_args_list]
                self.assertEqual(baseline[baseline.index(flag) + 1], value)
                other_flag = "--sample-count" if flag == "--sample-stride" else "--sample-stride"
                self.assertNotIn(other_flag, baseline)
                self.assertIn("--force", baseline)
                self.assertNotIn("--force", refine)


if __name__ == "__main__":
    unittest.main()
