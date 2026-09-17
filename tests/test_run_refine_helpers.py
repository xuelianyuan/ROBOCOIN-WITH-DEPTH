import json

import numpy as np

from scripts.depth_refine.run_refine import (
    _mapping_slope_report,
    _save_preencode_frames,
)


def test_mapping_slope_report_finds_steepest_quantile_segment():
    mapping = {
        "method": "quantile",
        "source_quantiles_mm": [0, 10, 20, 40],
        "target_quantiles_mm": [0, 10, 40, 60],
    }

    report = _mapping_slope_report(mapping, top_k=2)

    assert report["segment_count"] == 3
    assert report["slope_max"] == 3
    assert report["segments_above_2"] == 1
    assert report["steepest_segments"][0]["source_start_mm"] == 10
    assert report["steepest_segments"][0]["source_end_mm"] == 20


def test_save_preencode_frames_preserves_float_values(tmp_path):
    frames = [np.array([[1.25, 2.75]], dtype=np.float64)]

    _save_preencode_frames(frames, tmp_path / "episode_000000")

    saved = np.load(tmp_path / "episode_000000" / "frame_000000000.npy")
    manifest = json.loads((tmp_path / "episode_000000" / "manifest.json").read_text())
    assert saved.dtype == np.float32
    np.testing.assert_array_equal(saved, np.array([[1.25, 2.75]], dtype=np.float32))
    assert manifest["frame_count"] == 1
    assert manifest["frames"] == ["frame_000000000.npy"]
