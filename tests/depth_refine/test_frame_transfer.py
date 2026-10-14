import csv
import json

import numpy as np
import pytest

from scripts.depth_refine import run_refine
from scripts.depth_refine.summarize_timings import summarize_frames
from src.depth_refine.methods.histogram_transport import fit_histogram_transport
from src.depth_refine.pipeline import frame_transfer


def _dense(plan):
    matrix = np.zeros(plan["shape"])
    np.add.at(matrix, (plan["row"], plan["col"]), plan["mass"])
    return matrix


def test_transport_splits_mass_and_uses_matrix_barycenter():
    mapping = fit_histogram_transport(np.array([10, 10, 20, 20]), np.array([100, 200, 200, 200]), bins=2)
    matrix = _dense(mapping["transport"])
    np.testing.assert_allclose(matrix, [[0.25, 0.25], [0, 0.5]])
    np.testing.assert_allclose(mapping["mapped_depth_mm"], [150, 200])
    np.testing.assert_allclose(matrix.sum(axis=1), mapping["transport"]["source_mass"])
    np.testing.assert_allclose(matrix.sum(axis=0), mapping["transport"]["target_mass"])


@pytest.mark.parametrize("bins", [2, 16, 256])
def test_transport_conserves_mass_and_is_monotone(bins):
    rng = np.random.default_rng(1)
    mapping = fit_histogram_transport(rng.uniform(1, 4000, 307), rng.uniform(1, 7000, 509), bins)
    plan = mapping["transport"]
    matrix = _dense(plan)
    np.testing.assert_allclose(matrix.sum(axis=1), plan["source_mass"], atol=1e-14)
    np.testing.assert_allclose(matrix.sum(axis=0), plan["target_mass"], atol=1e-14)
    assert matrix.sum() == pytest.approx(1)
    assert np.all(np.diff(mapping["mapped_depth_mm"]) >= -1e-9)
    assert np.all(np.diff(plan["row"]) >= 0) and np.all(np.diff(plan["col"]) >= 0)
    assert len(plan["mass"]) <= sum(plan["shape"]) - 1


def test_constant_depth_and_invalid_output_pixels():
    mapping = fit_histogram_transport(np.full(5, 10), np.full(5, 123.25))
    result = run_refine._apply_mapping(np.array([[10, 0, np.nan, -1, np.inf]]), mapping, 100)
    np.testing.assert_array_equal(result, [[100, 0, 0, 0, 0]])
    np.testing.assert_array_equal(_dense(mapping["transport"]), [[1]])


@pytest.mark.parametrize("x,y,bins", [([], [1], 2), ([0], [1], 2), ([1], [np.nan], 2), ([1], [1], 1)])
def test_invalid_transport_inputs_fail(x, y, bins):
    with pytest.raises(ValueError):
        fit_histogram_transport(np.array(x), np.array(y), bins)


def test_frames_have_independent_matrices_and_actual_phase_timings(tmp_path, monkeypatch):
    reference_dir = tmp_path / "references"
    reference_dir.mkdir()
    for i, value in enumerate([100, 300]):
        np.save(reference_dir / f"frame_{i:09d}.npy", np.full((1, 2), value))
    # 固定时钟使统计范围可验证，不依赖运行速度或总耗时除以帧数。
    ticks = iter(range(100))
    monkeypatch.setattr(frame_transfer, "perf_counter", lambda: next(ticks))
    source = [np.full((1, 2), 10), np.full((1, 2), 10)]
    outputs, records = frame_transfer.transfer_frames(source, reference_dir, [0, 1], tmp_path / "reports", method="transport")
    np.testing.assert_array_equal(outputs[0], [[100, 100]])
    np.testing.assert_array_equal(outputs[1], [[300, 300]])
    for i, record in enumerate(records):
        with np.load(tmp_path / "reports" / record["matrix_path"]) as plan:
            assert plan["frame_index"] == i
            np.testing.assert_allclose(_dense(plan).sum(), 1)
        for field in ["reference_load_seconds", "pair_seconds", "fit_seconds", "apply_seconds", "artifact_write_seconds"]:
            assert record[field] == 1
        assert record["total_seconds"] == 11
    with (tmp_path / "reports/frame_timings.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["frame_index"]) for row in rows] == [0, 1]
    summary = json.loads((tmp_path / "reports/frame_timing_summary.json").read_text())
    assert summary["completed_frames"] == 2
    assert summary["total_seconds"]["sum"] == 22
    aggregated = summarize_frames([dict(camera="head", chunk="chunk-000", episode="episode_000000", stage="refine_full",
                                       frame_timings_csv=str(tmp_path / "reports/frame_timings.csv"))])
    assert len(aggregated) == 2 and aggregated[1]["stage"] == "refine_full"


def test_in_memory_dense_references_have_independent_mappings(tmp_path):
    source = [np.full((2, 2), 10), np.full((2, 2), 20)]
    references = [np.full((2, 2), 100), np.full((2, 2), 300)]
    outputs, records = frame_transfer.transfer_array_frames(
        source, references, [0, 1], tmp_path / "reports", method="transport")
    np.testing.assert_array_equal(outputs[0], np.full((2, 2), 100))
    np.testing.assert_array_equal(outputs[1], np.full((2, 2), 300))
    assert len(records) == 2
    assert all((tmp_path / "reports" / r["matrix_path"]).is_file() for r in records)


def test_in_memory_dense_references_require_matching_frame_count(tmp_path):
    with pytest.raises(ValueError, match="frame count mismatch"):
        frame_transfer.transfer_array_frames(
            [np.ones((1, 1))] * 2, [np.ones((1, 1))], [0, 1],
            tmp_path / "reports", method="transport")


def test_no_valid_pairs_preserves_failed_frame_timing(tmp_path):
    np.save(tmp_path / "frame_000000000.npy", np.ones((1, 2)))
    with pytest.raises(ValueError, match="frame 0: no valid"):
        frame_transfer.transfer_frames([np.zeros((1, 2))], tmp_path, [0], tmp_path / "reports", method="transport")
    with (tmp_path / "reports/frame_timings.csv").open(encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "failed" and float(row["total_seconds"]) > 0
    summary = json.loads((tmp_path / "reports/frame_timing_summary.json").read_text())
    assert summary["failed_frames"] == 1 and summary["total_seconds"] is None


def test_full_requires_every_declared_reference_and_rejects_stale_npy(tmp_path):
    for i in range(3):
        np.save(tmp_path / f"frame_{i:09d}.npy", np.ones((1, 1)))
    assert frame_transfer.require_frame_references(tmp_path, [2, 0], 3, full=False) == [0, 2]
    with pytest.raises(ValueError, match="sample-stride 1"):
        frame_transfer.require_frame_references(tmp_path, [0, 2], 3, full=True)
    assert frame_transfer.require_frame_references(tmp_path, [0, 1, 2], 3, full=True) == [0, 1, 2]
    with pytest.raises(ValueError, match="outside source"):
        frame_transfer.require_frame_references(tmp_path, [-1, 0], 3, full=False)
    with pytest.raises(ValueError, match="duplicated"):
        frame_transfer.require_frame_references(tmp_path, [0, 0], 3, full=False)
    (tmp_path / "frame_000000001.npy").unlink()
    with pytest.raises(ValueError, match="缺少 1"):
        frame_transfer.require_frame_references(tmp_path, [0, 1, 2], 3, full=True)


@pytest.mark.parametrize("method", ["quantile", "linear"])
def test_legacy_methods_can_fit_separately_per_frame(tmp_path, method):
    source = [np.arange(1, 21).reshape(4, 5)] * 2
    for i, offset in enumerate([100, 300]):
        np.save(tmp_path / f"frame_{i:09d}.npy", source[i] + offset)
    outputs, records = frame_transfer.transfer_frames(source, tmp_path, [0, 1], tmp_path / "reports", method=method)
    np.testing.assert_allclose(outputs[0], source[0] + 100)
    np.testing.assert_allclose(outputs[1], source[1] + 300)
    assert all(r["matrix_path"] == "" for r in records)


@pytest.mark.parametrize("stage", ["validate", "full"])
def test_cli_default_is_frame_transport_and_full_does_not_apply_twice(tmp_path, monkeypatch, stage):
    rgb = tmp_path / "videos/chunk-000/observation.images.head_rgb/episode_000000.mp4"
    depth = rgb.parent.parent / "observation.images.head_depth/episode_000000.mp4"
    rgb.parent.mkdir(parents=True)
    depth.parent.mkdir()
    rgb.touch()
    depth.touch()
    references = tmp_path / "depth_results/baseline/statistics/head/episode_000000"
    references.mkdir(parents=True)
    (references / "sample_indices.json").write_text(json.dumps({"sample_frame_indices": [0, 1]}))
    source = [np.full((2, 2), 10), np.full((2, 2), 20)]
    for i, value in enumerate([100, 300]):
        np.save(references / f"frame_{i:09d}.npy", np.full((2, 2), value))
    monkeypatch.setattr(run_refine, "_decode", lambda *args: source)
    encoded = []
    monkeypatch.setattr(run_refine.DepthVideoEncoder, "encode_from_arrays", lambda self, arrays, path: encoded.append(arrays) or 0)
    # full 若再次调用脚本里的 apply_mapping 就会失败；只能复用 pipeline 输出。
    monkeypatch.setattr(run_refine, "_apply_mapping", lambda *args: pytest.fail("duplicate mapping"))
    monkeypatch.setattr("sys.argv", ["run_refine.py", "--dataset-dir", str(tmp_path), "--camera", "head", "--stage", stage,
                                     "--allow-unvalidated", "--report-mapping-slope"])
    run_refine.main()
    report_dir = tmp_path / "depth_results/refine/reports/head/episode_000000"
    report = json.loads((report_dir / "refine_parameters.json").read_text())
    assert report["mapping_scope"] == "frame" and report["mapping"]["method"] == "transport"
    assert len(report["frame_mappings"]) == 2
    assert all((report_dir / r["matrix_path"]).is_file() for r in report["frame_mappings"])
    assert (report_dir / report["frame_timings_csv"]).is_file()
    if stage == "full":
        assert len(encoded) == 1
        np.testing.assert_array_equal(encoded[0][0], np.full((2, 2), 100))
        np.testing.assert_array_equal(encoded[0][1], np.full((2, 2), 300))
    else:
        assert not encoded


def test_cli_full_rejects_sparse_reference_and_episode_mode_remains_available(tmp_path, monkeypatch):
    rgb = tmp_path / "videos/chunk-000/observation.images.head_rgb/episode_000000.mp4"
    depth = rgb.parent.parent / "observation.images.head_depth/episode_000000.mp4"
    rgb.parent.mkdir(parents=True)
    depth.parent.mkdir()
    rgb.touch()
    depth.touch()
    references = tmp_path / "depth_results/baseline/statistics/head/episode_000000"
    references.mkdir(parents=True)
    (references / "sample_indices.json").write_text(json.dumps({"sample_frame_indices": [0]}))
    np.save(references / "frame_000000000.npy", np.full((2, 2), 100))
    monkeypatch.setattr(run_refine, "_decode", lambda *args: [np.full((2, 2), 10), np.full((2, 2), 20)])
    encoded = []
    monkeypatch.setattr(run_refine.DepthVideoEncoder, "encode_from_arrays", lambda self, arrays, path: encoded.append(arrays) or 0)
    args = ["run_refine.py", "--dataset-dir", str(tmp_path), "--camera", "head", "--stage", "full", "--allow-unvalidated"]
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(ValueError, match="缺少 1"):
        run_refine.main()
    assert not encoded
    monkeypatch.setattr("sys.argv", args + ["--mapping", "quantile", "--mapping-scope", "episode"])
    run_refine.main()
    assert len(encoded) == 1 and len(encoded[0]) == 2
    np.testing.assert_allclose(encoded[0], np.full((2, 2, 2), 100))


def test_cli_frame_rife_is_rejected_before_loading_models(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_refine.py", "--dataset-dir", "/unused", "--camera", "head", "--stage", "rife"])
    with pytest.raises(SystemExit) as error:
        run_refine.main()
    assert error.value.code == 2


def test_cli_rife_refine_interpolates_reference_then_maps_every_frame(tmp_path, monkeypatch):
    rgb_path = tmp_path / "videos/chunk-000/observation.images.head_rgb/episode_000000.mp4"
    depth_path = rgb_path.parent.parent / "observation.images.head_depth/episode_000000.mp4"
    rgb_path.parent.mkdir(parents=True)
    depth_path.parent.mkdir()
    rgb_path.touch()
    depth_path.touch()
    reference_dir = tmp_path / "depth_results/baseline/statistics/head/episode_000000"
    reference_dir.mkdir(parents=True)
    (reference_dir / "sample_indices.json").write_text(
        json.dumps({"sample_frame_indices": [0, 2]}))
    np.save(reference_dir / "frame_000000000.npy", np.full((2, 2), 100))
    np.save(reference_dir / "frame_000000002.npy", np.full((2, 2), 300))

    source = [np.full((2, 2), value) for value in (10, 20, 30)]
    rgb = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
    monkeypatch.setattr(
        run_refine, "_decode", lambda path, pix_fmt: rgb if pix_fmt == "rgb24" else source)
    monkeypatch.setattr(
        run_refine, "_interpolate_nested_reference",
        lambda frames, path, indices: [np.full((2, 2), value) for value in (100, 200, 300)])
    encoded = []
    monkeypatch.setattr(
        run_refine.DepthVideoEncoder, "encode_from_arrays",
        lambda self, arrays, path: encoded.append(arrays) or 0)
    monkeypatch.setattr(
        "sys.argv", ["run_refine.py", "--dataset-dir", str(tmp_path),
                     "--camera", "head", "--stage", "rife_refine"])

    run_refine.main()

    assert len(encoded) == 1
    np.testing.assert_array_equal(
        encoded[0], np.array([np.full((2, 2), value) for value in (100, 200, 300)]))
    report_dir = tmp_path / "depth_results/refine/reports/head/episode_000000"
    report = json.loads((report_dir / "refine_parameters.json").read_text())
    assert report["reference"]["kind"] == "rife_interpolated_nested"
    assert report["reference"]["anchor_indices"] == [0, 2]
    assert len(report["frame_mappings"]) == 3


def test_rife_refine_requires_first_and_last_nested_anchors(tmp_path):
    rgb = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
    with pytest.raises(ValueError, match="frame 0 and the final frame"):
        run_refine._interpolate_nested_reference(rgb, tmp_path, [0, 1])
