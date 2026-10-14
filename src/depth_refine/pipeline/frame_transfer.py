"""逐帧读取参考、拟合与应用映射，保存搬运矩阵和墙钟耗时。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..methods.mapping import _valid_pair, _fit_mapping, _apply_mapping


FRAME_TIMING_FIELDS = [
    "frame_index", "method", "status", "valid_pixels", "reference_load_seconds",
    "pair_seconds", "fit_seconds", "apply_seconds", "artifact_write_seconds",
    "total_seconds", "artifact_path", "matrix_path", "error",
]


def require_frame_references(reference_dir: Path, indices: list[int],
                             num_frames: int, *, full: bool) -> list[int]:
    """检查同帧参考；full 要求每个源帧都有 manifest 索引和 NPY。"""
    if num_frames <= 0:
        raise ValueError("source video has no frames")
    if len(set(indices)) != len(indices) or any(i < 0 or i >= num_frames for i in indices):
        raise ValueError("baseline frame indices are duplicated or outside source video")
    requested = list(range(num_frames)) if full else sorted(indices)
    declared = set(indices)
    missing = [i for i in requested if i not in declared
               or not (reference_dir / f"frame_{i:09d}.npy").is_file()]
    if missing:
        raise ValueError(
            f"逐帧搬运缺少 {len(missing)} 个同帧 Nested 参考，前 10 个帧号: {missing[:10]}。"
            "请运行 build_baseline.py --sample-stride 1 --force 后重试。"
        )
    if not requested:
        raise ValueError("baseline contains no reference frames")
    return requested


def _save_mapping(path: Path, frame_index: int, mapping: dict) -> None:
    if mapping["method"] == "transport":
        data = {key: np.asarray(value) for key, value in mapping["transport"].items()}
        data["mapped_depth_mm"] = np.asarray(mapping["mapped_depth_mm"])
    elif mapping["method"] == "quantile":
        data = {key: np.asarray(mapping[key]) for key in
                ("source_quantiles_mm", "target_quantiles_mm")}
    else:
        data = {"a": np.asarray(mapping["a"]), "b_mm": np.asarray(mapping["b_mm"])}
    np.savez_compressed(path, frame_index=np.int64(frame_index), method=mapping["method"], **data)


def frame_timing_summary(records: list[dict]) -> dict:
    """统计已完成帧的实际耗时（秒），失败帧只计入 failed_frames。"""
    completed = [r for r in records if r["status"] == "completed"]
    summary = {"completed_frames": len(completed), "failed_frames": len(records) - len(completed)}
    for field in FRAME_TIMING_FIELDS:
        if not field.endswith("_seconds"):
            continue
        values = np.array([r[field] for r in completed], dtype=np.float64)
        summary[field] = ({"sum": float(values.sum()), "mean": float(values.mean()),
                           "min": float(values.min()), "p50": float(np.percentile(values, 50)),
                           "p95": float(np.percentile(values, 95)), "max": float(values.max())}
                          if values.size else None)
    return summary


def transfer_frames(source: list[np.ndarray], reference_dir: Path, indices: list[int],
                    report_dir: Path, *, method: str, quantiles: int = 101,
                    transport_bins: int = 256, max_depth_mm: float = 4095) -> tuple[list[np.ndarray], list[dict]]:
    """每个索引独立拟合和保存一次；返回输出帧与逐帧参数/计时记录。

    帧总耗时包括参考读取、配对、拟合、应用、NPZ 写盘；不含视频解码、
    评估指标、趋势图、preencode 保存及视频编码，这些仍由 episode_timer 记录。
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = report_dir / "frame_mappings"
    artifact_dir.mkdir(exist_ok=True)
    outputs, records = [], []
    # CSV 每帧写入并 flush；中途异常也会留下已处理帧和失败帧的计时。
    with (report_dir / "frame_timings.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAME_TIMING_FIELDS, extrasaction="ignore")
        writer.writeheader()
        try:
            for frame_index in indices:
                record = dict.fromkeys(FRAME_TIMING_FIELDS, "")
                record.update(frame_index=frame_index, method=method, status="failed", valid_pixels=0)
                start = perf_counter()
                try:
                    # 一、读取同帧参考，并沿用原来的 resize 和有效像素交集规则。
                    phase = perf_counter()
                    nested = np.load(reference_dir / f"frame_{frame_index:09d}.npy", allow_pickle=False)
                    record["reference_load_seconds"] = perf_counter() - phase
                    phase = perf_counter()
                    x, y = _valid_pair(source[frame_index], nested)
                    record["pair_seconds"] = perf_counter() - phase
                    record["valid_pixels"] = int(x.size)
                    if not x.size:
                        raise ValueError(f"frame {frame_index}: no valid source/target pairs")

                    # 二、每帧独立拟合；不拼接其它帧，也不复用邻帧/全局矩阵。
                    phase = perf_counter()
                    mapping = _fit_mapping(x, y, method, quantiles, transport_bins)
                    record["fit_seconds"] = perf_counter() - phase
                    phase = perf_counter()
                    output = _apply_mapping(source[frame_index], mapping, max_depth_mm)
                    record["apply_seconds"] = perf_counter() - phase

                    # 三、保存可复算的矩阵/参数；报告里只留小型映射和文件路径。
                    phase = perf_counter()
                    artifact = artifact_dir / f"frame_{frame_index:09d}.npz"
                    _save_mapping(artifact, frame_index, mapping)
                    record["artifact_write_seconds"] = perf_counter() - phase
                    record["artifact_path"] = artifact.relative_to(report_dir).as_posix()
                    if method == "transport":
                        record["matrix_path"] = record["artifact_path"]
                    record["mapping"] = {k: v for k, v in mapping.items() if k != "transport"}
                    record["status"] = "completed"
                    outputs.append(output)
                except Exception as exc:
                    # 保留失败原因后重新抛出，调用方不会把未完成序列编码成完整视频。
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    record["total_seconds"] = perf_counter() - start
                    records.append(record)
                    writer.writerow(record)
                    stream.flush()
        finally:
            (report_dir / "frame_timing_summary.json").write_text(
                json.dumps(frame_timing_summary(records), indent=2) + "\n", encoding="utf-8")
    return outputs, records
