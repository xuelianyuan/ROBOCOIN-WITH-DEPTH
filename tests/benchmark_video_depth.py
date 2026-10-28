#!/usr/bin/env python3
"""
benchmark_video_depth.py — 视频深度质量与 1/2/4 卡性能评测入口

以“每张 GPU 独占一个模型实例和一组视频”的任务级并行执行；输出结构化
``metrics.json`` 与 ``timings.csv``，不作为普通 pytest 用例运行。
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.video_depth import VideoDepthConfig, VideoDepthPipeline

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评测 DA3 Nested + RIFE 视频深度")
    parser.add_argument("input_dir", type=Path, help="固定评测视频目录")
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "da3_nested_rife.yaml",
    )
    parser.add_argument("--gpu-count", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument(
        "--devices", help="物理 GPU ID，逗号分隔；默认从 0 开始，例如 0,1,2,3"
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--ground-truth-dir", type=Path, help="可选 <video_stem>.npz 真值目录")
    parser.add_argument("--run-id", help="可追溯运行 ID；默认包含日期、卡数和 stride")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = VideoDepthConfig.from_yaml(args.config)
    if args.repetitions < 1:
        raise ValueError("repetitions 必须 >= 1")
    devices = (
        [int(item) for item in args.devices.split(",")]
        if args.devices
        else list(range(args.gpu_count))
    )
    if len(devices) != args.gpu_count:
        raise ValueError("--devices 数量必须等于 --gpu-count")
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpu_count:
        raise RuntimeError(
            f"需要 {args.gpu_count} 张可用 GPU，PyTorch 当前检测到 {torch.cuda.device_count()} 张"
        )
    videos = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError(f"没有评测视频: {args.input_dir}")
    if len(videos) < args.gpu_count:
        raise ValueError("评测视频数必须不少于 GPU 数，避免空闲 worker 扭曲扩展效率")
    run_id = args.run_id or (
        f"{datetime.now():%Y%m%d_%H%M%S}_{args.gpu_count}xgpu_"
        f"stride{config.sampling.stride}"
    )
    result_dir = _PROJECT_ROOT / "benchmarks" / "results" / run_id
    if result_dir.exists() and any(result_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"评测目录已存在: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)

    assignments = [videos[index :: args.gpu_count] for index in range(args.gpu_count)]
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for worker_index, (device_id, assigned) in enumerate(zip(devices, assignments)):
        process = context.Process(
            target=_worker,
            args=(
                worker_index,
                device_id,
                [str(path) for path in assigned],
                config.to_dict(),
                args.repetitions,
                str(result_dir),
                str(args.ground_truth_dir) if args.ground_truth_dir else None,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failed = [process.pid for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"benchmark worker 失败: pid={failed}")

    rows: list[dict[str, Any]] = []
    for worker_index in range(args.gpu_count):
        worker_path = result_dir / f"worker_{worker_index}.json"
        rows.extend(json.loads(worker_path.read_text(encoding="utf-8")))
    _write_timings(result_dir / "timings.csv", rows)
    metrics = _summarize(rows, args.gpu_count, config, devices)
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def _worker(
    worker_index: int,
    device_id: int,
    videos: list[str],
    config_dict: dict[str, Any],
    repetitions: int,
    result_dir_string: str,
    ground_truth_dir_string: str | None,
) -> None:
    """单 GPU worker：只构造一个流水线实例并独占输出目录。"""

    config = VideoDepthConfig.from_dict(config_dict)
    config.device = f"cuda:{device_id}"
    config.validate()
    torch.cuda.set_device(device_id)
    pipeline = VideoDepthPipeline(config)
    result_dir = Path(result_dir_string)
    ground_truth_dir = Path(ground_truth_dir_string) if ground_truth_dir_string else None
    rows: list[dict[str, Any]] = []
    for video_string in videos:
        video = Path(video_string)
        ground_truth = _load_ground_truth(ground_truth_dir, video.stem)
        for repetition in range(repetitions):
            torch.cuda.reset_peak_memory_stats(device_id)
            output = result_dir / "outputs" / video.stem / f"rep_{repetition}" / "depth.mp4"
            result = pipeline.run(video, output, overwrite=True)
            torch.cuda.synchronize(device_id)
            row: dict[str, Any] = {
                "worker": worker_index,
                "device": device_id,
                "video": video.name,
                "repetition": repetition,
                "phase": "cold" if not rows else "steady",
                "frames": result.metadata.frame_count,
                "decode_s": result.timings_s["decode"],
                "inference_s": result.timings_s["inference"],
                "encode_s": result.timings_s["encode"],
                "total_s": result.timings_s["total"],
                "fps": result.metadata.frame_count / result.timings_s["total"],
                "peak_vram_mib": torch.cuda.max_memory_allocated(device_id) / 1024**2,
                "clip_ratio": result.clip_ratio,
                "output_bytes": output.stat().st_size,
            }
            if ground_truth is not None:
                row["quality"] = _quality_metrics(
                    result.depth_m, ground_truth, result.keyframe_indices
                )
            rows.append(row)
    (result_dir / f"worker_{worker_index}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_ground_truth(directory: Path | None, stem: str) -> np.ndarray | None:
    if directory is None:
        return None
    path = directory / f"{stem}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"缺少真值: {path}")
    with np.load(path) as payload:
        key = "depth_m" if "depth_m" in payload else "depth"
        return np.asarray(payload[key], dtype=np.float32)


def _quality_metrics(
    prediction: np.ndarray, target: np.ndarray, key_indices: list[int]
) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError(f"预测/真值 shape 不同: {prediction.shape} != {target.shape}")
    key_mask = np.zeros(prediction.shape[0], dtype=bool)
    key_mask[key_indices] = True
    metrics: dict[str, float] = {}
    for label, time_mask in (
        ("all", np.ones_like(key_mask)),
        ("keyframe", key_mask),
        ("interpolated", ~key_mask),
    ):
        pred = prediction[time_mask]
        truth = target[time_mask]
        valid = np.isfinite(pred) & np.isfinite(truth) & (pred > 0) & (truth > 0)
        if not valid.any():
            continue
        p, t = pred[valid], truth[valid]
        error = p - t
        ratio = np.maximum(p / t, t / p)
        values = {
            "abs_rel": np.mean(np.abs(error) / t),
            "sq_rel": np.mean(error**2 / t),
            "rmse": np.sqrt(np.mean(error**2)),
            "rmse_log": np.sqrt(np.mean((np.log(p) - np.log(t)) ** 2)),
            "delta1": np.mean(ratio < 1.25),
            "delta2": np.mean(ratio < 1.25**2),
            "delta3": np.mean(ratio < 1.25**3),
            "scale_bias": np.median(p / t),
            "invalid_ratio": 1.0 - valid.mean(),
        }
        metrics.update({f"{label}_{name}": float(value) for name, value in values.items()})
    return metrics


def _write_timings(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "worker",
        "device",
        "video",
        "repetition",
        "phase",
        "frames",
        "decode_s",
        "inference_s",
        "encode_s",
        "total_s",
        "fps",
        "peak_vram_mib",
        "clip_ratio",
        "output_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    rows: list[dict[str, Any]],
    gpu_count: int,
    config: VideoDepthConfig,
    devices: list[int],
) -> dict[str, Any]:
    steady = [row for row in rows if row["phase"] == "steady"] or rows
    quality_keys = sorted(
        {key for row in rows for key in row.get("quality", {}).keys()}
    )
    quality = {
        key: statistics.median(
            row["quality"][key]
            for row in rows
            if key in row.get("quality", {})
        )
        for key in quality_keys
    }
    worker_throughputs = []
    for worker in sorted({row["worker"] for row in steady}):
        worker_rows = [row for row in steady if row["worker"] == worker]
        worker_throughputs.append(
            sum(row["frames"] for row in worker_rows)
            / sum(row["total_s"] for row in worker_rows)
        )
    return {
        "strategy": config.strategy,
        "model_id": config.da3.model_id,
        "gpu_count": gpu_count,
        "devices": devices,
        "gpu_names": [torch.cuda.get_device_name(device) for device in devices],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "git_commit": _git_commit(),
        "config": config.to_dict(),
        "runs": len(rows),
        "steady_runs": len(steady),
        "median_video_fps": statistics.median(row["fps"] for row in steady),
        "aggregate_worker_fps": sum(worker_throughputs),
        "median_latency_s": statistics.median(row["total_s"] for row in steady),
        "peak_vram_mib_max": max(row["peak_vram_mib"] for row in rows),
        "quality_median": quality,
        "note": "多卡采用独立视频任务级并行；aggregate_worker_fps 不是单视频延迟。",
    }


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
