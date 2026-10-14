#!/usr/bin/env python3
"""RoboCOIN 深度分析工具。

读取 LeRobot 数据集中的 RGB 相机视频，默认对每个 episode 每 5 帧采样一次，
使用 DA3-Nested（或 DA3 Metric）估计深度，并保存：

* 采样深度帧（uint16 PNG/NPY，单位为毫米）；
* 全局及逐帧统计信息（``statistics.json``）；
* 可选的稀疏深度视频（12-bit H.265）。

该脚本只生成稀疏 baseline，不会伪造完整时间轴。结果默认写入
``<dataset-dir>/depth_results``，可交给 ``run_refine.py`` 继续验证或修复。

用法示例::

    python scripts/depth_refine/build_baseline.py --dataset-dir data/lerobot/my_dataset \\
        --camera head --sample-stride 5 --save-video
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.depth_refine.timing import episode_timer
from src.depth_encoder import DepthVideoEncoder
from src.depth_repair import create_depth_repair


def _sample_indices(n: int, count: int | None = None, *, stride: int = 5) -> list[int]:
    """默认按固定步长取源帧；显式传入 count 时沿用均匀定量采样。

    步长模式只取 ``0, stride, 2 * stride, ... < n``，不额外补尾帧。
    定量模式使用 ``linspace``；数量非正或超过总帧数时返回全部帧。
    """
    if count is None and stride <= 0:
        raise ValueError("sample stride must be positive")
    if n <= 0:
        return []
    if count is None:
        return list(range(0, n, stride))
    if count <= 0 or count >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, count, dtype=np.int64).tolist()))


def _decode_rgb_video(path: Path) -> list[np.ndarray]:
    """通过 ffprobe/ffmpeg 将 RGB 视频解码为 ``H x W x 3`` 的数组列表。

    这里使用原始字节流而不是临时图片文件，减少磁盘 I/O。探测或解码
    失败时返回空列表，由主流程负责跳过该视频。
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        width, height = map(int, probe.stdout.strip().split(","))
    except (ValueError, TypeError):
        return []
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=False,
    )
    if result.returncode:
        return []
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    size = width * height * 3
    return [raw[i:i + size].reshape(height, width, 3)
            for i in range(0, len(raw) - size + 1, size)]


def _stats(frames: list[np.ndarray]) -> dict:
    """汇总多帧深度的全局统计，只统计有限且大于零的像素（单位 mm）。"""
    if not frames:
        return {"count": 0, "valid_pixels": 0}
    values = np.concatenate([f[np.isfinite(f) & (f > 0)].ravel() for f in frames])
    if values.size == 0:
        return {"count": len(frames), "valid_pixels": 0}
    quantile_levels = np.linspace(0, 100, 101)
    return {
        "count": len(frames),
        "valid_pixels": int(values.size),
        "min_mm": float(values.min()),
        "max_mm": float(values.max()),
        "mean_mm": float(values.mean()),
        "std_mm": float(values.std()),
        "p01_mm": float(np.percentile(values, 1)),
        "p10_mm": float(np.percentile(values, 10)),
        "p50_mm": float(np.percentile(values, 50)),
        "p90_mm": float(np.percentile(values, 90)),
        "p99_mm": float(np.percentile(values, 99)),
        # 101 个分位点是一维最优传输/直方图匹配的紧凑表示。
        "quantile_levels": quantile_levels.tolist(),
        "quantiles_mm": np.percentile(values, quantile_levels).tolist(),
    }


def _frame_stats(frame: np.ndarray) -> dict:
    """计算单帧深度统计，字段与全局统计保持一致。"""
    valid = frame[np.isfinite(frame) & (frame > 0)]
    if valid.size == 0:
        return {"valid_pixels": 0}
    return {"valid_pixels": int(valid.size), "min_mm": float(valid.min()),
            "max_mm": float(valid.max()), "mean_mm": float(valid.mean()),
            "std_mm": float(valid.std()), "p10_mm": float(np.percentile(valid, 10)),
            "p50_mm": float(np.percentile(valid, 50)),
            "p90_mm": float(np.percentile(valid, 90)),
            "p99_mm": float(np.percentile(valid, 99))}


def main() -> None:
    """遍历数据集中的 RGB episode，运行稀疏 Nested 并保存分析产物。"""
    p = argparse.ArgumentParser(description="DA3-Nested 稀疏深度分析")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--camera", default="all", help="head/right/left/chest/all")
    sampling = p.add_mutually_exclusive_group()
    sampling.add_argument("--sample-stride", type=int, default=None,
                          help="源视频采样步长，必须大于 0，默认：每 5 帧一次")
    sampling.add_argument("--sample-count", type=int, default=None,
                          help="改用均匀定量采样；非正数表示全部帧，与 --sample-stride 互斥")
    p.add_argument("--strategy", default="da3_nested", choices=["da3_nested", "da3"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--result-root", default=None,
                   help="阶段化结果根目录；默认 <dataset>/depth_results")
    p.add_argument("--save-video", action="store_true", default=False)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=6)
    p.add_argument("--preset", default="medium")
    p.add_argument("--da3-model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    p.add_argument("--da3-process-res", type=int, default=504)
    p.add_argument("--da3-chunk-size", type=int, default=16)
    p.add_argument("--da3-overlap", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true", help="重新处理已有 statistics.json")
    args = p.parse_args()
    # 用 None 区分未传参和显式 --sample-stride 5，确保与 --sample-count 互斥。
    if args.sample_stride is None:
        args.sample_stride = 5
    if args.sample_count is None and args.sample_stride <= 0:
        p.error("--sample-stride 必须大于 0")
    sampling_config = {
        "sampling_mode": "stride" if args.sample_count is None else "count",
        "sample_stride": args.sample_stride if args.sample_count is None else None,
        "requested_sample_count": args.sample_count,
    }

    # 数据集通常包含 videos/、depth_statistics/ 等目录；所有相对路径
    # 均从用户传入的数据集根目录解析，便于把结果写入另一位置。
    root = Path(args.dataset_dir)
    result_root = Path(args.result_root) if args.result_root else root / "depth_results"
    stats_root = Path(args.output_dir) if args.output_dir else result_root / "baseline" / "statistics"
    videos_root = root / "videos"
    # LeRobot 版本差异较大：有的特征目录以 ``_rgb`` 结尾，有的直接使用
    # the camera feature name directly. Discover video directories from MP4s
    # and exclude generated depth variants instead of relying on one layout.
    rgb_dirs = sorted({p.parent for p in videos_root.glob("**/episode_*.mp4")
                       if "depth" not in p.parent.name.lower()})
    if args.camera != "all":
        rgb_dirs = [d for d in rgb_dirs if args.camera in d.name.lower()]
    if not rgb_dirs:
        raise FileNotFoundError(f"No RGB video dirs found under {videos_root}")

    # ``da3`` 策略默认切换到 Metric 模型；Nested 策略沿用用户指定模型。
    model_id = args.da3_model
    if args.strategy == "da3" and model_id == "depth-anything/DA3NESTED-GIANT-LARGE-1.1":
        model_id = "depth-anything/DA3METRIC-LARGE"
    repair = create_depth_repair(
        "da3", model_id=model_id, process_res=args.da3_process_res,
        chunk_size=args.da3_chunk_size, overlap=args.da3_overlap, device=args.device,
    )
    encoder = DepthVideoEncoder(fps=args.fps, crf=args.crf, preset=args.preset)

    # 每个相机目录下逐 episode 处理，并用 statistics.json 作为完成标记。
    for rgb_dir in rgb_dirs:
        camera_name = rgb_dir.name.replace("observation.images.", "").replace("_rgb", "")
        if camera_name == rgb_dir.name and camera_name.endswith("rgb"):
            camera_name = camera_name[:-3].rstrip("_.")
        if args.output_dir:
            sparse_dir = rgb_dir.parent / f"observation.images.{camera_name}_depth_nested_sparse"
        else:
            sparse_dir = (result_root / "baseline" / "videos" / rgb_dir.parent.name /
                          f"observation.images.{camera_name}_depth_nested_sparse")
        for video_path in sorted(rgb_dir.glob("episode_*.mp4")):
            episode = video_path.stem
            ep_out = stats_root / camera_name / episode
            marker = ep_out / "statistics.json"
            if marker.exists() and not args.force:
                continue
            with episode_timer(result_root, stage=args.strategy, camera=camera_name,
                               source=video_path, model_id=model_id,
                               sample_count=args.sample_count, **sampling_config) as timing:
                # 先完整解码 RGB，后续仅将采样帧送入深度修复器。
                rgb = _decode_rgb_video(video_path)
                if not rgb:
                    print(f"Skipping undecodable video: {video_path}", file=sys.stderr)
                    continue
                indices = _sample_indices(len(rgb), args.sample_count, stride=args.sample_stride)
                sampled_rgb = [rgb[i] for i in indices]
                # 只把采样后的 RGB 送入 Nested；这里不会生成伪造的完整时间轴。
                # 第二个参数保留为空，表示没有额外的真实深度约束；返回值
                # 与 sampled_rgb 一一对应，不能误当成完整视频帧序列。
                nested = repair.repair_frames(sampled_rgb, []) if sampled_rgb else []
                ep_out.mkdir(parents=True, exist_ok=True)
                # 每个采样结果以毫米为单位保存为 uint16 PNG 和 NPY。
                for frame_idx, depth in zip(indices, nested):
                    depth_u16 = np.clip(np.rint(depth), 0, 65535).astype(np.uint16)
                    cv2.imwrite(str(ep_out / f"frame_{frame_idx:09d}.png"), depth_u16)
                    np.save(ep_out / f"frame_{frame_idx:09d}.npy", depth_u16)
                with open(ep_out / "sample_indices.json", "w") as f:
                    json.dump({"source_video": str(video_path), "source_num_frames": len(rgb),
                               "sample_count": len(indices), "fps": args.fps,
                               **sampling_config,
                               "sample_frame_indices": indices}, f, indent=2)
                # 可选地把稀疏结果编码成灰度深度视频；视频帧数仍等于采样数。
                if args.save_video and nested:
                    sparse_dir.mkdir(parents=True, exist_ok=True)
                    rc = encoder.encode_from_arrays(nested, str(sparse_dir / video_path.name))
                    if rc != 0:
                        raise RuntimeError(f"ffmpeg failed ({rc}): {video_path}")
                    (ep_out / "manifest.json").write_text(json.dumps({
                        "source_video": str(video_path),
                        "sparse_video": str(sparse_dir / video_path.name),
                        "source_num_frames": len(rgb), "sparse_num_frames": len(nested),
                        "source_fps": args.fps, "sparse_fps": args.fps,
                        **sampling_config,
                        "sample_frame_indices": indices,
                    }, indent=2))
                # 写入汇总标记，记录模型、采样位置及全局/逐帧统计，支持断点续跑。
                with open(marker, "w") as f:
                    json.dump({"camera": camera_name, "episode": episode,
                               "strategy": args.strategy, "model_id": model_id,
                               "source_num_frames": len(rgb), "sample_indices": indices,
                               **sampling_config,
                               "stats": _stats(nested),
                               "sampled_frames": [
                                   {"frame_index": int(i), "npy": f"frame_{i:09d}.npy",
                                    "stats": _frame_stats(d)}
                                   for i, d in zip(indices, nested)
                               ],
                               "video_format": {"codec": "libx265", "pix_fmt": "gray12le",
                                                "fps": args.fps}}, f, indent=2)
                timing.update(status="completed", source_num_frames=len(rgb), sample_num_frames=len(nested))
    print(f"Nested analysis complete: {stats_root}")


if __name__ == "__main__":
    main()
