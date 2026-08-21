#!/usr/bin/env python3
"""稀疏 DA3-Nested 分析，并保存原始深度、统计信息和可视化视频。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.depth_encoder import DepthVideoEncoder
from src.depth_repair import create_depth_repair


def _sample_indices(n: int, count: int) -> list[int]:
    """生成近似等间隔的采样帧索引。

    ``linspace`` 能保证采样覆盖视频首尾；当请求数量超过总帧数时，
    直接返回全部索引，避免重复推理同一帧。
    """
    if n <= 0:
        return []
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
    p.add_argument("--sample-count", type=int, default=20)
    p.add_argument("--strategy", default="da3_nested", choices=["da3_nested", "da3"])
    p.add_argument("--output-dir", default=None)
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

    # 数据集通常包含 videos/、depth_statistics/ 等目录；所有相对路径
    # 均从用户传入的数据集根目录解析，便于把结果写入另一位置。
    root = Path(args.dataset_dir)
    stats_root = Path(args.output_dir) if args.output_dir else root / "depth_statistics"
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
        sparse_dir = rgb_dir.parent / f"observation.images.{camera_name}_depth_nested_sparse"
        for video_path in sorted(rgb_dir.glob("episode_*.mp4")):
            episode = video_path.stem
            ep_out = stats_root / camera_name / episode
            marker = ep_out / "statistics.json"
            if marker.exists() and not args.force:
                continue
            # 先完整解码 RGB，后续仅选取少量帧送入深度修复器。
            rgb = _decode_rgb_video(video_path)
            if not rgb:
                print(f"Skipping undecodable video: {video_path}", file=sys.stderr)
                continue
            # 等间隔采样并始终保留首尾帧，保证稀疏结果覆盖完整时间范围。
            indices = _sample_indices(len(rgb), args.sample_count)
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
                    "sample_frame_indices": indices,
                }, indent=2))
            # 写入汇总标记，记录模型、采样位置及全局/逐帧统计，支持断点续跑。
            with open(marker, "w") as f:
                json.dump({"camera": camera_name, "episode": episode,
                           "strategy": args.strategy, "model_id": model_id,
                           "source_num_frames": len(rgb), "sample_indices": indices,
                           "stats": _stats(nested),
                           "sampled_frames": [
                               {"frame_index": int(i), "npy": f"frame_{i:09d}.npy",
                                "stats": _frame_stats(d)}
                               for i, d in zip(indices, nested)
                           ],
                           "video_format": {"codec": "libx265", "pix_fmt": "gray12le",
                                            "fps": args.fps}}, f, indent=2)
    print(f"Nested analysis complete: {stats_root}")


if __name__ == "__main__":
    main()
