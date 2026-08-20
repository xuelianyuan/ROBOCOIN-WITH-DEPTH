#!/usr/bin/env python3
"""稀疏 DA3-Nested 分析，并保存原始深度、统计信息和可视化视频。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.depth_encoder import DepthVideoEncoder
from src.depth_repair import create_depth_repair


def _sample_indices(n: int, count: int) -> list[int]:
    if n <= 0:
        return []
    if count <= 0 or count >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, count, dtype=np.int64).tolist()))


def _decode_rgb_video(path: Path) -> list[np.ndarray]:
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
    if not frames:
        return {"count": 0, "valid_pixels": 0}
    values = np.concatenate([f[np.isfinite(f) & (f > 0)].ravel() for f in frames])
    if values.size == 0:
        return {"count": len(frames), "valid_pixels": 0}
    return {
        "count": len(frames),
        "valid_pixels": int(values.size),
        "min_mm": float(values.min()),
        "max_mm": float(values.max()),
        "mean_mm": float(values.mean()),
        "std_mm": float(values.std()),
        "p01_mm": float(np.percentile(values, 1)),
        "p50_mm": float(np.percentile(values, 50)),
        "p99_mm": float(np.percentile(values, 99)),
    }


def _frame_stats(frame: np.ndarray) -> dict:
    valid = frame[np.isfinite(frame) & (frame > 0)]
    if valid.size == 0:
        return {"valid_pixels": 0}
    return {"valid_pixels": int(valid.size), "min_mm": float(valid.min()),
            "max_mm": float(valid.max()), "mean_mm": float(valid.mean()),
            "std_mm": float(valid.std()), "p50_mm": float(np.percentile(valid, 50)),
            "p99_mm": float(np.percentile(valid, 99))}


def main() -> None:
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

    model_id = args.da3_model
    if args.strategy == "da3" and model_id == "depth-anything/DA3NESTED-GIANT-LARGE-1.1":
        model_id = "depth-anything/DA3METRIC-LARGE"
    repair = create_depth_repair(
        "da3", model_id=model_id, process_res=args.da3_process_res,
        chunk_size=args.da3_chunk_size, overlap=args.da3_overlap, device=args.device,
    )
    encoder = DepthVideoEncoder(fps=args.fps, crf=args.crf, preset=args.preset)

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
            rgb = _decode_rgb_video(video_path)
            if not rgb:
                print(f"Skipping undecodable video: {video_path}", file=sys.stderr)
                continue
            # 等间隔采样并始终保留首尾帧，保证稀疏结果覆盖完整时间范围。
            indices = _sample_indices(len(rgb), args.sample_count)
            sampled_rgb = [rgb[i] for i in indices]
            # 只把采样后的 RGB 送入 Nested；这里不会生成伪造的完整时间轴。
            nested = repair.repair_frames(sampled_rgb, []) if sampled_rgb else []
            ep_out.mkdir(parents=True, exist_ok=True)
            for frame_idx, depth in zip(indices, nested):
                np.save(ep_out / f"frame_{frame_idx:09d}.npy", depth.astype(np.float32))
            with open(ep_out / "sample_indices.json", "w") as f:
                json.dump({"source_video": str(video_path), "source_num_frames": len(rgb),
                           "sample_count": len(indices), "fps": args.fps,
                           "sample_frame_indices": indices}, f, indent=2)
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
