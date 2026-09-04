#!/usr/bin/env python3
"""用 Nested 稀疏参考迁移深度分布，验证后可选 RIFE 补齐时间轴。

处理流程：先读取项目 ``videos/`` 下的原始 Metric 深度视频，再读取
``depth_results/baseline/statistics/`` 中由 ``analyze_depth.py`` 生成的
Nested 采样帧；使用 ``quantile`` 或 ``linear`` 拟合映射并写出验证报告。
``validate`` 只评估，``full`` 将映射应用到每一帧，``rife`` 则用 RIFE
把迁移后的稀疏关键帧补齐为完整视频。

用法示例::

    python scripts/refine_depth.py \
        --dataset-dir data/refine/your_dataset \
        --result-root data/refine/your_dataset/depth_results \
        --camera head --mapping quantile --stage validate --plot-trend

输出：报告位于 ``depth_results/refine/reports/``，生成的视频位于
``depth_results/refine/videos/``。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.depth_encoder import DepthVideoEncoder


def _probe(path: Path) -> tuple[int, int]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return tuple(map(int, r.stdout.strip().split(",")))  # type: ignore[return-value]
    except ValueError as exc:
        raise RuntimeError(f"ffprobe failed: {path}") from exc


def _decode(path: Path, pix_fmt: str) -> list[np.ndarray]:
    """解码 RGB 或 12/16-bit 单通道视频。"""
    w, h = _probe(path)
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        capture_output=True, check=False,
    )
    if r.returncode:
        raise RuntimeError(f"ffmpeg decode failed: {path}")
    is_depth = pix_fmt in {"gray12le", "gray16le"}
    dtype, channels = (np.uint16, 1) if is_depth else (np.uint8, 3)
    raw = np.frombuffer(r.stdout, dtype=dtype)
    size = w * h * channels
    return [raw[i:i + size].reshape(h, w) if is_depth
            else raw[i:i + size].reshape(h, w, 3)
            for i in range(0, len(raw) - size + 1, size)]


def _valid_pair(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Nested 输出尺寸可能与原始深度视频不同；将参考深度缩放到源帧尺寸后
    # 才能建立逐像素对应关系。深度是连续数值，使用线性插值而非 RGB 插值。
    if pred.shape != target.shape:
        target = cv2.resize(target.astype(np.float32),
                            (pred.shape[1], pred.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    mask = np.isfinite(pred) & np.isfinite(target) & (pred > 0) & (target > 0)
    return pred[mask].astype(np.float64), target[mask].astype(np.float64)


def _wasserstein_1d(x: np.ndarray, y: np.ndarray, points: int = 1001) -> float | None:
    """用等概率分位点积分近似一维 Wasserstein-1/EMD。"""
    if not x.size or not y.size:
        return None
    q = np.linspace(0, 100, points)
    return float(np.mean(np.abs(np.percentile(x, q) - np.percentile(y, q))))


def _metrics(pred: list[np.ndarray], target: list[np.ndarray]) -> dict:
    pairs = [_valid_pair(p, t) for p, t in zip(pred, target)]
    pairs = [(x, y) for x, y in pairs if x.size]
    if not pairs:
        return {"valid_pixels": 0}
    x = np.concatenate([p[0] for p in pairs])
    y = np.concatenate([p[1] for p in pairs])
    error = x - y
    return {
        "valid_pixels": int(error.size),
        "mean_mm": float(x.mean()), "target_mean_mm": float(y.mean()),
        "mean_error_mm": float(x.mean() - y.mean()),
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(error * error))),
        "wasserstein_mm": _wasserstein_1d(x, y),
        "p10_mm": float(np.percentile(x, 10)),
        "p50_mm": float(np.percentile(x, 50)),
        "p90_mm": float(np.percentile(x, 90)),
    }


def _fit_mapping(x: np.ndarray, y: np.ndarray, method: str, quantiles: int) -> dict:
    if method == "linear":
        lo, hi = np.percentile(x, [1, 99])
        keep = (x >= lo) & (x <= hi)
        a, b = np.linalg.lstsq(
            np.stack([x[keep], np.ones(keep.sum())], axis=1), y[keep], rcond=None,
        )[0]
        return {"method": method, "a": float(a), "b_mm": float(b),
                "fit_samples": int(keep.sum())}
    q = np.linspace(0, 100, max(3, quantiles))
    source = np.percentile(x, q)
    target = np.percentile(y, q)
    source, unique_idx = np.unique(source, return_index=True)
    return {"method": method, "source_quantiles_mm": source.tolist(),
            "target_quantiles_mm": target[unique_idx].tolist(),
            "fit_samples": int(x.size)}


def _apply_mapping(frame: np.ndarray, mapping: dict, max_depth: float) -> np.ndarray:
    out = np.zeros(frame.shape, dtype=np.float32)
    valid = np.isfinite(frame) & (frame > 0)
    if mapping["method"] == "linear":
        out[valid] = mapping["a"] * frame[valid] + mapping["b_mm"]
    else:
        out[valid] = np.interp(
            frame[valid], mapping["source_quantiles_mm"], mapping["target_quantiles_mm"],
        )
    return np.clip(out, 0, max_depth)


def _statistics_episode(stat_root: Path, camera: str, episode: str) -> Path | None:
    for name in (camera, f"cam_{camera}", camera.removeprefix("cam_")):
        candidate = stat_root / name / episode
        if (candidate / "sample_indices.json").exists():
            return candidate
    return None


def _plot_trend(path: Path, ids: list[int], before: list[dict], after: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("绘图需要 matplotlib") from exc
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(ids, [x.get("target_mean_mm") for x in before], "k-", label="Nested")
    axes[0].plot(ids, [x.get("mean_mm") for x in before], "b--", label="Source")
    axes[0].plot(ids, [x.get("mean_mm") for x in after], "g-", label="Transferred")
    axes[0].set_ylabel("Mean depth (mm)"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(ids, [x.get("wasserstein_mm") for x in before], "b--", label="Before")
    axes[1].plot(ids, [x.get("wasserstein_mm") for x in after], "g-", label="After")
    axes[1].set_xlabel("Source frame index"); axes[1].set_ylabel("Wasserstein-1 (mm)")
    axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Nested 深度分布迁移、验证与 RIFE 补帧")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--statistics-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--result-root", default=None,
                   help="阶段化结果根目录；默认 <dataset>/depth_results")
    p.add_argument("--report-dir", default=None,
                   help="refine 报告目录；默认 <result-root>/refine/reports")
    p.add_argument("--input-depth-suffix", default="_depth")
    p.add_argument("--input-pix-fmt", default="gray12le", choices=["gray12le", "gray16le"])
    p.add_argument("--mapping", default="quantile", choices=["quantile", "linear"],
                   help="quantile 即一维最优传输/直方图匹配")
    p.add_argument("--quantiles", type=int, default=1001)
    p.add_argument("--stage", default="validate", choices=["validate", "full", "rife"],
                   help="validate 只评估；full 迁移完整输入；rife 从迁移后采样帧补齐")
    p.add_argument("--allow-unvalidated", action="store_true",
                   help="指标未同时改善时仍允许生成视频")
    p.add_argument("--max-depth-mm", type=float, default=4095)
    p.add_argument("--plot-trend", action="store_true")
    p.add_argument("--crf", type=int, default=6)
    p.add_argument("--preset", default="medium")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    root = Path(args.dataset_dir)
    result_root = Path(args.result_root) if args.result_root else root / "depth_results"
    stat_root = (Path(args.statistics_dir) if args.statistics_dir
                 else result_root / "baseline" / "statistics")
    out_root = (Path(args.output_dir) if args.output_dir
                else result_root / "refine" / "videos")
    report_root = (Path(args.report_dir) if args.report_dir
                   else result_root / "refine" / "reports")
    encoder = DepthVideoEncoder(args.fps, args.crf, args.preset)
    rgb_dirs = sorted({v.parent for v in (root / "videos").glob("**/episode_*.mp4")
                       if "depth" not in v.parent.name.lower()
                       and args.camera in v.parent.name.lower()})
    processed = 0
    for rgb_dir in rgb_dirs:
        camera = rgb_dir.name.replace("observation.images.", "").replace("_rgb", "")
        base_name = rgb_dir.name.replace("_rgb", "")
        source_dir = rgb_dir.parent / f"{base_name}{args.input_depth_suffix}"
        refined_dir = out_root / rgb_dir.parent.name / f"observation.images.{camera}_depth_refined"
        for source_path in sorted(source_dir.glob("episode_*.mp4")):
            ep_out = _statistics_episode(stat_root, camera, source_path.stem)
            if ep_out is None:
                continue
            metadata = json.loads((ep_out / "sample_indices.json").read_text())
            indices = [int(i) for i in metadata["sample_frame_indices"]]
            source = _decode(source_path, args.input_pix_fmt)
            valid_ids = [i for i in indices if i < len(source)
                         and (ep_out / f"frame_{i:09d}.npy").exists()]
            nested = [np.load(ep_out / f"frame_{i:09d}.npy") for i in valid_ids]
            sampled = [source[i] for i in valid_ids]
            pairs = [_valid_pair(x, y) for x, y in zip(sampled, nested)]
            pairs = [(x, y) for x, y in pairs if x.size]
            if not pairs:
                continue
            x = np.concatenate([v[0] for v in pairs])
            y = np.concatenate([v[1] for v in pairs])
            mapping = _fit_mapping(x, y, args.mapping, args.quantiles)
            transferred = [_apply_mapping(f, mapping, args.max_depth_mm) for f in sampled]
            before, after = _metrics(sampled, nested), _metrics(transferred, nested)
            improved = all(after.get(k, float("inf")) < before.get(k, float("inf"))
                           for k in ("mae_mm", "rmse_mm", "wasserstein_mm"))
            frame_before = [_metrics([a], [b]) for a, b in zip(sampled, nested)]
            frame_after = [_metrics([a], [b]) for a, b in zip(transferred, nested)]
            report = {"mapping": mapping, "stage": args.stage, "baseline": before,
                      "transferred": after, "improved": improved, "sample_indices": valid_ids,
                      "frame_metrics": [{"frame_index": i, "baseline": b, "transferred": a}
                                        for i, b, a in zip(valid_ids, frame_before, frame_after)]}
            report_path = report_root / camera / source_path.stem / "refine_parameters.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))
            if args.plot_trend:
                _plot_trend(report_path.with_name("refine_trend.png"), valid_ids, frame_before, frame_after)
            if args.stage == "validate":
                processed += 1
                continue
            if not improved and not args.allow_unvalidated:
                print(f"Skip unvalidated transfer: {source_path}", file=sys.stderr)
                continue
            if args.stage == "full":
                output = [_apply_mapping(f, mapping, args.max_depth_mm) for f in source]
            else:
                rgb = _decode(rgb_dir / source_path.name, "rgb24")
                if len(rgb) != len(source):
                    raise RuntimeError(f"RGB/depth frame count mismatch: {source_path}")
                from src.depth_repair.da3_rife import DA3RIFEDepthRepair
                rife = DA3RIFEDepthRepair(sample_stride=1)
                rife._load_rife()
                output = rife._interpolate(
                    [rgb[i] for i in valid_ids], transferred, valid_ids,
                    len(rgb), rgb[0].shape[0], rgb[0].shape[1],
                )
            refined_dir.mkdir(parents=True, exist_ok=True)
            out_path = refined_dir / source_path.name
            rc = encoder.encode_from_arrays(output, str(out_path))
            if rc:
                raise RuntimeError(f"ffmpeg encode failed ({rc}): {out_path}")
            report.update({"output_video": str(out_path), "output_num_frames": len(output)})
            report_path.write_text(json.dumps(report, indent=2))
            processed += 1
    print(f"Processed {processed} episode(s); results under {stat_root}")


if __name__ == "__main__":
    main()
