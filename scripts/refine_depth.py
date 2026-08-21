#!/usr/bin/env python3
"""利用 Nested 稀疏结果（或可选真实深度）校正 DA3 LARGE-1.1 深度视频。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.depth_encoder import DepthVideoEncoder


def _probe(path: Path) -> tuple[int, int]:
    """读取视频第一路视频流的宽高，供原始字节重排使用。"""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=False)
    return tuple(map(int, r.stdout.strip().split(",")))  # type: ignore[return-value]


def _decode(path: Path, pix_fmt: str) -> list[np.ndarray]:
    """按指定像素格式解码视频，并返回逐帧 NumPy 数组。

    深度视频使用单通道 ``gray12le/gray16le``；RGB 视频使用三通道格式。
    ``ffmpeg`` 输出的是连续字节，因此必须依据探测到的宽高逐帧切片。
    """
    w, h = _probe(path)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
                       capture_output=True, check=False)
    if r.returncode:
        raise RuntimeError(f"ffmpeg decode failed: {path}")
    raw = np.frombuffer(r.stdout, dtype=np.uint16 if pix_fmt == "gray12le" else np.uint8)
    channels = 1 if pix_fmt == "gray12le" else 3
    size = w * h * channels
    return [raw[i:i + size].reshape(h, w, channels)[..., 0] if channels == 1
            else raw[i:i + size].reshape(h, w, channels) for i in range(0, len(raw) - size + 1, size)]


def _metrics(pred: list[np.ndarray], target: list[np.ndarray]) -> dict:
    """汇总预测与参考帧的像素级 MAE、RMSE 和有符号偏差。"""
    errors = []
    for p, t in zip(pred, target):
        mask = np.isfinite(p) & np.isfinite(t) & (t > 0) & (p > 0)
        if mask.any(): errors.append((p[mask] - t[mask]).astype(np.float64))
    if not errors: return {"count": 0}
    e = np.concatenate(errors)
    return {"count": int(e.size), "mae_mm": float(np.mean(np.abs(e))),
            "rmse_mm": float(np.sqrt(np.mean(e * e))), "bias_mm": float(np.mean(e))}


def _frame_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """计算单帧均值、绝对误差和 RMSE，忽略无效/零深度像素。"""
    mask = np.isfinite(pred) & np.isfinite(target) & (pred > 0) & (target > 0)
    if not mask.any():
        return {"valid_pixels": 0, "pred_mean_mm": None, "target_mean_mm": None,
                "mae_mm": None, "rmse_mm": None}
    e = pred[mask].astype(np.float64) - target[mask].astype(np.float64)
    return {"valid_pixels": int(mask.sum()), "pred_mean_mm": float(pred[mask].mean()),
            "target_mean_mm": float(target[mask].mean()),
            "mae_mm": float(np.abs(e).mean()), "rmse_mm": float(np.sqrt(np.mean(e * e)))}


def _plot_trend(path: Path, frame_ids: list[int], before: list[dict], after: list[dict], title: str) -> None:
    """绘制逐帧均值和误差曲线；matplotlib 缺失时给出清晰提示。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("绘图需要 matplotlib，请先 pip install matplotlib") from exc
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(frame_ids, [x.get("target_mean_mm") for x in before], "k-", label="Nested参考均值")
    axes[0].plot(frame_ids, [x.get("pred_mean_mm") for x in before], "b--", label="DA3均值")
    axes[0].plot(frame_ids, [x.get("pred_mean_mm") for x in after], "g-", label="Refined均值")
    axes[0].set_ylabel("深度均值 (mm)"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(frame_ids, [x.get("mae_mm") for x in before], "b--", label="DA3 MAE")
    axes[1].plot(frame_ids, [x.get("mae_mm") for x in after], "g-", label="Refined MAE")
    axes[1].set_xlabel("原视频帧索引"); axes[1].set_ylabel("误差 (mm)"); axes[1].legend(); axes[1].grid(alpha=.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    """利用采样参考拟合线性关系，并将其应用到完整 DA3 深度视频。"""
    p = argparse.ArgumentParser(description="使用 Nested 参考校正 DA3 LARGE-1.1 深度视频")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--camera", required=True, help="head/right/left/chest")
    p.add_argument("--statistics-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--input-depth-suffix", default="_depth", help="DA3/RIFE 目录后缀")
    p.add_argument("--real-depth-suffix", default="_depth_real", help="真实 RGB-D 目录后缀")
    p.add_argument("--real-pix-fmt", default="gray12le", choices=["gray12le", "gray16le"])
    p.add_argument("--reference", choices=["nested", "real"], default="nested",
                    help="校正参考；nested 不需要真实深度")
    p.add_argument("--plot-trend", action="store_true", help="保存逐采样帧趋势曲线 PNG")
    p.add_argument("--crf", type=int, default=6)
    p.add_argument("--preset", default="medium")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    # 输入/输出目录可分离：默认在数据集原 videos 目录旁生成 refined 目录。
    root = Path(args.dataset_dir)
    stat_root = Path(args.statistics_dir) if args.statistics_dir else root / "depth_statistics"
    out_root = Path(args.output_dir) if args.output_dir else root / "videos"
    encoder = DepthVideoEncoder(args.fps, args.crf, args.preset)
    # 通过 RGB 目录定位 chunk，再推导 DA3、真实深度和 refined 目录名称。
    for rgb_dir in sorted((root / "videos").glob(f"chunk-*/observation.images.*{args.camera}*rgb")):
        chunk = rgb_dir.parent
        name = rgb_dir.name.replace("_rgb", "")
        da3_dir = chunk / f"{name}{args.input_depth_suffix}"
        real_dir = chunk / f"{name}{args.real_depth_suffix}"
        refined_dir = (out_root / chunk.name / f"{name}_depth_refined"
                       if args.output_dir else chunk / f"{name}_depth_refined")
        refined_dir.mkdir(parents=True, exist_ok=True)
        for da3_path in sorted(da3_dir.glob("episode_*.mp4")):
            ep = da3_path.stem
            real_path = real_dir / da3_path.name
            nested_ep = stat_root / (f"cam_{args.camera}" if not args.camera.startswith("cam_") else args.camera) / ep
            idx_path = nested_ep / "sample_indices.json"
            # 没有 Nested 采样索引，或选择真实参考但真实视频缺失时，无法拟合，直接跳过。
            if not idx_path.exists() or (args.reference == "real" and not real_path.exists()):
                continue
            indices = json.loads(idx_path.read_text())["sample_frame_indices"]
            da3 = _decode(da3_path, "gray12le")
            real = _decode(real_path, args.real_pix_fmt) if args.reference == "real" else None
            nested = [np.load(nested_ep / f"frame_{i:09d}.npy") for i in indices]
            # xs/ys 收集采样帧中逐像素的 DA3/参考配对，用于全局线性回归。
            xs, ys = [], []
            nested_valid = []
            for i, n in zip(indices, nested):
                if i >= len(da3): continue
                target = real[i] if real is not None and i < len(real) else n
                nested_valid.append(n)
                mask = np.isfinite(n) & (n > 0) & np.isfinite(target) & (target > 0) & (da3[i] > 0)
                xs.append(da3[i][mask].astype(np.float64)); ys.append(target[mask].astype(np.float64))
            if not xs: continue
            # 用采样帧的像素配对估计全局尺度和偏移；Nested 不是物理真值，
            # 因此该参数表达的是相对一致性校正，而非传感器标定。
            # 合并所有采样帧，并去除两端 1% 离群值，降低遮挡和异常深度的影响。
            x, y = np.concatenate(xs), np.concatenate(ys)
            lo, hi = np.percentile(x, [1, 99]); keep = (x >= lo) & (x <= hi)
            a, b = np.linalg.lstsq(np.stack([x[keep], np.ones(keep.sum())], 1), y[keep], rcond=None)[0]
            # 将拟合关系应用到完整 DA3 LARGE-1.1 视频，并限制到 12-bit 毫米量程。
            # 对完整 DA3 视频逐帧应用 y = a*x + b，并限制到 12-bit 深度范围。
            refined = [np.clip(a * f + b, 0, 4095).astype(np.float32) for f in da3]
            out_path = refined_dir / da3_path.name
            encoder.encode_from_arrays(refined, str(out_path))
            reference_frames = ([real[i] for i in indices if i < len(real)]
                                if real is not None else nested_valid)
            reference_pred = ([da3[i] for i in indices if i < len(da3)]
                              if real is None else da3)
            reference_refined = ([refined[i] for i in indices if i < len(refined)]
                                 if real is None else refined)
            # 同时计算校正前后误差；这些指标和逐帧趋势会写入参数 JSON。
            before = _metrics(reference_pred, reference_frames)
            after = _metrics(reference_refined, reference_frames)
            frame_ids = [i for i in indices if i < len(da3) and (real is None or i < len(real))]
            frame_targets = [nested[j] if real is None else real[i] for j, i in enumerate(indices) if i in frame_ids]
            frame_da3 = [da3[i] for i in frame_ids]
            frame_refined = [refined[i] for i in frame_ids]
            frame_before = [_frame_metrics(x, y) for x, y in zip(frame_da3, frame_targets)]
            frame_after = [_frame_metrics(x, y) for x, y in zip(frame_refined, frame_targets)]
            if args.plot_trend:
                _plot_trend(nested_ep / "refine_trend.png", frame_ids, frame_before, frame_after, ep)
            (nested_ep / "refine_parameters.json").write_text(json.dumps(
                {"a": float(a), "b_mm": float(b), "fit_samples": int(keep.sum()),
                 "reference": args.reference, "baseline": before, "refined": after,
                 "frame_metrics": [{"frame_index": i, "baseline": b0, "refined": a0}
                                    for i, b0, a0 in zip(frame_ids, frame_before, frame_after)],
                 "improved": bool(after.get("mae_mm", float("inf")) < before.get("mae_mm", float("inf"))
                              and after.get("rmse_mm", float("inf")) < before.get("rmse_mm", float("inf")))}, indent=2))
    print(f"Refined videos written under {out_root}")


if __name__ == "__main__": main()
