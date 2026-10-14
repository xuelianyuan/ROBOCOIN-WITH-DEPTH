#!/usr/bin/env python3
"""用 Nested 稀疏参考校准原始深度视频的数值分布。

重要：这里的“迁移”不是复制 Nested 的图像内容，也不是空间滤波。脚本把同一
episode 的所有采样像素汇总，拟合一条全局的一维函数 ``Nested=f(Source)``，
再将该函数逐像素应用于源深度。因此输出保留源深度的边缘、噪声和局部纹理；
分位数曲线斜率较陡或有平台时，还可能放大源图已有的小幅起伏。

数据链路：

1. 从 ``videos/`` 解码原始深度视频（默认按 gray12le 数值读取）。
2. 从 ``depth_results/baseline/statistics/`` 读取 Nested 采样帧 NPY；若尺寸
   不同，先把 Nested 双线性缩放至源深度尺寸，仅用于配对、拟合和验证。
3. ``quantile`` 将所有采样帧的有效像素合并后做全局直方图/分位数匹配；
   ``linear`` 则在源深度 1%--99% 区间拟合一条直线。
4. ``validate`` 只写报告；``full`` 对原视频每帧使用同一映射；``rife`` 只
   映射采样帧，再由 RGB 引导的 RIFE 补齐时间轴。
5. 输出先裁剪至 ``max_depth_mm``，随后编码器再裁剪至 12-bit [0, 4095]、
   转为 uint16，并用 gray12le/libx265 写视频；当前调用不会做 min-max 归一化。

诊断选项：``--report-mapping-slope`` 输出映射曲线各段的斜率摘要和最陡区间；
``--save-preencode-npy`` 将 full/rife 的编码前 float32 帧逐帧保存，便于和视频
解码结果比较，从而区分映射伪影与量化、编码伪影。

所以报告中的全局直方图改善不保证每一帧、每个空间区域都改善。诊断灰度色斑
时，应分别比较映射前、映射后编码前数组、视频解码值和显示归一化结果。

用法示例::

    python scripts/depth_refine/run_refine.py \
        --dataset-dir data/lerobot/your_dataset \
        --result-root data/lerobot/your_dataset/depth_results \
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
from src.depth_refine.timing import episode_timer
from src.depth_encoder import DepthVideoEncoder


# ============================== 一、视频解码 ==============================

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


# ======================== 二、有效像素配对与评估指标 ========================

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


# =========================== 三、拟合并应用数值映射 ===========================

def _fit_mapping(x: np.ndarray, y: np.ndarray, method: str, quantiles: int) -> dict:
    """拟合 episode 级全局映射；输入已合并所有采样帧，不保留空间/时间位置。"""
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
            # 12-bit 深度量化可能让多个分位点对应同一个源深度值。分别记录
            # 请求值和去重后的有效值，避免仅凭映射节点数误判实验配置。
            "requested_quantiles": int(max(3, quantiles)),
            "effective_quantile_nodes": int(source.size),
            "mapping_segments": int(max(0, source.size - 1)),
            "fit_samples": int(x.size)}


def _mapping_slope_report(mapping: dict, top_k: int = 20) -> dict:
    """汇总映射曲线斜率，并列出最可能放大源深度起伏的区间。"""
    if mapping["method"] == "linear":
        slope = float(mapping["a"])
        return {"method": "linear", "segment_count": 1,
                "slope_min": slope, "slope_p50": slope, "slope_p90": slope,
                "slope_p95": slope, "slope_p99": slope, "slope_max": slope,
                "steepest_segments": []}

    source = np.asarray(mapping["source_quantiles_mm"], dtype=np.float64)
    target = np.asarray(mapping["target_quantiles_mm"], dtype=np.float64)
    source_delta = np.diff(source)
    target_delta = np.diff(target)
    usable = np.isfinite(source_delta) & np.isfinite(target_delta) & (source_delta > 0)
    indices = np.flatnonzero(usable)
    slopes = target_delta[usable] / source_delta[usable]
    if not slopes.size:
        return {"method": "quantile", "segment_count": 0,
                "reason": "no_finite_positive_source_intervals",
                "steepest_segments": []}

    order = np.argsort(slopes)[::-1][:max(0, top_k)]
    steepest = []
    for rank in order:
        i = int(indices[rank])
        steepest.append({
            "source_start_mm": float(source[i]),
            "source_end_mm": float(source[i + 1]),
            "target_start_mm": float(target[i]),
            "target_end_mm": float(target[i + 1]),
            "source_delta_mm": float(source_delta[i]),
            "target_delta_mm": float(target_delta[i]),
            "slope": float(slopes[rank]),
        })
    percentile_values = np.percentile(slopes, [50, 90, 95, 99])
    return {
        "method": "quantile", "segment_count": int(slopes.size),
        "slope_min": float(slopes.min()),
        "slope_p50": float(percentile_values[0]),
        "slope_p90": float(percentile_values[1]),
        "slope_p95": float(percentile_values[2]),
        "slope_p99": float(percentile_values[3]),
        "slope_max": float(slopes.max()),
        "segments_above_1": int(np.count_nonzero(slopes > 1)),
        "segments_above_2": int(np.count_nonzero(slopes > 2)),
        "segments_above_5": int(np.count_nonzero(slopes > 5)),
        "steepest_segments": steepest,
    }


def _apply_mapping(frame: np.ndarray, mapping: dict, max_depth: float) -> np.ndarray:
    """逐像素应用同一标量映射；不参考邻域，也不执行空间去噪。"""
    out = np.zeros(frame.shape, dtype=np.float32)
    valid = np.isfinite(frame) & (frame > 0)
    if mapping["method"] == "linear":
        out[valid] = mapping["a"] * frame[valid] + mapping["b_mm"]
    # 将 Metric 的深度值映射到 Nested 的深度分布，并不考虑相邻像素，也不会用 Nested 的局部形状替换原图。因此：
    #   - 原图中的局部深度起伏会被保留。
    #   - 映射曲线较陡的区间，会把微小深度差放大，显示成更明显的灰度斑块。
    #   - Nested 自身更平滑，但匹配它的数值分布，并不能获得同样平滑的画面。
    else:
        out[valid] = np.interp(
            frame[valid], mapping["source_quantiles_mm"], mapping["target_quantiles_mm"],
        )
    return np.clip(out, 0, max_depth)


def _save_preencode_frames(frames: list[np.ndarray], output_dir: Path) -> None:
    """逐帧保存编码前 float32 深度，帧号与最终视频从 0 开始的帧号一致。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = []
    for frame_index, frame in enumerate(frames):
        filename = f"frame_{frame_index:09d}.npy"
        np.save(output_dir / filename,
                np.asarray(frame, dtype=np.float32), allow_pickle=False)
        filenames.append(filename)
    (output_dir / "manifest.json").write_text(json.dumps({
        "dtype": "float32", "frame_count": len(filenames), "frames": filenames,
        "note": "Values are after refine mapping/clip and before uint16/video encoding.",
    }, indent=2) + "\n")


# ========================= 四、参考定位与诊断趋势图 =========================

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


# ============================== 五、主处理流程 ==============================

def main() -> None:
    """逐相机、逐 episode 拟合一次全局映射，并按所选阶段验证或生成视频。"""
    # ---------- 5.1 参数：输入数值域、映射方法、阶段与编码设置 ----------
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
    p.add_argument("--quantiles", type=int, default=101,
                   help="quantile 映射请求的分位点数；默认 q101，重复源节点会去重")
    p.add_argument("--stage", default="validate", choices=["validate", "full", "rife"],
                   help="validate 只评估；full 迁移完整输入；rife 从迁移后采样帧补齐")
    p.add_argument("--allow-unvalidated", action="store_true",
                   help="指标未同时改善时仍允许生成视频")
    p.add_argument("--max-depth-mm", type=float, default=4095)
    p.add_argument("--plot-trend", action="store_true")
    p.add_argument("--report-mapping-slope", action="store_true",
                   help="分析映射斜率并保存 mapping_slope.json")
    p.add_argument("--save-preencode-npy", action="store_true",
                   help="full/rife 时逐帧保存编码前 float32 refine NPY")
    p.add_argument("--crf", type=int, default=6)
    p.add_argument("--preset", default="medium")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    # ---------- 5.2 解析目录并发现匹配相机的源 RGB/深度视频 ----------
    root = Path(args.dataset_dir)
    result_root = Path(args.result_root) if args.result_root else root / "depth_results"
    stat_root = (Path(args.statistics_dir) if args.statistics_dir
                 else result_root / "baseline" / "statistics")
    out_root = (Path(args.output_dir) if args.output_dir
                else result_root / "refine" / "videos")
    report_root = (Path(args.report_dir) if args.report_dir
                   else result_root / "refine" / "reports")
    preencode_root = result_root / "refine" / "preencode"
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
            # ---------- 5.3 读取源深度及相同帧号的 Nested 稀疏参考 ----------
            with episode_timer(result_root, stage=f"refine_{args.stage}", camera=camera,
                               source=source_path, mapping=args.mapping) as timing:
                ep_out = _statistics_episode(stat_root, camera, source_path.stem)
                if ep_out is None:
                    timing["reason"] = "missing_baseline"
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
                    timing["reason"] = "no_valid_pairs"
                    continue
                # 注意：这里把全部采样帧打平合并。因此得到的是 episode 级全局
                # 直方图映射，而不是逐帧映射，更不是空间位置一一搬运。
                x = np.concatenate([v[0] for v in pairs])
                y = np.concatenate([v[1] for v in pairs])
                mapping = _fit_mapping(x, y, args.mapping, args.quantiles)
                # ---------- 5.4 在采样帧验证映射，并写全局/逐帧指标 ----------
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
                if args.report_mapping_slope:
                    slope_report = _mapping_slope_report(mapping)
                    report["mapping_slope"] = slope_report
                    report_path.with_name("mapping_slope.json").write_text(
                        json.dumps(slope_report, indent=2) + "\n")
                report_path.write_text(json.dumps(report, indent=2))
                if args.plot_trend:
                    _plot_trend(report_path.with_name("refine_trend.png"), valid_ids, frame_before, frame_after)
                timing.update(source_num_frames=len(source), sample_num_frames=len(valid_ids))
                if args.stage == "validate":
                    timing["status"] = "completed"
                    processed += 1
                    continue
                if not improved and not args.allow_unvalidated:
                    timing["reason"] = "unvalidated_transfer"
                    print(f"Skip unvalidated transfer: {source_path}", file=sys.stderr)
                    continue
                # ---------- 5.5 full 逐帧映射；rife 从映射后的关键帧补时间轴 ----------
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
                # ---------- 5.6 编码为 12-bit 视频（clip 后 float 会截断为 uint16） ----------
                if args.save_preencode_npy:
                    preencode_dir = (preencode_root / rgb_dir.parent.name /
                                     f"observation.images.{camera}_depth_refined" /
                                     source_path.stem)
                    _save_preencode_frames(output, preencode_dir)
                    report["preencode_npy_dir"] = str(preencode_dir)
                refined_dir.mkdir(parents=True, exist_ok=True)
                out_path = refined_dir / source_path.name
                rc = encoder.encode_from_arrays(output, str(out_path))
                if rc:
                    raise RuntimeError(f"ffmpeg encode failed ({rc}): {out_path}")
                report.update({"output_video": str(out_path), "output_num_frames": len(output)})
                report_path.write_text(json.dumps(report, indent=2))
                processed += 1
                timing["status"] = "completed"
    print(f"Processed {processed} episode(s); results under {stat_root}")


# ============================== 六、命令行入口 ==============================

if __name__ == "__main__":
    main()
