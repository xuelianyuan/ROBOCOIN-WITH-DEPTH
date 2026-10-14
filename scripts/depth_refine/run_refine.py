#!/usr/bin/env python3
"""用 Nested 稀疏参考校准原始深度视频的数值分布。

默认逐帧搬运：每个源帧与同帧 Nested 参考独立拟合 ``Nested=f(Source)``，
transport 方法保存该帧直方图的稀疏 OT 矩阵，并通过行重心构造数值映射。
这不是复制 Nested 的图像内容，也不是空间滤波。因此输出保留源深度的边缘、噪声和局部纹理；
分位数曲线斜率较陡或有平台时，还可能放大源图已有的小幅起伏。

数据链路：

1. 从 ``videos/`` 解码原始深度视频（默认按 gray12le 数值读取）。
2. 从 ``depth_results/baseline/statistics/`` 读取 Nested 采样帧 NPY；若尺寸
   不同，先把 Nested 双线性缩放至源深度尺寸，仅用于配对、拟合和验证。
3. ``transport`` 求解每帧直方图最优传输；``quantile`` 做分位数匹配；
   ``linear`` 在源深度 1%--99% 区间拟合直线。``--mapping-scope episode``
   可恢复合并采样像素、拟合全局映射的旧行为。
4. ``validate`` 只处理参考帧并写矩阵/报告；逐帧 ``full`` 要求完整同帧参考。
   ``rife`` 仅适用于 episode scope，由 RGB 引导补齐时间轴。
5. 输出先裁剪至 ``max_depth_mm``，随后编码器再裁剪至 12-bit [0, 4095]、
   转为 uint16，并用 gray12le/libx265 写视频；当前调用不会做 min-max 归一化。

诊断选项：``--report-mapping-slope`` 输出映射曲线各段的斜率摘要和最陡区间；
``--save-preencode-npy`` 将 full/rife 的编码前 float32 帧逐帧保存，便于和视频
解码结果比较，从而区分映射伪影与量化、编码伪影。

所以报告中的直方图改善不保证每个空间区域都改善。诊断灰度色斑
时，应分别比较映射前、映射后编码前数组、视频解码值和显示归一化结果。

用法示例::

    python scripts/depth_refine/run_refine.py \
        --dataset-dir data/lerobot/your_dataset \
        --result-root data/lerobot/your_dataset/depth_results \
        --camera head --mapping transport --stage validate --plot-trend

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.depth_refine.methods.mapping import (
    _valid_pair, _fit_mapping, _apply_mapping, _mapping_slope_report,
)
from src.depth_refine.pipeline.frame_transfer import (
    transfer_frames, require_frame_references, frame_timing_summary,
)
from src.depth_refine.timing import episode_timer
from src.depth_refine.io_utils import iter_video_frames
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
    return list(iter_video_frames(path, pix_fmt))


# ======================== 二、有效像素配对与评估指标 ========================

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
    """逐相机、逐 episode 执行逐帧或兼容全局映射，再验证或生成视频。"""
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
    p.add_argument("--mapping", default="transport", choices=["transport", "quantile", "linear"],
                   help="transport 保存真实直方图 OT 矩阵；quantile 为分位数插值；linear 为直线")
    p.add_argument("--mapping-scope", default="frame", choices=["frame", "episode"],
                   help="默认每帧独立拟合；episode 恢复合并采样像素的旧行为")
    p.add_argument("--transport-bins", type=int, default=256,
                   help="transport 的直方图 bin 数，至少 2；默认 256")
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
    if args.transport_bins < 2:
        p.error("--transport-bins 必须至少为 2")
    if not np.isfinite(args.max_depth_mm) or args.max_depth_mm <= 0:
        p.error("--max-depth-mm 必须为有限正数")
    if args.stage == "rife" and args.mapping_scope == "frame":
        p.error("rife 插值帧没有独立搬运矩阵；逐帧搬运请使用 --stage full，"
                "旧 RIFE 流程请显式传入 --mapping-scope episode")

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
                               source=source_path, mapping=args.mapping,
                               mapping_scope=args.mapping_scope) as timing:
                ep_out = _statistics_episode(stat_root, camera, source_path.stem)
                if ep_out is None:
                    timing["reason"] = "missing_baseline"
                    continue
                metadata = json.loads((ep_out / "sample_indices.json").read_text())
                indices = [int(i) for i in metadata["sample_frame_indices"]]
                source = _decode(source_path, args.input_pix_fmt)
                if args.mapping_scope == "frame":
                    valid_ids = require_frame_references(
                        ep_out, indices, len(source), full=args.stage == "full")
                else:
                    valid_ids = [i for i in indices if 0 <= i < len(source)
                                 and (ep_out / f"frame_{i:09d}.npy").exists()]
                report_path = report_root / camera / source_path.stem / "refine_parameters.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                # 逐帧 full 的输出直接复用这里的结果，避免验证和输出重复搬运计时。
                frame_transfers = []
                if args.mapping_scope == "frame":
                    frame_report_dir = report_path.parent / args.stage
                    timing["frame_timings_csv"] = str((frame_report_dir / "frame_timings.csv").resolve())
                    transferred, frame_transfers = transfer_frames(
                        source, ep_out, valid_ids, frame_report_dir, method=args.mapping,
                        quantiles=args.quantiles, transport_bins=args.transport_bins,
                        max_depth_mm=args.max_depth_mm)
                    for record in frame_transfers:
                        for key in ("artifact_path", "matrix_path"):
                            if record[key]:
                                record[key] = f"{args.stage}/{record[key]}"
                    mapping = {"method": args.mapping, "scope": "frame", "frame_count": len(frame_transfers)}
                    timing["frame_timing_summary"] = frame_timing_summary(frame_transfers)
                nested = [np.load(ep_out / f"frame_{i:09d}.npy") for i in valid_ids]
                sampled = [source[i] for i in valid_ids]
                if args.mapping_scope == "episode":
                    pairs = [_valid_pair(x, y) for x, y in zip(sampled, nested)]
                    pairs = [(x, y) for x, y in pairs if x.size]
                    if not pairs:
                        timing["reason"] = "no_valid_pairs"
                        continue
                    # 仅兼容模式合并采样帧，所有输出帧共用这条全局映射。
                    x = np.concatenate([v[0] for v in pairs])
                    y = np.concatenate([v[1] for v in pairs])
                    mapping = _fit_mapping(x, y, args.mapping, args.quantiles, args.transport_bins)
                    transferred = [_apply_mapping(f, mapping, args.max_depth_mm) for f in sampled]
                # ---------- 5.4 在采样帧验证映射，并写全局/逐帧指标 ----------
                before, after = _metrics(sampled, nested), _metrics(transferred, nested)
                improved = all(after.get(k, float("inf")) < before.get(k, float("inf"))
                               for k in ("mae_mm", "rmse_mm", "wasserstein_mm"))
                frame_before = [_metrics([a], [b]) for a, b in zip(sampled, nested)]
                frame_after = [_metrics([a], [b]) for a, b in zip(transferred, nested)]
                report = {"mapping": mapping, "mapping_scope": args.mapping_scope,
                          "stage": args.stage, "baseline": before,
                          "transferred": after, "improved": improved, "sample_indices": valid_ids,
                          "frame_metrics": [{"frame_index": i, "baseline": b, "transferred": a}
                                            for i, b, a in zip(valid_ids, frame_before, frame_after)]}
                if frame_transfers:
                    report.update(frame_mappings=frame_transfers,
                                  frame_timings_csv=f"{args.stage}/frame_timings.csv",
                                  frame_timing_summary=frame_timing_summary(frame_transfers))
                if args.report_mapping_slope:
                    slope_report = ({"scope": "frame", "frames": [
                        {"frame_index": r["frame_index"], **_mapping_slope_report(r["mapping"])}
                        for r in frame_transfers]} if frame_transfers else _mapping_slope_report(mapping))
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
                    output = (transferred if args.mapping_scope == "frame" else
                              [_apply_mapping(f, mapping, args.max_depth_mm) for f in source])
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
