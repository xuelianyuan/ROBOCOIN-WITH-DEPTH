"""编排 DA3Metric → 固定 inverse-depth 归一化 → VDPP → 米制恢复。"""
from __future__ import annotations

import gc
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from ..io_utils import iter_video_frames, probe_video
from ..methods.da3_metric import MetricPredictor
from ..methods.metric_vdpp import (
    chunk_ranges,
    fit_positive_affine,
    load_vdpp_checkpoint,
    reliable_anchor_mask,
    smooth_affines,
)
from ..visualization.metric_preview import export_metric_preview

LOGGER = logging.getLogger(__name__)


@dataclass
class MetricVideoConfig:
    """单视频 POC 配置；K 必须对应输入视频的分辨率。"""

    video: str
    output: str
    da3_model: str
    vdpp_root: str
    vdpp_checkpoint: str
    intrinsics: tuple[float, float, float, float]
    intrinsics_source: str = "user-provided"
    process_res: int = 504
    batch_size: int = 2
    chunk_size: int = 32
    overlap: int = 8
    max_frames: int = 0
    device: str = "cuda"
    epsilon: float = 1e-6
    min_depth_m: float = 0.05
    max_depth_m: float = 100.0
    smooth_strength: float = 2.0


def run_metric_video(config: MetricVideoConfig) -> dict:
    """处理全部输入帧，保存 float32 米制 NPY、拟合参数、预览与运行报告。"""
    started = perf_counter()
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False))
    info = probe_video(Path(config.video))
    if config.max_frames < 0 or config.batch_size < 1 or config.epsilon <= 0:
        raise ValueError("batch_size/epsilon 必须为正，max_frames 必须非负")
    if not 0 < config.min_depth_m < config.max_depth_m:
        raise ValueError("深度上下界无效")
    fx, fy, cx, cy = config.intrinsics
    if min(fx, fy) <= 0 or not np.all(np.isfinite(config.intrinsics)):
        raise ValueError("需要有效的相机内参")
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)

    # ==================== 1. 复用公共 decoder，保留原始时间顺序 ====================
    rgb = []
    for frame in iter_video_frames(Path(config.video), "rgb24", info=info):
        rgb.append(frame)
        if config.max_frames and len(rgb) >= config.max_frames:
            break
    n, h, w = len(rgb), info["height"], info["width"]
    if not n or (not config.max_frames and info["declared_frames"] not in (None, n)):
        raise ValueError("解码帧数与视频声明不一致")
    ranges = chunk_ranges(n, config.chunk_size, config.overlap)
    LOGGER.info("解码完成: %d frames, %dx%d, %.3f fps", n, w, h, info["fps"])
    depth = np.lib.format.open_memmap(out / "da3_metric_depth_m.npy", mode="w+",
                                     dtype=np.float32, shape=(n, h, w))
    masks = np.lib.format.open_memmap(out / "anchor_mask.npy", mode="w+",
                                     dtype=bool, shape=(n, h, w))

    # ==================== 2. 用 resize 后内参转换 DA3 canonical 输出 ====================
    predictor = MetricPredictor(config.da3_model, config.process_res, config.device)
    for start in range(0, n, config.batch_size):
        end = min(n, start + config.batch_size)
        base, flipped, non_sky = predictor.predict(rgb[start:end], intrinsic)
        if not np.all(np.isfinite(base) & (base > 0)):
            raise ValueError(f"DA3 输出含非正或非有限深度: frames {start}:{end}")
        depth[start:end] = base
        for j, i in enumerate(range(start, end)):
            masks[i] = reliable_anchor_mask(rgb[i-1] if i else None, rgb[i],
                                             base[j], flipped[j], non_sky[j])
        if start % (config.batch_size * 10) == 0 or end == n:
            depth.flush()
            masks.flush()
            LOGGER.info("DA3Metric + flip + visibility: %d/%d", end, n)
    processed_k, output_keys = predictor.processed_intrinsics, predictor.output_keys
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    da3_seconds = perf_counter() - started

    # ==================== 3. 整段视频的固定 P2/P98，无逐帧 min-max ====================
    q = 1.0 / (np.asarray(depth) + config.epsilon)  # (T,H,W), float32, 单位 1/m
    p2, p98 = map(float, np.percentile(q, [2, 98]))
    if p98 - p2 < 1e-6:
        raise ValueError("整段 inverse depth 的 P2/P98 退化")
    masks[:] &= (q > p2) & (q < p98)
    masks.flush()
    normalized = np.clip((q - p2) / (p98 - p2), 0, 1)
    LOGGER.info("固定 inverse depth P2=%.6f, P98=%.6f 1/m", p2, p98)

    # ==================== 4. 直接调用 VDPP forward，替代其无约束 streaming 对齐 ====================
    sys.path.insert(0, str(Path(config.vdpp_root).resolve()))
    from vdpp.vdpp_model import VDPP
    vdpp = VDPP(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).to(config.device).eval()
    ignored_checkpoint_keys = load_vdpp_checkpoint(vdpp, config.vdpp_checkpoint)
    chunk_dir = out / "vdpp_chunks"
    chunk_dir.mkdir()
    anchors, overlaps, fit_records = [], [], []
    previous = None
    for ci, (start, end) in enumerate(ranges):
        # VDPP 要求空间尺寸为 14 的倍数；缩放后输出重新恢复为原视频尺寸。
        frames = torch.from_numpy(normalized[start:end]).to(config.device)[None]
        hh, ww = ((h + 13)//14)*14, ((w + 13)//14)*14
        frames = torch.nn.functional.interpolate(frames, size=(hh, ww), mode="bilinear", align_corners=True)
        device_type = torch.device(config.device).type
        autocast_dtype = torch.bfloat16 if device_type == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
            pred = vdpp(frames, downsize=True)
        pred = torch.nn.functional.interpolate(pred.float(), size=(h,w), mode="bilinear", align_corners=True)[0]
        current = pred.cpu().numpy()
        if not np.isfinite(current).all():
            raise ValueError(f"VDPP chunk {ci} 含非有限值")
        mask = masks[start:end]
        a, b, quality = fit_positive_affine(current[mask], q[start:end][mask])
        anchors.append((a, b))
        if previous is not None:
            ol = ranges[ci-1][1] - start
            overlap_mask = masks[start:start+ol]
            s, t, overlap_quality = fit_positive_affine(current[:ol][overlap_mask], previous[-ol:][overlap_mask])
            overlaps.append((s, t))
        else:
            overlap_quality = None
        np.save(chunk_dir / f"chunk_{ci:04d}.npy", current)
        fit_records.append({"start":start, "end":end, "a":a, "b":b,
                            "anchor_fit": quality, "overlap_fit":overlap_quality})
        previous = current
        LOGGER.info("VDPP chunk %d/%d [%d,%d): a=%.6f b=%.6f", ci+1, len(ranges), start,end,a,b)
    del vdpp, frames, pred, previous, normalized
    gc.collect()
    torch.cuda.empty_cache()

    # ==================== 5. 平滑参数并在 inverse depth 空间羽化重叠区域 ====================
    parameters = smooth_affines(anchors, overlaps, strength=config.smooth_strength)
    result = np.lib.format.open_memmap(out / "depth_m.npy", mode="w+", dtype=np.float32, shape=(n,h,w))
    weights = np.zeros(n, np.float32)
    result[:] = 0
    clipped_count = 0
    for ci, (start,end) in enumerate(ranges):
        current = np.load(chunk_dir / f"chunk_{ci:04d}.npy", mmap_mode="r")
        a,b = parameters[ci]
        recovered = a * current + b
        blend = np.ones(end-start, np.float32)
        if ci:
            ol = ranges[ci-1][1]-start
            blend[:ol] *= np.linspace(0,1,ol+2)[1:-1]
        if ci+1 < len(ranges):
            ol = end-ranges[ci+1][0]
            blend[-ol:] *= np.linspace(1,0,ol+2)[1:-1]
        result[start:end] += recovered * blend[:,None,None]
        weights[start:end] += blend
        fit_records[ci]["smoothed_a"], fit_records[ci]["smoothed_b"] = map(float,(a,b))
    for i in range(n):
        recovered = result[i] / weights[i]
        low_q = 1/(config.max_depth_m + config.epsilon)
        high_q = 1/(config.min_depth_m + config.epsilon)
        clipped_count += int(((recovered<low_q)|(recovered>high_q)).sum())
        result[i] = 1/np.clip(recovered, low_q, high_q) - config.epsilon
    result.flush()

    # ==================== 6. 保存可审查的统计、参数及固定色标预览 ====================
    # 以下只是对 DA3 的变化量和未补偿运动的相邻帧差；没有 GT，不能据此宣称精度提升。
    relative_change, temporal_base, temporal_result = [], [], []
    for i in range(n):
        relative_change.append(float(np.median(np.abs(result[i]-depth[i])/depth[i])))
        if i:
            temporal_base.append(float(np.mean(np.abs(depth[i]-depth[i-1]))))
            temporal_result.append(float(np.mean(np.abs(result[i]-result[i-1]))))
    limits = tuple(map(float,np.percentile(np.asarray(depth)[::5,::4,::4],[2,98])))
    export_metric_preview(rgb, depth, result, out / "comparison.mp4", info["fps"], limits)
    report = {"status":"completed", "frames":n, "shape":[n,h,w], "fps":info["fps"],
              "duration_s":n/info["fps"], "unit":"meters", "dtype":"float32",
              "intrinsics_original":intrinsic.tolist(), "intrinsics_processed":processed_k,
              "intrinsics_source":config.intrinsics_source,
              "vdpp_ignored_checkpoint_keys":ignored_checkpoint_keys,
              "da3_output_keys":output_keys, "confidence_source":"flip consistency proxy; no DA3Metric confidence head",
              "normalization":{"scope":"entire video", "p2_inverse_m":p2, "p98_inverse_m":p98,
                               "exact_percentiles":True, "per_frame_min_max":False},
              "anchor_fraction":float(np.count_nonzero(masks)/masks.size),
              "chunks":fit_records, "overlap_relations":overlaps,
              "clipped_depth_fraction":clipped_count/result.size,
              "median_frame_median_relative_change_to_da3":float(np.median(relative_change)),
              "raw_temporal_mae_m":{"da3":float(np.mean(temporal_base)) if temporal_base else None,
                                     "vdpp":float(np.mean(temporal_result)) if temporal_result else None,
                                     "motion_compensated":False},
              "preview_depth_limits_m":limits, "da3_stage_seconds":da3_seconds,
              "total_seconds":perf_counter()-started,
              "limitations":["默认相机内参未独立标定；metric scale 仅为近似",
                             "DA3Metric 无置信度头，以 flip 一致性作替代",
                             "无 ground truth；相邻帧差含真实运动，不能证明稳定性或精度提升",
                             "未重训 VDPP；局部几何可能改变"]}
    (out / "report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (out / "REPORT.md").write_text(
        f"# DA3Metric + 尺度保持 VDPP POC\n\n"
        f"输入：`{config.video}`\n\n"
        f"完成 {n} 帧，{w}×{h}，{info['fps']} fps；输出 `depth_m.npy` 为 float32 米制深度。\n\n"
        f"原始 DA3 米制深度：`da3_metric_depth_m.npy`；可靠区域：`anchor_mask.npy`；"
        f"对照：`comparison.mp4`（RGB / DA3Metric / VDPP，统一固定米制色标）。\n\n"
        f"整段视频精确 inverse-depth P2/P98：{p2:.8f} / {p98:.8f} m⁻¹；"
        f"{len(ranges)} 个 chunk，长度 {config.chunk_size}，重叠 {config.overlap} 帧。\n\n"
        f"DA3 原始 canonical depth × resize 后平均焦距 / 300 得到米制深度。"
        f"可靠区域采用 flip 相对误差<10%、非天空、非深度边缘及双向光流一致性<1 pixel。"
        f"各块全局正仿射 Huber 拟合，重叠参数联合平滑，再对 inverse depth 羽化混合。\n\n"
        f"可靠锚点比例：{report['anchor_fraction']:.2%}；"
        f"与 DA3 的帧中位相对变化的中位数：{report['median_frame_median_relative_change_to_da3']:.2%}；"
        f"输出深度边界裁剪比例：{report['clipped_depth_fraction']:.4%}。\n\n"
        + "\n".join(f"- {item}" for item in report["limitations"]) + "\n\n"
        + "完整内参、拟合系数、统计及运行时间见 `report.json`，复现配置见 `config.json`。\n")
    LOGGER.info("完成: %s (%.1fs)", out, report["total_seconds"])
    return report
