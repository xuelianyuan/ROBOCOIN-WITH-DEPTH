"""云状灰度斑块的启发式候选检测及连续区间统计，不进行模型归因。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CloudConfig:
    """阈值均在逐帧 P1–P99 归一化的数值域内；像素尺度按 analysis_width。"""

    analysis_width: int = 320
    amplitude: float = 0.008
    coverage: float = 0.02
    edge_gradient: float = 0.025
    anisotropy: float = 0.65
    min_component_pixels: int = 20
    min_assessable_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.analysis_width < 64 or self.min_component_pixels < 1:
            raise ValueError("analysis_width >= 64 and min_component_pixels >= 1 required")
        for name in ("amplitude", "coverage", "edge_gradient", "anisotropy",
                     "min_assessable_fraction"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie inside (0, 1)")


def detect_cloud(frame: np.ndarray, config: CloudConfig,
                 *, valid_max: float = 4095) -> tuple[dict, np.ndarray, np.ndarray]:
    """返回候选覆盖率、归一化图和候选 mask；不参考前后帧，避免强加连续性。

    使用两个尺度的带通起伏，排除强边缘及方向单一的轮廓，再过滤小连通域。
    coverage 的分母为全部有效像素，避免很小的可分析区域放大得分。
    自然曲面、弱轮廓仍可能误报，因此结果统一称为 suspected cloud。
    """
    if frame.ndim != 2 or min(frame.shape) < 32:
        raise ValueError("Depth frame must be a 2D image at least 32x32")
    depth = frame.astype(np.float32)
    if depth.shape[1] > config.analysis_width:
        height = round(depth.shape[0] * config.analysis_width / depth.shape[1])
        depth = cv2.resize(depth, (config.analysis_width, height), interpolation=cv2.INTER_AREA)
    valid = np.isfinite(depth) & (depth > 0) & (depth < valid_max)
    mask = np.zeros(depth.shape, dtype=bool)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    result = {"score": 0.0, "suspected": False, "assessable": False,
              "valid_fraction": float(valid.mean()), "assessable_fraction": 0.0,
              "cloud_pixels": 0, "valid_pixels": int(valid.sum()),
              "p01": None, "p99": None}
    if not valid.any():
        return result, gray, mask
    lo, hi = np.percentile(depth[valid], [1, 99])
    result.update(p01=float(lo), p99=float(hi))
    # 无效洞不参与滤波：先填入中位深度，随后排除洞及其滤波邻域。
    depth[~valid] = np.median(depth[valid])
    z = ((depth - lo) / max(float(hi - lo), 1.0)).astype(np.float32)
    gray[valid] = np.clip(z[valid] * 255, 0, 255).astype(np.uint8)
    fine = cv2.GaussianBlur(z, (0, 0), 1)
    gx = cv2.Sobel(fine, cv2.CV_32F, 1, 0) / 8
    gy = cv2.Sobel(fine, cv2.CV_32F, 0, 1) / 8
    excluded = (~valid) | (np.maximum(np.abs(gx), np.abs(gy)) > config.edge_gradient)
    interior = cv2.dilate(excluded.astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
    interior[:12] = False
    interior[-12:] = False
    interior[:, :12] = False
    interior[:, -12:] = False
    result["assessable_fraction"] = float(interior.mean())
    result["assessable"] = bool(interior.mean() >= config.min_assessable_fraction)
    if not result["assessable"] or hi - lo < 1:
        return result, gray, mask

    # 结构张量的方向性接近 1 时通常为单一边缘；云状纹理方向更分散。
    xx = cv2.GaussianBlur(gx * gx, (0, 0), 4)
    yy = cv2.GaussianBlur(gy * gy, (0, 0), 4)
    xy = cv2.GaussianBlur(gx * gy, (0, 0), 4)
    directionality = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2) / (xx + yy + 1e-10)
    broad = np.abs(fine - cv2.GaussianBlur(z, (0, 0), 6))
    detail = np.abs(fine - cv2.GaussianBlur(z, (0, 0), 3))
    candidate = (interior & (directionality < config.anisotropy)
                 & (broad > config.amplitude) & (detail > config.amplitude * 0.35))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= config.min_component_pixels
    keep[0] = False
    mask = keep[labels]
    result["cloud_pixels"] = int(mask.sum())
    result["score"] = result["cloud_pixels"] / result["valid_pixels"]
    result["suspected"] = bool(result["score"] >= config.coverage)
    return result, gray, mask


def cloud_segments(flags: Sequence[bool], fps: float) -> list[dict]:
    """逐视频提取连续阳性帧；帧号闭区间、秒数半开区间，不跨阴性或未知帧。"""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    padded = np.r_[False, np.asarray(flags, dtype=bool), False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return [{"start_frame": int(start), "end_frame": int(end),
             "frame_count": int(end - start + 1), "start_seconds": float(start / fps),
             "end_seconds_exclusive": float((end + 1) / fps),
             "duration_seconds": float((end - start + 1) / fps)}
            for start, end in zip(starts, ends)]


def summarize_segments(segments: Sequence[dict], total_frames: int,
                       assessable_frames: int) -> dict:
    """同时汇总单帧事件、连续事件以及各自包含的阳性帧数量。"""
    positive = sum(s["frame_count"] for s in segments)
    singles = sum(s["frame_count"] == 1 for s in segments)
    consecutive = positive - singles
    fraction = consecutive / positive if positive else None
    pattern = ("none_detected" if not positive else "segment_dominant" if fraction >= 0.8
               else "frame_dominant" if fraction <= 0.2 else "mixed")
    return {"total_frames": total_frames, "assessable_frames": assessable_frames,
            "unknown_frames": total_frames - assessable_frames,
            "suspected_frames": positive,
            "suspected_fraction_all_frames": positive / total_frames if total_frames else None,
            "suspected_fraction_assessable": positive / assessable_frames if assessable_frames else None,
            "events": len(segments), "isolated_single_frames": singles,
            "consecutive_events": len(segments) - singles,
            "frames_in_consecutive_events": consecutive,
            "consecutive_frame_fraction": fraction,
            "events_at_least_5_frames": sum(s["frame_count"] >= 5 for s in segments),
            "frames_in_events_at_least_5": sum(s["frame_count"] for s in segments if s["frame_count"] >= 5),
            "longest_event_frames": max((s["frame_count"] for s in segments), default=0),
            "longest_event_seconds": max((s["duration_seconds"] for s in segments), default=0),
            "pattern": pattern}


def config_dict(config: CloudConfig) -> dict:
    """将实际配置写入统计报告，保证结果可复现。"""
    return asdict(config)
