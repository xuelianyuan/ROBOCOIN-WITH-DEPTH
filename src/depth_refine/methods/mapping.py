"""深度有效像素配对、标量映射及斜率诊断。"""
from __future__ import annotations

import cv2
import numpy as np

from .histogram_transport import fit_histogram_transport


def _valid_pair(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Nested 输出尺寸可能与原始深度视频不同；将参考深度缩放到源帧尺寸后
    # 才能建立逐像素对应关系。深度是连续数值，使用线性插值而非 RGB 插值。
    if pred.shape != target.shape:
        target = cv2.resize(target.astype(np.float32),
                            (pred.shape[1], pred.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    mask = np.isfinite(pred) & np.isfinite(target) & (pred > 0) & (target > 0)
    return pred[mask].astype(np.float64), target[mask].astype(np.float64)


def _fit_mapping(x: np.ndarray, y: np.ndarray, method: str, quantiles: int,
                 transport_bins: int = 256) -> dict:
    """拟合输入像素的数值映射；调用方决定逐帧拟合或合并 episode。"""
    if not x.size or not y.size:
        raise ValueError("mapping requires valid source/target pairs")
    if method == "transport":
        return fit_histogram_transport(x, y, transport_bins)
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

    source_key, target_key = (("source_depth_mm", "mapped_depth_mm")
                              if mapping["method"] == "transport" else
                              ("source_quantiles_mm", "target_quantiles_mm"))
    source = np.asarray(mapping[source_key], dtype=np.float64)
    target = np.asarray(mapping[target_key], dtype=np.float64)
    source_delta = np.diff(source)
    target_delta = np.diff(target)
    usable = np.isfinite(source_delta) & np.isfinite(target_delta) & (source_delta > 0)
    indices = np.flatnonzero(usable)
    slopes = target_delta[usable] / source_delta[usable]
    if not slopes.size:
        return {"method": mapping["method"], "segment_count": 0,
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
        "method": mapping["method"], "segment_count": int(slopes.size),
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
    elif mapping["method"] == "transport":
        out[valid] = np.interp(frame[valid], mapping["source_depth_mm"], mapping["mapped_depth_mm"])
    # 将 Metric 的深度值映射到 Nested 的深度分布，并不考虑相邻像素，也不会用 Nested 的局部形状替换原图。因此：
    #   - 原图中的局部深度起伏会被保留。
    #   - 映射曲线较陡的区间，会把微小深度差放大，显示成更明显的灰度斑块。
    #   - Nested 自身更平滑，但匹配它的数值分布，并不能获得同样平滑的画面。
    else:
        out[valid] = np.interp(
            frame[valid], mapping["source_quantiles_mm"], mapping["target_quantiles_mm"],
        )
    return np.clip(out, 0, max_depth)
