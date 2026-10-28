"""方案 A 的正仿射尺度恢复、可靠区域筛选和重叠参数平滑。"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.optimize import lsq_linear


def load_vdpp_checkpoint(
    model: torch.nn.Module,
    checkpoint: str,
) -> tuple[str, ...]:
    """加载官方 v1.0 权重，只兼容已从当前模型删除的 shift head 参数。

    官方 release 权重仍包含两个 ``shift_head`` 参数，而当前源码从未定义对应层。
    这里不采用宽松的静默加载：缺失参数或其他未知参数仍会终止运行。
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_unexpected = {"shift_head.0.weight", "shift_head.0.bias"}
    unexpected = set(incompatible.unexpected_keys)
    if incompatible.missing_keys or not unexpected <= allowed_unexpected:
        raise RuntimeError(
            "VDPP checkpoint 与模型不兼容："
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return tuple(sorted(unexpected))


def fit_positive_affine(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int = 100000,
    iterations: int = 12,
) -> tuple[float, float, dict]:
    """用 Huber IRLS 拟合 y≈a*x+b，约束 a>0；退化输入直接报错。"""
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) > max_samples:
        ids = np.random.default_rng(0).choice(len(x), max_samples, replace=False)
        x, y = x[ids], y[ids]
    if len(x) < 100 or np.std(x) < 1e-7:
        raise ValueError("可靠锚点不足，或 VDPP 输出退化，不能恢复 metric scale")
    # 中心化和标准化改善正规方程条件数；正斜率约束不会约束截距。
    x, y = x.astype(np.float64), y.astype(np.float64)
    center, spread = x.mean(), x.std()
    design = np.column_stack(((x - center) / spread, np.ones_like(x)))
    weights = np.ones_like(x)
    for _ in range(iterations):
        root = np.sqrt(weights)
        result = lsq_linear(
            design * root[:, None],
            y * root,
            bounds=([1e-8, -np.inf], [np.inf, np.inf]),
        )
        residual = design @ result.x - y
        sigma = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-6)
        weights = np.minimum(1.0, 1.345 * sigma / np.maximum(np.abs(residual), 1e-12))
    a = float(result.x[0] / spread)
    b = float(result.x[1] - a * center)
    return a, b, {
        "samples": len(x),
        "median_abs_error": float(np.median(np.abs(residual))),
        "p95_abs_error": float(np.percentile(np.abs(residual), 95)),
    }


def reliable_anchor_mask(
    rgb_previous: np.ndarray | None,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    flipped_depth_m: np.ndarray,
    non_sky: np.ndarray,
    *,
    flip_threshold: float = 0.10,
    edge_threshold: float = 0.05,
) -> np.ndarray:
    """组合 flip 一致性、非天空、非深度边缘和双向光流可见性；非模型置信度。"""
    valid = np.isfinite(depth_m) & (depth_m > 0) & non_sky
    valid &= np.abs(depth_m - flipped_depth_m) / np.maximum(depth_m, 1e-6) < flip_threshold
    # 深度不连续附近容易发生遮挡，额外排除并腐蚀一圈。
    log_depth = np.log(np.maximum(depth_m, 1e-6))
    gy, gx = np.gradient(log_depth)
    valid &= np.hypot(gx, gy) < edge_threshold
    if rgb_previous is not None:
        h, w = depth_m.shape
        size = (max(32, w // 2), max(32, h // 2))
        prev = cv2.cvtColor(cv2.resize(rgb_previous, size), cv2.COLOR_RGB2GRAY)
        curr = cv2.cvtColor(cv2.resize(rgb, size), cv2.COLOR_RGB2GRAY)
        args = (None, 0.5, 3, 15, 3, 5, 1.2, 0)
        backward = cv2.calcOpticalFlowFarneback(curr, prev, *args)
        forward = cv2.calcOpticalFlowFarneback(prev, curr, *args)
        yy, xx = np.mgrid[:size[1], :size[0]].astype(np.float32)
        mx, my = xx + backward[..., 0], yy + backward[..., 1]
        reverse = cv2.remap(forward, mx, my, cv2.INTER_LINEAR)
        visible = (
            (mx >= 0)
            & (mx < size[0] - 1)
            & (my >= 0)
            & (my < size[1] - 1)
            & (np.linalg.norm(backward + reverse, axis=-1) < 1.0)
        )
        visible_full_size = cv2.resize(
            visible.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        valid &= visible_full_size
    return cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def chunk_ranges(count: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """生成有重叠、覆盖所有帧且不会增加纯重叠尾块的区间。"""
    if count <= 0 or size < 2 or not 0 < overlap < size:
        raise ValueError("需要 count>0 且 0<overlap<chunk_size")
    result, start = [], 0
    while True:
        end = min(start + size, count)
        result.append((start, end))
        if end == count:
            return result
        start += size - overlap


def smooth_affines(
    anchors: list[tuple[float, float]],
    overlaps: list[tuple[float, float]],
    *,
    strength: float = 2.0,
) -> np.ndarray:
    """联合各块 metric 锚点和重叠关系，平滑正斜率 a 与截距 b。

    overlaps[i] 表示 v_i≈s*v_(i+1)+t，因此约束
    a_(i+1)≈s*a_i、b_(i+1)≈t*a_i+b_i，避免无约束逐块漂移。
    """
    n = len(anchors)
    if not n or len(overlaps) != n - 1:
        raise ValueError("重叠参数数量应为块数减一")
    rows, targets = [], []
    for i, (a, b) in enumerate(anchors):
        for j, value in enumerate((a, b)):
            row = np.zeros(2 * n)
            row[2 * i + j] = 1
            rows.append(row)
            targets.append(value)
    for i, (s, t) in enumerate(overlaps):
        row = np.zeros(2 * n)
        row[2 * (i + 1)], row[2 * i] = 1, -s
        rows.append(row * strength)
        targets.append(0)
        row = np.zeros(2 * n)
        row[2 * (i + 1) + 1], row[2 * i + 1], row[2 * i] = 1, -1, -t
        rows.append(row * strength)
        targets.append(0)
    low = np.tile([1e-8, -np.inf], n)
    result = lsq_linear(np.asarray(rows), np.asarray(targets), bounds=(low, np.inf))
    if not result.success:
        raise RuntimeError(f"参数平滑失败: {result.message}")
    return result.x.reshape(n, 2)
