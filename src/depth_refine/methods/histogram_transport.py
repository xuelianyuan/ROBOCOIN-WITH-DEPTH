"""一维深度直方图的单调最优传输，保存可重建的稀疏质量矩阵。"""
from __future__ import annotations

import numpy as np


def _histogram(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    # 每个非空 bin 用实际像素均值代表，避免常量帧或窄范围被 bin 中点偏移。
    counts, edges = np.histogram(values, bins=bins)
    sums, _ = np.histogram(values, bins=edges, weights=values)
    occupied = counts > 0
    return sums[occupied] / counts[occupied], counts[occupied] / values.size


def fit_histogram_transport(x: np.ndarray, y: np.ndarray, bins: int = 256) -> dict:
    """拟合深度直方图 OT，并用矩阵的行重心构造标量映射（单位 mm）。

    transport 中 row/col/mass 表示 T[row, col] 的非零项，shape 为
    (源非空 bin 数, 目标非空 bin 数)。矩阵总质量为 1，行/列边缘分布
    分别等于源/目标直方图。bins 控制离散精度，非像素空间的搬运矩阵。
    """
    if bins < 2:
        raise ValueError("transport bins must be at least 2")
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    if (not x.size or not y.size or not np.all(np.isfinite(x) & (x > 0))
            or not np.all(np.isfinite(y) & (y > 0))):
        raise ValueError("transport requires finite positive depth samples")

    # 一、分别离散源/目标直方图；范围随输入变化，超出编码上限的参考仍参与拟合。
    source, source_mass = _histogram(x, bins)
    target, target_mass = _histogram(y, bins)

    # 二、两个 CDF 的区间交集就是单调 OT 质量。只存非零项，空间开销 O(bins)。
    # 对一维 |x-y| 或平方距离，这种单调耦合都是最优解，无需迭代 Sinkhorn。
    source_cdf = np.r_[0.0, np.cumsum(source_mass)]
    target_cdf = np.r_[0.0, np.cumsum(target_mass)]
    source_cdf[-1] = target_cdf[-1] = 1.0
    boundaries = np.unique(np.r_[source_cdf, target_cdf])
    mass = np.diff(boundaries)
    midpoints = boundaries[:-1] + mass / 2
    row = np.minimum(np.searchsorted(source_cdf[1:], midpoints, side="right"), source.size - 1)
    col = np.minimum(np.searchsorted(target_cdf[1:], midpoints, side="right"), target.size - 1)

    # 三、一个源 bin 可能分流到多个目标 bin；输出取该行的条件期望。
    # 这是确定性重心投影，重复源值不能凭空拆成多个输出值，因此不保证
    # 映射后的直方图精确等于目标；矩阵本身仍满足两侧的质量守恒。
    mapped = np.bincount(row, weights=mass * target[col], minlength=source.size) / source_mass
    return {
        "method": "transport", "transport_bins": bins, "fit_samples": int(x.size),
        "source_depth_mm": source.tolist(), "mapped_depth_mm": mapped.tolist(),
        "transport": {
            "row": row.tolist(), "col": col.tolist(), "mass": mass.tolist(),
            "shape": [int(source.size), int(target.size)],
            "source_depth_mm": source.tolist(), "target_depth_mm": target.tolist(),
            "source_mass": source_mass.tolist(), "target_mass": target_mass.tolist(),
        },
    }
