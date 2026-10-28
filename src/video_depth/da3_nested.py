"""
da3_nested.py — DA3 Nested 序列窗口推理与度量尺度融合

输入按时间排序的 RGB 关键帧，输出原分辨率 ``float32`` 米制深度和置信度。
模型只加载一次；长序列以重叠窗口推理，融合只允许受限的小尺度校正。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np
import torch

from src.video_depth.config import DA3Config
from src.video_depth.types import KeyframeDepth

LOGGER = logging.getLogger(__name__)


class DA3NestedEstimator:
    """延迟加载并复用 DA3 Nested 模型。

    Args:
        config: DA3 窗口与尺度校正配置。
        device: 显式 PyTorch 设备，例如 ``cuda:0`` 或 ``cpu``。
        model_factory: 测试注入点；接收模型 ID 并返回兼容 ``inference`` 的对象。
    """

    def __init__(
        self,
        config: DA3Config,
        device: str,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self._model_factory = model_factory
        self._model: Any | None = None

    def _load_model(self) -> Any:
        """首次使用时加载权重，之后返回同一实例。"""

        if self._model is not None:
            return self._model
        if self._model_factory is None:
            try:
                from depth_anything_3.api import DepthAnything3
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "DA3 运行依赖未安装；请在 Python 3.11 环境执行 "
                    "`pip install -e third_party/Depth-Anything-3`"
                ) from exc

            try:
                model = DepthAnything3.from_pretrained(self.config.model_id)
            except OSError as exc:
                raise RuntimeError(
                    f"无法加载 DA3 权重 {self.config.model_id}；请检查缓存或网络"
                ) from exc
        else:
            model = self._model_factory(self.config.model_id)
        model = model.to(self.device)
        model.eval()
        self._model = model
        return model

    def infer(self, frames: Sequence[np.ndarray]) -> KeyframeDepth:
        """按有序窗口推理关键帧序列。

        Args:
            frames: 按时间顺序排列的 ``uint8 RGB (H,W,3)`` 帧。
        Returns:
            米制 ``float32 (K,H,W)`` 深度与同 shape 置信度。
        """

        _validate_rgb_frames(frames)
        count = len(frames)
        height, width = frames[0].shape[:2]
        windows = make_windows(count, self.config.window_size, self.config.overlap)
        # 长视频在原分辨率累计；float32 足以承载有限窗口的加权和，并将三块
        # 累计缓存的内存占用减半，避免 CPU 验证时无谓占用数 GB 内存。
        depth_sum = np.zeros((count, height, width), dtype=np.float32)
        conf_sum = np.zeros_like(depth_sum)
        weight_sum = np.zeros_like(depth_sum)
        previous: tuple[int, int, np.ndarray, np.ndarray] | None = None
        model = self._load_model()

        for start, end in windows:
            with torch.inference_mode():
                prediction = model.inference(
                    list(frames[start:end]),
                    process_res=self.config.process_res,
                    process_res_method="upper_bound_resize",
                )
            if int(getattr(prediction, "is_metric", 1)) != 1:
                raise RuntimeError(
                    f"DA3 模型 {self.config.model_id} 未声明 metric 输出，拒绝当作米制深度"
                )
            depth = _resize_stack(prediction.depth, height, width)
            raw_conf = getattr(prediction, "conf", None)
            confidence = (
                np.ones_like(depth, dtype=np.float32)
                if raw_conf is None
                else _resize_stack(raw_conf, height, width)
            )
            confidence = np.asarray(confidence, dtype=np.float32)
            confidence[~np.isfinite(confidence) | (confidence < 0)] = 0.0
            depth = _sanitize_depth(depth)

            if previous is not None and self.config.align_scale:
                previous_start, previous_end, previous_depth, previous_conf = previous
                overlap_start = max(start, previous_start)
                overlap_end = min(end, previous_end)
                if overlap_end > overlap_start:
                    left = previous_depth[
                        overlap_start - previous_start : overlap_end - previous_start
                    ]
                    right = depth[overlap_start - start : overlap_end - start]
                    left_conf = previous_conf[
                        overlap_start - previous_start : overlap_end - previous_start
                    ]
                    right_conf = confidence[overlap_start - start : overlap_end - start]
                    scale = _robust_scale(
                        left,
                        right,
                        left_conf,
                        right_conf,
                        self.config.min_alignment_pixels,
                        self.config.max_scale_correction,
                    )
                    if scale != 1.0:
                        LOGGER.info(
                            "DA3 window scale correction start=%d end=%d scale=%.6f",
                            start,
                            end,
                            scale,
                        )
                        depth *= np.float32(scale)

            # 置信度仅作为融合权重；零置信度仍给极小权重，避免合法深度被丢弃。
            valid = np.isfinite(depth) & (depth > 0)
            weights = np.where(valid, np.maximum(confidence, 1e-6), 0.0)
            depth_sum[start:end] += depth * weights
            conf_sum[start:end] += confidence * weights
            weight_sum[start:end] += weights
            previous = (start, end, depth.copy(), confidence.copy())

        valid_weight = weight_sum > 0
        output_depth = np.zeros_like(weight_sum, dtype=np.float32)
        output_conf = np.zeros_like(weight_sum, dtype=np.float32)
        output_depth[valid_weight] = (
            depth_sum[valid_weight] / weight_sum[valid_weight]
        ).astype(np.float32)
        output_conf[valid_weight] = (
            conf_sum[valid_weight] / weight_sum[valid_weight]
        ).astype(np.float32)
        return KeyframeDepth(depth_m=output_depth, confidence=output_conf)


def make_windows(count: int, window_size: int, overlap: int) -> list[tuple[int, int]]:
    """生成覆盖完整序列的 ``[start,end)`` 重叠窗口。"""

    if count <= 0:
        return []
    if window_size <= 0 or not 0 <= overlap < window_size:
        raise ValueError("窗口参数必须满足 window_size > 0 且 0 <= overlap < window_size")
    if count <= window_size:
        return [(0, count)]
    step = window_size - overlap
    starts = list(range(0, count - window_size + 1, step))
    final_start = count - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, min(start + window_size, count)) for start in starts]


def _robust_scale(
    reference: np.ndarray,
    current: np.ndarray,
    reference_conf: np.ndarray,
    current_conf: np.ndarray,
    min_pixels: int,
    max_correction: float,
) -> float:
    valid = (
        np.isfinite(reference)
        & np.isfinite(current)
        & (reference > 0)
        & (current > 0)
        & (reference_conf > 0)
        & (current_conf > 0)
    )
    if int(valid.sum()) < min_pixels:
        return 1.0
    ratios = reference[valid] / current[valid]
    low, high = np.quantile(ratios, [0.1, 0.9])
    trimmed = ratios[(ratios >= low) & (ratios <= high)]
    if trimmed.size < min_pixels:
        return 1.0
    scale = float(np.median(trimmed))
    return float(np.clip(scale, 1.0 - max_correction, 1.0 + max_correction))


def _resize_stack(values: Any, height: int, width: int) -> np.ndarray:
    stack = np.asarray(values, dtype=np.float32)
    if stack.ndim != 3:
        raise RuntimeError(f"DA3 输出应为 (K,H,W)，实际为 {stack.shape}")
    if stack.shape[1:] == (height, width):
        return stack.copy()
    return np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in stack]
    ).astype(np.float32)


def _sanitize_depth(depth: np.ndarray) -> np.ndarray:
    result = np.asarray(depth, dtype=np.float32).copy()
    result[~np.isfinite(result) | (result <= 0)] = 0.0
    return result


def _validate_rgb_frames(frames: Sequence[np.ndarray]) -> None:
    if not frames:
        raise ValueError("RGB 帧序列为空")
    shape = frames[0].shape
    for index, frame in enumerate(frames):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"第 {index} 帧必须为 uint8 RGB (H,W,3)")
        if frame.shape != shape:
            raise ValueError("所有 RGB 帧必须具有相同 shape")
