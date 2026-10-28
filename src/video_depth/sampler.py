"""
sampler.py — 视频关键帧索引与场景边界处理

保证索引严格递增、无重复并包含首尾；场景切换两侧都会成为关键帧，从而禁止
RIFE 跨镜头估计运动。
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def select_keyframes(
    frame_count: int,
    stride: int,
    scene_boundaries: Sequence[int] | None = None,
) -> list[int]:
    """生成关键帧索引。

    Args:
        frame_count: 输入视频总帧数。
        stride: 均匀采样步长，必须大于零。
        scene_boundaries: 每个新镜头第一帧的索引。
    Returns:
        严格递增的索引；非空视频始终包含 0 和最后一帧。
    """

    if stride <= 0:
        raise ValueError("stride 必须 > 0")
    if frame_count < 0:
        raise ValueError("frame_count 不能为负")
    if frame_count == 0:
        return []
    indices = set(range(0, frame_count, stride))
    indices.update((0, frame_count - 1))
    for boundary in scene_boundaries or ():
        if 0 < boundary < frame_count:
            indices.add(boundary - 1)
            indices.add(boundary)
    return sorted(indices)


def detect_scene_changes(
    frames: Sequence[np.ndarray], threshold: float = 0.35
) -> list[int]:
    """以低分辨率灰度差检测硬切镜头。

    返回新镜头首帧索引。``threshold <= 0`` 表示关闭检测；该轻量方法只负责
    安全边界，不尝试识别渐变或复杂编辑。
    """

    if threshold <= 0 or len(frames) < 2:
        return []
    if threshold > 1:
        raise ValueError("场景切换阈值必须 <= 1")
    boundaries: list[int] = []
    previous = _scene_signature(frames[0])
    for index, frame in enumerate(frames[1:], start=1):
        current = _scene_signature(frame)
        score = float(np.mean(np.abs(current - previous)))
        if score >= threshold:
            boundaries.append(index)
        previous = current
    return boundaries


def _scene_signature(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("RGB 帧必须为 (H,W,3)")
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0
