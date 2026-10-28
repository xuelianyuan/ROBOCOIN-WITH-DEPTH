"""
types.py — 视频深度流水线的稳定数据结构

定义视频元数据、关键帧推理结果与米制逐帧深度结果；深度 shape 统一为
``(T, H, W)``，核心 API 的权威单位统一为米。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class VideoMetadata:
    """描述输入视频的时间轴和空间尺寸。

    ``timestamps_s`` 与解码帧一一对应；``is_vfr`` 表示输入时间戳是否为可变
    帧率。当前深度 MP4 编码为 CFR 时，会在运行元数据中明确记录该转换。
    """

    width: int
    height: int
    frame_count: int
    fps: float
    timestamps_s: list[float] = field(default_factory=list)
    is_vfr: bool = False
    source: Path | None = None
    codec_name: str | None = None


@dataclass(slots=True)
class KeyframeDepth:
    """DA3 Nested 的关键帧结果，深度单位为米。

    Attributes:
        depth_m: ``float32 (K,H,W)`` 米制深度。
        confidence: ``float32 (K,H,W)`` 模型置信度。
    """

    depth_m: np.ndarray
    confidence: np.ndarray


@dataclass(slots=True)
class VideoDepthResult:
    """完整视频深度结果。

    ``depth_m`` 和 ``confidence`` 均为 ``float32 (T,H,W)``。``unit`` 固定为
    ``"m"``，避免调用方依赖隐式单位约定。
    """

    depth_m: np.ndarray
    confidence: np.ndarray
    metadata: VideoMetadata
    keyframe_indices: list[int]
    unit: str = "m"
    output_path: Path | None = None
    npz_path: Path | None = None
    clip_ratio: float = 0.0
    timings_s: dict[str, float] = field(default_factory=dict)
