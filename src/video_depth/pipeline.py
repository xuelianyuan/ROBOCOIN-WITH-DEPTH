"""
pipeline.py — DA3 Nested + RIFE 视频深度估计总流水线

负责组织视频解码、关键帧采样、米制 DA3 推理、RIFE 补帧和原子写出。
输入为 RGB 视频；核心 API 输出 ``float32 (T,H,W)`` 米制深度序列。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.video_depth.config import VideoDepthConfig
from src.video_depth.da3_nested import DA3NestedEstimator
from src.video_depth.rife_depth import RIFEDepthInterpolator
from src.video_depth.sampler import detect_scene_changes, select_keyframes
from src.video_depth.types import VideoDepthResult, VideoMetadata
from src.video_depth.video_io import read_rgb_video, write_depth_npz, write_depth_video

LOGGER = logging.getLogger(__name__)


class VideoDepthPipeline:
    """管理模型生命周期并执行完整文件级推理。

    ``da3`` 和 ``rife`` 参数用于 CPU/mock 测试。正常使用时模型均按需构造；尤其
    ``stride=1`` 不会构造或加载 RIFE。
    """

    def __init__(
        self,
        config: VideoDepthConfig,
        *,
        da3: Any | None = None,
        rife: Any | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._da3 = da3
        self._rife = rife
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    def estimate_frames(
        self,
        frames: Sequence[np.ndarray],
        *,
        fps: float,
        timestamps_s: Sequence[float] | None = None,
        is_vfr: bool = False,
    ) -> VideoDepthResult:
        """从内存 RGB 帧估计逐帧深度，便于测试和上层系统集成。

        Args:
            frames: ``uint8 RGB (T,H,W,3)``，尺寸必须一致。
            fps: 正帧率。
            timestamps_s: 可选逐帧时间戳。
            is_vfr: 输入时间轴是否为 VFR。
        Returns:
            深度和置信度为 ``float32 (T,H,W)``，单位米。
        """

        _validate_rgb_sequence(frames, fps)
        count = len(frames)
        height, width = frames[0].shape[:2]
        boundaries = (
            []
            if self.config.sampling.stride == 1
            else detect_scene_changes(
                frames, threshold=self.config.sampling.scene_change_threshold
            )
        )
        key_indices = select_keyframes(
            count, self.config.sampling.stride, scene_boundaries=boundaries
        )
        key_frames = [frames[index] for index in key_indices]
        if self._da3 is None:
            self._da3 = DA3NestedEstimator(self.config.da3, self.config.device)
        key_result = self._da3.infer(key_frames)
        expected_key_shape = (len(key_indices), height, width)
        if key_result.depth_m.shape != expected_key_shape:
            raise RuntimeError(
                f"DA3 关键帧 shape 不匹配: {key_result.depth_m.shape} != {expected_key_shape}"
            )

        if len(key_indices) == count:
            # stride=1 时结果已经拥有独立数组；避免对长视频的深度和置信度各复制
            # 一次，减少数 GB 峰值内存，同时仍保持 RIFE 完全不加载。
            depth_m = key_result.depth_m
            confidence = key_result.confidence
        else:
            if self._rife is None:
                self._rife = RIFEDepthInterpolator(
                    device=self.config.device,
                    checkpoint=self.config.rife.checkpoint,
                    pad_multiple=self.config.rife.pad_multiple,
                    min_confidence=self.config.rife.min_confidence,
                )
            depth_m, confidence = self._rife.interpolate(
                frames,
                key_indices,
                key_result.depth_m,
                key_result.confidence,
            )
            # 防御第三方适配器误写关键帧；核心契约以 DA3 结果为准。
            depth_m[key_indices] = key_result.depth_m
            confidence[key_indices] = key_result.confidence

        depth_m = np.asarray(depth_m, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)
        _validate_result(depth_m, confidence, count, height, width)
        timestamps = (
            list(map(float, timestamps_s))
            if timestamps_s is not None and len(timestamps_s) == count
            else [index / fps for index in range(count)]
        )
        metadata = VideoMetadata(
            width=width,
            height=height,
            frame_count=count,
            fps=float(fps),
            timestamps_s=timestamps,
            is_vfr=is_vfr,
        )
        return VideoDepthResult(
            depth_m=depth_m,
            confidence=confidence,
            metadata=metadata,
            keyframe_indices=key_indices,
        )

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> VideoDepthResult:
        """一条调用从 RGB 视频生成深度 MP4、可选 NPZ 和运行元数据。"""

        source = Path(input_path)
        destination = Path(output_path)
        artifact_dir = destination.parent
        sidecar_prefix = "" if destination.name == "depth.mp4" else f"{destination.stem}."
        npz_path = (
            artifact_dir / "depth.npz"
            if destination.name == "depth.mp4"
            else destination.with_suffix(".npz")
        )
        config_path = artifact_dir / f"{sidecar_prefix}run_config.yaml"
        metadata_path = artifact_dir / f"{sidecar_prefix}run_metadata.json"
        planned = [destination, config_path, metadata_path]
        if self.config.output.save_npz:
            planned.append(npz_path)
        existing = [path for path in planned if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                "输出已存在；如需覆盖请显式使用 --overwrite: "
                + ", ".join(map(str, existing))
            )

        started = time.perf_counter()
        frames, input_metadata = read_rgb_video(source)
        decoded = time.perf_counter()
        LOGGER.info(
            "video=%s device=%s model=%s frames=%d resolution=%dx%d input_fps=%.6f "
            "output_fps=%.6f stride=%d window=%d overlap=%d unit=m",
            source.name,
            self.config.device,
            self.config.da3.model_id,
            input_metadata.frame_count,
            input_metadata.width,
            input_metadata.height,
            input_metadata.fps,
            input_metadata.fps,
            self.config.sampling.stride,
            self.config.da3.window_size,
            self.config.da3.overlap,
        )
        result = self.estimate_frames(
            frames,
            fps=input_metadata.fps,
            timestamps_s=input_metadata.timestamps_s,
            is_vfr=input_metadata.is_vfr,
        )
        estimated = time.perf_counter()
        result.metadata.source = source.resolve()
        result.clip_ratio = write_depth_video(
            result.depth_m,
            destination,
            result.metadata,
            self.config.output,
            overwrite=overwrite,
        )
        result.output_path = destination.resolve()
        if self.config.output.save_npz:
            result.npz_path = write_depth_npz(
                result.depth_m,
                npz_path,
                result.metadata,
                confidence=result.confidence,
                overwrite=overwrite,
            ).resolve()
        encoded = time.perf_counter()
        result.timings_s = {
            "decode": decoded - started,
            "inference": estimated - decoded,
            "encode": encoded - estimated,
            "total": encoded - started,
        }

        run_metadata = {
            "strategy": self.config.strategy,
            "input": {
                "name": source.name,
                "sha256": _sha256(source),
                "width": result.metadata.width,
                "height": result.metadata.height,
                "frame_count": result.metadata.frame_count,
                "fps": result.metadata.fps,
                "is_vfr": result.metadata.is_vfr,
                "timeline_conversion": "vfr_to_cfr" if result.metadata.is_vfr else "none",
            },
            "output": {
                "video": destination.name,
                "npz": result.npz_path.name if result.npz_path else None,
                "unit": "m",
                "encoded_unit": "mm",
                "clip_ratio": result.clip_ratio,
            },
            "keyframe_indices": result.keyframe_indices,
            "timings_s": result.timings_s,
            "elapsed_s": time.perf_counter() - started,
            "git_commit": _git_commit(),
            "environment": _environment_summary(),
        }
        _atomic_write_text(config_path, yaml.safe_dump(self.config.to_dict(), sort_keys=False))
        _atomic_write_text(
            metadata_path,
            json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        )
        return result


def _validate_rgb_sequence(frames: Sequence[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("输入为空视频或没有可解码帧")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps 必须为正数")
    shape = frames[0].shape
    for index, frame in enumerate(frames):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"第 {index} 帧必须为 uint8 RGB (H,W,3)")
        if frame.shape != shape:
            raise ValueError("所有输入帧必须具有相同 shape")


def _validate_result(
    depth_m: np.ndarray,
    confidence: np.ndarray,
    count: int,
    height: int,
    width: int,
) -> None:
    expected = (count, height, width)
    if depth_m.shape != expected or confidence.shape != expected:
        raise RuntimeError(f"输出 shape 必须为 {expected}")
    if depth_m.dtype != np.float32 or confidence.dtype != np.float32:
        raise RuntimeError("输出深度和置信度必须为 float32")
    if not np.isfinite(depth_m).all() or np.any(depth_m < 0):
        raise RuntimeError("输出深度包含 NaN/Inf 或负值")
    if not np.isfinite(confidence).all() or np.any(confidence < 0):
        raise RuntimeError("输出置信度包含 NaN/Inf 或负值")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() or None


def _environment_summary() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
