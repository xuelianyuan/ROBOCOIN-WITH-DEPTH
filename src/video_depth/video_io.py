"""
video_io.py — RGB 视频解码、时间轴探测与深度视频原子编码

解码输出为 ``uint8 RGB (T,H,W,3)``；核心深度输入为米制 ``float32``，只在
12-bit ``gray12le`` 编码边界转换一次到毫米，并返回超量程裁剪比例。
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from src.video_depth.config import OutputConfig
from src.video_depth.types import VideoMetadata


def probe_video(path: str | Path) -> VideoMetadata:
    """读取尺寸、帧数、FPS 和逐帧时间戳，并检测 VFR。"""

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"输入视频不存在: {video_path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_frames",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames:frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        timestamps = [
            float(frame["best_effort_timestamp_time"])
            for frame in payload.get("frames", [])
            if "best_effort_timestamp_time" in frame
        ]
        average_fps = _fraction(stream.get("avg_frame_rate"))
        nominal_fps = _fraction(stream.get("r_frame_rate"))
        fps = average_fps or nominal_fps
        frame_count = len(timestamps) or int(stream.get("nb_frames") or 0)
        is_vfr = _is_vfr(timestamps) or (
            bool(average_fps and nominal_fps) and abs(average_fps - nominal_fps) > 1e-4
        )
        if fps <= 0:
            raise ValueError("ffprobe 未返回有效 FPS")
        return VideoMetadata(
            width=int(stream["width"]),
            height=int(stream["height"]),
            frame_count=frame_count,
            fps=float(fps),
            timestamps_s=timestamps,
            is_vfr=is_vfr,
            source=video_path.resolve(),
            codec_name=stream.get("codec_name"),
        )
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError):
        return _probe_with_opencv(video_path)


def read_rgb_video(path: str | Path) -> tuple[list[np.ndarray], VideoMetadata]:
    """解码完整 RGB 视频，失败或空视频时给出明确错误。"""

    video_path = Path(path)
    metadata = probe_video(video_path)
    # 当前部分 OpenCV 构建会错误地要求 AV1 硬件解码；对 AV1 直接使用系统
    # ffmpeg 软件解码。其他编码先走 OpenCV，失败或帧数异常时同样安全回退。
    frames = (
        _read_rgb_with_ffmpeg(video_path, metadata)
        if metadata.codec_name == "av1"
        else _read_rgb_with_opencv(video_path)
    )
    if not frames or (
        metadata.frame_count > 0 and len(frames) != metadata.frame_count
    ):
        frames = _read_rgb_with_ffmpeg(video_path, metadata)
    if not frames:
        raise ValueError(f"输入视频没有可解码帧: {video_path}")
    first_shape = frames[0].shape
    if any(frame.shape != first_shape for frame in frames):
        raise ValueError("输入视频帧尺寸在时间轴上发生变化，当前不支持")
    metadata.width = first_shape[1]
    metadata.height = first_shape[0]
    if metadata.frame_count > 0 and metadata.frame_count != len(frames):
        raise RuntimeError(
            f"视频解码帧数不匹配: decoded={len(frames)} probed={metadata.frame_count}"
        )
    metadata.frame_count = len(frames)
    if len(metadata.timestamps_s) != len(frames):
        metadata.timestamps_s = [index / metadata.fps for index in range(len(frames))]
    return frames, metadata


def validate_depth_frames(
    depth_m: np.ndarray,
    expected_frames: int,
    height: int,
    width: int,
) -> None:
    """编码前检查深度 shape、dtype、有限性和非负性。"""

    if depth_m.shape != (expected_frames, height, width):
        raise ValueError(
            f"深度 shape 不匹配: {depth_m.shape} != {(expected_frames, height, width)}"
        )
    if depth_m.dtype != np.float32:
        raise ValueError(f"深度必须为 float32，实际为 {depth_m.dtype}")
    if not np.isfinite(depth_m).all():
        raise ValueError("深度包含 NaN/Inf，拒绝编码")
    if np.any(depth_m < 0):
        raise ValueError("深度包含负值，拒绝编码")


def write_depth_video(
    depth_m: np.ndarray,
    output_path: str | Path,
    metadata: VideoMetadata,
    config: OutputConfig,
    *,
    overwrite: bool = False,
) -> float:
    """原子写出 12-bit 深度 MP4，并返回被量程裁剪的像素比例。

    米到毫米只在本函数执行一次。VFR 输入当前会按 ``metadata.fps`` 写成 CFR，
    调用方应把 ``metadata.is_vfr`` 记录到运行元数据。
    """

    validate_depth_frames(
        depth_m, metadata.frame_count, metadata.height, metadata.width
    )
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"输出已存在；如需覆盖请显式使用 --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"
    )
    depth_mm = depth_m * np.float32(1000.0)
    clip_ratio = float(np.count_nonzero(depth_mm > config.max_depth_mm) / depth_mm.size)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{metadata.width}x{metadata.height}",
        "-pix_fmt",
        config.pixel_format,
        "-r",
        f"{metadata.fps:.12g}",
        "-i",
        "-",
        "-r",
        f"{metadata.fps:.12g}",
        "-vsync",
        "cfr",
        "-video_track_timescale",
        "90000",
        "-c:v",
        config.codec,
        "-crf",
        str(config.crf),
        "-preset",
        config.preset,
        "-pix_fmt",
        config.pixel_format,
        "-g",
        str(max(1, round(metadata.fps))),
        "-keyint_min",
        str(max(1, round(metadata.fps))),
        "-x265-params",
        "bframes=0",
        str(temporary),
    ]
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        assert process.stdin is not None
        for frame in depth_mm:
            encoded = np.clip(frame, 0, config.max_depth_mm).astype(np.uint16)
            process.stdin.write(encoded.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg 深度编码失败 (exit={return_code}): {stderr.strip()}")
        os.replace(temporary, destination)
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)
        raise
    return clip_ratio


def write_depth_npz(
    depth_m: np.ndarray,
    output_path: str | Path,
    metadata: VideoMetadata,
    *,
    confidence: np.ndarray | None = None,
    overwrite: bool = False,
) -> Path:
    """原子保存无损米制深度、时间戳和可选置信度。"""

    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"NPZ 输出已存在: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp.npz"
    )
    try:
        payload = {
            "depth_m": depth_m,
            "timestamps_s": np.asarray(metadata.timestamps_s, dtype=np.float64),
            "unit": np.asarray("m"),
        }
        if confidence is not None:
            if confidence.shape != depth_m.shape or confidence.dtype != np.float32:
                raise ValueError("confidence 必须与 depth_m 同 shape 且为 float32")
            payload["confidence"] = confidence
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _probe_with_opencv(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法探测视频: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"视频元数据无效: {path}")
    timestamps = [index / fps for index in range(max(frame_count, 0))]
    return VideoMetadata(width, height, frame_count, fps, timestamps, False, path.resolve())


def _read_rgb_with_opencv(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if bgr is None or bgr.ndim != 3:
                return []
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def _read_rgb_with_ffmpeg(path: Path, metadata: VideoMetadata) -> list[np.ndarray]:
    """通过 ffmpeg 软件路径逐帧解码 RGB，主要用于 AV1 和 OpenCV 回退。"""

    frame_bytes = metadata.width * metadata.height * 3
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.stdout is not None
    frames: list[np.ndarray] = []
    while True:
        payload = process.stdout.read(frame_bytes)
        if not payload:
            break
        if len(payload) != frame_bytes:
            process.kill()
            process.wait()
            raise RuntimeError(
                f"ffmpeg 返回不完整 RGB 帧: {len(payload)} != {frame_bytes}"
            )
        frame = np.frombuffer(payload, dtype=np.uint8).reshape(
            metadata.height, metadata.width, 3
        )
        frames.append(frame.copy())
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg RGB 解码失败 (exit={return_code}): {stderr.strip()}"
        )
    return frames


def _fraction(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    return float(Fraction(value))


def _is_vfr(timestamps: list[float]) -> bool:
    if len(timestamps) < 3:
        return False
    intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
    median = float(np.median(intervals))
    tolerance = max(1e-4, abs(median) * 1e-3)
    return bool(np.any(np.abs(intervals - median) > tolerance))
