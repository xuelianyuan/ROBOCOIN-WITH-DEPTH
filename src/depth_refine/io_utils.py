"""深度修正工具共用的路径、尺度和校验函数。"""
import json
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def probe_video(path: Path) -> dict:
    """读取视频尺寸、原生格式、帧率与声明帧数，不解码整段视频。"""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)["streams"][0]
    num, den = map(int, data.get("avg_frame_rate", "0/1").split("/"))
    data["fps"] = num / den if den else 0.0
    data["declared_frames"] = (int(data["nb_frames"])
                               if data.get("nb_frames", "N/A") != "N/A" else None)
    return data


def iter_video_frames(path: Path, pix_fmt: str, *, info: dict | None = None,
                      max_width: int | None = None) -> Iterator[np.ndarray]:
    """顺序解码 H×W 深度或 H×W×3 RGB 帧；可缩小空间尺寸，始终保留全部帧。

    深度格式必须与输入一致，防止 gray12le→gray16le 隐式缩放数值。
    stderr 写临时文件，避免管道填满导致长视频解码死锁。
    """
    if pix_fmt not in {"gray12le", "gray16le", "rgb24"}:
        raise ValueError(f"Unsupported pixel format: {pix_fmt}")
    info = info or probe_video(path)
    is_depth = pix_fmt != "rgb24"
    if is_depth and info["pix_fmt"] != pix_fmt:
        raise ValueError(f"Depth pixel format mismatch: {info['pix_fmt']} != {pix_fmt}")
    w, h = int(info["width"]), int(info["height"])
    cmd = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
           "-map", "0:v:0", "-an", "-vsync", "0", "-threads", "1"]
    if max_width is not None:
        if max_width < 32:
            raise ValueError("max_width must be at least 32")
        if w > max_width:
            h, w = max(1, round(h * max_width / w)), max_width
            cmd += ["-vf", f"scale={w}:{h}:flags=area", "-filter_threads", "1"]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    dtype, channels = ("<u2", 1) if is_depth else ("u1", 3)
    size = w * h * channels * np.dtype(dtype).itemsize
    shape = (h, w) if is_depth else (h, w, 3)
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors)
        assert proc.stdout is not None
        try:
            while True:
                raw = proc.stdout.read(size)
                if not raw:
                    break
                if len(raw) != size:
                    raise RuntimeError(f"Incomplete video frame: {path}")
                yield np.frombuffer(raw, dtype=dtype).reshape(shape)
            if proc.wait():
                errors.seek(0)
                raise RuntimeError(f"Decode failed: {path}: {errors.read().decode(errors='replace')}")
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.terminate()
            proc.wait()


# ==================== 视频路径处理 ====================
def resolve_video(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        videos = sorted(path.glob("*.mp4"))
        if len(videos) != 1:
            raise FileNotFoundError(f"视频目录应恰好包含一个 mp4: {path}")
        return videos[0]
    if not path.is_file():
        raise FileNotFoundError(f"视频不存在: {path}")
    return path


def matching_rgb_path(depth_path: Path) -> Path:
    return Path(str(depth_path).replace("_depth_nested_sparse", "_rgb").replace("_depth_refined", "_rgb").replace("_depth", "_rgb"))


# ==================== 点云输出路径 ====================
def pcd_output_root(depth_path: Path, pcd_root: Path) -> Path:
    parts = Path(depth_path).parts
    try:
        i = next(i for i, part in enumerate(parts) if part.endswith("_depth_preview"))
        return Path(pcd_root) / parts[i] / parts[i + 1]
    except (StopIteration, IndexError) as exc:
        raise ValueError(f"无法从路径解析数据集/任务名: {depth_path}") from exc


# ==================== 深度尺度校验 ====================
def validate_depth_scale(scale: float) -> float:
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"depth scale 必须大于 0: {scale}")
    return scale


# ==================== 基线帧索引校验 ====================
def load_baseline_indices(depth_path: Path, depth_count: int, rgb_count: int):
    base = depth_path.parents[3] if len(depth_path.parents) > 3 else depth_path.parent
    camera = depth_path.parent.name.replace("observation.images.", "").replace("_depth_nested_sparse", "")
    manifest = base / "statistics" / camera / depth_path.stem / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"baseline manifest 不存在: {manifest}")
    indices = json.loads(manifest.read_text()).get("sample_frame_indices")
    if not indices or len(indices) < depth_count or max(indices[:depth_count]) >= rgb_count:
        raise ValueError(f"manifest 帧索引无法匹配视频: {manifest}")
    return indices[:depth_count], manifest
