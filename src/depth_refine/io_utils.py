"""深度修正工具共用的路径、尺度和校验函数。"""
import json
from pathlib import Path


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
