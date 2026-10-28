"""
video_depth — RGB 视频到绝对米制深度视频的稳定公共 API

只导出配置、总流水线和结果类型；DA3/RIFE 上游私有接口不属于公共 API。
"""

from src.video_depth.config import VideoDepthConfig
from src.video_depth.pipeline import VideoDepthPipeline
from src.video_depth.types import VideoDepthResult

__all__ = ["VideoDepthConfig", "VideoDepthPipeline", "VideoDepthResult"]
