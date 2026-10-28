#!/usr/bin/env python3
"""
estimate_video_depth.py — 单视频或目录的唯一产品级推理入口

读取默认 YAML，可用少量 CLI 参数覆盖 stride/device；一条命令生成逐帧对齐的
12-bit 深度 MP4、可选无损米制 NPZ、最终配置、元数据和日志。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.video_depth import VideoDepthConfig, VideoDepthPipeline
from src.video_depth.sampler import detect_scene_changes, select_keyframes
from src.video_depth.video_io import read_rgb_video

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def build_parser() -> argparse.ArgumentParser:
    """构建稳定、精简的命令行接口。"""

    parser = argparse.ArgumentParser(
        description="使用 DA3 Nested 1.1 + RIFE 生成绝对米制深度视频"
    )
    parser.add_argument("input", type=Path, help="RGB 视频或包含视频的目录")
    parser.add_argument("-o", "--output", type=Path, help="输出 MP4 或输出目录")
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "da3_nested_rife.yaml",
        help="配置 YAML",
    )
    parser.add_argument("--stride", type=int, help="覆盖关键帧步长")
    parser.add_argument("--device", help="覆盖设备，如 cuda:0 或 cpu")
    parser.add_argument("--overwrite", action="store_true", help="显式允许覆盖已有输出")
    parser.add_argument(
        "--dry-run", action="store_true", help="只解析配置、视频元数据和关键帧，不加载模型"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数并依次处理输入；失败返回非零退出码。"""

    args = build_parser().parse_args(argv)
    try:
        config = VideoDepthConfig.from_yaml(args.config)
        if args.stride is not None:
            config.sampling.stride = args.stride
        if args.device is not None:
            config.device = args.device
        config.validate()
        inputs = _discover_inputs(args.input)
        outputs = _resolve_outputs(inputs, args.input, args.output)

        if args.dry_run:
            for source, destination in zip(inputs, outputs):
                frames, metadata = read_rgb_video(source)
                boundaries = detect_scene_changes(
                    frames, config.sampling.scene_change_threshold
                )
                indices = select_keyframes(
                    len(frames), config.sampling.stride, boundaries
                )
                print(
                    yaml.safe_dump(
                        {
                            "config": config.to_dict(),
                            "input": {
                                "path": str(source),
                                "frames": metadata.frame_count,
                                "resolution": [metadata.width, metadata.height],
                                "fps": metadata.fps,
                                "is_vfr": metadata.is_vfr,
                            },
                            "output": str(destination),
                            "scene_boundaries": boundaries,
                            "keyframe_count": len(indices),
                            "keyframe_indices": indices,
                        },
                        sort_keys=False,
                        allow_unicode=True,
                    )
                )
            return 0

        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"请求设备 {config.device}，但 PyTorch 未检测到可用 NVIDIA 驱动；"
                "可先用 --device cpu 做接口验证"
            )
        pipeline = VideoDepthPipeline(config)
        for source, destination in zip(inputs, outputs):
            _configure_logging(destination)
            result = pipeline.run(source, destination, overwrite=args.overwrite)
            logging.getLogger(__name__).info(
                "完成 output=%s frames=%d clip_ratio=%.8f npz=%s",
                result.output_path,
                result.metadata.frame_count,
                result.clip_ratio,
                result.npz_path,
            )
        return 0
    except Exception as exc:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
        logging.getLogger(__name__).error("%s", exc)
        return 2


def _discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"不支持的视频后缀: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        videos = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
        if not videos:
            raise ValueError(f"目录中没有支持的视频: {input_path}")
        return videos
    raise FileNotFoundError(f"输入不存在: {input_path}")


def _resolve_outputs(
    inputs: list[Path], input_argument: Path, output_argument: Path | None
) -> list[Path]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_argument is None:
        return [
            _PROJECT_ROOT / "outputs" / source.stem / run_id / "depth.mp4"
            for source in inputs
        ]
    if len(inputs) == 1 and input_argument.is_file() and output_argument.suffix:
        return [output_argument]
    if output_argument.suffix:
        raise ValueError("目录/多视频输入时，--output 必须是输出目录而不是文件")
    return [output_argument / source.stem / run_id / "depth.mp4" for source in inputs]


def _configure_logging(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_name = "run.log" if destination.name == "depth.mp4" else f"{destination.stem}.run.log"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(destination.parent / log_name, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s level=%(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
