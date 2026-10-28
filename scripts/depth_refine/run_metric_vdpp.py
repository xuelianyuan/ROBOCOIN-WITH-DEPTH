"""DA3Metric + 尺度保持 VDPP 的单视频命令行入口。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.depth_refine.pipeline.metric_video import MetricVideoConfig, run_metric_video


def main() -> None:
    """解析路径、内参与 chunk 配置，运行方案 A。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "output", "da3-model", "vdpp-root", "vdpp-checkpoint"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument(
        "--intrinsics",
        nargs=4,
        type=float,
        required=True,
        metavar=("FX", "FY", "CX", "CY"),
    )
    parser.add_argument("--intrinsics-source", default="user-provided")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示整段视频")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=100.0)
    parser.add_argument("--smooth-strength", type=float, default=2.0)
    args = vars(parser.parse_args())
    args["intrinsics"] = tuple(args["intrinsics"])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_metric_video(MetricVideoConfig(**args))


if __name__ == "__main__":
    main()
