#!/usr/bin/env python3
"""统计原始深度视频中疑似云斑的逐帧出现和连续区间。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.depth_refine.evaluation.cloud_artifacts import CloudConfig
from src.depth_refine.pipeline.cloud_statistics import run_cloud_statistics


def main() -> None:
    """解析可复现的阈值和范围，调用同一统计流程并显示进度。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs") / f"cloud_statistics_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d_%H%M%S}")
    parser.add_argument("--reference-npy", nargs="*", type=Path, default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--resume", action="store_true", help="恢复相同输入和参数的未完成扫描")
    parser.add_argument("--analysis-width", type=int, default=320)
    parser.add_argument("--amplitude", type=float, default=0.008)
    parser.add_argument("--coverage", type=float, default=0.02)
    parser.add_argument("--edge-gradient", type=float, default=0.025)
    parser.add_argument("--anisotropy", type=float, default=0.65)
    parser.add_argument("--min-component-pixels", type=int, default=20)
    args = parser.parse_args()
    config = CloudConfig(analysis_width=args.analysis_width, amplitude=args.amplitude,
                         coverage=args.coverage, edge_gradient=args.edge_gradient,
                         anisotropy=args.anisotropy, min_component_pixels=args.min_component_pixels)

    def progress(done: int, total: int, report: dict) -> None:
        print(f"[{done}/{total}] {report['status']} {report['dataset']}/{report['task']}/{report['camera']} "
              f"suspected={report.get('suspected_frames', '?')}/{report.get('total_frames', '?')}", flush=True)

    summary = run_cloud_statistics(args.dataset_roots, args.output, config,
        references=args.reference_npy, workers=args.workers, limit=args.limit_videos,
        resume=args.resume, progress=progress)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output / 'REPORT.md'}", flush=True)
    if summary["overall"]["failed_videos"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
