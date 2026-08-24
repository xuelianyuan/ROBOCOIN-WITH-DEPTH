#!/usr/bin/env python3
"""Batch Nested analysis/refinement for every LeRobot project under a root."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CAMERA_KEYS = {"head": "head", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}


def projects(root: Path) -> list[Path]:
    return sorted({p.parent.parent for p in root.glob("**/videos/chunk-*") if p.is_dir()})


def main() -> None:
    parser = argparse.ArgumentParser(description="批量处理 data/lerobot 下所有项目和指定视角")
    parser.add_argument("--root", default="data/lerobot")
    parser.add_argument("--cameras", nargs="+", default=list(CAMERA_KEYS),
                        choices=sorted(CAMERA_KEYS), help="head left_wrist right_wrist")
    parser.add_argument("--stage", choices=["baseline", "refine", "all"], default="baseline")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--refine-stage", choices=["validate", "full", "rife"], default="validate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    found = projects(root)
    if not found:
        raise SystemExit(f"未找到包含 videos/chunk-* 的项目: {root}")
    for project in found:
        result_root = project / "depth_results"
        for camera in args.cameras:
            cmd = [sys.executable, "scripts/analyze_depth.py", "--dataset-dir", str(project),
                   "--camera", CAMERA_KEYS[camera], "--sample-count", str(args.sample_count),
                   "--result-root", str(result_root)]
            if args.save_video:
                cmd.append("--save-video")
            if args.stage in ("baseline", "all"):
                print("[baseline]", project.name, camera)
                if not args.dry_run:
                    subprocess.run(cmd, check=True)
            if args.stage in ("refine", "all"):
                refine = [sys.executable, "scripts/refine_depth.py", "--dataset-dir", str(project),
                          "--camera", CAMERA_KEYS[camera], "--result-root", str(result_root),
                          "--stage", args.refine_stage]
                print("[refine]", project.name, camera)
                if not args.dry_run:
                    subprocess.run(refine, check=True)


if __name__ == "__main__":
    main()
