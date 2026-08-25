#!/usr/bin/env python3
"""批量遍历 LeRobot 数据集项目并执行深度 baseline/refine 流程。

脚本会在指定根目录下查找包含 ``videos/chunk-*`` 的项目，按用户选择的
相机视角调用 ``analyze_depth.py`` 和 ``refine_depth.py``。每个项目的结果
独立写入该项目下的 ``depth_results/baseline`` 和 ``depth_results/refine``，
因此不会与原始 LeRobot 的 ``videos``、``meta`` 目录混在一起。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CAMERA_KEYS = {"head": "head", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}


def has_camera(project: Path, camera: str) -> bool:
    """检查项目中是否存在指定视角的 RGB 视频目录。

    不同数据集的目录名可能是 ``cam_front_head`` 或 ``camera_head``，所以
    这里只按视角关键词匹配；目录名包含 ``depth`` 的目录会被排除，避免
    把深度视频误当成 RGB 输入。
    """
    return any(
        camera in directory.name.lower()
        for directory in project.glob("videos/**/observation.images.*")
        if directory.is_dir() and "depth" not in directory.name.lower()
    )


def projects(root: Path) -> list[Path]:
    """查找根目录下的原始 LeRobot 项目，并过滤本脚本生成的结果目录。

    一个项目通常形如 ``<project>/videos/chunk-000``。由于输出目录中也会
    创建类似的 ``videos/chunk-*`` 结构，必须排除路径中包含
    ``depth_results`` 的目录，否则重复运行时会把输出目录误认为新项目。
    使用集合去重，以防一个项目包含多个 chunk 时被重复处理。
    """
    found = set()
    for chunk in root.glob("**/videos/chunk-*"):
        if not chunk.is_dir():
            continue
        try:
            relative = chunk.relative_to(root)
        except ValueError:
            continue
        if "depth_results" in relative.parts:
            continue
        found.add(chunk.parent.parent)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量处理根目录下所有 LeRobot 项目和指定相机视角")
    parser.add_argument("--root", default="data/lerobot",
                        help="数据集根目录，默认：data/lerobot")
    parser.add_argument("--cameras", nargs="+", default=list(CAMERA_KEYS),
                        choices=sorted(CAMERA_KEYS),
                        help="要处理的视角：head、left_wrist、right_wrist")
    parser.add_argument("--stage", choices=["baseline", "refine", "all"], default="baseline",
                        help="执行阶段：baseline、refine 或 all，默认：baseline")
    parser.add_argument("--sample-count", type=int, default=20,
                        help="每个视频均匀采样的帧数，默认：20")
    parser.add_argument("--save-video", action="store_true",
                        help="保存 Nested 稀疏 baseline 深度视频")
    parser.add_argument("--refine-stage", choices=["validate", "full", "rife"], default="validate",
                        help="refine 的具体模式：validate、full 或 rife，默认：validate")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将处理的项目和视角，不实际运行子脚本")
    args = parser.parse_args()
    # 转为绝对路径，确保从任意当前工作目录启动时，子脚本都能收到稳定路径。
    root = Path(args.root).resolve()
    found = projects(root)
    if not found:
        raise SystemExit(f"未找到包含 videos/chunk-* 的项目: {root}")
    for project in found:
        # 每个项目单独保存结果，避免不同任务的 episode 相互覆盖。
        result_root = project / "depth_results"
        for camera in args.cameras:
            if not has_camera(project, CAMERA_KEYS[camera]):
                print("[跳过]", project.name, camera, "（未找到对应 RGB 相机）")
                continue
            # baseline 使用 Nested 分析脚本，输出到 depth_results/baseline。
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
                # all 模式下 refine 会在 baseline 成功后执行；单独使用 refine
                # 时要求项目中已经存在可读取的 baseline 统计结果。
                refine = [sys.executable, "scripts/refine_depth.py", "--dataset-dir", str(project),
                          "--camera", CAMERA_KEYS[camera], "--result-root", str(result_root),
                          "--stage", args.refine_stage]
                print("[refine]", project.name, camera)
                if not args.dry_run:
                    subprocess.run(refine, check=True)


if __name__ == "__main__":
    main()
