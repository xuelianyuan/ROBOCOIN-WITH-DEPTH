#!/usr/bin/env python3
"""批量遍历 LeRobot 数据集项目并执行深度 baseline/refine 流程。

脚本会在指定根目录下查找包含 ``videos/chunk-*`` 的项目，按用户选择的
相机视角调用 ``build_baseline.py`` 和 ``run_refine.py``。每个项目的结果
独立写入该项目下的 ``depth_results/baseline`` 和 ``depth_results/refine``，
因此不会与原始 LeRobot 的 ``videos``、``meta`` 目录混在一起。

用法示例::

    python scripts/depth_refine/run_batch.py \
        --root data/refine \
        --cameras head left_wrist right_wrist chest \
        --stage all --save-video --refine-stage validate

``baseline`` 先运行 ``build_baseline.py``；``refine`` 需要已有 baseline
统计；``all`` 按顺序执行两阶段。使用 ``--dry-run`` 可只检查将处理的项目。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import json
from pathlib import Path


CAMERA_KEYS = {
    "head": "head",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "chest": "chest",
}


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


def replace_depth_video(project: Path, camera: str, generated: Path) -> Path | None:
    """将生成的深度视频替换为数据集中的对应深度视频。"""
    if not generated.exists():
        return None
    episode = generated.name
    # ===== 替换目标定位：沿用生成结果中的 chunk，查找对应深度目录 =====
    chunk = generated.parent.parent
    depth_dirs = [d for d in chunk.glob("observation.images.*")
                  if d.is_dir() and "depth" in d.name.lower() and camera in d.name.lower()]
    if not depth_dirs:
        return None
    target = depth_dirs[0] / episode
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.copy2(generated, tmp)
    os.replace(tmp, target)
    return target


def plot_mean_curve(project: Path, camera: str, result_root: Path) -> None:
    """根据 refine 报告绘制原始、Nested、迁移后深度均值曲线。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("绘制均值曲线需要安装 matplotlib") from exc
    report_root = result_root / "refine" / "reports"
    reports = sorted(
        path for path in report_root.glob("**/episode_*/refine_parameters.json")
        if camera in path.parent.parent.name.lower()
    )
    if not reports:
        print("[警告] 未找到均值曲线所需的 refine 报告", project.name, camera)
        return
    output_root = result_root / "refine" / "plots"
    for report_path in reports:
        data = json.loads(report_path.read_text())
        metrics = data.get("frame_metrics", [])
        if not metrics:
            continue
        frames = [item["frame_index"] for item in metrics]
        before = [item.get("baseline", {}).get("mean_mm") for item in metrics]
        nested = [item.get("baseline", {}).get("target_mean_mm") for item in metrics]
        after = [item.get("transferred", {}).get("mean_mm") for item in metrics]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(frames, before, "b--", label="原始 Metric")
        ax.plot(frames, nested, "orange", label="DA3 Nested")
        ax.plot(frames, after, "g-", label="Refined Metric")
        ax.set(xlabel="帧号", ylabel="平均深度 (mm)", title=f"{camera} {report_path.parent.name} 深度均值曲线")
        ax.grid(True, alpha=0.3)
        ax.legend()
        output = output_root / report_path.parent.parent.name / report_path.parent.name / "mean_curve.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[均值曲线]", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量处理根目录下所有 LeRobot 项目和指定相机视角")
    parser.add_argument("--root", default="data/refine",
                        help="数据集根目录，默认：data/refine")
    parser.add_argument("--cameras", nargs="+", default=list(CAMERA_KEYS),
                        choices=sorted(CAMERA_KEYS),
                        help="要处理的视角：head、left_wrist、right_wrist、chest")
    parser.add_argument("--stage", choices=["baseline", "refine", "all", "plot"], default="baseline",
                        help="执行阶段：baseline、refine、all 或 plot（仅绘图），默认：baseline")
    parser.add_argument("--sample-count", type=int, default=20,
                        help="每个视频均匀采样的帧数，默认：20")
    parser.add_argument("--save-video", action="store_true",
                        help="保存 Nested 稀疏 baseline 深度视频")
    parser.add_argument("--refine-stage", choices=["validate", "full", "rife"], default="validate",
                        help="refine 的具体模式：validate、full 或 rife，默认：validate")
    parser.add_argument("--replace-depth-video", action="store_true",
                        help="将 refine 生成的完整深度视频替换数据集 videos 中对应的原始深度视频")
    parser.add_argument("--plot-mean-curve", action="store_true",
                        help="根据 refine 报告绘制采样帧深度均值曲线")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将处理的项目和视角，不实际运行子脚本")
    args = parser.parse_args()
    # ===== 路径准备：转为绝对路径，保证子脚本收到稳定路径 =====
    root = Path(args.root).resolve()
    found = projects(root)
    if not found:
        raise SystemExit(f"未找到包含 videos/chunk-* 的项目: {root}")
    for project in found:
        # ===== 项目级处理：每个项目使用独立的结果目录 =====
        result_root = project / "depth_results"
        for camera in args.cameras:
            if not has_camera(project, CAMERA_KEYS[camera]):
                print("[跳过]", project.name, camera, "（未找到对应 RGB 相机）")
                continue
            # ===== 绘图阶段：只读取已有 refine 报告，不重新执行深度处理 =====
            if args.stage == "plot":
                print("[绘图]", project.name, camera)
                if not args.dry_run:
                    plot_mean_curve(project, CAMERA_KEYS[camera], result_root)
                continue
            # ===== Baseline 阶段：运行深度分析并输出统计结果 =====
            cmd = [sys.executable, "scripts/depth_refine/build_baseline.py", "--dataset-dir", str(project),
                   "--camera", CAMERA_KEYS[camera], "--sample-count", str(args.sample_count),
                   "--result-root", str(result_root)]
            if args.save_video:
                cmd.append("--save-video")
            if args.stage in ("baseline", "all"):
                print("[baseline]", project.name, camera)
                if not args.dry_run:
                    subprocess.run(cmd, check=True)
            if args.stage in ("refine", "all"):
                # ===== Refine 阶段：依赖 baseline 统计结果生成完整深度视频 =====
                # all 模式会在 baseline 成功后执行；单独执行时必须已有统计结果。
                refine = [sys.executable, "scripts/depth_refine/run_refine.py", "--dataset-dir", str(project),
                          "--camera", CAMERA_KEYS[camera], "--result-root", str(result_root),
                          "--stage", args.refine_stage]
                print("[refine]", project.name, camera)
                if not args.dry_run:
                    subprocess.run(refine, check=True)
                    if args.plot_mean_curve:
                        plot_mean_curve(project, CAMERA_KEYS[camera], result_root)
                    # ===== 数据集替换：将 refine 视频覆盖回原始深度目录 =====
                    if args.replace_depth_video:
                        if args.refine_stage == "validate":
                            print("[警告] validate 不生成完整视频，无法替换")
                        else:
                            generated_root = result_root / "refine" / "videos"
                            replaced = 0
                            for generated in generated_root.glob("**/observation.images.*_depth_refined/episode_*.mp4"):
                                if CAMERA_KEYS[camera] not in generated.parent.name.lower():
                                    continue
                                target = replace_depth_video(project, CAMERA_KEYS[camera], generated)
                                if target:
                                    print("[替换]", target)
                                    replaced += 1
                            if not replaced:
                                print("[警告] 未找到可替换的深度视频", project.name, camera)


if __name__ == "__main__":
    main()
