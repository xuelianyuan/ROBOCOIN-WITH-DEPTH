#!/usr/bin/env python3
"""对比原始 Metric、Refined Metric 和原始 Nested 深度。

脚本读取同一项目、同一 episode 的原始 Metric 视频、Refined 视频以及
对应 Nested ``.npy`` 采样帧，生成一张 2x3 对比图：上排是三张灰度图，
下排是使用相同深度范围和 bins 的三个直方图。

用法示例::

    python scripts/plot_depth_comparison.py \
        --before data/lerobot/your_dataset/videos/chunk-000/observation.images.cam_front_head_depth/episode_000000.mp4 \
        --after data/lerobot/your_dataset/depth_results/refine/videos/chunk-000/observation.images.cam_front_head_depth_refined/episode_000000.mp4 \
        --nested data/lerobot/your_dataset/depth_results/baseline/statistics/cam_front_head/episode_000000/frame_000000000.npy \
        --frame 0

默认输出到 ``depth_results/refine/plots/``，也可用 ``--output`` 指定路径。

批量绘制 baseline episode 的全部采样帧::

    python scripts/plot_depth_comparison.py \
        --before <项目>/videos/chunk-000/observation.images.cam_front_head_depth/episode_000000.mp4 \
        --after <项目>/depth_results/refine/videos/chunk-000/observation.images.cam_front_head_depth_refined/episode_000000.mp4 \
        --nested-dir <项目>/depth_results/baseline/statistics/cam_front_head/episode_000000 \
        --all-sampled
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def configure_chinese_font() -> None:
    """选择系统中可用的中文字体，避免保存图片时出现缺字警告。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = (
        "Noto Sans CJK SC",
        "Noto Sans CJK HK",
        "Droid Sans Fallback",
        "AR PL UMing CN",
        "AR PL UKai CN",
    )
    selected = next((name for name in preferred if name in available), None)
    if selected:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def decode(path: Path, pix_fmt: str = "gray12le") -> list[np.ndarray]:
    """使用 ffprobe/ffmpeg 解码单通道深度视频为 uint16 帧。"""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    width, height = map(int, probe.stdout.strip().split(","))
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", pix_fmt, "-"], capture_output=True, check=True,
    )
    raw = np.frombuffer(result.stdout, dtype=np.uint16)
    size = width * height
    return [raw[i:i + size].reshape(height, width)
            for i in range(0, len(raw) - size + 1, size)]


def main() -> None:
    configure_chinese_font()
    parser = argparse.ArgumentParser(description="绘制 Metric、Refined Metric、Nested 对比图")
    parser.add_argument("--before", required=True, help="原始 Metric 深度视频")
    parser.add_argument("--after", required=True, help="基于 Nested 统计迁移后的深度视频")
    parser.add_argument("--nested", default=None,
                        help="原始 Nested 深度；推荐对应帧的 .npy，也支持深度视频")
    parser.add_argument("--all-sampled", action="store_true",
                        help="根据 --nested-dir/sample_indices.json 批量绘制所有采样帧")
    parser.add_argument("--nested-dir", default=None,
                        help="baseline episode 目录，包含 sample_indices.json 和 frame_*.npy")
    parser.add_argument("--frame", type=int, default=0, help="显示的帧号，默认 0")
    parser.add_argument("--output", default=None, help="输出 PNG 路径")
    parser.add_argument("--max-depth", type=float, default=4095,
                        help="直方图和灰度显示的最大深度，默认 4095 mm")
    args = parser.parse_args()

    before_path, after_path = Path(args.before), Path(args.after)
    if args.all_sampled:
        if not args.nested_dir:
            raise SystemExit("--all-sampled 必须同时指定 --nested-dir")
        nested_dir = Path(args.nested_dir)
        metadata_path = nested_dir / "sample_indices.json"
        if not metadata_path.is_file():
            raise SystemExit(f"未找到采样索引文件：{metadata_path}")
        indices = json.loads(metadata_path.read_text()).get("sample_frame_indices", [])
        if not indices:
            raise SystemExit(f"采样索引为空：{metadata_path}")
        for frame in indices:
            nested_frame = nested_dir / f"frame_{int(frame):09d}.npy"
            if not nested_frame.is_file():
                print(f"[跳过] 缺少 Nested 帧：{nested_frame}")
                continue
            output = Path(args.output) if args.output else (
                nested_dir.parents[3] / "refine" / "plots" /
                f"frame_{int(frame):09d}_comparison.png"
            )
            if args.output:
                output = output.parent / f"{output.stem}_frame_{int(frame):09d}{output.suffix}"
            command = [
                "--before", str(before_path), "--after", str(after_path),
                "--nested", str(nested_frame), "--frame", str(int(frame)),
                "--output", str(output), "--max-depth", str(args.max_depth),
            ]
            subprocess.run([sys.executable, __file__, *command], check=True)
        return
    if not args.nested:
        raise SystemExit("单帧模式必须指定 --nested")
    nested_path = Path(args.nested)
    for label, path in (("原始 Metric", before_path), ("Refined Metric", after_path),
                        ("原始 Nested", nested_path)):
        if not path.is_file():
            raise SystemExit(
                f"未找到{label}：{path}\n"
                "请确认 refine 使用了相同的项目、相机视角和 --result-root。"
            )
    before, after = decode(before_path), decode(after_path)
    if args.frame < 0 or args.frame >= min(len(before), len(after)):
        raise SystemExit(f"帧号超出范围：before={len(before)}, after={len(after)}")
    b, a = before[args.frame].astype(float), after[args.frame].astype(float)
    if nested_path.suffix.lower() == ".npy":
        n = np.load(nested_path).astype(float)
    else:
        nested_frames = decode(nested_path)
        if args.frame >= len(nested_frames):
            raise SystemExit(f"Nested 视频帧号超出范围：{len(nested_frames)}")
        n = nested_frames[args.frame].astype(float)
    if args.output:
        output = Path(args.output)
    else:
        # refined 视频通常位于 <项目>/depth_results/refine/videos/chunk-*。
        # 将图片放到同一项目的 depth_results/refine/plots 下。
        try:
            result_root = after_path.parents[4]  # .../depth_results
            output = result_root / "refine" / "plots" / f"frame_{args.frame:09d}_comparison.png"
        except IndexError:
            output = after_path.with_name(f"frame_{args.frame:09d}_comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    images = (b, a, n)
    titles = ("原始 Metric", "Refined Metric（Nested 统计迁移）", "原始 Nested")
    colors = ("tab:blue", "tab:green", "tab:orange")
    fig, axes = plt.subplots(2, 3, figsize=(20, 9), constrained_layout=True)
    for ax, image, title in zip(axes[0], images, titles):
        display = np.ma.masked_where((image <= 0) | (image > args.max_depth), image)
        ax.imshow(display, cmap="gray", vmin=0, vmax=args.max_depth)
        ax.set_title(f"{title} 灰度图 | frame {args.frame}")
        ax.axis("off")
    bins = np.linspace(0, args.max_depth, 101)
    for ax, image, title, color in zip(axes[1], images, titles, colors):
        valid = image[(image > 0) & (image <= args.max_depth)]
        ax.hist(valid, bins=bins, color=color, alpha=.75)
        ax.set_title(f"{title}直方图 | 有效像素 {len(valid)}")
        ax.set_xlabel("深度 (mm)"); ax.set_ylabel("像素数"); ax.grid(alpha=.25)
    fig.suptitle("原始 Metric / Refined Metric / 原始 Nested 深度对比")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"对比图已保存：{output}")


if __name__ == "__main__":
    main()
