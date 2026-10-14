"""为云斑候选输出可人工复核的静态对照图及报告。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_review(path: Path, samples: list[tuple[str, dict]], transition: list[dict]) -> None:
    """每行同时展示原深度灰度图及红色候选；底部展示首次阳性相邻帧。"""
    if not samples:
        return
    width = max(s["gray"].shape[1] for _, s in samples)
    height = max(s["gray"].shape[0] for _, s in samples)
    sheet = Image.new("RGB", (width * 2, (height + 30) * len(samples) + (height // 2 + 30) * bool(transition)), "#202020")
    draw = ImageDraw.Draw(sheet)
    for row, (name, sample) in enumerate(samples):
        y = row * (height + 30)
        draw.text((5, y + 5), f"{name} | frame {sample['index']} | score {sample['score']:.4f} | {'unknown' if not sample['assessable'] else 'suspected' if sample['suspected'] else 'negative'}", fill="white")
        gray = sample["gray"]
        overlay = np.repeat(gray[..., None], 3, axis=2)
        overlay[sample["mask"]] = [255, 48, 48]
        sheet.paste(Image.fromarray(gray), (0, y + 30))
        sheet.paste(Image.fromarray(overlay), (width, y + 30))
    if transition:
        y = len(samples) * (height + 30)
        thumb_w = width * 2 // len(transition)
        for col, sample in enumerate(transition):
            draw.text((col * thumb_w + 2, y + 5), f"{sample['index']}:{'+' if sample['suspected'] else '? ' if not sample['assessable'] else '-'}", fill="white")
            thumb = Image.fromarray(sample["gray"]).resize((thumb_w, height // 2))
            sheet.paste(thumb, (col * thumb_w, y + 30))
    sheet.save(path)


def write_overview(output: Path, summary: dict, reports: list[dict]) -> None:
    """写中文口径、分组汇总和逐视频复核链接；结果仍为启发式疑似标签。"""
    lines = ["# 深度视频云斑时间分布统计", "", "**本报告为启发式疑似云斑统计，未经全量人工标注验证。**",
             "扫描原始 `任务/videos/chunk-*/*_depth/episode_*.mp4`，不扫描 baseline/refine 派生视频。",
             "逐帧 P1–P99 归一化后，检测空间带通纹理，排除强边缘和方向单一的轮廓。",
             "连续 ≥2 帧算一段；单帧算孤立事件；另列 ≥5 帧区间。阴性/未知帧会切断区间，不补间隙。",
             "时间使用帧号/平均帧率；VFR 视频时间为近似值。失败视频不计入成功统计。", ""]
    fraction = summary['overall']['consecutive_frame_fraction']
    short = summary['fraction_of_consecutive_events_lasting_2_or_3_frames']
    if fraction is not None and short is not None:
        lines += [f"默认阈值下，**{fraction:.2%}** 的疑似帧位于连续区间内；连续区间中 **{short:.2%}** 只持续 2–3 帧。",
                  "这表明孤立单帧和连续短段并存；上述比例是启发式标签的时间分布，不能直接当作人工确认的云斑频率。", ""]
    lines += [
             "任务按 `<数据集>/<任务目录>` 计数，任一视角任一帧命中即算含疑似云斑任务。",
             "云斑次数按时间事件计数：孤立单帧计 1 次，连续阳性区间计 1 次；同帧多个空间斑块不另计。",
             "视频占比 = 含疑似云斑视频数 / 成功扫描视频数；任务占比按全部发现的任务目录计算。",
             "跨摄像头的事件分别计数，任务级求和不做同一时刻去重。", "",
             "| 数据集 | 含疑似云斑任务/总任务 | 任务占比 | 含疑似云斑视频/完成视频 | 视频占比 |",
             "|---|---:|---:|---:|---:|"]
    for name, s in [("全部", summary["overall"]), *summary["datasets"].items()]:
        task_ratio = f"{s['task_suspected_fraction']:.2%}" if s['task_suspected_fraction'] is not None else "N/A"
        video_ratio = f"{s['video_suspected_fraction']:.2%}" if s['video_suspected_fraction'] is not None else "N/A"
        lines.append(f"| {name} | {s['tasks_with_suspected_cloud']}/{s['tasks']} | {task_ratio} | {s['videos_with_suspected_cloud']}/{s['completed_videos']} | {video_ratio} |")
    lines += ["", "| 数据集 | 含云斑视频的总帧数 | 其中疑似云斑帧 | 云斑帧占这些视频的比例 | 时间事件总数 | 独立事件 | 连续事件 | 连续事件包含的帧数 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in [("全部", summary["overall"]), *summary["datasets"].items()]:
        ratio = f"{s['suspected_fraction_in_affected_videos']:.2%}" if s['suspected_fraction_in_affected_videos'] is not None else "N/A"
        lines.append(f"| {name} | {s['frames_in_videos_with_suspected_cloud']} | {s['suspected_frames']} | {ratio} | {s['events']} | {s['isolated_single_frames']} | {s['consecutive_events']} | {s['frames_in_consecutive_events']} |")
    lines += ["",
             "| 数据集 | 完成/总视频 | 疑似帧/已分析帧 | 未知帧 | 单帧事件 | 连续区间 | 连续区间内疑似帧占比 | 最长区间 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in [("全部", summary["overall"]), *summary["datasets"].items()]:
        fraction = s["consecutive_frame_fraction"]
        ratio = f"{fraction:.2%}" if fraction is not None else "N/A"
        lines.append(f"| {name} | {s['completed_videos']}/{s['videos']} | {s['suspected_frames']}/{s['total_frames']} | {s['unknown_frames']} | {s['isolated_single_frames']} | {s['consecutive_events']} | {ratio} | {s['longest_event_frames']} 帧 / {s['longest_event_seconds']:.2f}s |")
    lines += ["", "## 阈值敏感性", "", "| 覆盖率阈值 | 疑似帧 | 孤立单帧 | 连续区间 | 连续区间内疑似帧占比 |", "|---:|---:|---:|---:|---:|"]
    for name, s in summary["sensitivity"].items():
        threshold = summary["config"]["coverage"] * {"low": 0.75, "default": 1, "high": 1.25}[name]
        ratio = f"{s['consecutive_frame_fraction']:.2%}" if s["consecutive_frame_fraction"] is not None else "N/A"
        lines.append(f"| {threshold:.3f} | {s['suspected_frames']} | {s['isolated_single_frames']} | {s['consecutive_events']} | {ratio} |")
    lines += ["", "覆盖率阈值改变时，出现频率和最长区间长度可能明显变化，请结合复核图判断。", "",
              "## 区间长度分布", "", "| 区间帧数 | 事件数量 |", "|---:|---:|"]
    for length, count in summary['event_length_histogram'].items():
        lines.append(f"| {length} | {count} |")
    lines += ["", "## 检查结果", "", "[tasks.csv](tasks.csv) 为逐任务汇总，[videos.csv](videos.csv) 为全部视频汇总，[videos_with_cloud.csv](videos_with_cloud.csv) 仅列出有疑似云斑的视频。",
              "每个视频目录含 `frames.csv`、`segments.csv`、`report.json`、`review.png`。",
              "复核图左列为深度灰度图，右列红色为候选纹理；peak/boundary/low 分别为最高分、阈值附近、最低分可分析帧。",
              "自然曲面和弱轮廓可能误报。两个正例不能估计跨场景准确率；建议先检查边界样本。", "",
              "| 数据集 / 任务 / 视角 | 视频总帧 | 云斑帧 | 时间事件 | 独立事件 | 连续事件 | 时间模式/状态 | 复核图 |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for r in reports:
        link = f"[review]({r['output']}/review.png)" if r["status"] == "completed" else r.get("error", "failed")
        lines.append(f"| {r['dataset']}/{r['task']}/{r['camera']}/{r['episode']} | {r.get('total_frames', '')} | {r.get('suspected_frames', '')} | {r.get('events', '')} | {r.get('isolated_single_frames', '')} | {r.get('consecutive_events', '')} | {r.get('pattern', r['status'])} | {link} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
