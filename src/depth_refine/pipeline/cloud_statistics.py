"""按任务和视角扫描原始深度视频，输出逐帧候选和连续区间报告。"""
from __future__ import annotations

import csv
import json
import logging
import multiprocessing
import subprocess
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from src.depth_refine.evaluation.cloud_artifacts import (
    CloudConfig, cloud_segments, config_dict, detect_cloud, summarize_segments,
)
from src.depth_refine.io_utils import iter_video_frames, probe_video
from src.depth_refine.visualization.cloud_review import save_review, write_overview


def discover_depth_videos(roots: Sequence[Path]) -> list[dict]:
    """仅匹配 <root>/<task>/videos/chunk-*/*_depth/episode_*.mp4，排除派生结果。"""
    videos, seen = [], set()
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.glob("*/videos/chunk-*/*_depth/episode_*.mp4")):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            relative = path.relative_to(root)
            videos.append({"path": str(path), "dataset": root.name,
                           "task": relative.parts[0], "chunk": relative.parts[2],
                           "camera": path.parent.name.removeprefix("observation.images."),
                           "episode": path.stem, "size_bytes": path.stat().st_size,
                           "mtime_ns": path.stat().st_mtime_ns})
    return videos


def _write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _analyze_video(item: dict, output: Path, config: CloudConfig) -> dict:
    """串行和并行均调用此函数；每个视频有独立目录，进程间不共享帧数组。"""
    cv2.setNumThreads(1)
    folder = output / "videos" / item["dataset"] / item["task"] / item["chunk"] / item["camera"] / item["episode"]
    if folder.exists():
        # 仅在显式恢复时进入此分支；保存中断/失败尝试，避免覆盖其诊断信息。
        attempt = 1
        while folder.with_name(f"{folder.name}.previous_attempt_{attempt}").exists():
            attempt += 1
        folder.rename(folder.with_name(f"{folder.name}.previous_attempt_{attempt}"))
    folder.mkdir(parents=True, exist_ok=False)
    report = {**item, "status": "failed", "output": str(folder.relative_to(output))}
    scores, assessable, rows = [], [], []
    samples = {}
    context = deque(maxlen=7)
    transition = []
    transition_until = None
    try:
        info = probe_video(Path(item["path"]))
        if info["pix_fmt"] not in {"gray12le", "gray16le"}:
            raise ValueError(f"Unsupported native depth format: {info['pix_fmt']}")
        if not np.isfinite(info["fps"]) or info["fps"] <= 0:
            raise ValueError("Invalid video fps")
        report["video_info"] = info
        # gray16le 沿用其原生数值范围；不把输入强制转换到 12-bit。
        valid_max = 4095 if info["pix_fmt"] == "gray12le" else 65535
        for index, frame in enumerate(iter_video_frames(Path(item["path"]), info["pix_fmt"],
                                                        info=info, max_width=config.analysis_width)):
            result, gray, mask = detect_cloud(frame, config, valid_max=valid_max)
            score = result["score"]
            rows.append({"frame_index": index, "seconds": index / info["fps"], **result})
            scores.append(score)
            assessable.append(result["assessable"])
            sample = {"index": index, "score": score, "gray": gray, "mask": mask,
                      "suspected": result["suspected"], "assessable": result["assessable"]}
            if "peak" not in samples or score > samples["peak"]["score"]:
                samples["peak"] = sample
            if result["assessable"]:
                if "boundary" not in samples or abs(score - config.coverage) < abs(samples["boundary"]["score"] - config.coverage):
                    samples["boundary"] = sample
                if "low" not in samples or score < samples["low"]["score"]:
                    samples["low"] = sample
            context.append(sample)
            # 保存首次阳性的邻近帧，复核连续性；不使用时间平滑或补齐阴性间隔。
            if result["suspected"] and transition_until is None:
                transition = list(context)[-4:]
                transition_until = index + 3
            elif transition_until is not None and index <= transition_until:
                transition.append(sample)
        if not rows:
            raise ValueError("No decoded frames")
        declared = info["declared_frames"]
        if declared is not None and declared != len(rows):
            raise ValueError(f"Frame count mismatch: declared={declared}, decoded={len(rows)}")
        report["status"] = "completed"
        report["sensitivity"] = {}
        for name, threshold in (("low", config.coverage * 0.75), ("default", config.coverage),
                                ("high", config.coverage * 1.25)):
            flags = np.asarray(assessable) & (np.asarray(scores) >= threshold)
            segments = cloud_segments(flags, info["fps"])
            report["sensitivity"][name] = {"threshold": threshold,
                **summarize_segments(segments, len(rows), sum(assessable))}
            if name == "default":
                _write_csv(folder / "segments.csv", segments)
        report.update(report["sensitivity"]["default"])
        save_review(folder / "review.png", list(samples.items()), transition)
    except (OSError, ValueError, RuntimeError, KeyError, IndexError, subprocess.SubprocessError,
            cv2.error) as exc:
        # 保存失败信息和已读帧，但失败视频不计入成功统计，避免伪装完整扫描。
        report["error"] = str(exc)
        report["partial_decoded_frames"] = len(rows)
        logging.exception("Cloud statistics failed: %s", item["path"])
    _write_csv(folder / "frames.csv", rows)
    _write_json(folder / "report.json", report)
    return report


def _aggregate(reports: list[dict], sensitivity: str = "default") -> dict:
    completed = [r for r in reports if r["status"] == "completed"]
    counts = [r["sensitivity"][sensitivity] for r in completed]
    affected = [r for r in completed if r["sensitivity"][sensitivity]["suspected_frames"] > 0]
    all_tasks = {(r["dataset"], r["task"]) for r in reports}
    affected_tasks = {(r["dataset"], r["task"]) for r in affected}
    affected_frames = sum(r["total_frames"] for r in affected)
    fields = ["total_frames", "assessable_frames", "unknown_frames", "suspected_frames",
              "events", "isolated_single_frames", "consecutive_events",
              "frames_in_consecutive_events", "events_at_least_5_frames", "frames_in_events_at_least_5"]
    result = {field: sum(r[field] for r in counts) for field in fields}
    positive, total, usable = (result[f] for f in ("suspected_frames", "total_frames", "assessable_frames"))
    result.update(videos=len(reports), completed_videos=len(completed), failed_videos=len(reports) - len(completed),
                  tasks=len(all_tasks), tasks_with_suspected_cloud=len(affected_tasks),
                  task_suspected_fraction=len(affected_tasks) / len(all_tasks) if all_tasks else None,
                  videos_with_suspected_cloud=len(affected),
                  video_suspected_fraction=len(affected) / len(completed) if completed else None,
                  frames_in_videos_with_suspected_cloud=affected_frames,
                  suspected_fraction_in_affected_videos=positive / affected_frames if affected_frames else None,
                  suspected_fraction_all_frames=positive / total if total else None,
                  suspected_fraction_assessable=positive / usable if usable else None,
                  consecutive_frame_fraction=result["frames_in_consecutive_events"] / positive if positive else None,
                  longest_event_frames=max((r["longest_event_frames"] for r in counts), default=0),
                  longest_event_seconds=max((r["longest_event_seconds"] for r in counts), default=0),
                  video_patterns={pattern: sum(r["pattern"] == pattern for r in counts)
                                  for pattern in ("none_detected", "frame_dominant", "segment_dominant", "mixed")})
    return result


def run_cloud_statistics(roots: Sequence[Path], output: Path, config: CloudConfig,
                         *, references: Sequence[Path] = (), workers: int = 1,
                         limit: int | None = None,
                         resume: bool = False,
                         progress: Callable[[int, int, dict], None] | None = None) -> dict:
    """扫描全部视频，写独立报告；默认拒绝覆盖任何已有输出目录。"""
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    videos = discover_depth_videos(roots)
    if not videos:
        raise ValueError("No original depth videos found")
    if limit is not None:
        videos = videos[:limit]
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        previous_config = json.loads((output / "config.json").read_text())
        previous_inventory = json.loads((output / "inventory.json").read_text())
        if (previous_config["config"] != config_dict(config)
                or previous_config["roots"] != [str(p.resolve()) for p in roots]
                or previous_inventory != videos
                or [r['path'] for r in previous_config['references']] != [str(p.resolve()) for p in references]):
            raise ValueError("Resume requires identical input inventory, references, roots and configuration")
    else:
        output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "inventory.json", videos)
    reference_reports = []
    for index, path in enumerate(references):
        result, gray, mask = detect_cloud(np.load(path, allow_pickle=False), config)
        reference_reports.append({"path": str(path.resolve()), **result})
        save_review(output / f"reference_{index}.png", [("confirmed positive", {
            "index": index, "score": result["score"], "gray": gray, "mask": mask,
            "suspected": result["suspected"], "assessable": result["assessable"]})], [])
    metadata = {"config": config_dict(config), "roots": [str(p.resolve()) for p in roots],
                "references": reference_reports, "scope": "original videos only",
                "limited_videos": limit,
                "notes": ["Heuristic candidates, not manually verified cloud labels or model attribution.",
                          "Two positive references do not measure precision or recall across datasets.",
                          "Scores use per-frame P1-P99 normalization; display range can affect apparent cloud strength.",
                          "Consecutive means >=2 positive frames, sustained means >=5; no gap bridging or temporal smoothing.",
                          "Unknown frames split events and are reported separately; failures excluded from totals.",
                          "Seconds are frame_index/avg_fps, approximate for variable frame rate."]}
    _write_json(output / "config.json", metadata)
    if reference_reports and not all(r["suspected"] for r in reference_reports):
        raise ValueError("Confirmed references did not pass detection; inspect reference images and adjust config")
    reports, pending = [], []
    for item in videos:
        path = output / "videos" / item["dataset"] / item["task"] / item["chunk"] / item["camera"] / item["episode"] / "report.json"
        if resume and path.is_file():
            existing = json.loads(path.read_text())
            if existing["status"] == "completed":
                reports.append(existing)
                continue
        pending.append(item)
    if workers == 1:
        for item in pending:
            report = _analyze_video(item, output, config)
            reports.append(report)
            if progress:
                progress(len(reports), len(videos), report)
    else:
        # OpenCV 正例检查可能已经初始化线程池；spawn 避免 fork 继承锁而死锁。
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(_analyze_video, item, output, config) for item in pending]
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                if progress:
                    progress(len(reports), len(videos), report)
    reports.sort(key=lambda r: r["path"])
    _write_json(output / "video_reports.json", reports)
    flat = [{**{k: r[k] for k in ("dataset", "task", "camera", "chunk", "episode", "path", "status", "output")},
             "has_suspected_cloud": r["suspected_frames"] > 0 if r["status"] == "completed" else None,
             "fps": r.get("video_info", {}).get("fps"),
             "suspected_duration_seconds": (r["suspected_frames"] / r["video_info"]["fps"]
                                             if r["status"] == "completed" else None),
             **{k: r.get(k) for k in ("total_frames", "assessable_frames", "unknown_frames", "suspected_frames",
                  "suspected_fraction_all_frames", "events", "isolated_single_frames", "consecutive_events",
                  "frames_in_consecutive_events", "events_at_least_5_frames",
                  "consecutive_frame_fraction", "longest_event_frames", "longest_event_seconds", "pattern", "error")}}
            for r in reports]
    _write_csv(output / "videos.csv", flat)
    _write_csv(output / "videos_with_cloud.csv", [r for r in flat if r["has_suspected_cloud"]])
    # 任务以目录为单位，任一视角/episode 有阳性即可命中；不把不同视角的事件合并。
    task_reports = [{"dataset": dataset, "task": task,
                     **_aggregate([r for r in reports if (r["dataset"], r["task"]) == (dataset, task)])}
                    for dataset, task in sorted({(r["dataset"], r["task"]) for r in reports})]
    _write_json(output / "task_reports.json", task_reports)
    _write_csv(output / "tasks.csv", [{k: v for k, v in r.items() if k != "video_patterns"}
                                     for r in task_reports])
    summary = {**metadata, "overall": _aggregate(reports),
               "datasets": {name: _aggregate([r for r in reports if r["dataset"] == name])
                            for name in sorted({r["dataset"] for r in reports})},
               "cameras": {name: _aggregate([r for r in reports if f"{r['dataset']}/{r['camera']}" == name])
                           for name in sorted({f"{r['dataset']}/{r['camera']}" for r in reports})},
               "sensitivity": {name: _aggregate(reports, name) for name in ("low", "default", "high")},
               "failures": [{"path": r["path"], "error": r.get("error")} for r in reports if r["status"] != "completed"]}
    lengths = Counter()
    for report in reports:
        path = output / report["output"] / "segments.csv"
        if report["status"] == "completed" and path.stat().st_size:
            with path.open() as handle:
                lengths.update(int(row["frame_count"]) for row in csv.DictReader(handle))
    summary["event_length_histogram"] = {str(length): count for length, count in sorted(lengths.items())}
    continuous = summary["overall"]["consecutive_events"]
    summary["fraction_of_consecutive_events_lasting_2_or_3_frames"] = (
        (lengths[2] + lengths[3]) / continuous if continuous else None)
    _write_json(output / "summary.json", summary)
    write_overview(output, summary, reports)
    return summary
