#!/usr/bin/env python3
"""Export latest per-view attempts and complete per-episode camera totals."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def summarize(result_root, cameras):
    rows = []
    for path in sorted((result_root / "timings").glob("*/*/*/episode_*.json")):
        record = json.loads(path.read_text())
        if record["camera"] in cameras:
            rows.append(record)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["chunk"], row["episode"], row["stage"])].append(row)
    totals = []
    for (chunk, episode, stage), records in sorted(groups.items()):
        completed = {r["camera"]: r for r in records if r["status"] == "completed"}
        missing = sorted(set(cameras) - completed.keys())
        totals.append(dict(chunk=chunk, episode=episode, stage=stage,
                           completed_views=len(completed), expected_views=len(cameras),
                           missing_views=";".join(missing),
                           total_seconds=(sum(r["elapsed_seconds"] for r in completed.values())
                                          if not missing else "")))
    return rows, totals


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="逐 episode、逐视角 Nested/refine 耗时汇总（秒）")
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+",
                        default=["cam_front_head", "cam_left_wrist", "cam_right_wrist"],
                        help="完整相机名，默认头部、左腕、右腕")
    args = parser.parse_args()
    rows, totals = summarize(args.result_root, args.cameras)
    if not rows:
        parser.exit(message="没有计时记录：请先运行新版 baseline/refine；旧结果不能反推耗时。\n")
    out = args.result_root / "timings"
    write_csv(out / "per_view.csv", rows,
              ["chunk", "episode", "camera", "stage", "status", "elapsed_seconds",
               "source_num_frames", "sample_num_frames", "started_at", "reason", "error"])
    write_csv(out / "per_episode.csv", totals,
              ["chunk", "episode", "stage", "completed_views", "expected_views",
               "missing_views", "total_seconds"])
    for row in rows:
        print(f"{row['chunk']}/{row['episode']} {row['camera']} {row['stage']}: "
              f"{row['elapsed_seconds']:.3f}s [{row['status']}]")
    print(f"CSV: {out / 'per_view.csv'}\nCSV: {out / 'per_episode.csv'}")


if __name__ == "__main__":
    main()
