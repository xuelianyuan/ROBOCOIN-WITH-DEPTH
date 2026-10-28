"""使用全视频固定色标导出 RGB / DA3Metric / VDPP 对照视频。"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def export_metric_preview(rgb: list[np.ndarray], baseline: np.ndarray, refined: np.ndarray,
                          path: Path, fps: float, limits_m: tuple[float, float]) -> None:
    """保持所有帧、固定米制色标，同时保存三张对照图供人工检查。"""
    h, w = rgb[0].shape[:2]
    cmd = ["ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{3*w}x{h+40}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
           "-threads", "2", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    low, high = limits_m
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        for i, frame in enumerate(rgb):
            panels = [frame]
            for depth in (baseline[i], refined[i]):
                gray = (np.clip((depth - low) / (high - low), 0, 1) * 255).astype(np.uint8)
                panels.append(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[..., ::-1])
            canvas = np.zeros((h + 40, 3 * w, 3), np.uint8)
            canvas[40:] = np.concatenate(panels, axis=1)
            for j, name in enumerate(("RGB", "DA3Metric (m)", "DA3Metric + VDPP (m)")):
                cv2.putText(canvas, name, (j*w + 12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(canvas, f"#{i} | fixed scale {low:.2f}-{high:.2f} m", (w+200,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            proc.stdin.write(canvas.tobytes())
            if i in (0, len(rgb)//2, len(rgb)-1):
                cv2.imwrite(str(path.parent / f"comparison_{i:06d}.png"), canvas[..., ::-1])
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("预览视频编码失败")
