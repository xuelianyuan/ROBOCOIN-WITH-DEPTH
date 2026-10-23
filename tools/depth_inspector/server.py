#!/usr/bin/env python3
"""Local depth/PCD inspection server. Requires ffmpeg, numpy and Pillow."""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import threading
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
OUTPUT_HTML = PROJECT_ROOT / "outputs" / "depth_inspector.html"
VIEWS = ("chest", "head", "left", "right")
METHODS = ("metric", "nested", "refine")


# ============================== 视频与数据探测 ==============================

def probe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(path)]
    data = json.loads(subprocess.check_output(cmd))["streams"][0]
    num, den = (int(x) for x in data.get("avg_frame_rate", "30/1").split("/"))
    fps = num / den if den else 30.0
    frames = int(data.get("nb_frames") or round(float(data.get("duration", 0)) * fps))
    return {"width": int(data["width"]), "height": int(data["height"]), "fps": fps, "frames": frames}


def camera_name(paths: list[Path], view: str) -> str | None:
    keys = {"chest": ("front_chest", "camera_chest", "cam_chest"),
            "head": ("front_head", "camera_head", "cam_head"),
            "left": ("left_wrist", "camera_left", "cam_left"),
            "right": ("right_wrist", "camera_right", "cam_right")}[view]
    for path in paths:
        if any(k in str(path).lower() for k in keys):
            return path.parent.name
    return None


def discover_pcds(root: Path, view: str, method: str) -> list[str]:
    """发现视角和方法均匹配的点云，避免三种方法错误复用同一文件。"""
    keys = {"chest": ("front_chest", "cam_chest"),
            "head": ("front_head", "cam_head"),
            "left": ("left_wrist", "cam_left"),
            "right": ("right_wrist", "cam_right")}[view]
    candidates = list(root.rglob("*.pcd"))
    # Generated PCDs may live outside the dataset under pcds/<dataset>/<task>/<method>.
    try:
        dataset_dir = next(part for part in root.parts if part.endswith("_depth_preview"))
        task = root.name
        pcd_method = {"metric": "original", "nested": "baseline", "refine": "refine"}[method]
        pcd_base = PROJECT_ROOT / "pcds" / dataset_dir / task / pcd_method
        camera_dir = {"chest": "cam_front_chest", "head": "cam_front_head",
                      "left": "cam_left_wrist", "right": "cam_right_wrist"}[view]
        scoped = sorted((pcd_base / camera_dir).glob("*.pcd"))
        if scoped:
            return [str(path.resolve()) for path in scoped]
        # Legacy layout without camera subdirectories is intentionally not
        # reused across views because it cannot be identified correctly.
    except StopIteration:
        pass
    candidates += list((PROJECT_ROOT / "pcds").glob("*.pcd"))
    view_files = [path for path in sorted(candidates) if any(key in path.stem.lower() for key in keys)]
    explicit = [path for path in view_files if method in path.stem.lower() or method in str(path.parent).lower()]
    # 无方法标识的历史 PCD 来自原始 metric 深度，只允许放在 Metric 行。
    if method == "metric" and not explicit:
        explicit = [path for path in view_files if not any(name in path.stem.lower() for name in ("nested", "refine"))]
    return [str(path.resolve()) for path in explicit]


def discover(root: Path) -> dict:
    videos = list(root.glob("videos/chunk-*/**/episode_*.mp4"))
    nested = list(root.glob("depth_results/baseline/videos/chunk-*/**/episode_*.mp4"))
    refined = list(root.glob("depth_results/refine/videos/chunk-*/**/episode_*.mp4"))

    # Validation datasets keep the three methods in separate repository trees:
    # original metric depth, sparse Nested output, and the final refined depth.
    # Accepting the original task directory here keeps the documented command
    # repository-relative and avoids one-off manifests with machine-local paths.
    dataset = root.parent.name
    if dataset.endswith("_depth_preview"):
        task = root.name
        nested += list((PROJECT_ROOT / "outputs" / "refine_validation" / dataset / task /
                       "baseline" / "videos").glob("chunk-*/**/episode_*.mp4"))
        refined += list((PROJECT_ROOT / "data" / "refine" / dataset / task /
                        "videos").glob("chunk-*/**/episode_*.mp4"))
    cfg = {"title": root.name, "rgb": {}, "groups": {},
           "invalid_depth": {"min_exclusive": 0, "max_exclusive": 4095}}
    available_views = []
    for view in VIEWS:
        rgb_name = camera_name([p for p in videos if p.parent.name.endswith("_rgb")], view)
        depth_name = camera_name([p for p in videos if p.parent.name.endswith("_depth")], view)
        if rgb_name:
            cfg["rgb"][view] = str(next(p for p in videos if p.parent.name == rgb_name).resolve())
        candidates = {"metric": next((p for p in videos if p.parent.name == depth_name), None),
                      "nested": next((p for p in nested if camera_name([p], view)), None),
                      "refine": next((p for p in refined if camera_name([p], view)), None)}
        for method, path in candidates.items():
            pcd_files = discover_pcds(root, view, method)
            cfg["groups"][f"{view}.{method}"] = {"depth": str(path.resolve()) if path else None, "pcd": pcd_files or None}
        if rgb_name or any(cfg["groups"][f"{view}.{method}"]["depth"] for method in METHODS):
            available_views.append(view)
    cfg["views"] = available_views
    return cfg


class Inspector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.meta: dict = {"title": cfg.get("title", "Depth inspector"), "views": cfg.get("views", list(VIEWS)),
                           "groups": {}, "rgb": {}, "ready": False}
        self.frames: dict[str, dict] = {}
        self.preview_dir = tempfile.TemporaryDirectory(prefix="depth-inspector-")
        self.previews: dict[str, Path] = {}
        self.lock = threading.Lock()
        for view, value in cfg.get("rgb", {}).items():
            p = Path(value)
            if p.exists():
                self.meta["rgb"][view] = {**probe(p), "url": f"/media/rgb/{view}"}
        threading.Thread(target=self._prepare, daemon=True).start()

    def _prepare(self):
        try:
            self._analyze()
        except Exception as exc:
            self.meta["error"] = str(exc)

    def _analyze(self):
        """统计所有有效帧，并为同一视角的三种方法生成统一归一化范围。"""
        invalid = self.cfg.get("invalid_depth", {})
        lo, hi = invalid.get("min_exclusive", 0), invalid.get("max_exclusive", 4095)
        for key, group in self.cfg.get("groups", {}).items():
            path = Path(group["depth"]) if group.get("depth") else None
            if not path or not path.exists():
                self.meta["groups"][key] = {"available": False, "error": "Depth video not found"}
                continue
            info = probe(path); n = info["frames"]
            cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray12le", "-"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            size = info["width"] * info["height"] * 2
            means, video_min, video_max = [], np.inf, -np.inf
            for idx in range(n):
                raw = proc.stdout.read(size)
                if len(raw) != size: break
                frame = np.frombuffer(raw, np.uint16)
                valid = frame[(frame > lo) & (frame < hi)]
                means.append(float(valid.mean()) if valid.size else None)
                if valid.size:
                    video_min = min(video_min, float(valid.min()))
                    video_max = max(video_max, float(valid.max()))
            proc.wait()
            if not np.isfinite(video_min) or video_max <= video_min:
                video_min, video_max = 0.0, 1.0
            pcd_source = group.get("pcd")
            pcd_video_source = group.get("pcd_video")
            pcd_video_path = Path(pcd_video_source) if pcd_video_source else None
            pcd_frames = []
            if isinstance(pcd_source, list):
                pcd_frames = sorted({int(match.group(1)) for value in pcd_source
                                     if (match := re.search(r"frame[_-]?(\d+)", Path(value).stem))})
            item = {**info, "available": True, "min": video_min, "max": video_max,
                    "means": means, "pcd": bool(pcd_source), "pcd_frames": pcd_frames,
                    "pcd_pattern": pcd_source if isinstance(pcd_source, str) else None,
                    "pcd_video": bool(pcd_video_path and pcd_video_path.is_file()),
                    "pcd_video_url": f"/media/pcd/{key}" if pcd_video_path and pcd_video_path.is_file() else None}
            with self.lock: self.meta["groups"][key] = item

        # 每个视角内 Metric/Nested/Refine 共用三个视频的总 min/max。
        for view in VIEWS:
            items = [self.meta["groups"].get(f"{view}.{method}") for method in METHODS]
            items = [item for item in items if item and item.get("available")]
            if not items:
                continue
            normalize_min = min(item["min"] for item in items)
            normalize_max = max(item["max"] for item in items)
            for item in items:
                item["normalize_min"] = normalize_min
                item["normalize_max"] = normalize_max
        for key, info in self.meta["groups"].items():
            if info.get("available"):
                self.meta["status"] = f"正在生成流畅播放预览：{key}"
                self._preview(key)
        self.meta["ready"] = True

    def _normalize(self, depth: np.ndarray, info: dict) -> np.ndarray:
        invalid = self.cfg.get("invalid_depth", {})
        valid = (depth > invalid.get("min_exclusive", 0)) & (depth < invalid.get("max_exclusive", 4095))
        normalized = np.zeros(depth.shape, np.uint8)
        lo, hi = info["normalize_min"], info["normalize_max"]
        normalized[valid] = np.clip((depth[valid].astype(float) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
        return normalized

    def _preview(self, key: str):
        """一次顺序解码并生成浏览器预览，播放期间无需逐帧启动 FFmpeg。"""
        info = self.meta["groups"][key]
        path = Path(self.preview_dir.name) / f"{key}.mp4"
        size = info["width"] * info["height"] * 2
        with subprocess.Popen([
            "ffmpeg", "-v", "error", "-threads", "1", "-i", self.cfg["groups"][key]["depth"],
            "-f", "rawvideo", "-pix_fmt", "gray12le", "-",
        ], stdout=subprocess.PIPE) as decoder:
            try:
                with subprocess.Popen([
                    "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                    "-s", f"{info['width']}x{info['height']}", "-r", str(info["fps"]), "-i", "-",
                    "-an", "-c:v", "libx264", "-threads", "1", "-preset", "ultrafast", "-crf", "18",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
                ], stdin=subprocess.PIPE) as encoder:
                    try:
                        while raw := decoder.stdout.read(size):
                            if len(raw) != size:
                                raise ValueError(f"Incomplete depth frame: {key}")
                            depth = np.frombuffer(raw, np.uint16).reshape(info["height"], info["width"])
                            encoder.stdin.write(self._normalize(depth, info).tobytes())
                    finally:
                        encoder.stdin.close()
                    if encoder.wait():
                        raise RuntimeError(f"Preview encoding failed: {key}")
                if decoder.wait():
                    raise RuntimeError(f"Preview decoding failed: {key}")
            finally:
                decoder.stdout.close()
                if decoder.poll() is None:
                    decoder.terminate()
        self.previews[key] = path
        info["url"] = f"/media/depth/{key}"

    # ============================== 帧数据接口 ==============================

    def depth_png(self, key: str, frame: int) -> bytes:
        group = self.cfg["groups"][key]; path = Path(group["depth"]); info = self.meta["groups"][key]
        frame = max(0, min(frame, info["frames"] - 1))
        # 先按时间定位到临近 GOP，再只解码目标帧，避免每次从视频开头扫描。
        timestamp = frame / info["fps"]
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{timestamp:.6f}", "-i", str(path),
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray12le", "-"]
        raw = subprocess.check_output(cmd)
        depth = np.frombuffer(raw, np.uint16).reshape(info["height"], info["width"])
        valid = (depth > self.cfg.get("invalid_depth", {}).get("min_exclusive", 0)) & (depth < self.cfg.get("invalid_depth", {}).get("max_exclusive", 4095))
        normalized = np.zeros(depth.shape, np.uint8)
        norm_min = info.get("normalize_min", info["min"])
        norm_max = info.get("normalize_max", info["max"])
        normalized[valid] = np.clip((depth[valid].astype(float) - norm_min) * 255 / (norm_max - norm_min), 0, 255).astype(np.uint8)
        out = io.BytesIO(); Image.fromarray(normalized).save(out, "PNG"); return out.getvalue()

    def pcd(self, key: str, frame: int) -> dict:
        source = self.cfg["groups"][key].get("pcd")
        if not source: raise FileNotFoundError("未配置 PCD")
        if isinstance(source, list):
            indexed = [(int(match.group(1)), Path(value)) for value in source
                       if (match := re.search(r"frame[_-]?(\d+)", Path(value).stem))]
            if not indexed: raise FileNotFoundError("PCD 文件名中缺少 frame 帧号")
            _, path = min(indexed, key=lambda item: abs(item[0] - frame))
        else:
            path = Path(source.format(frame=frame))
        if not path.exists(): raise FileNotFoundError(str(path))
        raw = path.read_bytes(); marker = raw.find(b"DATA ")
        end = raw.find(b"\n", marker); header = raw[:end].decode("ascii", "replace"); body = raw[end + 1:]
        fields = re.search(r"^FIELDS (.+)$", header, re.M).group(1).split()
        mode = re.search(r"^DATA (.+)$", header, re.M).group(1).strip()
        if mode == "ascii":
            data = np.loadtxt(io.BytesIO(body), ndmin=2)
            xyz = data[:, [fields.index(axis) for axis in ("x", "y", "z")]]
        elif mode == "binary":
            sizes = [int(x) for x in re.search(r"^SIZE (.+)$", header, re.M).group(1).split()]
            types = re.search(r"^TYPE (.+)$", header, re.M).group(1).split()
            counts_match = re.search(r"^COUNT (.+)$", header, re.M)
            counts = [int(x) for x in counts_match.group(1).split()] if counts_match else [1] * len(fields)
            formats = []
            type_map = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4"}
            for field, size, kind, count in zip(fields, sizes, types, counts):
                base = type_map[(kind, size)]; formats.append((field, base) if count == 1 else (field, base, (count,)))
            data = np.frombuffer(body, dtype=np.dtype(formats))
            xyz = np.column_stack((data["x"], data["y"], data["z"]))
        else:
            raise ValueError(f"暂不支持 DATA {mode} PCD")
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz) > 25000: xyz = xyz[::max(1, len(xyz) // 25000)]
        return {"points": xyz.tolist(), "file": path.name}


INSPECTOR: Inspector


# ============================== HTTP 服务 ==============================
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs): super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)
    def send_data(self, data: bytes, content_type: str, status=200):
        self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        parsed = urlparse(self.path); q = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self.send_response(302); self.send_header("Location", "/outputs/depth_inspector.html"); self.end_headers(); return
            if parsed.path == "/api/meta": return self.send_data(json.dumps(INSPECTOR.meta).encode(), "application/json")
            if parsed.path == "/api/depth": return self.send_data(INSPECTOR.depth_png(q["id"][0], int(q["frame"][0])), "image/png")
            if parsed.path == "/api/pcd": return self.send_data(json.dumps(INSPECTOR.pcd(q["id"][0], int(q["frame"][0]))).encode(), "application/json")
            if parsed.path.startswith("/media/rgb/"):
                view = parsed.path.rsplit("/", 1)[-1]; self.path = INSPECTOR.cfg["rgb"][view]; return self._file(Path(self.path), "video/mp4")
            if parsed.path.startswith("/media/depth/"):
                return self._file(INSPECTOR.previews[parsed.path.rsplit("/", 1)[-1]], "video/mp4")
            if parsed.path.startswith("/media/pcd/"):
                key = parsed.path.rsplit("/", 1)[-1]
                return self._file(Path(INSPECTOR.cfg["groups"][key]["pcd_video"]), "video/mp4")
            return super().do_GET()
        except Exception as exc: self.send_data(json.dumps({"error": str(exc)}).encode(), "application/json", 404)
    def _file(self, path: Path, content_type: str):
        size = path.stat().st_size; start, end = 0, size - 1
        if self.headers.get("Range"):
            match = re.match(r"bytes=(\d*)-(\d*)", self.headers["Range"]); start = int(match.group(1) or 0); end = int(match.group(2) or end)
            self.send_response(206); self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else: self.send_response(200)
        self.send_header("Content-Type", content_type); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(end-start+1)); self.end_headers()
        with path.open("rb") as f: f.seek(start); self.wfile.write(f.read(end-start+1))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path); parser.add_argument("--manifest", type=Path)
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8765); args = parser.parse_args()
    if not args.dataset and not args.manifest: parser.error("provide --dataset or --manifest")
    cfg = json.loads(args.manifest.read_text()) if args.manifest else discover(args.dataset.resolve())
    # Keep a convenient report copy under outputs/ for opening or archiving.
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    report_html = (HERE / "index.html").read_text(encoding="utf-8")
    report_html = report_html.replace('href="style.css"', 'href="../tools/depth_inspector/style.css"')
    report_html = report_html.replace('src="app.js"', 'src="../tools/depth_inspector/app.js"')
    OUTPUT_HTML.write_text(report_html, encoding="utf-8")
    global INSPECTOR; INSPECTOR = Inspector(cfg)
    print(f"Depth inspector: http://{args.host}:{args.port}/outputs/depth_inspector.html", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__": main()
