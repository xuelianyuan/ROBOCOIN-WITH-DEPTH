#!/usr/bin/env python3
"""
LeRobot 数据集可视化：RGB + Depth + Point Cloud（方向键切换帧）。

用法:
python scripts/load_lerobot.py \
    --repo-id AI2_Alphabot_2_arrange_teaset_0_qced_hardlink_ep000000 \
    --root data/refine/ai2_depth_preview/AI2_Alphabot_2_arrange_teaset_0_qced_hardlink_ep000000

python scripts/load_lerobot.py \
  --generate-pcd \
  --pcd-root pcd \
  --original-depth "/home/user/RoboCOIN-with-Depth/data/original/ai2_depth_preview/AI2_Alphabot_2_arrange_teaset_0_qced_hardlink_ep000000/videos/chunk-000/observation.images.cam_front_chest_depth" \
  --baseline-depth "/home/user/RoboCOIN-with-Depth/data/refine/ai2_depth_preview/AI2_Alphabot_2_arrange_teaset_0_qced_hardlink_ep000000/depth_results/baseline/videos/chunk-000/observation.images.cam_front_chest_depth_nested_sparse" \
  --refine-depth "/home/user/RoboCOIN-with-Depth/data/refine/ai2_depth_preview/AI2_Alphabot_2_arrange_teaset_0_qced_hardlink_ep000000/videos/chunk-000/observation.images.cam_front_chest_depth" \
  --rgb-video "/home/user/RoboCOIN-with-Depth/data/refine/ai2_depth_preview/AI2_Alphabot_2_arrange_teaset_0_qced_hardlink_ep000000/videos/chunk-000/observation.images.cam_front_chest_rgb/episode_000000.mp4" \
  --pcd-stride 2 \
  --pcd-max-depth 5000
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# ==================== 路径与基础依赖 ====================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _pcd_dataset_output_root(depth_path: Path, pcd_root: Path) -> Path:
    """Return ``pcd/<dataset-name>/<task-name>`` for a depth video path."""
    parts = depth_path.parts
    try:
        # Dataset roots follow ``<dataset>_depth_preview/<task>`` (for
        # example ``ai2_depth_preview`` or ``leju_depth_preview``).
        dataset_idx = next(i for i, part in enumerate(parts)
                           if part.endswith("_depth_preview"))
        dataset_root = parts[dataset_idx]
        task_name = parts[dataset_idx + 1]
    except (StopIteration, IndexError) as exc:
        raise ValueError(
            f"无法从深度路径解析数据集/任务名（需要包含 <dataset>_depth_preview/<task>）: {depth_path}"
        ) from exc
    return Path(pcd_root) / dataset_root / task_name


def _generate_pcd_videos(depth_paths, pcd_root, rgb_path=None, stride=2,
                         max_depth_mm=5000, depth_scale=1.0):
    """Convert one or more depth videos to per-frame binary PCD files."""
    from tools.depth_to_pointcloud import read_depth_video, read_rgb_video, depth_to_pointcloud

    pcd_root = Path(pcd_root)
    for kind, depth_path in depth_paths.items():
        depth_path = Path(depth_path)
        if depth_path.is_dir():
            videos = sorted(depth_path.glob("*.mp4"))
            if len(videos) != 1:
                raise FileNotFoundError(f"深度目录应恰好包含一个 mp4: {depth_path}")
            depth_path = videos[0]
        if not depth_path.is_file():
            raise FileNotFoundError(f"深度视频不存在: {depth_path}")
        # LeRobot stores the matching RGB stream beside the depth stream.
        rgb = Path(rgb_path) if rgb_path else Path(str(depth_path).replace("_depth_nested_sparse", "_rgb").replace("_depth_refined", "_rgb").replace("_depth", "_rgb"))
        if rgb.is_dir():
            videos = sorted(rgb.glob("*.mp4"))
            rgb = videos[0] if len(videos) == 1 else rgb
        if not rgb.is_file():
            raise FileNotFoundError(f"未找到对应 RGB 视频，请用 --rgb-video 指定: {rgb}")
        camera_key = depth_path.parent.name.replace("observation.images.", "")
        camera_key = re.sub(r"_depth(?:_nested_sparse|_refined)?$", "", camera_key)
        if not camera_key:
            raise ValueError(f"无法从深度路径解析相机 key: {depth_path}")
        print(f"[{kind}] decoding {depth_path}")
        depths = read_depth_video(str(depth_path))
        rgbs = read_rgb_video(str(rgb))
        frame_names = list(range(min(len(depths), len(rgbs))))
        if kind == "baseline":
            # Sparse baseline frames correspond to source-video indices listed
            # in statistics/**/episode_*/manifest.json.
            base = depth_path.parents[3] if len(depth_path.parents) > 3 else depth_path.parent
            camera_dir = depth_path.parent.name.replace("observation.images.", "").replace("_depth_nested_sparse", "")
            episode_name = depth_path.stem
            manifest = base / "statistics" / camera_dir / episode_name / "manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(f"baseline manifest 不存在: {manifest}")
            info = json.loads(manifest.read_text())
            frame_names = info.get("sample_frame_indices")
            if not frame_names or len(frame_names) < len(depths):
                raise ValueError(f"manifest 帧索引不足: {manifest}（深度帧数={len(depths)}）")
            frame_names = frame_names[:len(depths)]
            if max(frame_names, default=-1) >= len(rgbs):
                raise ValueError(f"RGB 视频帧数不足以匹配 manifest: {manifest}")
            rgbs = [rgbs[j] for j in frame_names]
            print(f"[{kind}] using source frame indices from {manifest}")
        n = min(len(depths), len(rgbs), len(frame_names))
        # Keep camera identity in the output path; otherwise head/left/right
        # runs overwrite or get misidentified as the same point cloud.
        out_dir = _pcd_dataset_output_root(depth_path, pcd_root) / kind / camera_key
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            pts, cols = depth_to_pointcloud(depths[i], rgbs[i], stride=stride,
                                            max_depth_mm=max_depth_mm,
                                            depth_scale=depth_scale)
            fname = out_dir / f"frame_{int(frame_names[i]):06d}.pcd"
            _write_pcd(fname, pts, cols.astype(np.float32) / 255.0 if cols is not None else None)
        print(f"[{kind}] wrote {n} frames to {out_dir}")


# ==================== 图像与深度数据转换 ====================
def _to_hwc(rgb):
    if hasattr(rgb, "numpy"):
        rgb = rgb.numpy()
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    return np.ascontiguousarray(rgb.astype(np.uint8))


def _depth_to_color(depth):
    import cv2
    d = np.asarray(depth, dtype=np.float32)
    valid = (d > 0) & np.isfinite(d)
    if valid.any():
        vmax = np.percentile(d[valid], 98)
        d = np.clip(d, 0, vmax) / max(vmax, 1)
    d = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)


# ==================== 点云渲染器 ====================
def _make_pcd_renderer(H=480, W=640):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import io, cv2

    dpi = 100
    fig = Figure(figsize=(W / dpi, H / dpi), facecolor="#f0f0f0", dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(projection="3d", facecolor="#f0f0f0")
    ax.set_axis_off(); ax.grid(False)
    ax.dist = 2
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    _scat = None

    def render(pts, azim=45, elev=25):
        nonlocal _scat
        if hasattr(pts, "numpy"):
            pts = pts.numpy()
        pts = np.asarray(pts, dtype=np.float32)

        has_rgb = pts.shape[1] >= 6
        if has_rgb:
            xyz = pts[:, :3]; colors = np.clip(pts[:, 3:6], 0, 1)
        else:
            xyz = pts[:, :3]; colors = None

        valid_mask = (xyz[:, 2] > 0) & np.isfinite(xyz[:, 2])
        if valid_mask.sum() < 10:
            return np.full((H, W, 3), 240, dtype=np.uint8)

        valid = xyz[valid_mask]
        valid_colors = colors[valid_mask] if colors is not None else None
        if len(valid) > 200000:
            idx = np.random.RandomState(0).choice(len(valid), 200000, replace=False)
            valid = valid[idx]
            valid_colors = valid_colors[idx] if valid_colors is not None else None

        centroid = valid.mean(axis=0); valid = valid - centroid
        half = max(abs(valid[:, 0]).max(), abs(valid[:, 1]).max(),
                   abs(valid[:, 2]).max()) * 1.05

        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(-half, half)
        ax.view_init(elev=elev, azim=azim)
        if _scat is not None:
            _scat.remove()
        _scat = ax.scatter(valid[:, 0], valid[:, 1], valid[:, 2],
                           s=1.0, marker=".", alpha=0.8,
                           c=valid_colors, depthshade=False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="#f0f0f0",
                    bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H))
        else:
            img = np.full((H, W, 3), 240, dtype=np.uint8)
        return img
    return render


# ==================== 点云文件保存 ====================
def _save_pcd_all(sample, cameras, state, out_dir):
    """保存所有相机的点云为 PCD 文件。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for cam in cameras:
        pcd_key = f"observation.pointcloud.{cam}_pcd"
        if pcd_key not in sample:
            continue
        pts = sample[pcd_key]
        if hasattr(pts, "numpy"):
            pts = pts.numpy()
        pts = np.asarray(pts, dtype=np.float32)
        valid = (pts[:, 2] > 0) & np.isfinite(pts[:, 2])
        pts = pts[valid]

        fname = out / f"ep{state['ep']:04d}_frame{state['fi']:06d}_{cam}.pcd"
        _write_pcd(fname, pts[:, :3],
                   pts[:, 3:6] if pts.shape[1] >= 6 else None)
        print(f"  -> {fname}")


def _write_pcd(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None):
    """写入 PCD 文件（binary format, xyz + optional rgb）。"""
    n = len(xyz)
    has_rgb = rgb is not None and rgb.shape[0] == n
    fields = "x y z" + (" rgb" if has_rgb else "")
    sizes = "4 4 4" + (" 4" if has_rgb else "")
    types = "F F F" + (" F" if has_rgb else "")
    counts = "1 1 1" + (" 1" if has_rgb else "")

    header = (
        f"# .PCD v0.7 - Point Cloud Data file format\n"
        f"VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA binary\n"
    )

    if has_rgb:
        rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint32)
        rgb_packed = ((rgb_u8[:, 0].astype(np.uint32) << 16) |
                      (rgb_u8[:, 1].astype(np.uint32) << 8) |
                      rgb_u8[:, 2].astype(np.uint32))
        data = np.column_stack([xyz, rgb_packed.view(np.float32).reshape(-1, 1)])
    else:
        data = xyz

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.astype(np.float32).tobytes())


def _cam_from_key(key: str) -> str:
    return key.rsplit(".", 1)[-1].replace("_rgb", "")


def _camera_from_depth_path(path: Path) -> str:
    match = re.fullmatch(r"observation\.images\.(.+)_depth(?:_refined)?", path.parent.name)
    if match is None:
        raise ValueError(f"无法从深度视频目录解析相机名: {path.parent.name}")
    return match.group(1)


def _decode_external_depth(path: Path, frame_index: int, fps: float, backend: str):
    from src.video_utils import decode_video_frames

    depth = decode_video_frames(path, [frame_index / fps], 1.0 / fps, backend).squeeze(0)
    if depth.ndim == 3:
        depth = depth[0]
    return depth


# ==================== 主程序：参数、数据集与窗口初始化 ====================
def main():
    # ---------- 命令行参数 ----------
    parser = argparse.ArgumentParser(description="LeRobot 数据集可视化")
    parser.add_argument("--repo-id", default=None,
                        help="LeRobot 数据集 ID（交互可视化模式必填）")
    parser.add_argument("--root", default="data/refine")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--camera", default="all")
    parser.add_argument("--native", action="store_true",
                        help="使用原生 LeRobotDataset（无 PCD）")
    parser.add_argument("--save-pcd", default=None,
                        help="PCD 保存目录（按 's' 键保存当前帧）")
    parser.add_argument("--generate-pcd", action="store_true",
                        help="批量将三套外部深度视频转换为逐帧 PCD 后退出")
    parser.add_argument("--pcd-root", default="pcd",
                        help="批量 PCD 输出根目录，默认 pcd")
    parser.add_argument("--original-depth", type=Path)
    parser.add_argument("--baseline-depth", type=Path)
    parser.add_argument("--refine-depth", type=Path)
    parser.add_argument("--rgb-video", type=Path,
                        help="三套深度共用的 RGB 视频；省略时从深度路径自动推断")
    parser.add_argument("--pcd-stride", type=int, default=2)
    parser.add_argument("--pcd-max-depth", type=float, default=5000)
    parser.add_argument("--pcd-depth-scale", type=float, default=1.0)
    parser.add_argument("--compare-depth-before", type=Path,
                        help="比较模式：refine 前的深度视频")
    parser.add_argument("--compare-depth-after", type=Path,
                        help="比较模式：refine 后的深度视频")
    args = parser.parse_args()

    if args.generate_pcd:
        paths = {k: p for k, p in (("original", args.original_depth),
                                   ("baseline", args.baseline_depth),
                                   ("refine", args.refine_depth)) if p is not None}
        if len(paths) != 3:
            parser.error("--generate-pcd 需要同时提供 --original-depth、--baseline-depth、--refine-depth")
        _generate_pcd_videos(paths, args.pcd_root, args.rgb_video,
                             args.pcd_stride, args.pcd_max_depth,
                             args.pcd_depth_scale)
        return

    if not args.repo_id:
        parser.error("交互可视化模式需要提供 --repo-id；仅批量生成 PCD 时使用 --generate-pcd")

    if bool(args.compare_depth_before) != bool(args.compare_depth_after):
        parser.error("--compare-depth-before 和 --compare-depth-after 必须同时提供")

    # ---------- 加载数据集 ----------
    if args.native:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(args.repo_id, root=args.root)
        has_pcd = False
        print("Using native LeRobotDataset (no PCD)")
    else:
        from src.lerobot_with_depth_dataset import LeRobotWithDepthDataset as LeRobotDataset
        ds = LeRobotDataset(args.repo_id, root=args.root, enable_pcd=True, pcd_with_rgb=True)
        has_pcd = True

    print(f"Episodes: {ds.num_episodes}  Frames: {ds.num_frames}")
    print("Keys: ← → ±1 frame, Shift ±10, Ctrl ±100, ↑ ↓ episode")
    if has_pcd:
        print("PCD: w/s elev, a/d azim, p save PCD")

    if args.compare_depth_before:
        before_cam = _camera_from_depth_path(args.compare_depth_before)
        after_cam = _camera_from_depth_path(args.compare_depth_after)
        if before_cam != after_cam:
            parser.error(
                "before/after 必须来自同一相机，当前分别为 "
                f"{before_cam!r} 和 {after_cam!r}"
            )
        for path in (args.compare_depth_before, args.compare_depth_after):
            if not path.is_file():
                parser.error(f"深度视频不存在: {path}")
        args.camera = before_cam

    # ---------- 解析相机与初始帧 ----------
    rgb_keys = sorted(k for k in ds.features if "rgb" in k and "images" in k)
    cameras = [_cam_from_key(k) for k in rgb_keys]
    if args.camera != "all":
        cameras = [c for c in cameras if c == args.camera]

    ep_from = int(ds.episode_data_index["from"][args.episode])
    ep_to = int(ds.episode_data_index["to"][args.episode])
    state = {"ep": args.episode, "fi": min(ep_from + args.frame, ep_to - 1)}

    # ---------- 创建 RGB、深度和点云显示窗口 ----------
    import matplotlib.pyplot as plt

    compare_mode = args.compare_depth_before is not None
    n_rows = 2 if compare_mode else len(cameras)
    n_cols = 2 if compare_mode else (3 if has_pcd else 2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    imgs = {}
    for ri, cam in enumerate(cameras if not compare_mode else [cameras[0], cameras[0]]):
        if compare_mode:
            label = "Before" if ri == 0 else "After (refined)"
            imgs[(ri, 0)] = axes[ri][0].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            axes[ri][0].set_title(f"{label} Depth", fontsize=10)
            axes[ri][0].set_axis_off()
            imgs[(ri, 1)] = axes[ri][1].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            axes[ri][1].set_title(f"{label} PCD", fontsize=10)
            axes[ri][1].set_axis_off()
            continue
        imgs[(ri, 0)] = axes[ri][0].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        axes[ri][0].set_title(f"{cam} RGB", fontsize=10)
        axes[ri][0].set_axis_off()
        imgs[(ri, 1)] = axes[ri][1].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        axes[ri][1].set_title(f"{cam} Depth", fontsize=10)
        axes[ri][1].set_axis_off()
        if has_pcd:
            imgs[(ri, 2)] = axes[ri][2].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            axes[ri][2].set_title(f"{cam} PCD", fontsize=10)
            axes[ri][2].set_axis_off()

    title = plt.suptitle("", fontsize=14)
    render_pcd = _make_pcd_renderer() if has_pcd else None
    pcd_view = {"azim": 45, "elev": 25}
    drag = {"active": False, "x": 0.0, "y": 0.0}

    # ---------- 界面刷新与键盘交互 ----------
    def _redraw():
        ep = state["ep"]
        ep0 = int(ds.episode_data_index["from"][ep])
        ep1 = int(ds.episode_data_index["to"][ep])
        fi = state["fi"]
        sample = ds[fi]
        state["_sample"] = sample

        if compare_mode:
            local_fi = fi - ep0
            cam = cameras[0]
            rgb = sample[f"observation.images.{cam}_rgb"]
            cfg = ds._pcd_cameras.get(cam.replace("cam_", ""), {})
            K = cfg.get("K", np.array([[605, 0, 323], [0, 605, 252], [0, 0, 1]], dtype=np.float32))
            for ri, path in enumerate((args.compare_depth_before, args.compare_depth_after)):
                depth = _decode_external_depth(path, local_fi, ds.fps, ds.video_backend)
                pts = ds._depth_to_pointcloud(depth, K, rgb=rgb, stride=ds._pcd_stride)
                imgs[(ri, 0)].set_data(_depth_to_color(depth))
                imgs[(ri, 1)].set_data(render_pcd(
                    pts, azim=pcd_view["azim"], elev=pcd_view["elev"]
                ))
            title.set_text(
                f"{cam} Before vs Refined | Episode {ep} | Frame {local_fi}/{ep1 - ep0 - 1}"
            )
            fig.canvas.draw_idle()
            return

        for ri, cam in enumerate(cameras):
            rgb_key = f"observation.images.{cam}_rgb"
            depth_key = f"observation.images.{cam}_depth"
            pcd_key = f"observation.pointcloud.{cam}_pcd"

            if rgb_key in sample:
                imgs[(ri, 0)].set_data(_to_hwc(sample[rgb_key]))
            if depth_key in sample:
                depth = sample[depth_key]
                if hasattr(depth, "numpy"):
                    depth = depth.numpy()
                if depth.ndim == 3 and depth.shape[0] in (1, 3, 4):
                    depth = depth[0]
                imgs[(ri, 1)].set_data(_depth_to_color(depth))
            if has_pcd and pcd_key in sample:
                imgs[(ri, 2)].set_data(render_pcd(sample[pcd_key],
                    azim=pcd_view["azim"], elev=pcd_view["elev"]))

        title.set_text(f"Episode {ep}  Frame {fi - ep0}/{ep1 - ep0 - 1}  (global {fi})")
        fig.canvas.draw_idle()

    def _on_key(event):
        ep = state["ep"]
        ep0 = int(ds.episode_data_index["from"][ep])
        ep1 = int(ds.episode_data_index["to"][ep])

        key = (event.key or "").lower()
        if "ctrl" in key:       step = 100
        elif "shift" in key:    step = 10
        else:                   step = 1

        if "right" in key:
            state["fi"] = min(state["fi"] + step, ep1 - 1)
        elif "left" in key:
            state["fi"] = max(state["fi"] - step, ep0)
        elif key == "up":
            if ep < ds.num_episodes - 1:
                state["ep"] += 1
                state["fi"] = int(ds.episode_data_index["from"][state["ep"]])
        elif event.key == "down":
            if ep > 0:
                state["ep"] -= 1
                state["fi"] = int(ds.episode_data_index["from"][state["ep"]])
        elif has_pcd and args.save_pcd and key == "p":
            _save_pcd_all(state["_sample"], cameras, state, args.save_pcd)
            print(f"  [saved] ep{state['ep']} frame{state['fi']}")
        elif has_pcd and key == "a":
            pcd_view["azim"] = (pcd_view["azim"] + 30) % 360
        elif has_pcd and key == "d":
            pcd_view["azim"] = (pcd_view["azim"] - 30) % 360
        elif has_pcd and key == "w":
            pcd_view["elev"] += 30
        elif has_pcd and key == "s":
            pcd_view["elev"] -= 30
        elif event.key in ("q", "escape"):
            plt.close(fig)
            return
        _redraw()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    def _on_press(event):
        pcd_col = 1 if compare_mode else 2
        if has_pcd and event.inaxes in [axes[ri][pcd_col] for ri in range(n_rows)] and event.button == 1:
            drag.update(active=True, x=event.x, y=event.y)

    def _on_motion(event):
        if not (has_pcd and drag["active"] and event.x is not None and event.y is not None):
            return
        dx, dy = event.x - drag["x"], event.y - drag["y"]
        drag.update(x=event.x, y=event.y)
        pcd_view["azim"] = (pcd_view["azim"] + dx * 0.5) % 360
        pcd_view["elev"] = float(np.clip(pcd_view["elev"] + dy * 0.5, -85, 85))
        _redraw()

    def _on_release(event):
        drag["active"] = False

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    _redraw()
    plt.show()


if __name__ == "__main__":
    main()
