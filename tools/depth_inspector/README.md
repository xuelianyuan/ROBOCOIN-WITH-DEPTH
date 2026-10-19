# Depth Inspector

在仓库根目录运行本地检查网页（需要 `ffmpeg`、`numpy`、`Pillow`）。传入
`data/original` 中的任务目录后，检查器会自动关联仓库中的原始深度、Nested
验证结果、refine 深度以及 PCD。

AI2 四视角验证：

```bash
cd /home/user/RoboCOIN-with-Depth
conda activate depth

python tools/depth_inspector/server.py \
  --dataset data/original/ai2_depth_preview/AI2_Alphabot_2_clink_glasses_0_qced_hardlink_ep000000 \
  --port 8765

# 浏览器打开 http://127.0.0.1:8765/outputs/depth_inspector.html
```

Leju 三视角验证（在另一个终端运行）：

```bash
cd /home/user/RoboCOIN-with-Depth
conda activate depth

python tools/depth_inspector/server.py \
  --dataset data/original/leju_depth_preview/Leju_Kuavo_4_box_storage_parcel_0_qced_hardlink_ep000000 \
  --port 8766

# 浏览器打开 http://127.0.0.1:8766/outputs/depth_inspector.html
```

也可以用 `manifest.example.json` 指定任意 RGB、gray12le 深度视频和 PCD 模板：

```bash
python tools/depth_inspector/server.py --manifest tools/depth_inspector/manifest.example.json
```

启动时会同步生成报告入口 `outputs/depth_inspector.html`；交互 API 仍由本地服务提供，因此请保持服务运行后从 `http://127.0.0.1:8765/outputs/depth_inspector.html` 打开。

功能：三视角×Metric/Nested/Refine 表格、共享时间轴、Depth/颜色直方图/PCD 同时展示、全局固定点云帧选择（抽取帧提示、任意帧输入并自动匹配最近的已有 PCD）、每组独立偏移、PCD 旋转/缩放、全视频逐帧 min/max、NaN/Inf/无效值过滤、同一视角三种方法统一 0–255 归一化、均值曲线、自动播放并在三遍循环后暂停。

播放说明：

- 启动时会在系统临时目录生成 H.264 深度预览，准备完成后按原始帧率连续播放；首次打开需等待统计和转码完成。需使用支持 `libx264` 的 FFmpeg。
- 深度预览沿用同视角统一归一化范围。直方图取自预览画面，受视频压缩影响；min/max 和均值曲线仍由原始深度计算。
- 稀疏结果（例如仅 20 帧的 Nested）按采样帧对齐显示，不能补出原视频的全部动作帧。
- RGB 作为播放时钟，拖动时间轴时同步定位各视频；深度独立帧偏移仍有效。
- 自动发现点云的目录为 `pcds/<dataset>/<task>/{original,baseline,refine}/cam_*/frame_*.pcd`。PCD 首次打开自动加载，右上角选择固定帧，并匹配最近的已有点云，不随视频逐帧切换。
- 修改代码后请重启 `server.py` 并强制刷新浏览器（Ctrl+Shift+R）。`Depth video not found` 表示对应方法/视角没有找到深度视频，需要先生成结果或在 manifest 中指定路径。

点云操作：按住鼠标左键拖拽旋转（支持触屏单指拖动），同一相机视角下 Metric/Nested/Refine 同步旋转，HEAD/LEFT WRIST/RIGHT WRIST 互不影响。滚轮同步缩放同视角的所有点云；双击重置同视角的旋转和缩放。切换点云帧或其他方法加载完成后，仍保持该视角的旋转角度和缩放倍率。
