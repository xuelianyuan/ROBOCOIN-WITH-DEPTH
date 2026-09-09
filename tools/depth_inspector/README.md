# Depth Inspector

直接运行本地检查网页（需要 `ffmpeg`、`numpy`、`Pillow`）：

```bash
python tools/depth_inspector/server.py \
  --dataset data/lerobot/ai2_depth_preview/AI2_Alphabot_2_arrange_teaset_0_qced_hardlink_ep000000 \
  --port 8765

# 浏览器打开 http://127.0.0.1:8765/outputs/depth_inspector.html
```

也可以用 `manifest.example.json` 指定任意 RGB、gray12le 深度视频和 PCD 模板：

```bash
python tools/depth_inspector/server.py --manifest tools/depth_inspector/manifest.example.json
```

启动时会同步生成报告入口 `outputs/depth_inspector.html`；交互 API 仍由本地服务提供，因此请保持服务运行后从 `http://127.0.0.1:8765/outputs/depth_inspector.html` 打开。

功能：三视角×Metric/Nested/Refine 表格、共享时间轴、Depth/颜色直方图/PCD 同时展示、全局固定点云帧选择（抽取帧提示、任意帧输入并自动匹配最近的已有 PCD）、每组独立偏移、PCD 旋转/缩放、全视频逐帧 min/max、NaN/Inf/无效值过滤、同一视角三种方法统一 0–255 归一化、均值曲线、自动播放并在三遍循环后暂停。
