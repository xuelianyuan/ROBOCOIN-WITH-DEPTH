# 深度斑块来源比较

运行脚本 `compare_depth_artifacts.py`，依赖 Python、numpy、Pillow，以及 PATH 中的 ffmpeg/ffprobe。它是手动诊断工具，不会运行模型，也不属于 pytest 测试。

## 1. 先导出原始深度视频帧

仓库根目录执行：

```bash
python tests/compare_depth_artifacts.py \
  --depth-video data/lerobot/ai2_depth_preview/AI2_Alphabot_2_grind_ink_0_qced_hardlink_ep000000/videos/chunk-000/observation.images.cam_front_head_depth/episode_000000.mp4 \
  --frame 278 --range 287 1475 \
  --output outputs/depth_artifacts/head_278
```

帧号从 0 开始；这里的显示范围取自问题截图 HEAD 的统一范围。换数据时请换成页面显示的范围。省略 `--range` 时使用当前解码帧的有效值 min/max，不能再把亮度直接与网页比较。

可加 `--crop 20 20 100 80`，按 X、Y、宽、高选取墙面等区域。裁剪不改变归一化范围。读取视频按帧序号从头解码，避免时间定位落到不同帧。

## 2. 加入编码前模型输出

在上述命令加：

```bash
--raw-npy /path/to/model_depth.npy
```

支持单帧 H×W 或视频 T×H×W 数组。后者默认取 `--frame`，可通过 `--raw-frame` 指定数组索引。必须是同一帧、同一分辨率、同一预测流程的编码前输出，不能用另一模型的 Nested 结果代替 Metric 的编码前输出。

原始数组须与视频里的数值域一致。可用 `--raw-scale S --raw-offset O` 转换为 `raw*S+O`。例如原始数据为米、视频直接存毫米，使用 `--raw-scale 1000`。若编码用了 min/max 归一化，则 S=4095/(max-min)，O=-min*S；请用当时保存的真实参数，不要根据当前帧重新拟合。

脚本按当前 DepthVideoEncoder 的 `clip(0,4095)` 后截断到整数模拟编码前量化。若实际编码流程不同，需调整这一假设。深度误差统计使用原始值和解码值均处于 (0,4095) 的像素；原始裁剪/无效像素数量另存报告，避免把饱和值混入误差。

## 3. 加入网页预览

再加 `--preview-video /path/to/head.metric.mp4 --range MIN MAX`。必须提供 Inspector 使用的精确归一化范围；如使用了自定义无效值阈值，本脚本默认 (0,4095) 也需要相应调整。

这是 Inspector 临时目录中的 H.264 文件，不是原始深度 MP4。可从运行中的本地服务保存：

```bash
curl --fail http://127.0.0.1:8765/media/depth/head.metric -o /tmp/head.metric.preview.mp4
```

确保该服务正打开目标数据集，且预览生成完成。稀疏预览可通过 `--preview-frame` 指定对应索引；脚本不自动推断采样映射。预览误差在 0–255 显示值域内计算，包含视频压缩和颜色转换，不包含浏览器 GPU 渲染。

## 输出与判断

- `comparison.png`：固定深度灰度范围的并排图；误差图单独放大到各自最大绝对误差，白色对应值见报告，不能跨误差图直接比较亮度。
- `A_raw.npy/png`、`A_quantized.npy/png`、`B_decoded.npy/png`：裁剪后的原始浮点、量化和解码数据/图像（A 仅在提供 NPY 时生成）。
- `*_error.npy`：带符号误差，无效位置为 NaN；`*_abs_error.png` 为绝对误差可视化。
- `report.json`：MAE、RMSE、P99、最大绝对误差、偏差及参数。深度误差单位为编码数值域，不一定是毫米。
- `profile.csv`：裁剪中间一行的深度曲线数值，行坐标见报告。

A 已有斑块，说明编码前已存在，需继续区分模型与编码前处理；A 平滑但量化后有斑块，说明量化有影响；量化图平滑但 B 有斑块，说明编码有影响；只有 C 明显，说明预览链路有影响。多阶段也可能共同加重斑块。没有编码前 NPY 时不能区分模型与原始深度编码，脚本不会自动给出归因结论。
