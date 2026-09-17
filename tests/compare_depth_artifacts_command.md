# 深度斑块比较命令速查

对应脚本：`tests/compare_depth_artifacts.py`。原理与限制见 [compare_depth_artifacts.md](compare_depth_artifacts.md)。这是手动诊断工具，不运行模型，也不属于 pytest 测试。

以下命令在**仓库根目录、同一个 Bash 终端**执行。先设置公共变量，再按需选择命令。

## 1. 公共配置

```bash
DEPTH_VIDEO="data/lerobot/ai2_depth_preview/AI2_Alphabot_2_grind_ink_0_qced_hardlink_ep000000/videos/chunk-000/observation.images.cam_front_head_depth/episode_000000.mp4"
FRAME=278
DEPTH_MIN=287
DEPTH_MAX=1475
OUTPUT_ROOT="outputs/depth_artifacts/head_${FRAME}"
```

帧号从 0 开始。更换数据或视角时，同步修改视频路径、帧号、输出目录和 Inspector 页面显示的归一化范围。修改 `FRAME` 后需重新设置 `OUTPUT_ROOT`。

依赖检查与参数帮助：

```bash
python -c 'import numpy; from PIL import Image; print("numpy / Pillow OK")'
ffmpeg -version
ffprobe -version
python tests/compare_depth_artifacts.py --help
```

## 2. 导出原始深度视频帧

```bash
python tests/compare_depth_artifacts.py \
  --depth-video "$DEPTH_VIDEO" \
  --frame "$FRAME" --range "$DEPTH_MIN" "$DEPTH_MAX" \
  --output "$OUTPUT_ROOT/decoded"
```

省略 `--range` 会使用当前解码帧有效值的 min/max，此时亮度不能直接与网页比较。

## 3. 裁剪局部区域

```bash
python tests/compare_depth_artifacts.py \
  --depth-video "$DEPTH_VIDEO" \
  --frame "$FRAME" --range "$DEPTH_MIN" "$DEPTH_MAX" \
  --crop 20 20 100 80 \
  --output "$OUTPUT_ROOT/crop"
```

`--crop X Y W H` 指定左上角和宽高；按实际图像调整。裁剪不改变归一化范围。

## 4. 对比编码前模型输出

先替换 NPY 路径并设置数值转换参数：

```bash
RAW_NPY="/path/to/model_depth.npy"
RAW_SCALE=1
RAW_OFFSET=0

python tests/compare_depth_artifacts.py \
  --depth-video "$DEPTH_VIDEO" \
  --frame "$FRAME" --range "$DEPTH_MIN" "$DEPTH_MAX" \
  --raw-npy "$RAW_NPY" \
  --raw-scale "$RAW_SCALE" --raw-offset "$RAW_OFFSET" \
  --output "$OUTPUT_ROOT/raw_vs_decoded"
```

- NPY 支持单帧 `H×W` 或多帧 `T×H×W`；多帧默认取 `FRAME`，索引不同时追加 `--raw-frame 实际索引`。
- 必须使用同一帧、同一分辨率、同一预测流程的编码前输出，不能用 Nested 输出代替 Metric 编码前输出。
- 转换公式为 `raw * RAW_SCALE + RAW_OFFSET`。原始单位为米、视频直接存毫米时设置 `RAW_SCALE=1000`。
- 若编码使用 min/max 归一化，使用编码时保存的真实参数：`S=4095/(max-min)`、`O=-min*S`，不要按当前帧重新拟合。

## 5. 对比 Inspector 网页预览

确认本地 Inspector 服务已打开目标数据集，并完成对应预览生成，然后保存 H.264 预览：

```bash
PREVIEW_VIDEO="/tmp/head.metric.preview.mp4"
curl --fail "http://127.0.0.1:8765/media/depth/head.metric" \
  -o "$PREVIEW_VIDEO"
```

上述地址对应 HEAD Metric；更换视角或服务端口时调整地址和保存路径。预览文件必须来自 Inspector，不能使用原始深度 MP4 代替。

```bash
python tests/compare_depth_artifacts.py \
  --depth-video "$DEPTH_VIDEO" \
  --frame "$FRAME" --range "$DEPTH_MIN" "$DEPTH_MAX" \
  --preview-video "$PREVIEW_VIDEO" \
  --output "$OUTPUT_ROOT/preview_vs_decoded"
```

必须提供 Inspector 使用的精确 `--range`。稀疏预览需追加 `--preview-frame 实际索引`，脚本不会自动推断采样映射；完整预览默认使用 `FRAME`。

## 6. 完整链路对比：模型输出 → 深度视频 → 网页预览

先完成第 4 节的 NPY 配置和第 5 节的预览下载，再执行：

```bash
python tests/compare_depth_artifacts.py \
  --depth-video "$DEPTH_VIDEO" \
  --frame "$FRAME" --range "$DEPTH_MIN" "$DEPTH_MAX" \
  --raw-npy "$RAW_NPY" \
  --raw-scale "$RAW_SCALE" --raw-offset "$RAW_OFFSET" \
  --preview-video "$PREVIEW_VIDEO" \
  --output "$OUTPUT_ROOT/full"
```

可按需追加 `--crop X Y W H`、`--raw-frame 实际索引`、`--preview-frame 实际索引`。示例中的不同任务分别保存到子目录；重复执行同一命令时，建议更换输出目录以保留此前结果。

## 7. 查看结果

以完整链路对比为例：

```bash
python -m json.tool "$OUTPUT_ROOT/full/report.json"
```

| 文件 | 内容 |
| --- | --- |
| `comparison.png` | 并排对比图 |
| `A_raw.npy/png`、`A_quantized.npy/png` | 编码前浮点与模拟量化结果，仅提供 NPY 时生成 |
| `B_decoded.npy/png` | 原始深度视频解码结果 |
| `*_error.npy`、`*_abs_error.png` | 带符号误差数组与绝对误差图 |
| `report.json` | 参数、MAE、RMSE、P99、最大绝对误差、偏差等 |
| `profile.csv` | 裁剪区域中间一行的深度曲线 |

误差图各自按最大绝对误差放大，不能直接比较亮度。深度误差单位取决于编码数值域；预览误差在 0–255 显示值域计算，不包含浏览器 GPU 渲染。

脚本按 `clip(0,4095)` 后截断到整数模拟量化，深度误差仅统计原始值和解码值均处于 `(0,4095)` 的像素。实际编码流程或 Inspector 无效值阈值不同时，需要调整脚本假设。

没有编码前 NPY 时，不能区分模型与原始深度编码造成的斑块；具体判断方法见原说明文档。
