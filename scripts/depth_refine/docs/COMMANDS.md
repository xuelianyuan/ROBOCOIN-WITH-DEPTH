# 命令速查

先执行公共配置，再按需复制对应命令。完整参数和限制见 [README.md](README.md)。

## 公共配置
以下配置使用 **Bash**，请在同一个终端执行配置、切换和后续命令。

在仓库根目录运行，将数据集、相机、chunk 和 episode 改为实际值：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
DATASET="data/lerobot/ai2_depth_preview/AI2_Alphabot_2_heat_test_tube_0_qced_hardlink_ep000000"
RESULT_ROOT="$DATASET/depth_results"
CHUNK="chunk-000"
EPISODE="episode_000000"

# 切换视角时，同时刷新所有依赖相机的路径
use_camera() {
  case "${1:-}" in
    head|cam_front_head) CAMERA="cam_front_head"; CAMERA_KEY="head" ;;
    left_wrist|cam_left_wrist) CAMERA="cam_left_wrist"; CAMERA_KEY="left_wrist" ;;
    right_wrist|cam_right_wrist) CAMERA="cam_right_wrist"; CAMERA_KEY="right_wrist" ;;
    *) printf '用法：use_camera head|left_wrist|right_wrist\n' >&2; return 1 ;;
  esac
  RGB="$DATASET/videos/$CHUNK/observation.images.${CAMERA}_rgb/$EPISODE.mp4"
  BEFORE="$DATASET/videos/$CHUNK/observation.images.${CAMERA}_depth/$EPISODE.mp4"
  AFTER="$RESULT_ROOT/refine/videos/$CHUNK/observation.images.${CAMERA}_depth_refined/$EPISODE.mp4"
  SPARSE="$RESULT_ROOT/baseline/videos/$CHUNK/observation.images.${CAMERA}_depth_nested_sparse/$EPISODE.mp4"
  NESTED_DIR="$RESULT_ROOT/baseline/statistics/$CAMERA/$EPISODE"
  printf '当前视角：%s\nRGB：%s\n原始深度：%s\n修正深度：%s\n' "$CAMERA" "$RGB" "$BEFORE" "$AFTER"
}

use_camera right_wrist  # 默认头部视角
```

### 快速切换三个相机视角

配置完成后，每次只需执行下面**一行**，再运行后文需要的命令：

```bash
use_camera head         # 头部：cam_front_head
use_camera left_wrist   # 左腕：cam_left_wrist
use_camera right_wrist  # 右腕：cam_right_wrist
```

三行是备选项，全部执行会停在右腕视角。不要只修改 `CAMERA`：Bash 已赋值的路径不会随它自动更新。修改 `DATASET`（同时修改 `RESULT_ROOT`）、`CHUNK` 或 `EPISODE` 后，也需重新执行 `use_camera`。如果实际目录中的相机名称不同，请修改函数中的完整相机名。

切换后，生成参考、refine、绘图、浏览及 PCD 导出均使用当前视角；已打开的浏览窗口需退出后重新运行命令。`CHUNK` / `EPISODE` 只指定绘图和检查的输入，baseline/refine 仍处理当前数据集内该相机的所有 episode。

## 1. build_baseline.py：生成参考

```bash
python scripts/depth_refine/build_baseline.py \
  --dataset-dir "$DATASET" --result-root "$RESULT_ROOT" \
  --camera "$CAMERA" --sample-stride 5 --save-video
```

默认按源帧 `0, 5, 10, ...` 采样，不额外补尾帧。已有统计、切换采样配置或补存视频时，追加 `--force`。需要旧的均匀定量采样时，将 `--sample-stride 5` 换成 `--sample-count 60`，两者不能同时指定。

## 2. run_refine.py：验证 / 生成完整结果

```bash
# 验证并绘制趋势图
python scripts/depth_refine/run_refine.py \
  --dataset-dir "$DATASET" --result-root "$RESULT_ROOT" \
  --camera "$CAMERA" --mapping quantile --stage validate --plot-trend

# 逐帧迁移，生成 AFTER 对应的视频
python scripts/depth_refine/run_refine.py \
  --dataset-dir "$DATASET" --result-root "$RESULT_ROOT" \
  --camera "$CAMERA" --mapping quantile --stage full \
  --report-mapping-slope --save-preencode-npy

# 可选：RIFE 补帧，单独保存以便与 full 比较
python scripts/depth_refine/run_refine.py \
  --dataset-dir "$DATASET" --result-root "$DATASET/depth_results_rife" \
  --statistics-dir "$RESULT_ROOT/baseline/statistics" \
  --camera "$CAMERA" --mapping quantile --stage rife
```

full/rife 默认仅在 MAE、RMSE、Wasserstein 全部改善时生成视频。RIFE 需额外模型依赖；后续命令中的 `AFTER` 仍指向 full 结果。

## 3. run_batch.py：批量处理

默认批量处理头部、左腕、右腕三个视角。先选择范围，再运行下面的命令：

```bash
BATCH_CAMERAS=(head left_wrist right_wrist)  # 三个视角
# BATCH_CAMERAS=("$CAMERA_KEY")             # 或仅处理 use_camera 当前选中的视角
```

`BATCH_CAMERAS` 独立于单视角切换；仅处理当前视角时，每次切换后需重新执行 `BATCH_CAMERAS=("$CAMERA_KEY")`。

```bash
# 先预览项目及相机
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras "${BATCH_CAMERAS[@]}" \
  --stage all --dry-run

# 批量生成参考并验证
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras "${BATCH_CAMERAS[@]}" \
  --stage all --sample-stride 5 --save-video \
  --refine-stage validate --plot-mean-curve

# 基于已有参考生成完整结果
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras "${BATCH_CAMERAS[@]}" \
  --stage refine --refine-stage full

# 仅根据已有报告绘图
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras "${BATCH_CAMERAS[@]}" --stage plot
```

仅处理当前项目时，将 `--root data/lerobot` 换成 `--root "$DATASET"`。切换采样配置后追加 `--force` 重新生成已有 baseline。

## 4. plot_comparison.py：三路对比图

```bash
# 单帧：源视频第 0 帧与对应的 Nested NPY
python scripts/depth_refine/plot_comparison.py \
  --before "$BEFORE" --after "$AFTER" \
  --nested "$NESTED_DIR/frame_000000000.npy" --frame 0 \
  --output "$RESULT_ROOT/refine/plots/$CAMERA/$EPISODE/comparison.png"

# 全部采样帧
python scripts/depth_refine/plot_comparison.py \
  --before "$BEFORE" --after "$AFTER" \
  --nested-dir "$NESTED_DIR" --all-sampled \
  --output "$RESULT_ROOT/refine/plots/$CAMERA/$EPISODE/comparison.png"
```

需要 Matplotlib，三组深度尺寸须一致；真实内参可通过 `--fx --fy --cx --cy` 指定。

## 5. inspect_dataset.py：浏览 / 对比 / 导出点云

```bash
# 浏览 RGB、深度和点云；按 p 保存当前帧点云
python scripts/depth_refine/inspect_dataset.py \
  --repo-id "${DATASET##*/}" --root "$DATASET" \
  --camera "$CAMERA" --episode 0 --frame 0 \
  --save-pcd "pcds/manual_preview"

# 交互对比原始与 full 结果；episode 索引须与外部视频对应
python scripts/depth_refine/inspect_dataset.py \
  --repo-id "${DATASET##*/}" --root "$DATASET" --episode 0 \
  --compare-depth-before "$BEFORE" --compare-depth-after "$AFTER"

# 导出三套逐帧 PCD，按 chunk/episode 隔离输出
python scripts/depth_refine/inspect_dataset.py \
  --generate-pcd --pcd-root "pcds/$CHUNK/$EPISODE" \
  --original-depth "$BEFORE" --baseline-depth "$SPARSE" \
  --refine-depth "$AFTER" --rgb-video "$RGB" \
  --pcd-stride 2 --pcd-max-depth 5000 --pcd-depth-scale 1.0
```

交互快捷键：`←/→` 切帧，`Shift` 加速 10 帧，`Ctrl` 加速 100 帧，`↑/↓` 切 episode，`w/s/a/d` 旋转点云，`p` 保存，`q` 退出。外部视频对比时留在对应 episode 内。

PCD 导出要求数据路径含 `<dataset>_depth_preview/<task>`，且 baseline 的稀疏视频与 `manifest.json` 均已生成。

## 6. 每条 episode 的 Nested / refine 耗时

新版 baseline/refine 自动记录每个 episode、每个视角的处理耗时（秒），无需额外开关。
沿用前面的公共配置，处理当前数据集的三个视角并生成完整 refine 视频：

```bash
for VIEW in cam_front_head cam_left_wrist cam_right_wrist; do
  python scripts/depth_refine/build_baseline.py \
    --dataset-dir "$DATASET" --result-root "$RESULT_ROOT" \
    --camera "$VIEW" --sample-stride 5 --save-video --force || break
  python scripts/depth_refine/run_refine.py \
    --dataset-dir "$DATASET" --result-root "$RESULT_ROOT" \
    --camera "$VIEW" --mapping quantile --stage full || break
done

# 三视角明细 + 每条 episode 的三视角合计
python scripts/depth_refine/summarize_timings.py --result-root "$RESULT_ROOT"

# 或只统计 use_camera 当前选中的视角
python scripts/depth_refine/summarize_timings.py \
  --result-root "$RESULT_ROOT" --cameras "$CAMERA"
```

- `timings/per_view.csv`：每条 episode × 单视角 × 阶段的秒数、状态和帧数。
- `timings/per_episode.csv`：每条 episode 按阶段汇总所选视角；缺失、跳过或失败时合计留空，并列出缺失视角。
- `timings/<stage>/<chunk>/<camera>/<episode>.json`：保留该阶段最近一次实际尝试。阶段分别为 `da3_nested`（或 `da3`）、`refine_validate`、`refine_full`、`refine_rife`，不会混算。

耗时为单视频处理的墙钟时间，包括读取/解码、计算、统计、结果写入及可选编码/绘图；不包括进程启动和 episode 外的初始化。模型按需加载，因此首次推理的模型加载开销计入首条 episode。三视角合计为各视角耗时之和，不代表并行处理时的实际等待时间。

旧结果没有计时数据，不能用文件时间戳准确还原。上述 `--force` 会重新生成 baseline 并覆盖其结果和计时；不加时已有 baseline 直接跳过、保留原计时。refine 每次重新处理并更新对应阶段计时。验证未通过而跳过视频的 full/rife 记录标为 `skipped`，不算完整 refine。汇总命令每次覆盖两份 CSV，原始 JSON 保留。
