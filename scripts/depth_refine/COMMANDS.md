# 命令速查

先执行公共配置，再按需复制对应命令。完整参数和限制见 [README.md](README.md)。

## 公共配置
Bash/Shell 脚本里的环境变量和路径变量定义。

在仓库根目录运行，将数据集、相机、chunk 和 episode 改为实际值：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
DATASET="data/lerobot/ai2_depth_preview/AI2_Alphabot_2_arrange_teaset_0_qced_hardlink_ep000000"
RESULT_ROOT="$DATASET/depth_results"
CAMERA="cam_front_head"
CHUNK="chunk-000"
EPISODE="episode_000000"

RGB="$DATASET/videos/$CHUNK/observation.images.${CAMERA}_rgb/$EPISODE.mp4"
BEFORE="$DATASET/videos/$CHUNK/observation.images.${CAMERA}_depth/$EPISODE.mp4"
AFTER="$RESULT_ROOT/refine/videos/$CHUNK/observation.images.${CAMERA}_depth_refined/$EPISODE.mp4"
SPARSE="$RESULT_ROOT/baseline/videos/$CHUNK/observation.images.${CAMERA}_depth_nested_sparse/$EPISODE.mp4"
NESTED_DIR="$RESULT_ROOT/baseline/statistics/$CAMERA/$EPISODE"
```

## 1. build_baseline.py：生成参考

```bash
python scripts/depth_refine/build_baseline.py \
  --dataset-dir "$DATASET" --result-root "$RESULT_ROOT" \
  --camera "$CAMERA" --sample-count 20 --save-video
```

已有统计、需要重新生成或补存视频时，追加 `--force`。

## 2. run_refine.py：验证 / 生成完整结果

```bash
# 验证并绘制趋势图
python scripts/depth_refine/run_refine.py \
  --dataset-dir "$DATASET" --result-root "$RESULT_ROOT" \
  --camera "$CAMERA" --mapping quantile --stage validate --plot-trend

# 逐帧迁移，生成 AFTER 对应的视频
python scripts/depth_refine/run_refine.py \
  --dataset-dir "$DATASET" --result-root "$RESULT_ROOT" \
  --camera "$CAMERA" --mapping quantile --stage full

# 可选：RIFE 补帧，单独保存以便与 full 比较
python scripts/depth_refine/run_refine.py \
  --dataset-dir "$DATASET" --result-root "$DATASET/depth_results_rife" \
  --statistics-dir "$RESULT_ROOT/baseline/statistics" \
  --camera "$CAMERA" --mapping quantile --stage rife
```

full/rife 默认仅在 MAE、RMSE、Wasserstein 全部改善时生成视频。RIFE 需额外模型依赖；后续命令中的 `AFTER` 仍指向 full 结果。

## 3. run_batch.py：批量处理

```bash
# 先预览项目及相机
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras head left_wrist right_wrist chest \
  --stage all --dry-run

# 批量生成参考并验证
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras head left_wrist right_wrist chest \
  --stage all --sample-count 20 --save-video \
  --refine-stage validate --plot-mean-curve

# 基于已有参考生成完整结果
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras head left_wrist right_wrist chest \
  --stage refine --refine-stage full

# 仅根据已有报告绘图
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras head left_wrist right_wrist chest --stage plot
```

仅处理当前项目时，将 `--root data/lerobot` 换成 `--root "$DATASET"`。

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
