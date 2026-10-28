# DA3Metric + 尺度保持 VDPP 使用说明

本方案先用 DA3Metric 从 RGB 视频得到米制深度，再把深度转换为“近处值大”的
inverse depth 送入 VDPP 做时序稳定，最后利用 DA3Metric 的可靠区域恢复米制尺度。
目标是在不重训 VDPP 的前提下完成 POC，并避免逐帧归一化导致的尺度闪烁。

## 1. 处理流程

输入视频的每一帧依次经过以下步骤：

1. 使用原视频分辨率对应的相机内参。DA3 官方预处理 resize 图像时同步 resize
   内参，程序用 resize 后的平均焦距执行
   `metric_depth = raw_depth × focal / 300`，输出单位为米。
2. 同一批 RGB 同时执行原图和水平翻转推理。程序排除天空、深度边缘、翻转预测
   不一致和双向光流不一致的像素，剩余像素作为尺度锚点。
3. 计算 `q = 1 / (D + ε)`。整段视频只计算一次精确 P2/P98，并用这一组固定值
   归一化所有帧。代码不做逐帧 min-max。
4. 按 32 帧、重叠 8 帧切分 normalized inverse depth，调用 VDPP 的 `forward`。
   不调用 VDPP 原始脚本中的逐段无约束尺度拼接。
5. 每个 chunk 在可靠锚点上使用 Huber IRLS 拟合
   `q_metric ≈ a × q_vdpp + b`，并约束 `a > 0`，防止近远关系翻转。
6. 相邻 chunk 在重叠帧上估计仿射关系，联合平滑各 chunk 的 `a、b`。重叠区在
   inverse depth 空间羽化融合，最后再取倒数恢复米制深度。

DA3METRIC-LARGE 没有可直接使用的像素置信度输出，因此这里的“可靠区域”是
flip 一致性、非天空、非深度边缘和光流可见性的组合代理，不等同于模型置信度。

## 2. 环境和模型准备

从仓库根目录执行：

```bash
git submodule update --init --recursive \
  third_party/Depth-Anything-3 third_party/VDPP

# 在用于推理的 Python 环境中安装。若环境已有匹配的 PyTorch/CUDA，
# 安装 VDPP requirements 前应先核对其中的 torch 版本，避免覆盖现有 CUDA 版本。
pip install -e third_party/Depth-Anything-3
pip install scipy opencv-python easydict einops timm

mkdir -p third_party/VDPP/checkpoints
wget https://github.com/injun-baek/VDPP/releases/download/v1.0/vdpp.pth \
  -O third_party/VDPP/checkpoints/vdpp.pth
```

DA3Metric 可使用 Hugging Face 模型 ID，首次运行时自动下载：

```text
depth-anything/DA3METRIC-LARGE
```

离线运行时，可将 `--da3-model` 指向本地 snapshot 目录。VDPP checkpoint 和所有
运行结果都被 `.gitignore` 排除，不随 Git 提交。

官方 v1.0 `vdpp.pth` 比当前源码多出已删除的 `shift_head.0.weight` 和
`shift_head.0.bias`。本入口只忽略这两个已知旧参数；若 checkpoint 缺少当前模型
参数或出现其他未知参数，会立即报错，避免静默加载不兼容权重。

运行时还需保证 `ffmpeg`、`ffprobe` 可在 `PATH` 中调用。推荐 GPU 显存不低于
16 GB；显存不足时先减小 `--batch-size`，必要时再缩短 `--chunk-size`。VDPP 的
训练窗口是 32 帧，因此默认保持 `--chunk-size 32`。

## 3. 本次目标视频命令

目标视频为 640×480、30 fps、1,059 帧。数据中没有独立相机标定文件，本次采用
仓库 head RGB 默认内参，并已由用户确认：

```text
fx=605.4, fy=605.2, cx=323.3, cy=252.0
```

完整视频命令：

```bash
python scripts/depth_refine/run_metric_vdpp.py \
  --video data/original/ai2_depth_preview/AI2_Alphabot_2_grind_ink_0_qced_hardlink_ep000000/videos/chunk-000/observation.images.cam_front_head_rgb/episode_000000.mp4 \
  --output outputs/da3_metric_vdpp_grind_ink_20261028 \
  --da3-model depth-anything/DA3METRIC-LARGE \
  --vdpp-root third_party/VDPP \
  --vdpp-checkpoint third_party/VDPP/checkpoints/vdpp.pth \
  --intrinsics 605.4 605.2 323.3 252.0 \
  --intrinsics-source repository-head-default-user-confirmed \
  --process-res 504 --batch-size 2 \
  --chunk-size 32 --overlap 8 --device cuda
```

先做少量帧的环境和显存检查时，可追加 `--max-frames 40`，并使用另一个全新的
输出目录。输出目录必须不存在，防止误覆盖已经完成的结果。

## 4. 参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--video` | 必填 | 输入 RGB 视频 |
| `--output` | 必填 | 新建的结果目录；已存在时拒绝覆盖 |
| `--da3-model` | 必填 | DA3Metric 模型 ID 或本地模型目录 |
| `--vdpp-root` | 必填 | VDPP 源码目录 |
| `--vdpp-checkpoint` | 必填 | `vdpp.pth` 路径 |
| `--intrinsics FX FY CX CY` | 必填 | 输入视频原分辨率对应的相机内参 |
| `--intrinsics-source` | `user-provided` | 写入报告的内参来源说明 |
| `--process-res` | `504` | DA3 预处理分辨率上限 |
| `--batch-size` | `2` | DA3 一次处理的原图帧数；实际还会加入同批 flip 帧 |
| `--chunk-size` | `32` | VDPP 时间窗口长度 |
| `--overlap` | `8` | 相邻 VDPP chunk 的重叠帧数，必须小于 chunk size |
| `--max-frames` | `0` | 仅用于 smoke test；0 表示处理完整视频 |
| `--device` | `cuda` | PyTorch 设备，可使用 `cuda:0` |
| `--epsilon` | `1e-6` | inverse depth 分母的稳定项，单位米 |
| `--min-depth-m` | `0.05` | 输出深度下界，单位米 |
| `--max-depth-m` | `100.0` | 输出深度上界，单位米 |
| `--smooth-strength` | `2.0` | chunk 重叠约束相对于 metric 锚点的权重 |

内参必须与输入视频实际分辨率匹配。若 RGB 在入库前做过 crop、resize 或畸变校正，
需要先对内参做相同变换，再传给命令。

## 5. 输出文件

```text
<output>/
├── config.json                 # 完整复现参数
├── report.json                 # 内参、P2/P98、chunk 拟合、耗时和限制
├── REPORT.md                   # 本次运行的中文摘要
├── da3_metric_depth_m.npy      # (T,H,W) float32，DA3Metric 米制深度
├── depth_m.npy                 # (T,H,W) float32，VDPP 后恢复的米制深度
├── anchor_mask.npy             # (T,H,W) bool，尺度拟合可靠区域
├── comparison.mp4              # RGB / DA3Metric / VDPP 三路固定色标预览
├── comparison_000000.png       # 首帧静态对比
├── comparison_<中间帧>.png
├── comparison_<末帧>.png
└── vdpp_chunks/
    └── chunk_*.npy             # VDPP normalized inverse-depth 中间结果
```

`depth_m.npy` 和 `da3_metric_depth_m.npy` 的单位都是米，不能按毫米直接解释。
`comparison.mp4` 的两路深度使用同一组固定米制色标，适合观察时序闪烁和尺度跳变；
它是有损可视化，数值分析应读取 NPY。

`report.json` 中的 `raw_temporal_mae_m` 是没有做运动补偿的相邻帧差，包含相机和
物体真实运动，只能作为运行间的辅助比较，不能证明深度精度或稳定性提升。

## 6. 结果检查

运行完成后至少检查：

1. `report.json` 的 `status` 为 `completed`，帧数与输入视频一致。
2. `normalization.per_frame_min_max` 为 `false`，并记录整段视频唯一的 P2/P98。
3. `anchor_fraction` 不是 0；过低时应检查内参、天空 mask、flip 阈值和场景运动。
4. 各 chunk 的 `smoothed_a` 都大于 0，重叠拟合误差没有突然增大。
5. `clipped_depth_fraction` 足够低；比例较高通常说明仿射恢复或深度上下界不合适。
6. 人工查看固定色标 `comparison.mp4` 的边界、运动区域和 chunk 交界位置。

## 7. 已知限制

- 默认 head 内参不是该视频的独立标定结果，绝对尺度只能视为近似 metric。
- VDPP 会修改局部深度结构；全局正仿射恢复无法保证每个局部区域仍保持 DA3Metric
  的几何精度。
- DA3Metric 没有置信度头，本实现的 flip/光流可靠区域只是工程代理。
- 当前实现将整段 RGB、inverse depth 和米制结果保存在内存或 memmap 中。长视频
  需要预留足够的 RAM 和磁盘；运行中间结果不会自动续跑。
- 没有 ground truth 时，只能检查一致性、拟合残差和视觉效果，不能据此宣称绝对
  深度精度提高。
- 正式落地前建议加入 metric anchor loss 并重训/微调 VDPP，再用有真值的数据集
  评估 AbsRel、RMSE、边缘误差和运动补偿后的时序指标。
