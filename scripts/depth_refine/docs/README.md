# 深度参考生成、修正与检查工具

常用命令可直接复制：[命令速查 COMMANDS.md](COMMANDS.md)。

本目录提供 LeRobot 数据集的稀疏深度参考生成、深度分布迁移、批处理和可视化入口。本文参考 Git 已提交版本的根目录 README 中“深度处理 Pipeline”和点云检查说明，并按重构后的脚本实现更新路径、参数和使用限制。

## 1. 脚本总览

| 脚本 | 作用 | 输入 | 主要输出 |
|---|---|---|---|
| [build_baseline.py](../build_baseline.py) | 默认每 5 帧采样 RGB，使用 DA3 Nested（或 Metric）生成参考深度 | 单个数据集的 RGB 视频 | PNG/NPY 采样帧、统计 JSON、可选稀疏深度视频 |
| [run_refine.py](../run_refine.py) | 拟合原深度到参考深度的映射，验证并生成完整结果 | 原始深度视频、baseline 采样帧 | 指标报告、可选趋势图和 refined 视频 |
| [run_batch.py](../run_batch.py) | 遍历多个项目、多个相机，依次调用 baseline/refine 或绘制均值曲线 | 包含多个数据集的根目录 | 各项目独立的 `depth_results/` |
| [plot_comparison.py](../plot_comparison.py) | 绘制原始、修正后和 Nested 的 3×3 对比图 | 两个完整深度视频、对应 Nested 采样帧 | 点云、深度图、直方图组成的 PNG |
| [inspect_dataset.py](../inspect_dataset.py) | 交互浏览 RGB/深度/点云，对比外部深度，或批量导出 PCD | LeRobot 数据集或三套深度视频 | Matplotlib 窗口、binary PCD 文件 |

公共路径和采样索引工具位于 [src/depth_refine/io_utils.py](../../../src/depth_refine/io_utils.py)。原始 LeRobot 加载入口仍为 [scripts/load_lerobot.py](../../load_lerobot.py)；本目录的扩展检查功能使用 `inspect_dataset.py`。

典型处理顺序：

```text
RGB 视频 ── build_baseline.py ── 稀疏参考 NPY / 统计 / 可选视频
                                    │
原始 Metric 深度 ────────────────────┤
                                    ▼
                         run_refine.py --stage validate
                                    │
                         run_refine.py --stage full 或 rife
                                    │
                   plot_comparison.py / inspect_dataset.py
```

这里的 baseline 是模型生成的稀疏参考，不是真实传感器标注；refine 是对原始深度做分布迁移。指标反映与参考的接近程度，不能直接当作真实物理精度。

## 2. 运行准备与数据约定

### 环境和工作目录

先按[项目 README](../../../README.md)准备基础环境。各功能实际依赖如下：

| 功能 | 依赖 |
|---|---|
| baseline 推理 | NumPy、OpenCV、PyTorch、DA3 及对应模型权重 |
| 深度视频读写 | PATH 中可调用的 `ffmpeg`、`ffprobe`；ffmpeg 支持 `libx265` 和 `gray12le` |
| refine 的 validate/full | NumPy、OpenCV、项目深度编码器；绘图另需 Matplotlib |
| refine 的 rife | 另需 `src/depth_repair/da3_rife.py` 使用的 RIFE 实现及权重 |
| 静态对比/交互浏览 | Matplotlib；交互浏览另需 LeRobot、项目深度数据集类及其视频解码依赖 |
| 批量 PCD | `tools/depth_to_pointcloud.py` 及其 NumPy、OpenCV、tqdm 等依赖 |

所有命令从**仓库根目录**运行。当前部分脚本移动目录后仍将 `scripts/` 加入模块搜索路径，运行前显式加入仓库根目录，保证 `src`、`tools` 及批处理子进程可导入：

```bash
cd /home/user/RoboCOIN-with-Depth
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# 用真实的数据集目录替换此示例，目录中应包含 meta/ 和 videos/
DATASET="data/lerobot/ai2_depth_preview/AI2_Alphabot_2_arrange_teaset_0_qced_hardlink_ep000000"
RESULT_ROOT="$DATASET/depth_results"
CAMERA="cam_front_head"
EPISODE="episode_000000"
```

后续示例沿用上述变量；相机名、chunk 和 episode 必须对应实际文件。

### 输入结构

```text
<dataset>/
├── meta/
│   └── info.json                 # 以及 LeRobot 所需的其他元数据
├── data/
│   └── chunk-000/episode_000000.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.cam_front_head_rgb/
        │   └── episode_000000.mp4
        └── observation.images.cam_front_head_depth/
            └── episode_000000.mp4
```

- `build_baseline.py` 只需要 RGB；`run_refine.py` 还需要已经存在的原深度及 baseline 统计，它不会先从 RGB 生成完整 Metric 视频。
- 原始深度应为以毫米表示的单通道深度，RGB/depth 对应同一相机、episode 和时间轴。直接读取视频的脚本不会按 LeRobot 元数据还原 min-max 归一化深度，归一化视频不能直接按毫米解释。
- baseline/refine 输出视频为 12-bit H.265，编码器将数值裁剪到 `0～4095`。NPY/PNG 参考帧为 `uint16` 毫米，允许保存到 `65535`；超出 4095 mm 的参考值在稀疏视频中会被裁剪。
- `--fps` 默认 30，并非自动探测输出帧率；应与源数据保持一致。
- baseline/refine 的 `--camera` 按目录名子串匹配，例如 `head` 可匹配 `cam_front_head`。检查器交互模式则要求完整相机 key，例如 `cam_front_head`。

### 默认结果结构

```text
<dataset>/depth_results/
├── baseline/
│   ├── statistics/<camera>/<episode>/
│   │   ├── frame_000000000.npy
│   │   ├── frame_000000000.png
│   │   ├── sample_indices.json
│   │   ├── statistics.json
│   │   └── manifest.json             # 仅保存稀疏视频时生成
│   └── videos/chunk-000/observation.images.<camera>_depth_nested_sparse/
│       └── episode_000000.mp4        # 仅 --save-video 时生成
└── refine/
    ├── reports/<camera>/<episode>/
    │   ├── refine_parameters.json
    │   ├── refine_trend.png          # 仅 --plot-trend 时生成
    │   └── <validate|full>/          # 逐帧模式，各阶段独立保存
    │       ├── frame_mappings/frame_000000000.npz
    │       ├── frame_timings.csv
    │       └── frame_timing_summary.json
    ├── videos/chunk-000/observation.images.<camera>_depth_refined/
    │   └── episode_000000.mp4        # full/rife 通过检查后生成
    └── plots/
        ├── frame_000000000_comparison.png
        └── <camera>/<episode>/mean_curve.png
```

默认输出不会注册到原数据集的 `meta/` 中，因此普通数据集加载不会自动切换为 refined 深度；可使用后文的外部深度对比模式查看。

## 3. build_baseline.py：生成稀疏参考

脚本发现 RGB 视频后，完整解码每个视频，默认按步长 5 抽取源帧 `0, 5, 10, ...` 执行 DA3 推理。只取小于总帧数的索引，不额外补尾帧。可用 `--sample-stride` 修改步长，必须大于 0。

显式指定 `--sample-count` 时沿用均匀定量采样，通常覆盖首尾；`--sample-count 1` 只采首帧，设为非正数或不小于总帧数则使用全部帧。两个采样参数互斥。

```bash
python scripts/depth_refine/build_baseline.py \
  --dataset-dir "$DATASET" --camera head \
  --sample-stride 5 --strategy da3_nested \
  --save-video --result-root "$RESULT_ROOT"
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset-dir` | 必填 | 单个数据集根目录 |
| `--camera` | `all` | RGB 相机目录的匹配关键词；`all` 处理所有发现的 RGB 目录 |
| `--sample-stride` | `5` | 源帧采样步长，必须大于 0；与 `--sample-count` 互斥 |
| `--sample-count` | 未指定 | 改用均匀定量采样，非正数表示全部帧 |
| `--strategy` | `da3_nested` | `da3_nested` 或 `da3`；后者在未修改默认模型时切换到 `depth-anything/DA3METRIC-LARGE` |
| `--da3-model` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | 模型 ID |
| `--da3-process-res` | `504` | DA3 处理分辨率 |
| `--da3-chunk-size` / `--da3-overlap` | `16` / `0` | 推理分块大小及重叠帧数 |
| `--device` | `auto` | 推理设备，传给 DA3 实现 |
| `--result-root` | `<dataset>/depth_results` | 结果根目录 |
| `--output-dir` | 未指定 | 覆盖 baseline 统计根目录，另见下方路径说明 |
| `--save-video` | 关闭 | 额外保存稀疏深度视频和 manifest |
| `--fps` / `--crf` / `--preset` | `30` / `6` / `medium` | 视频帧率和编码参数 |
| `--force` | 关闭 | 即使已有 `statistics.json` 也重新处理 |

`sample_indices.json` 记录源视频、源帧数、采样方式、步长和 `sample_frame_indices`。例如采样索引为 `[0, 5, 10]` 时，稀疏视频的第 1 帧对应源视频第 5 帧，不是源视频第 1 帧。NPY/PNG 文件名直接使用源帧号，补齐为 9 位。

`statistics.json` 记录模型、采样位置、全局/逐帧有效深度统计和分位数；统计仅包含有限且大于零的深度。`manifest.json` 记录稀疏视频与源帧的对应关系，供 PCD 导出使用。

已有 `statistics.json` 时会跳过整个 episode。切换采样方式或步长后必须加 `--force` 重新生成。第一次未加 `--save-video`，后来需要补视频时，应同时加 `--force --save-video`。

**自定义位置优先使用 `--result-root`。** 当前 `--output-dir` 不仅改变统计位置，还会让稀疏视频写到源 `videos/chunk-*` 旁的 `_depth_nested_sparse` 目录，且不符合 PCD 导出默认查找 manifest 的结构。即使策略为 `da3`，输出目录后缀仍叫 `_depth_nested_sparse`。

## 4. run_refine.py：验证和迁移深度分布

默认 `--mapping-scope frame --mapping transport`：每个源帧只与同帧参考 NPY 配对，独立求解直方图最优传输矩阵并应用映射。参考尺寸不同时，先将参考线性插值到原深度尺寸；拟合使用两者有限且大于零的像素交集。`--mapping-scope episode` 可恢复合并所有采样像素、各帧共用一套映射的旧行为。

| 映射 | 作用 |
|---|---|
| `transport` | 一维直方图单调最优传输，保存源 bin → 目标 bin 的概率质量矩阵；用每行目标深度的加权均值构造映射 |
| `quantile` | 分别计算源/参考分位数，以插值完成一维分布匹配；超出拟合范围的值取端点值 |
| `linear` | 拟合 `target ≈ a × source + b`，拟合时剔除源深度低于 P1、高于 P99 的像素 |

映射只作用于有限且大于零的源像素，其他位置保持 0；结果裁剪到 `--max-depth-mm`。

### 先验证，再生成完整视频

```bash
# 只生成报告及趋势图
python scripts/depth_refine/run_refine.py \
  --dataset-dir "$DATASET" --result-root "$RESULT_ROOT" \
  --camera head --mapping transport --mapping-scope frame --stage validate --plot-trend

# 先生成每一帧的 Nested 参考；已有稀疏统计时用 --force 重新生成
python scripts/depth_refine/build_baseline.py \
  --dataset-dir "$DATASET" --result-root "$RESULT_ROOT" \
  --camera head --sample-stride 1 --force

# 每一帧独立搬运、保存矩阵和计时，生成完整视频
python scripts/depth_refine/run_refine.py \
  --dataset-dir "$DATASET" --result-root "$RESULT_ROOT" \
  --camera head --mapping transport --mapping-scope frame --stage full
```

如需使用 RIFE，从迁移后的稀疏关键帧补齐时间轴：

```bash
python scripts/depth_refine/run_refine.py \
  --dataset-dir "$DATASET" --result-root "$DATASET/depth_results_rife" \
  --statistics-dir "$RESULT_ROOT/baseline/statistics" \
  --camera head --mapping quantile --mapping-scope episode --stage rife
```

示例将 RIFE 结果放在独立目录，便于与 full 比较。同一路径运行不同阶段/映射会重写报告和同名结果；每次运行都会重新拟合，不会直接加载上次验证报告中的映射。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset-dir` / `--camera` | 必填 | 数据集根目录 / 相机目录关键词；这里没有特殊的 `all` 模式 |
| `--result-root` | `<dataset>/depth_results` | 结果根目录 |
| `--statistics-dir` | `<result-root>/baseline/statistics` | 参考统计根目录，下面是 `<camera>/<episode>/` |
| `--output-dir` | `<result-root>/refine/videos` | refined 视频根目录 |
| `--report-dir` | `<result-root>/refine/reports` | 报告根目录 |
| `--input-depth-suffix` | `_depth` | 从 RGB 目录推导源深度目录时使用的后缀 |
| `--input-pix-fmt` | `gray12le` | 源深度解码格式，可选 `gray16le` |
| `--mapping` | `transport` | `transport`、`quantile` 或 `linear`；只有 transport 产生概率质量矩阵 |
| `--mapping-scope` | `frame` | frame 每帧独立；episode 沿用全局映射 |
| `--transport-bins` | `256` | transport 直方图离散 bin 数，至少 2；只保存非空 bin |
| `--quantiles` | `101` | quantile 拟合请求的分位点数，实际至少 3 个；重复源节点会去重 |
| `--stage` | `validate` | validate 搬运并评估已有参考帧；逐帧 full 要求完整同帧参考；rife 仅支持 episode scope |
| `--allow-unvalidated` | 关闭 | 未满足指标改善条件时仍生成视频 |
| `--max-depth-mm` | `4095` | 映射输出上限；编码器仍会裁剪到 4095 |
| `--plot-trend` | 关闭 | 保存逐采样帧均值及 Wasserstein 趋势图 |
| `--report-mapping-slope` | 关闭 | 保存映射斜率摘要及最陡区间，并写入验证报告 |
| `--save-preencode-npy` | 关闭 | full/rife 逐帧保存编码前 float32 深度 NPY |
| `--fps` / `--crf` / `--preset` | `30` / `6` / `medium` | 输出视频编码参数 |

### 如何阅读验证报告

`refine_parameters.json` 包含映射参数、采样索引、迁移前 `baseline` 指标、迁移后 `transferred` 指标及逐帧 `frame_metrics`。quantile 映射还会记录 `requested_quantiles`、`effective_quantile_nodes` 和 `mapping_segments`，用于区分请求的 q101 配置与量化去重后的实际节点数。此处报告字段 `baseline` 指的是迁移前的原深度指标，与磁盘上的 Nested baseline 目录含义不同。

逐帧模式在 `frame_mappings` 中记录每帧参数、矩阵路径和耗时。每个阶段独立保存到 `<report-dir>/<camera>/<episode>/<validate|full>/`：

- `frame_mappings/frame_000000000.npz`：每帧的矩阵/映射，帧号从 0 开始，与源视频一致。
- `frame_timings.csv`：每帧参考读取、resize/有效像素配对、直方图和矩阵拟合、应用映射、NPZ 写盘、合计耗时（秒）。
- `frame_timing_summary.json`：完成/失败帧数及各项耗时的总和、均值、最小值、P50、P95、最大值。

完整逐帧 full 不会复用全局/邻帧矩阵；任何同帧参考缺失都会报错。validate 可以使用稀疏参考，但只为实际参考帧生成矩阵和计时。RIFE 插值帧没有独立求解的矩阵，因此不用于完整逐帧搬运。

transport NPZ 使用 COO 稀疏存储：`row`、`col`、`mass`、`shape`，并包含 `source_depth_mm`、`target_depth_mm`、`source_mass`、`target_mass`、`mapped_depth_mm`、`frame_index`。行/列边缘分别等于源/目标概率直方图，总质量为 1；bin 深度为该 bin 内实际像素均值。可重建普通矩阵：

```python
with np.load("frame_000000000.npz", allow_pickle=False) as plan:
    T = np.zeros(tuple(plan["shape"]))
    np.add.at(T, (plan["row"], plan["col"]), plan["mass"])
```

输出是矩阵行重心的确定性深度映射，按源 bin 代表深度线性插值，再裁剪；当同一个源深度需要分流到多个目标深度时，重心投影不会精确复现目标直方图。提高 `--transport-bins` 可提高离散精度。旧 quantile/linear 的 NPZ 只保存各自映射参数，不称为 OT 质量矩阵。

排查灰度色斑时可在 full 命令后追加：

```bash
--report-mapping-slope --save-preencode-npy
```

斜率结果位于报告目录的 `mapping_slope.json`，并同时写入
`refine_parameters.json` 的 `mapping_slope` 字段。`slope > 1` 表示该深度区间
会放大源深度变化。编码前帧位于
`<result-root>/refine/preencode/<chunk>/<camera-depth-refined>/<episode>/`，每帧为
一个 float32 NPY；它们尚未经历 uint16 截断及 H.265 编码。以同目录的
`manifest.json` 为准读取本次运行的帧数和文件列表。

| 指标 | 含义 |
|---|---|
| `mae_mm` | 与参考的逐像素平均绝对误差 |
| `rmse_mm` | 均方根误差，对大误差更敏感 |
| `wasserstein_mm` | 基于分位点近似的一维分布距离 |
| `mean_mm` / `target_mean_mm` | 当前结果 / 参考的平均深度 |
| `mean_error_mm` | 两者均值之差 |
| `p10_mm` / `p50_mm` / `p90_mm` | 当前结果的深度分位数 |
| `valid_pixels` | 参与统计的有效配对像素数 |

仅当 MAE、RMSE 和 Wasserstein **全部严格降低**时，`improved` 才为 `true`；否则 full/rife 默认跳过视频输出，并打印 `Skip unvalidated transfer`。报告仍会保留。验证使用拟合时的同一批采样帧，并非独立测试集，也不直接衡量时间一致性。

找不到参考索引/NPY 或没有有效像素的 episode 会被跳过。RIFE 模式额外要求 RGB 与原深度总帧数相同。

## 5. run_batch.py：批量处理项目和相机

递归发现 `--root` 下含 `videos/chunk-*` 的项目，排除路径含 `depth_results` 的输出目录并去重，随后按项目、相机顺序执行子脚本。它是串行批处理器，子进程失败会中止后续处理。

```bash
# 预览发现的项目及相机
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras head left_wrist right_wrist chest \
  --stage all --dry-run

# 生成参考并验证，保存稀疏视频及均值曲线
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras head left_wrist right_wrist \
  --stage all --sample-stride 5 --save-video \
  --refine-stage validate --plot-mean-curve

# 基于已有完整逐帧参考生成完整结果
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras head left_wrist right_wrist \
  --stage refine --refine-stage full --plot-mean-curve

# 只读取已有报告绘制均值曲线
python scripts/depth_refine/run_batch.py \
  --root data/lerobot --cameras head --stage plot
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--root` | `data/lerobot` | 扫描根目录，也可指向单个项目 |
| `--cameras` | `head left_wrist right_wrist chest` | 空格分隔，仅接受这四个选项；缺失相机跳过 |
| `--stage` | `baseline` | `baseline`、`refine`、`all`、`plot` |
| `--sample-stride` | `5` | 传给 baseline；与 `--sample-count` 互斥 |
| `--sample-count` | 未指定 | 传给 baseline，改用均匀定量采样 |
| `--force` | 关闭 | 传给 baseline，重新生成已有统计 |
| `--save-video` | 关闭 | 传给 baseline，保存稀疏视频 |
| `--refine-stage` | `validate` | 传给 refine：`validate`、`full`、`rife` |
| `--mapping` / `--mapping-scope` / `--transport-bins` | `transport` / `frame` / `256` | 传给 refine；逐帧 full 的 baseline 必须用 `--sample-stride 1` 或 `--sample-count 0` |
| `--plot-mean-curve` | 关闭 | refine 后根据报告绘制原始/参考/迁移后均值曲线 |
| `--dry-run` | 关闭 | 只打印项目、相机和阶段，不执行子脚本或绘图 |
| `--replace-depth-video` | 关闭 | 存在替换实现问题，见下文 |

当前代码**支持 chest，且默认包含 chest**，与旧 README 中“不参与批处理”的说法不同。结果固定为各项目的 `depth_results/`；切换采样配置后用 `--force` 重新生成 baseline。批处理器没有暴露模型、设备、帧率、自定义结果路径或 `--allow-unvalidated`，需要这些参数时直接运行对应单项目脚本。

`--replace-depth-video` 的帮助文本表示替换原数据集深度，但当前函数实际从生成视频所在的结果 chunk 查找目标，没有使用 `project/videos` 定位，可能只是把结果复制回自身。因此不要依赖此选项完成原深度替换；本文示例均保留独立输出。validate 本身不生成可替换视频。

## 6. plot_comparison.py：绘制三路对比图

输出三列分别来自 Original、Refined 和 Nested，三行分别为点云、深度图和直方图。点云共享内参、视角和坐标范围，深度图共享色阶，直方图共享深度范围和 bins。

先定义同一相机、episode 的输入：

```bash
BEFORE="$DATASET/videos/chunk-000/observation.images.${CAMERA}_depth/$EPISODE.mp4"
AFTER="$RESULT_ROOT/refine/videos/chunk-000/observation.images.${CAMERA}_depth_refined/$EPISODE.mp4"
NESTED_DIR="$RESULT_ROOT/baseline/statistics/$CAMERA/$EPISODE"

# 单帧：--frame 对应源视频帧号，NPY 必须是同一帧
python scripts/depth_refine/plot_comparison.py \
  --before "$BEFORE" --after "$AFTER" \
  --nested "$NESTED_DIR/frame_000000000.npy" --frame 0 \
  --output "$RESULT_ROOT/refine/plots/$CAMERA/$EPISODE/comparison.png"

# 遍历 sample_indices.json 中的全部采样帧
python scripts/depth_refine/plot_comparison.py \
  --before "$BEFORE" --after "$AFTER" \
  --nested-dir "$NESTED_DIR" --all-sampled \
  --output "$RESULT_ROOT/refine/plots/$CAMERA/$EPISODE/comparison.png"
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--before` / `--after` | 必填 | 原始 / 修正后完整深度视频 |
| `--nested` | 未指定 | 单帧模式必填，使用对应源帧的 NPY；也接受深度视频 |
| `--nested-dir` / `--all-sampled` | 未指定 / 关闭 | 成对使用，批量绘制统计目录内的采样帧 |
| `--frame` | `0` | 单帧模式的源视频帧号，从 0 开始 |
| `--output` | 自动推导 | PNG 路径；批量模式在文件 stem 后追加 `_frame_XXXXXXXXX` |
| `--max-depth` | `4095` | 显示、直方图和有效点云的深度上限，单位 mm |
| `--fx` / `--fy` | `605` / `605` | 水平/垂直焦距，单位像素 |
| `--cx` / `--cy` | `323` / `252` | 主点位置，单位像素 |
| `--pcd-stride` | `3` | 点云采样步长，至少 1，越小点越密集 |
| `--azim` / `--elev` | `45` / `25` | 点云观察方位角/俯仰角，单位度 |

点云坐标为相机坐标系下的米。实际相机应传入对应分辨率的标定内参；默认值不会自动适配相机。

使用时注意：

- 三组深度的宽高必须一致，否则脚本报错；refine 验证中的自动缩放不会修改磁盘上的 Nested NPY。
- `--nested` 传视频时直接取该视频的 `--frame`，不会读取稀疏 manifest 做索引转换，因此稀疏参考优先用 NPY 或 `--all-sampled`。
- 当前图片第二列标题固定为 `DA3 LARGE`，实际内容是 `--after` 指定的结果，脚本本身不会额外运行 DA3 LARGE。
- 默认图片路径只有帧号，没有相机/episode 隔离。多相机、多 episode 绘图时像示例一样指定 `--output`，避免同名覆盖。
- 视频解码固定为 `gray12le`；无显示器环境保存静态图时可设置 `export MPLBACKEND=Agg`，交互检查时需恢复可交互后端。

## 7. inspect_dataset.py：交互检查与 PCD 导出

### 7.1 浏览 LeRobot 数据集

```bash
python scripts/depth_refine/inspect_dataset.py \
  --repo-id "${DATASET##*/}" --root "$DATASET" \
  --episode 0 --frame 0 --camera all \
  --save-pcd "pcds/manual_preview"

# 原生 LeRobot 模式：显示 RGB 和深度
python scripts/depth_refine/inspect_dataset.py \
  --repo-id "${DATASET##*/}" --root "$DATASET" --native
```

`--root` 直接传包含 `meta/info.json` 的数据集目录；不会自动将 `--repo-id` 拼到 root 后面。默认使用项目 `LeRobotWithDepthDataset`，显示 RGB、深度及彩色点云。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--repo-id` | 未指定 | 交互模式必填，LeRobot 数据集 ID |
| `--root` | `data/lerobot` | 数据集实际根目录 |
| `--episode` / `--frame` | `0` / `0` | 初始 episode 索引 / episode 内帧号 |
| `--camera` | `all` | `all` 或完整相机 key，如 `cam_front_head`、`cam_left_wrist` |
| `--native` | 关闭 | 使用原生 LeRobotDataset，不生成 PCD |
| `--save-pcd` | 未指定 | 按 `p` 保存当前样本点云的目录 |
| `--compare-depth-before` / `--compare-depth-after` | 未指定 | 同时指定，进入外部深度对比模式 |

| 操作 | 功能 |
|---|---|
| `←` / `→` | 前后 1 帧 |
| `Shift + ←/→` | 前后 10 帧 |
| `Ctrl + ←/→` | 前后 100 帧 |
| `↑` / `↓` | 下一个 / 上一个 episode |
| `w` / `s` | 调整点云俯仰角 |
| `a` / `d` | 调整点云方位角 |
| 在点云区域按住鼠标左键拖动 | 旋转点云 |
| `p` | 已设置 `--save-pcd` 时保存当前样本所有选中相机的点云 |
| `q` / `Escape` | 退出 |

保存文件名为 `ep0000_frame000000_<camera>.pcd`，其中 frame 使用数据集全局帧号。脚本旧 help 中“按 s 保存”不准确，以实际按键 `p` 为准。

### 7.2 交互对比原始与 refined 深度

沿用第 6 节的 `BEFORE`、`AFTER`：

```bash
python scripts/depth_refine/inspect_dataset.py \
  --repo-id "${DATASET##*/}" --root "$DATASET" \
  --episode 0 --frame 0 \
  --compare-depth-before "$BEFORE" \
  --compare-depth-after "$AFTER"
```

窗口为两行两列：Before / After 的深度图及点云。两条路径必须同时提供、来自同一相机，目录名应为 `observation.images.<camera>_depth` 或 `_depth_refined`；相机会从路径自动解析。

该模式依赖增强数据集的点云内部接口，**不要与 `--native` 组合**。外部视频固定为传入的 episode，浏览时应留在对应 episode 内。此模式的 `p` 仍保存数据集样本点云，不会导出窗口中的两套外部结果；外部结果导出使用下一节。

### 7.3 将 Original / Baseline / Refine 导出为逐帧 PCD

此模式不需要 `--repo-id`，完成导出后退出。先用 baseline 的 `--save-video` 生成稀疏视频及 manifest，再用 full/rife 生成完整 refined 视频：

```bash
python scripts/depth_refine/inspect_dataset.py \
  --generate-pcd --pcd-root pcds \
  --original-depth "$DATASET/videos/chunk-000/observation.images.${CAMERA}_depth/$EPISODE.mp4" \
  --baseline-depth "$RESULT_ROOT/baseline/videos/chunk-000/observation.images.${CAMERA}_depth_nested_sparse/$EPISODE.mp4" \
  --refine-depth "$RESULT_ROOT/refine/videos/chunk-000/observation.images.${CAMERA}_depth_refined/$EPISODE.mp4" \
  --rgb-video "$DATASET/videos/chunk-000/observation.images.${CAMERA}_rgb/$EPISODE.mp4" \
  --pcd-stride 2 --pcd-max-depth 5000 --pcd-depth-scale 1.0
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--generate-pcd` | 关闭 | 批量导出三套视频的点云并退出 |
| `--original-depth` / `--baseline-depth` / `--refine-depth` | 未指定 | 导出模式三者均必填，视频文件或恰好包含一个 MP4 的目录 |
| `--rgb-video` | 自动推导 | 三套深度共用的 RGB；使用独立结果目录时建议显式指定 |
| `--pcd-root` | `pcds` | PCD 输出根目录；旧 help 中的 `pcd` 不是实际默认值 |
| `--pcd-stride` | `2` | 像素采样步长，使用正整数 |
| `--pcd-max-depth` | `5000` | 最大有效深度，单位 mm |
| `--pcd-depth-scale` | `1.0` | 传给点云工具的深度比例参数，必须大于 0；标准毫米输入用 1.0 |

将 `CAMERA` 改为 `cam_left_wrist`、`cam_right_wrist` 或实际相机 key 后重复运行即可导出其他视角。

输出路径按数据集、任务、类型和相机区分：

```text
pcds/<dataset>_depth_preview/<task>/
├── original/<camera>/frame_000000.pcd
├── baseline/<camera>/frame_000000.pcd
└── refine/<camera>/frame_000000.pcd
```

当前路径解析要求输入路径含 `<dataset>_depth_preview/<task>` 两级，例如本文的 `ai2_depth_preview/<任务名>`。任意普通 `<dataset>` 路径可以做 baseline/refine，但不一定满足此 PCD 导出的路径约定。

Baseline 根据 `baseline/statistics/<camera>/<episode>/manifest.json` 将稀疏视频帧对齐到源 RGB，并以**源帧号**命名 PCD。Original/refine 按顺序配对 RGB，帧数不同时取较短长度。自动 RGB 推导仅替换路径中的深度后缀，不会从 `depth_results` 返回源数据集，所以示例显式传 `--rgb-video`。

PCD 输出目录本身不含 episode/chunk；同一任务、相机导出多个 episode 时，应为每个 episode 指定不同的 `--pcd-root`，避免重复的源帧号覆盖。

## 8. 常见问题与检查顺序

| 现象 | 检查方法 |
|---|---|
| `No module named src` 或 `tools` | 回到仓库根目录，按第 2 节设置 `PYTHONPATH`，批处理子进程也会继承 |
| baseline 没有新增视频 | 检查是否用了 `--save-video`；已有统计标记时加 `--force` |
| refine 显示处理 0 个 episode | 核对相机关键词、源深度后缀、统计根目录、采样索引和 NPY 是否存在 |
| 有报告但没有 refined 视频 | 确认不是 validate；查看 `improved` 及 `Skip unvalidated transfer` |
| 图像尺寸不一致 | 检查源深度与 Nested 输出尺寸，绘图脚本不会自动缩放 |
| PCD 找不到 RGB 或 manifest | 显式传 `--rgb-video`，使用标准 baseline 目录并保留 manifest |
| 检查器相机为空 | 用完整相机 key，不能将 baseline 的关键词 `head` 直接当交互相机名 |
| 不同结果互相覆盖 | 为不同方法指定结果根目录，为图像和多 episode PCD 指定独立输出路径 |

完整参数也可查看各入口帮助：

```bash
python scripts/depth_refine/build_baseline.py --help
python scripts/depth_refine/run_refine.py --help
python scripts/depth_refine/run_batch.py --help
python scripts/depth_refine/plot_comparison.py --help
python scripts/depth_refine/inspect_dataset.py --help
```

## 9. 逐 episode、逐视角耗时

baseline/refine 自动将最近一次处理的墙钟耗时写入 `depth_results/timings/`，包含每个视角的独立记录，区分 Nested/Metric 和 refine 的 validate/full/rife 阶段。运行 `python scripts/depth_refine/summarize_timings.py --result-root "$RESULT_ROOT"` 导出单视角明细 `per_view.csv`、三视角合计 `per_episode.csv` 及逐帧明细 `per_frame.csv`。缺失或未完成的视角不会按零秒计入完整合计。

逐帧 CSV 的 `fit_seconds` 是直方图构建和搬运矩阵求解耗时，`apply_seconds` 是应用映射耗时，`total_seconds` 包含参考读取、配对、拟合、应用和矩阵写盘。视频解码、评估、绘图、preencode 保存及编码计入 episode 总耗时，不分摊到各帧；旧结果无法反推真实逐帧耗时。

三视角运行、单视角筛选和计时范围见[命令速查第 6 节](COMMANDS.md#6-每条-episode-的-nested--refine-耗时)。旧结果需要重新运行才能得到真实耗时。
