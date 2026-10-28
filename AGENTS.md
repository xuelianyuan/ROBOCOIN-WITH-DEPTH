# RoboCOIN-with-Depth Agent Guide

## 1. 项目目标

本仓库下一阶段的主线是实现一条从 RGB 视频到绝对度量深度视频的单目端到端推理流水线：

```text
连续 RGB 视频
  -> 按步长选取关键帧
  -> DA3NESTED-GIANT-LARGE-1.1 对关键帧序列做度量深度估计
  -> RIFE 根据相邻 RGB 关键帧估计运动并补齐中间深度帧
  -> 与输入逐帧对齐的连续绝对深度视频
```

这里的“端到端”指用户只需提供一个 RGB 视频并直接得到深度视频，不表示现阶段要联合训练 DA3 与 RIFE。默认模型必须使用刷新后的 `depth-anything/DA3NESTED-GIANT-LARGE-1.1`，不要退回已弃用的无 `-1.1` 权重。

首阶段在当前电脑完成正确性验证和最小可运行部署；第二阶段迁移到最多 4 张 NVIDIA A800 80GB，系统评测 1/2/4 卡下的质量、吞吐、延迟、显存和扩展效率。

## 2. 当前仓库事实

- `src/depth_repair/da3.py` 是现有 DA3 通用实现，内部深度单位为毫米。
- `src/depth_repair/da3_rife.py` 当前实现的是 DA3 Metric 关键帧推理加 RIFE 深度 warp。它是已有基线，不应被悄悄改成新方案。
- `third_party/Depth-Anything-3/` 和 `third_party/Practical-RIFE/` 是上游代码；除非确有必要，不直接修改。
- `scripts/add_depth_fs_parallel.py` 服务于 LeRobot 数据集批处理，不应承担新的视频产品入口。
- `tests/test_da3.py`、`tests/test_da3_rife.py` 和 `tests/conftest.py` 给出了当前测试及 GPU benchmark 习惯。
- 当前环境可能没有可用 NVIDIA 驱动。任何 GPU 结论必须来自实际命令输出，不得根据机器名称猜测。

## 3. 范围与非目标

### 本阶段必须完成

- 接受常见 RGB 视频文件，保持原分辨率、帧数、帧率和时间顺序。
- 输出每帧一个 `float32` 绝对深度图；推理内部统一以米为权威单位，写入现有 12-bit 深度视频前只转换一次到毫米。
- DA3 Nested 必须以有时间顺序的关键帧序列推理，而不是把所有关键帧当作互不相关的单图。
- RIFE 使用 RGB 估计中间时刻运动，将相邻关键帧深度 warp 并做遮挡融合；关键帧深度必须原样保留。
- 长视频以有重叠的窗口处理，窗口间对齐和融合不能破坏绝对尺度。
- 支持当前电脑单卡运行，并为 1/2/4 张 A800 的批量视频并行评测预留清晰接口。

### 暂不默认实施

- 不训练或微调 DA3、RIFE。
- 不承诺逐帧低延迟的实时在线流；先实现文件级连续视频推理，再单独评估流式能力。
- 不以 RIFE 生成 RGB 帧后再对每个补帧运行 DA3；那会抵消关键帧采样的主要算力收益。
- 不把多卡吞吐提升误称为单视频时延提升。
- 不改写现有 `da3_rife` 策略的语义；新方案必须是独立策略。

## 4. 新增代码的固定命名与路径

新增文件应遵循下面的最小结构。若没有明确收益，不再增加同义入口或散落脚本。

```text
configs/
└── da3_nested_rife.yaml          # 唯一默认配置，记录模型、采样、窗口、RIFE 和输出参数
src/
└── video_depth/
    ├── __init__.py               # 仅导出稳定公共 API
    ├── config.py                 # 配置 dataclass、校验、YAML 加载
    ├── pipeline.py               # DA3 Nested + RIFE 的总编排
    ├── sampler.py                # 关键帧索引、尾帧、场景切换处理
    ├── da3_nested.py             # DA3 Nested 加载、窗口推理、窗口融合
    ├── rife_depth.py             # RGB 光流、深度 warp、遮挡融合
    ├── video_io.py               # 视频解码、时间信息与深度视频编码
    └── types.py                  # 帧、元数据和推理结果的数据结构
scripts/
└── estimate_video_depth.py       # 唯一面向用户的单视频/目录 CLI
tests/
├── video_depth/
│   ├── test_config.py
│   ├── test_sampler.py
│   ├── test_rife_depth.py
│   ├── test_pipeline.py
│   └── test_video_io.py
└── benchmark_video_depth.py      # 质量与性能评测入口，不作为普通 pytest 用例
benchmarks/
├── results/                      # 自动生成的原始评测数据，不提交 Git
└── reports/                      # 从原始数据汇总出的评测报告
outputs/                          # 正式推理输出，不提交 Git
```

命名规则：

- Python 文件、函数、参数使用 `snake_case`，类使用 `PascalCase`。
- 新策略的注册名统一为 `da3_nested_rife`。
- 不使用 `new.py`、`final.py`、`test2.py`、`utils2.py`、`tmp.py` 等无语义名称。
- 通用代码只有在两个以上模块真实复用时才进入辅助模块；禁止创建包罗万象的 `utils.py`。
- 实验输出放到 `outputs/` 或用户指定目录；权重放到 `checkpoints/`，两者都不得提交 Git。
- benchmark 结果放到 `benchmarks/results/`，文件名包含日期、GPU 数量和配置短名，例如 `20261028_4xa800_stride5.json`。

如实现中证明上述结构需要调整，先在变更说明中陈述理由；保持一个概念只有一个权威实现。

### 4.1 中文注释与文件可读性

所有新增 Python 文件必须在第一行附近使用中文模块 docstring 说明文件用途。至少写清楚“本文件负责什么”；涉及模型或数据流时，再注明主要输入、输出和单位。例如：

```python
"""
pipeline.py — DA3 Nested + RIFE 视频深度估计总流水线

负责组织视频解码、关键帧采样、米制深度推理、RIFE 补帧和结果写出。
输入为 RGB 视频，核心 API 输出 float32 米制深度序列。
"""
```

代码中使用中文分区注释组织较长文件，分区名称必须能直接表达职责：

```python
# ============================================================================
# 关键帧采样与边界处理
# ============================================================================
```

具体要求：

- 公共类、公共函数和复杂内部函数使用中文 docstring，写明职责、关键参数、返回值以及 shape/单位。
- 变量、函数和文件名仍使用有意义的英文标识符；中文用于解释设计与数据契约，不用拼音命名。
- 注释重点解释“为什么这样做”、单位转换、张量 shape、边界条件和回退策略；不要给每行显而易见的代码添加注释。
- 一个文件只承担一个清晰职责。仅看文件名、文件头说明和分区标题，应能快速判断代码作用。
- 修改现有文件时，只为本次触及的复杂区域补充必要中文说明，不为了统一风格重写整个文件。

### 4.2 输出文件与临时产物位置

三类输出严格分开：

```text
outputs/
└── <input_stem>/<run_id>/
    ├── depth.mp4                 # 正式深度视频
    ├── depth.npz                 # 可选的无损米制深度
    ├── run_config.yaml           # 本次解析后的最终配置
    ├── run_metadata.json         # 输入信息、环境、commit、单位和耗时
    └── run.log                   # 本次运行日志

benchmarks/results/
└── <run_id>/
    ├── metrics.json              # 质量和汇总性能指标
    ├── timings.csv               # 分视频、分阶段耗时
    └── visualizations/           # 少量固定评测可视化

benchmarks/reports/
└── <date>_<experiment>.md        # 汇总比较、结论和复现命令

tests/output/
└── <test_or_case_name>/          # 仅保留人工检查所需的测试图片/视频
```

- 正式运行结果默认写入 `outputs/`；如果用户显式传入 `-o/--output`，以用户路径为准。
- 机器可读的评测原始结果写入 `benchmarks/results/`，便于后续汇总 1/2/4 卡对比。
- 可阅读的最终评测结论写入 `benchmarks/reports/`；报告引用对应 result 目录、配置和 Git commit。
- 普通单元测试的临时文件优先使用 pytest 的 `tmp_path`，测试结束自动清理，不写入仓库。
- 只有需要人工检查的深度视频、点云或对比图才写入现有 `tests/output/`，并按测试名建立子目录，禁止直接散落在项目根目录。
- `outputs/`、`benchmarks/results/`、`tests/output/`、临时文件和大型可视化均不得提交 Git；开始实现时同步确认 `.gitignore` 已覆盖这些路径。
- `benchmarks/reports/` 中的小型 Markdown 报告可以提交，但生成的视频、NPZ 和大图不提交。
- 所有自动生成文件使用稳定、可追溯的名称；禁止使用 `result1`、`new_output`、`tmp_final` 等名称。

### 4.3 最小化原仓库改动

- 新功能优先落在独立的 `src/video_depth/` 包和单一 CLI 中；现有模块只做必要的注册、导出或兼容接线。
- 不重命名、不移动、不大规模重构现有文件，不改变已有 `da3`、`da3_rife`、`vda` 策略的默认行为。
- 能通过适配层复用时，不复制 DA3、RIFE、编码器或视频读取实现；适配层应隔离上游 API 变化。
- `third_party/` 默认只读。确需修补上游时，补丁保持最小且独立记录原因。
- 不进行与当前任务无关的格式化、注释清理、依赖升级或性能重写。
- 每次实现前列出预计新增和修改的文件；若需要偏离本文件规定的路径，先说明原因。
- 提交前检查 `git diff --stat` 和 `git diff`，确保修改范围与任务一致，并运行现有策略的最小回归测试。

## 5. 简洁执行接口

面向用户只提供一个主命令：

```bash
python scripts/estimate_video_depth.py input.mp4 -o output_depth.mp4
```

默认自动读取 `configs/da3_nested_rife.yaml`。常用覆盖参数保持少而稳定：

```bash
python scripts/estimate_video_depth.py input.mp4 -o output_depth.mp4 \
  --stride 5 --device cuda:0
```

配置文件承担模型 ID、窗口大小、重叠、处理分辨率、精度模式、RIFE 权重、编码参数等高级选项。CLI 参数优先于 YAML。不要要求用户设置一串环境变量，也不要让用户依次手工运行抽帧、DA3、RIFE、编码四个脚本。

CLI 至少应支持：

- `input`：单个视频或包含视频的目录。
- `-o/--output`：输出文件或输出目录。
- `--config`：可选配置路径。
- `--stride`：关键帧步长，默认值来自配置。
- `--device`：如 `cuda:0`、`cpu`；CPU 主要用于接口测试，不承诺可接受性能。
- `--overwrite`：显式允许覆盖，默认拒绝覆盖已有输出。
- `--dry-run`：只打印解析后的配置、输入元数据和预计关键帧数，不加载模型。

失败时返回非零退出码并给出可执行的错误信息。启动日志必须打印模型、设备、输入/输出 FPS、帧数、分辨率、stride、窗口和单位。

## 6. 算法与数据契约

### 6.1 视频与索引

- 解码后 RGB 为 `(H, W, 3)`、`uint8`、RGB 通道顺序。
- 关键帧索引必须严格递增、无重复、始终包含第 0 帧和最后一帧。
- `stride <= 0` 必须报错；`stride == 1` 必须绕过 RIFE，输出直接来自 DA3。
- 输出帧数必须严格等于输入帧数，输出帧 `i` 必须对应输入帧 `i`。
- 优先保留输入时间戳；若当前编码链只支持 CFR，必须检测并明确记录 VFR 到 CFR 的转换，不能静默漂移。
- 场景切换处禁止跨镜头插值。检测到切换时，将边界帧提升为关键帧，或采用明确记录的安全回退。

### 6.2 DA3 Nested

- 默认 checkpoint 为 `depth-anything/DA3NESTED-GIANT-LARGE-1.1`。
- 模型原始输出按米解释。所有 resize、窗口融合和置信度处理在米制浮点深度上完成。
- 不沿用为相对深度设计的任意 scale/shift 对齐。绝对深度窗口只允许稳健的小尺度校正，且必须有重叠帧、有效掩码、阈值限制和可关闭开关。
- 长序列使用带重叠窗口；重叠帧融合需保留度量尺度并避免接缝。窗口大小和 overlap 必须满足 `0 <= overlap < window_size`。
- 保留 DA3 的置信度供融合、诊断和评测使用，不要在 API 边界提前丢弃。
- 模型加载只能发生一次；推理全程使用 `eval()` 和 `torch.inference_mode()`。

### 6.3 RIFE 深度补帧

- RIFE 的运动输入是相邻 RGB 关键帧，深度不是用于估计光流的纹理输入。
- 对每个中间时刻同时 warp 前后两张关键帧深度，并使用 RIFE mask/遮挡信息融合。
- 只对有效、有限且大于零的深度参与融合；空洞不能被零值平均成虚假的近距离表面。
- 深度融合保持线性米制值，不对深度做 RGB 式归一化，不在逆深度和深度之间隐式切换。
- RIFE 只生成非关键帧；输出中的关键帧必须与 DA3 输出逐值一致（允许最终编码量化误差）。
- 对超大运动、遮挡、反射、运动模糊和场景切换提供置信度或回退路径，并在 benchmark 中单独观察。

### 6.4 输出

- 核心 Python API 返回 `float32` 米制深度和显式 `unit="m"` 元数据。
- 接入现有 LeRobot/12-bit 编码器时再转换为毫米；禁止出现重复 `* 1000`。
- 普通视频输出保持输入宽高、帧数和 FPS。编码前检查所有帧 shape、dtype、finite 比例和非负性。
- 绝对深度超出 12-bit 毫米量程时不得静默截断。默认记录裁剪比例；需要完整量程时提供无损数组输出（如 NPZ）或显式归一化元数据。
- 写文件采用临时文件加原子重命名；中断后不得留下看似完整的最终输出。

## 7. 公共 API 建议

稳定入口保持精简：

```python
from src.video_depth import VideoDepthPipeline, VideoDepthConfig

config = VideoDepthConfig.from_yaml("configs/da3_nested_rife.yaml")
pipeline = VideoDepthPipeline(config)
result = pipeline.run("input.mp4", "output_depth.mp4")
```

不要把 DA3 或 RIFE 上游私有方法暴露为本项目公共 API。配置对象负责全量校验，流水线负责资源生命周期，I/O 层负责时间轴与编码，模型适配层只处理张量和推理。

## 8. 测试门槛

每次变更先运行与改动最相关的最小测试，再扩大范围。

无需 GPU 的快速测试：

```bash
pytest -q tests/video_depth -m "not slow"
pytest -q tests/test_base.py
```

GPU 集成测试只在模型和权重可用时运行：

```bash
pytest -q tests/video_depth/test_pipeline.py --run-slow
```

最低正确性断言：

- 输入空视频、单帧、短于 stride、非整除尾段、奇数宽高、损坏视频有明确行为。
- 关键帧索引包含首尾且不重复。
- 输入输出帧数、尺寸、FPS/时间戳一致。
- `stride=1` 不加载 RIFE，结果与 DA3 Nested 基线一致。
- 关键帧在补帧前后不改变。
- 深度为非负 `float32`，单位为米，NaN/Inf 比例受控。
- 临时文件、覆盖保护和中断恢复符合预期。
- 使用 mock/fake 模型覆盖大部分单元测试，避免每次测试都下载权重或占用 GPU。

修复 bug 时必须先增加能复现该 bug 的回归测试。慢测试标记为 `slow`，不能让默认测试意外下载大型模型。

## 9. 评测设计

质量和性能必须由同一份、可版本化的配置驱动。至少保留以下基线：

1. DA3 Nested 全帧推理（质量上界，stride=1）。
2. DA3 Metric + RIFE（仓库现有高效基线）。
3. DA3 Nested + RIFE，stride 为 2/3/5 或经预实验筛选的集合。

### 质量指标

- 有真值时：AbsRel、SqRel、RMSE、RMSE-log、δ1/δ2/δ3、尺度偏差。
- 视频时序：相邻帧 warp 后误差、静态区域 flicker、深度变化分位数。
- 分开统计 DA3 关键帧与 RIFE 插值帧，避免平均值掩盖补帧退化。
- 统计无效像素率、裁剪率、边界误差，并保存少量固定可视化样例。
- 机器人数据若有传感器深度，明确有效掩码、量程和传感器空洞策略。

### 性能指标

- 端到端 wall time、稳态 FPS、首帧/首段延迟、峰值 GPU 显存、峰值 CPU 内存、GPU 利用率和输出大小。
- 分阶段记录 decode、DA3、RIFE、encode 时间。
- 模型加载和 warm-up 与稳态推理分开报告；每项至少重复 3 次，报告中位数及波动。
- 记录 GPU 型号/数量、驱动、CUDA、PyTorch、精度模式、分辨率、帧数、stride、window/overlap 和 Git commit。

### 1/2/4 张 A800

- 首选“一个 GPU 处理一个独立视频/episode”的任务级并行，以获得简单、可靠的吞吐扩展。
- 单视频默认绑定一张 GPU；只有实测单卡显存不足或单视频延迟确有收益时才引入模型并行。
- 2/4 卡测试使用完全相同的数据集和配置，报告总吞吐、单视频延迟、加速比与扩展效率。
- 明确设置设备映射，禁止所有 worker 默认落到 `cuda:0`。
- 每个 worker 独占模型实例和输出目标；主进程汇总结构化 JSON，不能让多个进程同时改写同一 metadata 文件。

## 10. 配置、复现与日志

- 默认参数只存在于 `configs/da3_nested_rife.yaml`；代码中的默认值仅用于 schema 必填或向后兼容，避免两套默认值漂移。
- 每次运行把解析后的最终配置保存到输出旁边，附输入文件哈希或稳定标识、Git commit 和环境摘要。
- 随机过程固定 seed；若使用非确定性 CUDA 算子，在结果中记录。
- 日志使用结构化字段，至少包含 video、frame/window、device、stage、elapsed、peak_vram。
- 不在日志中输出密钥、访问令牌或完整私人数据路径；错误日志保留足够上下文。

## 11. 代码与依赖规范

- 支持 Python 3.11，与仓库现有安装说明一致。
- 为公共函数和关键张量写类型标注；注释解释单位、shape、坐标系和设计原因。
- GPU 代码禁止裸 `.cuda()`；统一使用显式 `device`，以便 CPU 测试和多 GPU 调度。
- 避免调用上游模型的私有方法；若性能原因必须调用，封装在适配层并用版本检查和测试保护。
- 不在推理热路径重复加载权重、反复创建 CUDA context 或频繁进行 GPU/CPU 小张量同步。
- 新依赖必须说明用途并固定兼容范围；不要为了一个小函数引入大型框架。
- 不提交模型权重、输入数据、生成视频、benchmark 大文件、缓存或密钥。
- 不对 `third_party/` 做顺手格式化。确需修改上游代码时，保持补丁最小，并在文档记录上游版本和原因。

## 12. Agent 工作流程

1. 开始前阅读本文件、`README.md`、相关源码和测试，并检查 `git status`；保留用户已有改动。
2. 先陈述本次变更的输入/输出、单位、shape、时间轴和设备假设。
3. 优先修改最少文件；新增文件必须符合第 4 节目录，不创建临时同义实现。
4. 先写或更新测试，再实现；先跑快速测试，再决定是否运行 GPU 慢测试。
5. GPU、模型或数据不可用时，完成可验证的 CPU/mock 部分，并清楚列出未验证项；禁止伪造性能数据。
6. 完成后报告改动文件、执行命令、测试结果、已知风险和下一步。命令应可直接复制执行。
7. 未经明确要求，不下载大型权重、不修改系统 CUDA/驱动、不删除数据、不覆盖已有输出、不提交或推送 Git。

## 13. 完成定义

一项 DA3 Nested + RIFE 工作只有在以下条件同时满足时才算完成：

- 新增文件命名和路径符合本规范，且只有一个简洁主入口。
- RGB 视频可以一条命令生成逐帧对齐的绝对深度视频。
- 单位、shape、帧数、FPS/时间戳和首尾帧均通过自动测试。
- stride=1 与采样加 RIFE 两条路径都有测试，关键帧不被改写。
- 当前电脑上的可运行部分已验证；无法验证的 GPU 项明确标注。
- A800 评测可通过配置复现，并能比较 1/2/4 卡，而不改变算法默认值。
- README 中有最小安装、单命令运行和输出解释，结构化 benchmark 结果可追溯到配置和 commit。
