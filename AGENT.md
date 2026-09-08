# RoboCOIN-with-Depth 开发规范

本文件定义 Codex 及其他 Coding Agent 在本仓库中进行代码修改时必须遵守的开发规范。

在进行任何代码修改之前，必须先阅读：

1. 根目录 `AGENTS.md`
2. 与任务相关的现有代码
3. `src/depth_refine/` 下与本次任务相关的模块
4. 相关测试、配置和调用方

不要在不了解现有实现、调用关系和数据流的情况下直接新增代码。

如果本文件中的规则与普通 Prompt 指令冲突，默认优先遵守本文件；只有当 Prompt **明确要求覆盖某条规则**时，才允许偏离。

---

# 1. 项目开发原则

本仓库基于现有 RoboCOIN-with-Depth 项目进行扩展。

新增的深度处理、深度优化、评估、可视化和数据集生成相关功能，应尽可能集中在：

```text
src/depth_refine/
```

对应的命令行入口应集中在：

```text
scripts/depth_refine/
```

测试应集中在：

```text
tests/depth_refine/
```

总体原则：

> 保留原仓库已有结构和行为，通过边界清晰的 `depth_refine` 模块进行增量开发，而不是为了新功能对整个原仓库进行大规模重构。

修改代码时优先：

```text
理解现有实现
    ↓
梳理真实数据流
    ↓
确认代码责任归属
    ↓
寻找已有接口
    ↓
扩展已有实现
    ↓
必要时新增模块
```

不要反过来先创建目录、类或抽象，再强行让现有代码适配它。

---

# 2. 目录与模块职责

`depth_refine` 模块原则上按照以下结构组织：

```text
src/depth_refine/
├── __init__.py
│
├── io/
├── methods/
├── evaluation/
├── geometry/
├── visualization/
└── pipeline/

scripts/
└── depth_refine/

tests/
└── depth_refine/
```

各模块职责如下。

| 模块                      | 主要职责                      | 不应包含          |
| ----------------------- | ------------------------- | ------------- |
| `io/`                   | 数据集、视频、metadata 等数据读写     | 算法逻辑、绘图逻辑     |
| `methods/`              | 深度处理、baseline、refine 等算法  | CLI、数据集遍历、绘图  |
| `evaluation/`           | 指标、统计、方法比较                | 深度优化算法实现      |
| `geometry/`             | 相机模型、坐标转换、点云等几何计算         | 数据集专用 I/O     |
| `visualization/`        | 曲线、深度图、preview、Web export | 深度优化算法        |
| `pipeline/`             | 组织完整处理流程                  | 底层数学实现、具体算法细节 |
| `scripts/depth_refine/` | CLI 参数解析和调用 pipeline      | 核心实现          |

如果某项功能无法明确归属于上述模块：

1. 先检查仓库中是否已经存在负责该功能的模块；
2. 再判断是否应该扩展现有模块；
3. 如果确实属于新的职责，再提出新的模块；
4. 不要为了方便直接塞入某个不相关文件。

新增模块必须具有**单一且明确的职责**。

---

# 3. 架构与依赖方向

代码依赖应尽量保持单向。

推荐的整体依赖方向：

```text
scripts
   ↓
pipeline
   ↓
methods / evaluation / visualization
   ↓
io / geometry
```

这里表达的是总体架构方向，而不是要求所有模块形成严格线性关系。

允许的典型依赖：

```text
scripts → pipeline

pipeline → io
pipeline → methods
pipeline → evaluation
pipeline → geometry
pipeline → visualization

methods → geometry

evaluation → geometry
visualization → geometry
```

应避免：

```text
io → pipeline

geometry → pipeline

methods → scripts

evaluation → scripts

底层模块 → CLI
```

严禁为了方便造成循环依赖。

如果出现：

```text
A imports B
B imports A
```

不要通过局部 import、运行时 import 等方式简单掩盖问题。

首先检查模块边界是否划分错误。

---

# 4. Pipeline 设计原则

`pipeline/` 负责**流程编排（orchestration）**，而不是承担所有业务逻辑。

一个典型 pipeline 可以负责：

```text
读取数据
    ↓
选择处理方法
    ↓
执行 depth processing
    ↓
执行 evaluation
    ↓
生成 visualization
    ↓
保存结果
```

但以下逻辑不应该直接写在 pipeline 中：

```text
视频解码算法
深度 refine 数学实现
点云投影公式
指标具体计算公式
绘图具体实现
```

这些逻辑应该分别下沉到：

```text
io/
methods/
geometry/
evaluation/
visualization/
```

Pipeline 的主要职责是：

> 决定“什么时候调用谁”，而不是实现“每一步具体怎么算”。

---

# 5. Original / Baseline / Refine 统一接口

Original、Baseline、Refine 等不同处理方式应尽量使用同一套处理流程。

推荐：

```text
                    ┌→ original
dataset → pipeline ─┼→ baseline
                    └→ refine
```

而不是：

```text
dataset → original_pipeline

dataset → baseline_pipeline

dataset → refine_pipeline
```

不同方法之间的差异应该通过：

* 配置
* 参数
* 方法对象
* 统一接口
* strategy / method selection

进行选择。

不要因为算法不同就复制整套 pipeline。

---

# 6. 串行与并行统一实现

串行处理和并行处理不得维护两套独立业务实现。

避免：

```text
process_dataset.py
process_dataset_parallel.py
```

内部包含大量重复逻辑。

优先：

```python
process_dataset(..., num_workers=1)
```

或者等价的配置方式。

例如：

```text
num_workers = 1
    ↓
串行执行

num_workers > 1
    ↓
并行执行
```

并行化应该改变**执行方式**，而不是复制一份业务逻辑。

---

# 7. I/O 统一管理

所有与以下内容相关的逻辑：

* depth video decoding
* RGB/depth video reading
* LeRobot dataset reading
* metadata reading
* dataset writing
* serialization

应优先放在：

```text
src/depth_refine/io/
```

不要在不同 pipeline、script 或 evaluation 文件中重复实现 decoder。

在创建新的 I/O 函数之前，必须先搜索现有实现。

如果已有 decoder：

> 优先扩展现有 decoder，而不是创建第二套实现。

---

# 8. Camera 配置必须保持单一数据源

Camera 名称、camera list、camera mapping 等信息不得散落在多个 Python 文件中重复定义。

禁止出现：

```python
CAMERAS = [...]
```

在多个模块分别维护。

Camera 配置必须具有一个明确的 Single Source of Truth。

具体位置应根据仓库现有结构决定，例如：

```text
io/cameras.py
```

或：

```text
config
```

不要在没有检查现有配置体系之前自行创建新的配置机制。

其他模块应该 import / 读取统一配置，而不是复制 camera list。

---

# 9. 配置管理

以下运行参数不得散落在实现代码中：

* dataset path
* output path
* camera list
* threshold
* worker 数量
* refinement 参数
* evaluation 参数
* visualization 参数

禁止硬编码机器相关路径，例如：

```python
dataset_path = "/home/user/data/..."
```

应通过仓库已有配置方式管理，例如：

```text
CLI arguments
configuration file
dataclass
environment variables
```

具体采用哪一种方式：

> 先检查仓库当前使用的配置模式，再保持一致。

不要因为新增一个参数就额外创造一套独立配置系统。

---

# 10. 数据安全

原始 LeRobot 数据集默认视为：

```text
READ ONLY
```

除非任务明确要求，否则：

* 不得原地修改原始 dataset；
* 不得覆盖原始 depth video；
* 不得静默删除原始文件；
* 不得把 refinement 结果直接写回原始数据；
* 不得因为重复运行脚本而破坏已有数据。

Refined dataset、实验结果和中间产物应该写入独立输出位置。

例如：

```text
data/
outputs/
experiments/
```

具体目录以仓库现有约定为准。

可能产生覆盖行为时，应显式处理。

---

# 11. Generated Data 不得进入 src

以下内容不得放入：

```text
src/
```

包括：

* dataset
* generated depth video
* preview images
* logs
* checkpoints
* statistics output
* plots
* HTML/Web export
* temporary files
* experiment results

`src/` 只保存源代码及必要的代码资源。

实验产物应写入仓库已有的数据/输出目录。

如果仓库尚无明确约定，可优先考虑：

```text
data/
outputs/
experiments/
```

并根据需要加入 `.gitignore`。

---

# 12. Scripts 规则

`scripts/depth_refine/` 中的文件只作为程序入口。

推荐结构：

```text
parse CLI arguments
        ↓
load / build config
        ↓
call pipeline
        ↓
display final result
```

Script 中可以包含：

* argparse / CLI parser
* config loading
* 用户输入处理
* 调用 pipeline
* CLI 用户提示

不得包含：

* depth refinement 算法
* video decoder
* dataset 核心处理
* point cloud 数学
* evaluation 公式
* 大量业务逻辑

如果 script 开始快速增长，首先检查其中逻辑是否应该移动到 `src/depth_refine/`。

---

# 13. 代码复用原则

新增代码之前必须搜索仓库。

重点检查：

* 是否已有类似函数；
* 是否已有 decoder；
* 是否已有 dataset wrapper；
* 是否已有 camera configuration；
* 是否已有 geometry helper；
* 是否已有 evaluation metric；
* 是否已有 pipeline；
* 是否已有 base class / interface。

优先级：

```text
复用已有实现
    ↓
扩展已有接口
    ↓
提取公共逻辑
    ↓
新增独立实现
```

避免：

```text
function()
function_v2()
function_new()
function_parallel()
function_refine()
```

如果这些函数本质上只是参数不同，应优先统一接口。

---

# 14. 按“变化原因”划分模块

不要仅仅因为两段代码“看起来属于同一个处理阶段”就放在一起。

判断模块边界时，应考虑：

> 这两段代码是否因为同一种原因发生变化？

例如：

```text
camera decoding
```

通常因为：

```text
数据格式 / codec / dataset format
```

变化。

而：

```text
depth refinement
```

通常因为：

```text
算法 / 参数 / 模型
```

变化。

即使它们在流水线上相邻，也不应该放在同一个模块。

判断是否需要拆分时，可以问：

> “修一个 bug 或增加一种方法时，这两块代码通常需要一起修改吗？”

如果变化原因不同，应倾向于保持模块独立。

---

# 15. Coding Conventions

## Python 版本

不要凭经验假设 Python 版本。

修改前检查仓库实际环境定义，例如：

```text
pyproject.toml
setup.cfg
setup.py
environment.yml
requirements.txt
```

以仓库实际配置为准。

---

## Type Hints

所有新增 public：

* function
* method
* class interface

应提供合理的 type hints。

例如：

```python
def load_depth_video(path: Path) -> np.ndarray:
    ...
```

---

## Docstring

所有新增 public：

* module
* function
* class
* method

应提供 docstring。

Docstring 风格应优先遵循当前文件或仓库已有风格。

不要为了统一格式而无关地修改整个旧文件。

---

## Exception

禁止：

```python
try:
    ...
except:
    ...
```

应捕获明确异常：

```python
try:
    ...
except FileNotFoundError:
    ...
```

如果确实需要捕获 `Exception`，必须有明确理由，并正确记录或重新抛出。

---

## Logging

核心模块中不要随意使用：

```python
print(...)
```

优先使用仓库已有 logging 机制。

CLI 用户可见输出可以放在：

```text
scripts/depth_refine/
```

如果仓库目前没有统一 logger，不要仅为了满足本规则自行构建复杂 logging framework；优先沿用现有风格，并在必要时进行最小改进。

---

## Naming

遵循 Python 常规命名：

```text
function / variable
    ↓
snake_case

Class
    ↓
PascalCase

CONSTANT
    ↓
UPPER_CASE
```

模块名称应直接表达其职责。

---
# 16. 中文注释与代码解释规范

本仓库新增或修改的核心代码，应优先采用**中文注释进行逻辑解释**。

注释的主要目标不是重复代码本身，而是帮助开发者快速理解：

* 这一段代码负责什么；
* 输入数据当前处于什么状态；
* 为什么需要执行这一步；
* 数据经过这一步之后发生了什么变化；
* 某些参数、shape、单位或坐标系代表什么；
* 为什么采用当前实现而不是看起来更直接的实现；
* 哪些地方存在边界条件、兼容性要求或历史原因。

总体原则：

> 注释应该解释“意图、数据流、约束和原因”，而不仅仅是把 Python 代码翻译成中文。

---

## 16.1 注释语言

项目中新增加的说明性注释，默认使用中文。

例如推荐：

```python
# 读取当前 episode 的深度视频，并统一转换为毫米单位。
depth_frames = load_depth_video(depth_path)
```

而不是：

```python
# Load depth video.
depth_frames = load_depth_video(depth_path)
```

对于以下内容可以保留英文：

* 类名；
* 函数名；
* API 名称；
* library 名称；
* tensor / ndarray；
* RGB / Depth；
* intrinsic / extrinsic；
* world frame / camera frame；
* baseline / refine；
* LeRobot；
* codec；
* dtype；
* shape。

不要为了全部翻译成中文而创造不常用或容易造成歧义的中文术语。

例如推荐：

```python
# 将 depth 从 camera frame 投影到 3D point cloud。
```

而不是强行写成难以阅读的完全中文术语。

---

## 16.2 注释应按逻辑阶段分区

对于较长函数、pipeline、数据处理过程，不要只在零散代码行后添加短注释。

优先按照实际数据流，将函数划分成若干清晰的逻辑区域。

推荐格式：

```python
# ============================================================
# 1. 读取当前 episode 的输入数据
# ============================================================

...

# ============================================================
# 2. 对深度帧进行预处理
# ============================================================

...

# ============================================================
# 3. 执行 refine 方法
# ============================================================

...

# ============================================================
# 4. 计算统计指标
# ============================================================

...

# ============================================================
# 5. 保存处理结果
# ============================================================

...
```

分区标题应描述**业务阶段**，而不是描述具体 Python 操作。

推荐：

```python
# ============================================================
# 2. 将无效深度区域转换为统一 mask
# ============================================================
```

不推荐：

```python
# ============================================================
# 2. 创建数组
# ============================================================
```

因为“创建数组”只是实现细节，并不能帮助理解完整数据流。

---

## 16.3 分区粒度

不要把每 2～3 行代码都划成一个分区。

一个分区通常应该对应一个相对完整的处理阶段，例如：

```text
读取数据
↓
数据标准化
↓
构造 mask
↓
执行算法
↓
计算指标
↓
保存结果
```

一个好的判断方法是：

> 如果把这段代码折叠起来，分区标题是否足以让我知道这一阶段在干什么？

如果答案是“是”，说明分区粒度比较合理。

---

## 16.4 注释重点解释“为什么”

以下代码：

```python
depth = depth.astype(np.float32)
```

低价值注释：

```python
# 转换成 float32。
depth = depth.astype(np.float32)
```

因为代码本身已经说明了这一点。

更好的注释：

```python
# 后续会执行除法和插值运算，因此先转换为 float32，
# 避免 uint16 运算产生截断或溢出。
depth = depth.astype(np.float32)
```

同样：

```python
depth[depth == 0] = np.nan
```

不要只写：

```python
# 把 0 改成 NaN。
```

应解释：

```python
# 数据集中 0 表示传感器没有有效深度观测。
# 使用 NaN 表示缺失值，便于后续统计时通过 nanmean / nanmedian 自动忽略。
depth[depth == 0] = np.nan
```

---

## 16.5 数据 shape 必须在关键位置说明

本项目涉及大量：

* image；
* depth；
* video；
* point cloud；
* camera；
* episode；
* frame；

因此在 shape 容易产生歧义的位置，应明确写出数据维度含义。

例如：

```python
# depth_frames.shape = (T, H, W)
# T: 当前 episode 的帧数
# H/W: 深度图高度和宽度
depth_frames = load_depth_video(path)
```

对于 RGB：

```python
# rgb_frames.shape = (T, H, W, 3)
# 最后一维按照 RGB 顺序存储三个颜色通道。
```

对于点云：

```python
# points.shape = (N, 3)
# 每一行为一个三维点，列顺序为 [x, y, z]。
```

如果 shape 在进入某个函数前发生变化，应解释转换前后：

```python
# 输入 depth shape: (T, H, W)
# 为了使用逐帧接口，这里取出单帧后变为 (H, W)。
frame = depth_frames[frame_idx]
```

不要假设读者会自动知道某一维代表什么。

---

## 16.6 dtype 必须在重要转换处说明

深度数据中的 dtype 往往决定：

* 数值范围；
* 单位；
* 精度；
* 是否允许 NaN；
* OpenCV / NumPy 是否能够正确处理。

因此发生重要 dtype 转换时应说明原因。

例如：

```python
# 原始深度以 uint16 保存，单位为 mm。
# refine 中需要使用 NaN 和浮点运算，因此转换为 float32。
depth = depth.astype(np.float32)
```

保存前：

```python
# 输出视频仍沿用原数据格式，保存前将单位为 mm 的结果裁剪到
# uint16 可表示范围，再转换回 uint16。
depth_uint16 = np.clip(depth, 0, np.iinfo(np.uint16).max).astype(np.uint16)
```

---

## 16.7 单位必须明确

涉及物理量时，注释或变量命名必须能够明确单位。

重点包括：

* depth：mm / m；
* translation：m；
* angle：rad / degree；
* timestamp：s / ms / ns；
* camera intrinsic 参数；
* point cloud 坐标；
* threshold。

推荐：

```python
# 深度值单位：mm
depth_mm = ...
```

或者：

```python
# 将原始毫米深度转换成米，供点云投影使用。
depth_m = depth_mm / 1000.0
```

不推荐：

```python
depth = depth / 1000
```

因为读者无法立即判断：

* 为什么除以 1000；
* 原始单位是什么；
* 转换之后是什么单位。

---

## 16.8 坐标系必须详细注释

`geometry/` 中涉及：

* world frame；
* camera frame；
* robot frame；
* point cloud；
* extrinsic；
* pose；
* rotation；
* translation；

时，应优先详细标明坐标系方向。

例如：

```python
# 将 camera frame 中的点转换到 world frame。
#
# p_camera: (N, 3)
# R_wc: camera frame -> world frame 的旋转矩阵
# t_wc: camera 原点在 world frame 中的位置
#
# 采用：
#     p_world = R_wc @ p_camera + t_wc
#
# 注意不要与 world -> camera 的 extrinsic 定义混淆。
points_world = points_camera @ rotation_wc.T + translation_wc
```

对于四元数，还应说明顺序：

```python
# 四元数使用 [w, x, y, z] 顺序。
```

或者：

```python
# scipy Rotation 接口使用 [x, y, z, w]，
# 因此这里需要从项目中的 [w, x, y, z] 顺序重新排列。
```

坐标系相关代码属于高风险区域，应优先写详细注释。

---

## 16.9 Camera 参数应解释含义

例如：

```python
# 相机内参矩阵：
#
#     [fx  0  cx]
# K = [ 0 fy  cy]
#     [ 0  0   1]
#
# fx/fy: x/y 方向焦距（pixel）
# cx/cy: 主点坐标（pixel）
```

对于投影：

```python
# 根据 pinhole camera model：
#
# X = (u - cx) * Z / fx
# Y = (v - cy) * Z / fy
# Z = depth
#
# 最终得到 camera frame 中的三维坐标。
```

这类数学逻辑不要只留下代码，应提供必要公式和变量含义。

---

## 16.10 算法代码必须解释输入、核心步骤和输出

在 `methods/` 中新增算法时，至少要让读者能够通过注释理解：

```text
输入是什么
↓
无效值怎么处理
↓
算法核心思想是什么
↓
关键参数是什么意思
↓
输出是什么
```

例如：

```python
# ============================================================
# 1. 构造无效深度 mask
# ============================================================
#
# 原始数据中：
#   depth == 0 表示无有效观测。
#
# 后续只对这些位置执行补全，避免修改原本有效的深度值。
invalid_mask = depth == 0


# ============================================================
# 2. 使用邻域有效像素估计缺失深度
# ============================================================
#
# 对每个无效像素，搜索 radius 范围内的有效深度。
# 使用 median 而不是 mean，可以降低边缘位置异常值对结果的影响。
...
```

不要要求读者必须完整阅读算法实现之后才能猜出算法逻辑。

---

## 16.11 Baseline / Refine 差异必须解释

如果 baseline 和 refine 共享统一接口，应在关键差异位置说明为什么结果不同。

例如：

```python
# baseline 只根据当前帧空间邻域进行补全。
#
# refine 在 baseline 基础上额外利用时序一致性，
# 因此这里需要同时读取前后帧。
```

不要通过复制代码来“解释差异”。

应通过注释 + 参数 + 统一接口明确不同方法的行为。

---

## 16.12 Pipeline 注释重点解释数据流

`pipeline/` 中的注释重点应该是：

* 当前数据从哪里来；
* 传给了哪个模块；
* 返回结果是什么；
* 下一阶段如何使用。

例如：

```python
# ============================================================
# 3. 执行当前配置指定的 depth method
# ============================================================
#
# method 只负责单帧/单 episode 的算法处理，
# dataset 遍历和结果保存仍由 pipeline 管理。
refined_depth = method.process(depth_frames)
```

这里不需要重新解释 `method.process()` 内部算法。

Pipeline 注释应站在更高一层描述流程。

---

## 16.13 I/O 注释重点解释数据格式

`io/` 中应重点解释：

* 文件格式；
* codec；
* shape；
* dtype；
* 单位；
* metadata；
* 文件与 episode / camera 的映射关系。

例如：

```python
# 一个 depth video 对应：
#
# dataset/
# └── episode_xxx/
#     └── camera_name/
#         └── depth.mp4
#
# decoder 输出统一为：
#     shape = (T, H, W)
#     dtype = uint16
#     unit  = mm
```

具体目录格式应以仓库真实结构为准，不能凭空假设。

---

## 16.14 Evaluation 注释重点解释指标含义

不要只写：

```python
mae = np.mean(np.abs(pred - target))
```

应解释指标测量什么。

例如：

```python
# MAE 表示 refine 深度与参考深度之间的平均绝对误差。
# 单位与输入 depth 保持一致；当前输入为 mm，因此 MAE 也是 mm。
mae = np.mean(np.abs(pred - target))
```

对于 mask：

```python
# 只在 reference 有效的位置计算指标。
# 无效 reference 没有真实深度意义，因此不应参与误差统计。
valid_mask = reference > 0
```

---

## 16.15 Visualization 注释重点解释展示目的

绘图代码中，应解释：

* 图想比较什么；
* x/y 轴含义；
* 是否归一化；
* 是否共享颜色范围；
* 为什么采用当前显示范围。

例如：

```python
# Original / Baseline / Refine 使用相同的 vmin/vmax，
# 避免因为每张图自动缩放而造成视觉上的错误比较。
```

这种注释比：

```python
# 设置 vmin/vmax
```

更有价值。

---

## 16.16 并行代码必须解释共享与隔离边界

涉及 multiprocessing / concurrent futures / workers 时，应解释：

* worker 接收什么；
* worker 返回什么；
* 哪些数据是只读的；
* 哪些数据不能跨进程共享；
* 最终由谁写文件。

例如：

```python
# worker 只负责计算，不直接写最终 dataset。
#
# 这样可以避免多个进程同时修改同一输出文件，
# 最终写入操作统一由主进程完成。
```

如果某个对象不能 pickle，也应说明原因：

```python
# VideoCapture 对象不能安全地跨进程传递，
# 因此每个 worker 内部独立创建 decoder。
```

---

## 16.17 边界条件必须写清楚

以下情况应优先添加注释：

* 空 episode；
* video 只有 1 帧；
* camera 缺失；
* depth 全为 0；
* 不同 camera 帧数不同；
* NaN / Inf；
* 文件不存在；
* output 已存在；
* worker 数量为 1；
* invalid frame；
* unsupported codec。

例如：

```python
# 某些 episode 可能缺少指定 camera。
# 这里选择跳过该 camera，而不是让整个 dataset processing 失败。
```

如果选择抛异常，也应说明原因。

---

## 16.18 Magic Number 禁止无解释出现

禁止：

```python
valid = depth < 10000
```

至少应改成：

```python
# 深度超过 10 m 视为无效观测。
MAX_VALID_DEPTH_MM = 10_000
valid = depth < MAX_VALID_DEPTH_MM
```

更推荐将其放入统一 config，如果该值属于可调参数。

任何具有业务含义的数字都不应仅以裸数字形式散落。

---

## 16.19 临时 workaround 必须明确标注

如果因为：

* 第三方库 bug；
* codec 限制；
* LeRobot 版本兼容；
* platform 问题；
* legacy behavior；

必须使用 workaround，应明确说明。

推荐：

```python
# WORKAROUND:
# 当前 xxx 版本在读取 16-bit depth video 时会返回错误 dtype，
# 因此这里显式转换。
#
# 如果未来升级依赖后确认问题已经解决，可以删除此兼容逻辑。
```

必要时可加入相关 issue / documentation 名称，但不要随意粘贴大量外部链接。

---

## 16.20 TODO 必须具体

避免：

```python
# TODO: improve this
```

推荐：

```python
# TODO:
# 当前逐帧读取会频繁创建 decoder。
# 后续可以在确认线程安全后，将 decoder 生命周期提升到 episode 级别。
```

TODO 应说明：

```text
当前问题是什么
为什么暂时没有解决
未来应该往什么方向处理
```

不要留下没有上下文的 TODO。

---

## 16.21 不要写无意义逐行注释

禁止这种风格：

```python
# 创建列表
results = []

# 遍历所有 frame
for frame in frames:

    # 调用函数
    result = process(frame)

    # 添加到列表
    results.append(result)
```

除非这些操作本身具有特殊原因，否则这些注释没有信息增量。

简单代码应让代码本身表达含义。

推荐：

```python
# 对 episode 内所有帧使用同一方法处理，
# 保持输出顺序与原视频完全一致。
results = [process(frame) for frame in frames]
```

---

## 16.22 注释不能替代清晰代码

如果一段代码需要非常长的注释才能解释：

```python
a = ...
b = ...
c = ...
```

首先考虑：

* 变量命名是否清晰；
* 是否应该拆函数；
* 是否应该抽取数据结构；
* 是否存在过度复杂逻辑。

优先：

```python
valid_depth_mask = ...
camera_points = ...
world_points = ...
```

而不是：

```python
a = ...
b = ...
c = ...
```

然后依赖大量注释解释 `a/b/c` 是什么。

原则：

> 好代码负责表达“是什么”，注释负责解释“为什么”。

---

## 16.23 Public Function Docstring 要比普通注释更完整

新增 public function 的 docstring 应尽量包含：

* 功能；
* Args；
* Returns；
* 重要 shape；
* dtype；
* unit；
* 异常情况；
* 特殊约束。

例如：

```python
def project_depth_to_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """将深度图投影为 camera frame 下的三维点云。

    Args:
        depth:
            深度图，shape 为 ``(H, W)``。
            dtype 可以为 float32 或 float64，单位为米。
        intrinsics:
            对应相机的内参参数。

    Returns:
        shape 为 ``(N, 3)`` 的点云，其中每行为 ``[x, y, z]``，
        坐标位于 camera frame 下，单位为米。

    Notes:
        输入中的 NaN 和非正深度会被视为无效像素，不包含在返回点云中。
    """
```

Docstring 具体格式仍应优先遵循当前仓库已有风格。

---

## 16.24 Private Helper 不要求机械写长 Docstring

对于非常简单的 private helper，例如：

```python
def _normalize_path(...):
```

如果函数职责非常明显，可以只写简短 docstring 或必要注释。

不要因为“要求详细注释”而给所有 3 行 helper 写十几行说明。

详细程度应与：

```text
逻辑复杂度
+
出错风险
+
领域知识要求
```

成正比。

---

## 16.25 高风险代码需要更详细注释

以下代码属于本项目中的高风险区域，应优先详细解释：

1. camera intrinsic / extrinsic；
2. coordinate transform；
3. quaternion reorder；
4. point-cloud projection；
5. depth unit conversion；
6. uint16 / float conversion；
7. invalid depth handling；
8. video codec / decoding；
9. multiprocessing；
10. dataset writing；
11. original → refined dataset mapping；
12. frame synchronization；
13. statistical masking；
14. baseline/refine 差异；
15. overwrite / output policy。

在这些区域中：

> 宁可多写几行有信息量的中文解释，也不要依赖读者自行推测。

---

## 16.26 推荐的函数内部注释结构

对于中等以上复杂度函数，优先采用：

```python
def process_episode(...):
    """处理单个 episode。"""

    # ============================================================
    # 1. 校验输入和准备路径
    # ============================================================
    # 说明输入数据应该满足什么条件，以及为什么需要提前校验。
    ...

    # ============================================================
    # 2. 加载 episode 数据
    # ============================================================
    # 说明读取哪些 camera / metadata，以及得到的数据格式。
    ...

    # ============================================================
    # 3. 统一数据格式
    # ============================================================
    # 说明 dtype / shape / unit 的转换。
    ...

    # ============================================================
    # 4. 执行 depth method
    # ============================================================
    # 这里只负责编排，具体算法由 methods/ 实现。
    ...

    # ============================================================
    # 5. 计算评估结果
    # ============================================================
    ...

    # ============================================================
    # 6. 写入独立输出目录
    # ============================================================
    # 明确说明不会覆盖原始 dataset。
    ...
```

这是一种推荐结构，而不是要求所有函数机械拥有六个分区。

应根据实际数据流调整。

---

## 16.27 简单函数不要过度分区

例如：

```python
def is_valid_depth(depth: np.ndarray) -> np.ndarray:
    """返回有效深度 mask。"""

    # 非正值和 NaN 都不作为有效深度参与后续处理。
    return np.isfinite(depth) & (depth > 0)
```

这种函数不需要写：

```text
1. 读取
2. 判断
3. 返回
```

注释详细并不意味着注释数量越多越好。

---

## 16.28 修改旧代码时的注释原则

修改已有文件时：

1. 优先保持原文件整体风格；
2. 对本次修改涉及的复杂区域补充必要中文解释；
3. 不要为了统一注释语言而一次性重写整个文件；
4. 不要仅为了加注释造成巨大的无关 diff；
5. 对明显容易误解的旧逻辑，可以在本次修改范围内补充解释。

采用渐进式改善。

---

## 16.29 注释必须与代码保持一致

修改代码后必须同步检查相关注释。

以下情况属于错误：

```python
# depth 单位为 mm
depth = depth / 1000.0
```

如果下面代码已经把它转换成 m，那么原注释可能已经不再成立。

Agent 修改：

* shape；
* dtype；
* unit；
* algorithm；
* path structure；
* method behavior；

时，应同时检查相关 docstring 和注释。

> 错误注释比没有注释更危险。

---

## 16.30 Self-Check 中加入注释检查

任务结束前，还必须检查：

* [ ] 核心新增代码具有合理的中文解释。
* [ ] 中等以上复杂度函数已经按逻辑阶段进行中文分区。
* [ ] 注释主要解释意图和原因，而不是逐行翻译代码。
* [ ] 关键 tensor / ndarray 的 shape 已在容易混淆的位置说明。
* [ ] 深度、坐标、时间等物理量的单位明确。
* [ ] 坐标变换明确说明 source frame 和 target frame。
* [ ] quaternion 顺序如果存在歧义，已经明确标注。
* [ ] 重要 dtype 转换已经解释原因。
* [ ] Magic Number 已提取或解释。
* [ ] 边界条件和异常处理具有必要说明。
* [ ] workaround / TODO 都有具体上下文。
* [ ] 修改代码后相关注释和 docstring 仍然准确。
* [ ] 没有为了“详细注释”产生大量无信息价值的逐行注释。

# 17. 测试规范

Depth refine 相关测试统一放在：

```text
tests/depth_refine/
```

建议与源码结构对应：

```text
tests/depth_refine/
├── io/
├── methods/
├── evaluation/
├── geometry/
├── visualization/
└── pipeline/
```

新增 `methods/` 算法时，应增加对应 unit test。

新增 `io/` 功能时，应使用：

* synthetic data
* minimal fixture
* 小型测试视频/数据

不要依赖完整真实 dataset 才能运行基础测试。

不得提交大型 binary fixture。

Bug fix 在条件允许时应增加 regression test。

---

# 18. 修改前必须执行的检查

开始修改前，Agent 应：

1. 阅读本 `AGENTS.md`；
2. 阅读任务相关模块；
3. 搜索已有类似实现；
4. 查找调用该代码的 caller；
5. 检查相关测试；
6. 理解一次完整数据流；
7. 判断代码应该属于哪个模块；
8. 检查是否会造成反向依赖或循环依赖。

不要仅根据文件名猜测代码职责。

应该根据：

```text
实际数据流
+
变化原因
+
依赖方向
```

共同判断。

---

# 19. 修改策略

默认采用：

> Smallest Coherent Change

即：

> 完成需求所需要的最小、完整、结构合理的修改。

不要因为看到旧代码“不够漂亮”就顺手重构。

不要在一个 feature change 中同时进行：

```text
feature
+
repository-wide refactor
+
rename
+
format cleanup
```

除非任务明确要求。

对于 legacy code：

* 保持已有行为；
* 优先渐进式修改；
* 尽量保持兼容；
* 新功能与无关旧代码分离。

如果发现值得重构的问题，但当前任务并不需要：

> 指出问题，而不是自动扩大修改范围。

---

# 20. 新建文件规则

创建新文件之前必须回答：

1. 为什么现有模块不能承担这个职责？
2. 是否已有类似实现？
3. 新文件的单一职责是什么？
4. 谁会依赖它？
5. 它会依赖谁？
6. 是否会形成重复实现？
7. 是否会产生新的平行 pipeline？

如果无法明确回答，应优先继续理解现有代码，而不是立即创建文件。

---

# 21. 完成任务后的验证

修改完成后，至少检查：

```text
代码是否能 import
        ↓
相关 unit tests
        ↓
相关 integration tests
        ↓
必要时完整 test suite
```

优先先运行与修改直接相关的测试，以快速发现问题。

任务结束前，应尽可能运行仓库现有测试。

如果因为：

* 缺少 dataset
* GPU 环境
* 外部模型
* dependency
* hardware
* runtime 限制

无法运行某些测试，必须明确说明：

```text
哪些测试没有运行
为什么没有运行
已经运行了哪些替代检查
```

不得在未运行测试时声称：

```text
all tests pass
```

---

# 22. Agent 工作方式

Agent 在本仓库中工作时：

### 应该

* 先读代码，再修改；
* 先搜索已有实现，再新增；
* 理解 caller 和 dependency；
* 保持修改范围最小；
* 保持模块单一职责；
* 保持依赖方向清晰；
* 复用已有接口；
* 优先参数化差异；
* 保护原始数据；
* 修改后运行相关测试。

### 不应该

* 猜测仓库结构；
* 未搜索就创建重复 helper；
* 创建第二套 decoder；
* 创建 parallel 专用业务实现；
* 为 baseline/refine 创建独立 pipeline；
* 把算法塞进 scripts；
* 把大量实现塞进 pipeline；
* 把数据放进 src；
* 硬编码机器路径；
* 顺手大规模重构无关代码。

---

# 23. 完成任务前 Self-Check

Agent 在结束任务之前必须对照以下清单自查：

* [ ] 修改位于正确模块。
* [ ] 已阅读相关现有实现和调用方。
* [ ] 新增文件前已确认不存在相同职责模块。
* [ ] 没有引入重复 decoder。
* [ ] 没有重复定义 camera list。
* [ ] 没有硬编码 dataset path。
* [ ] 没有创建 serial / parallel 两套业务实现。
* [ ] 没有为 original / baseline / refine 创建平行 pipeline。
* [ ] `pipeline/` 只负责 orchestration。
* [ ] `scripts/` 中没有新增核心业务逻辑。
* [ ] `src/` 中没有生成数据、日志、checkpoint 或实验结果。
* [ ] 原始 LeRobot dataset 没有被原地修改。
* [ ] 新 public API 具有合理 type hints。
* [ ] 新 public API 具有 docstring。
* [ ] 没有引入不必要的新 dependency。
* [ ] 没有产生明显循环依赖。
* [ ] 已运行与修改相关的测试。
* [ ] 新功能具有必要测试。
* [ ] 未运行的测试及原因已经明确说明。

---

# 24. Agent 最终回复要求

完成代码修改后，最终回复应简要说明：

### 修改内容

说明实现了什么。

### 主要修改文件

列出核心文件及其职责。

### 架构影响

说明是否：

* 新增模块；
* 修改接口；
* 改变 dependency；
* 改变 dataset format。

如果没有，也可以明确说明。

### 验证

说明实际执行了哪些：

```text
tests
lint
type check
manual check
```

以及结果。

### 未解决问题

如果存在：

* 环境限制；
* 未运行测试；
* 潜在技术债；
* 后续建议；

应明确说明。

不要隐藏失败的测试或环境问题。

---

# 25. 推荐 Prompt 模板

使用 Codex 修改本仓库时，可以采用：

```text
先阅读根目录 AGENTS.md，以及本次任务涉及的
src/depth_refine/ 相关代码、调用方和测试。

先理解现有数据流和接口，不要直接创建新文件。

这次只实现：

<任务描述>

要求：
1. 遵守 AGENT.md 的模块职责和依赖方向；
2. 优先扩展现有接口，不创建重复实现；
3. 不创建新的平行 pipeline；
4. 不复制已有 video decode / camera config；
5. 不修改原始 LeRobot dataset；
6. 不在 scripts 中实现核心逻辑；
7. 保持修改范围最小。

完成后：
1. 运行相关测试；
2. 对照 AGENTS.md Self-Check；
3. 汇报修改文件、架构影响、测试结果和未解决问题。
```

---

# 26. 核心原则总结

当不确定代码应该怎么组织时，按照以下顺序判断：

```text
① 真实数据流
      ↓
代码实际上是怎么运行的？

② 变化原因
      ↓
这些代码为什么会一起变化？

③ 模块职责
      ↓
谁应该拥有这段逻辑？

④ 依赖方向
      ↓
谁允许依赖谁？

⑤ 现有接口
      ↓
能否扩展，而不是复制？

⑥ 最小修改
      ↓
怎样以最小范围完成需求？
```

最终目标不是追求目录数量或形式上的“架构漂亮”，而是：

> **让代码职责清晰、依赖可控、实现可复用、修改影响范围有限，并避免随着实验增加不断产生重复和平行实现。**
