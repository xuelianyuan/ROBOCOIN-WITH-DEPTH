# 数据集对照表

这两个目录都符合 LeRobot 风格，但来源、相机配置和元数据完整度不一样。

## 概览

| 项目 | `leju_depth_preview` | `ai2_depth_preview` |
|---|---|---|
| 数据集来源 | Leju 机器人 | AI2 Alphabot 机器人 |
| 目录样例 | `Leju_Kuavo_4_box_storage_parcel_8_qced_hardlink_ep000000` | `AI2_Alphabot_2_stir_spoon_0_qced_hardlink_ep000000` |
| LeRobot 结构 | 是 | 是 |
| `meta/info.json` | 有 | 有 |
| `data/chunk-000/episode_*.parquet` | 有 | 有 |
| `videos/.../*.mp4` | 有 | 有 |
| `annotations/qc_report.json` | 有 | 有 |
| `annotations/camera_params.json` | 没有 | 部分样本有 |

## 关键差异

| 项目 | `leju_depth_preview` | `ai2_depth_preview` |
|---|---|---|
| RGB 相机数 | 3 路 | 4 路 |
| RGB 相机名 | `cam_front_head_rgb`、`cam_left_wrist_rgb`、`cam_right_wrist_rgb` | `cam_front_chest_rgb`、`cam_front_head_rgb`、`cam_left_wrist_rgb`、`cam_right_wrist_rgb` |
| RGB 分辨率 | `848x480` | `640x480` |
| RGB 编码 | `h264` | `av1` |
| 状态维度 | `52` | `34` |
| 动作维度 | `40` | `34` |
| 深度字段 | 已包含 `*_depth` | 已包含 `*_depth` |
| 深度编码 | `libx265 + gray12le` | `libx265 + gray12le` |

## 结论

- `leju_depth_preview` 更像 Leju 机器人的深度预览数据，结构标准，但没有单独相机参数文件。
- `ai2_depth_preview` 也是 LeRobot 格式，但相机信息更完整，分辨率和视频编码也不同。
- 如果目标是后续做几何恢复或更严谨的深度分析，`ai2_depth_preview` 通常更方便一些。

## LeRobot 数据集格式总结

LeRobot 数据集可以理解为一个多模态机器人轨迹数据集：

> 一个 `episode` 表示一次完整的机器人操作过程；`parquet` 保存结构化时序数据，`mp4` 保存相机视频，`meta` 保存数据集说明和索引。

典型目录结构如下：

```text
dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── episode_000002.parquet
├── meta/
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── info.json
│   └── tasks.jsonl
└── videos/
    └── chunk-000/
        ├── observation.images.cam_head_rgb/
        │   ├── episode_000000.mp4
        │   └── episode_000001.mp4
        ├── observation.images.cam_left_wrist_rgb/
        └── observation.images.cam_right_wrist_rgb/
```

### 核心概念

| 概念 | 含义 |
|---|---|
| `episode` | 一次完整的机器人操作示范，例如完成一次抓取、放置或整理任务 |
| `frame` | episode 中的一个时间点，通常由 `fps` 决定，例如 30 FPS 表示每秒 30 帧 |
| `observation.state` | 机器人当前状态，例如关节角、末端位姿、夹爪状态等 |
| `action` | 当前时刻执行或应该执行的动作，例如目标关节角、末端位姿增量或夹爪控制 |
| `observation.images.*` | 相机观测，可以是 RGB 视频，也可以是深度视频 |
| `task_index` | 当前 episode 对应的任务编号，通常可以在 `tasks.jsonl` 中找到任务文本 |

### 文件含义

| 文件/目录 | 作用 | 主要内容 |
|---|---|---|
| `data/` | 结构化机器人时序数据 | state、action、timestamp、frame_index 等 |
| `episode_XXXXXX.parquet` | 单个 episode 的表格数据 | 每一帧对应一行机器人状态和动作 |
| `videos/` | 相机视频数据 | 多路 RGB 或 Depth 视频 |
| `episode_XXXXXX.mp4` | 某个 episode 的某一路相机视频 | 与同编号 parquet 在时间上对应 |
| `meta/info.json` | 数据集总说明 | robot_type、fps、features、视频信息、路径模板等 |
| `meta/episodes.jsonl` | episode 索引 | 每个 episode 的长度、任务编号和数据范围 |
| `meta/episodes_stats.jsonl` | 统计信息 | state/action/depth 等字段的 mean、std、min、max |
| `meta/tasks.jsonl` | 任务说明 | task_index 与语言任务描述 |
| `chunk-000` | 数据分块 | 用于管理大规模数据集，避免单目录文件过多 |

### 数据如何对齐

同一个 episode 会同时出现在 `data/` 和 `videos/` 中：

```text
episode_000000
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/<video_key>/episode_000000.mp4
```

它们通过下面几个字段在时间上对应：

| 字段 | 作用 |
|---|---|
| `episode_index` | 表示属于哪一次操作 |
| `frame_index` | 表示 episode 内的第几帧 |
| `timestamp` | 表示当前帧对应的时间 |

也就是说，LeRobot 数据集不是简单的一堆视频，而是：

```text
Vision + Robot State + Action + Task + Time
```

组成的机器人操作轨迹。

### 推荐阅读顺序

第一次检查一个 LeRobot 数据集时，建议按这个顺序看：

1. `meta/info.json`：先确认有哪些 feature、相机、fps、state/action 维度。
2. `meta/tasks.jsonl`：确认任务文本和 `task_index`。
3. `meta/episodes.jsonl`：确认 episode 数量、长度和任务对应关系。
4. `data/chunk-000/episode_000000.parquet`：查看一条实际轨迹的结构化字段。
5. `videos/chunk-000/.../episode_000000.mp4`：检查视频是否和 parquet 帧数、时间对齐。

对深度估计任务来说，最需要关注的是 `info.json` 里的 `observation.images.*_rgb` 和 `observation.images.*_depth` 字段，以及是否存在 `camera_params.json` 这类相机标定文件。
