# 云斑按帧或按段出现的统计

仓库根目录运行，依赖 NumPy、OpenCV、Pillow 及 PATH 中的 FFmpeg/FFprobe；不运行深度模型。

```bash
python scripts/depth_refine/stat_cloud_artifacts.py \
  --dataset-roots data/lerobot/ai2_depth_preview data/lerobot/leju_depth_preview \
  --workers 8
```

默认输出到 `outputs/cloud_statistics_<北京时间>`。可用 `--output` 指定新目录，已存在的目录会报错，避免覆盖实验结果。

中断后可用相同参数和 `--output 原目录 --resume` 恢复。脚本核对输入文件清单、修改时间及参数，复用已完成的视频报告，重新扫描未完成或失败视频；其旧目录会改名保存为 `*.previous_attempt_*`。

仅扫描 `<数据集>/<任务>/videos/chunk-*/*_depth/episode_*.mp4`，摄像头名称直接从路径读取。排除 `depth_results*` 下的 baseline/refine 视频，不跨视频拼接区间。

已确认的正例可以添加 `--reference-npy 文件1.npy 文件2.npy`。脚本会输出正例得分和复核图；任一正例未过阈值时停止扫描，需检查参数。正例用于检查灵敏度，不构成准确率验证。

## 判定与统计口径

- 深度按原生 `gray12le` 或 `gray16le` 读取，禁止隐式位深转换。只缩小空间尺寸至默认宽度 320，逐帧分析全部时间轴，不跳帧。
- 每帧有效深度 P1–P99 用于归一化。检测中等尺度的灰度起伏，排除强边缘、方向单一轮廓和不足 20 像素的小连通域。
- 得分为候选纹理像素 / 全部有效深度像素。默认 `--coverage 0.02`，带通起伏阈值 `--amplitude 0.008`。
- 阳性连续 **≥2 帧**算连续区间；只有 1 帧算孤立事件。同时统计 **≥5 帧**的区间。相邻阴性或不可分析帧会切断区间，不平滑标签、不补间隙。
- “云斑数”按**时间事件**计数，事件总数 = 孤立单帧事件 + 连续区间事件，不统计同一帧多个空间斑块的个数；云斑帧数另列。
- 任务按 `<数据集>/<任务目录>` 计数：任一视角/episode 至少有一帧阳性，即为含疑似云斑任务，不合并类似名称的任务目录。视频占比 = 含疑似云斑视频 / 成功扫描视频。跨视角事件分别累计。
- `frames_in_videos_with_suspected_cloud` 是所有含云斑视频的完整帧数之和；`suspected_frames` 是其中阳性帧之和，两者不要混淆。
- 连续区间内的阳性帧占全部阳性帧 **≥80%**记为 `segment_dominant`；**≤20%**记为 `frame_dominant`；其余为 `mixed`。没有阳性为 `none_detected`。
- 有效内部区域不足图像面积 10% 时为 unknown，报告单独统计；失败视频不计入成功汇总。没有检测到不等于已证实没有云斑。
- 帧号从 0 开始，起止帧为闭区间；时间终点为 `(end_frame+1)/fps`，时长为 `帧数/fps`。VFR 视频的秒数为按平均帧率估算。
- 覆盖率阈值敏感性同时汇总默认值的 0.75、1、1.25 倍；不是分类置信度。

自然曲面、弱轮廓和真实表面纹理可能误报；云斑尺度、强度不同时也可能漏检。结果是**疑似云斑的启发式统计**，不能作为已完成人工标注的精确云斑频率，也不能用于判断模型、映射或编码的责任。

## 输出

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 中文汇总、阈值敏感性、全部视频复核图链接 |
| `summary.json` | 全部、数据集、视角汇总，任务/视频占比，含云斑视频总帧数、失败清单 |
| `tasks.csv` / `task_reports.json` | 一行/记录一个任务目录，列出任务内命中视频占比、总帧数和事件数 |
| `videos.csv` | 全部视频逐条统计：视频总帧、云斑帧、事件总数、孤立事件、连续事件、时长 |
| `videos_with_cloud.csv` | 仅列出含疑似云斑的视频，保留同样字段 |
| `video_reports.json` | 全部视频的完整报告 |
| `inventory.json` / `config.json` | 输入文件路径、大小、修改时间、实际参数、正例检查结果 |
| `videos/.../frames.csv` | 每一帧的得分、suspected、assessable、归一化范围 |
| `videos/.../segments.csv` | 连续阳性事件的起止帧和时长，包含单帧事件 |
| `videos/.../review.png` | 最高分、阈值附近、最低分可分析帧和首次阳性前后帧 |

复核图左侧为逐帧归一化深度灰度图，右侧红色为候选纹理。底部 `+/-/?` 分别为疑似/阴性/不可分析。

实现位于 `evaluation/cloud_artifacts.py`，编排位于 `pipeline/cloud_statistics.py`；共用 `io_utils.iter_video_frames`，`run_refine._decode` 也委托给这一解码器。

修复方法与 RIFE 的可行性仅做调研，见 [CLOUD_REPAIR_RESEARCH.md](CLOUD_REPAIR_RESEARCH.md)。

基础检查（此环境的系统 Python 带有 pytest，禁用不兼容的外部插件和模型 fixtures）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest \
  --confcutdir=tests/depth_refine \
  tests/depth_refine/test_cloud_statistics.py tests/test_run_refine_helpers.py -q
```
