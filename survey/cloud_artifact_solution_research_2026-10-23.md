# 基于《深度转化视频云斑问题说明》的方案调研

_2026-10-23_

## 1. 判断

报告的证据已经足够把问题分成两层：

1. **上游不一致**：DA3METRIC-LARGE 对相隔 5 帧、几乎相同的静态区域给出了 13–20 mm 的主导平移，叠加约 2–3 mm 的位置相关起伏。
2. **云斑生成机制**：RIFE 的 mask 在中间时刻由偏向前帧切换到偏向后帧；当两张深度本来差十几毫米时，mask 切换就被显式写成了移动斑块。报告中 98% 的跳变像素是 mask 项大于 warp 项，这已经排除了“光流主要把深度搬错位置”的解释。

因此，以下方向不应继续作为主线：加大高斯模糊、只调 RIFE mask、只减小光流位移，或把步长从 5 改成 10。最后一种还会让关键帧间差异更难控制。

## 2. 额外发现：当前 DA3METRIC 缺少官方的焦距换算

DA3 官方说明中，`DA3METRIC-LARGE` 网络输出是以 300 px 焦距为基准的 cano nical depth，米制深度应为：

`depth_m = net_output * focal_px_at_model_input / 300`

当前 `src/depth_repair/da3.py` 在第 93–94 行把 `pred.depth` 直接乘 1000 当作毫米，没有乘 `focal/300`。这会导致：

- 绝对尺度与官方定义不一致；
- 头/胸/腕的焦距不同，直接比较深度会有系统性偏差；
- 用 Nested 作参考的后续映射会部分掩盖这个问题。

焦距换算是必须修正的独立问题，但它对同一相机是常数倍率，**不会自动消除相邻关键帧的加性偏移和空间倾斜**。应先修正它，再评估下面的时序方案。

## 3. 方案排序

| 优先级 | 方案 | 质量潜力 | 效率 | 工程风险 | 定位 |
|---|---|---:|---:|---:|---|
| P0 | DA3 关键帧低维时序配准 + RIFE | 中高 | 高 | 低 | 最快落地、保留现有链路 |
| P0 | Metric-VDA-Small/Base offline | 高 | 高 | 中 | 最可能成为新主线 |
| P1 | 相对 VDA/SCOPE + 稀疏米制锚点 | 高 | 中 | 中高 | Metric-VDA 尺度不达标时的混合方案 |
| P2 | DA3 any-view/DA3-Streaming + 尺度标定 | 高 | 低到中 | 高 | 高质量慢速备选 |
| 排除主线 | DepthCrafter/ChronoDepth/Align3R | 高 | 低 | 高 | 调研对照，不适合高效批处理 |

## 4. P0-A：保留现有链路的低成本修复

### 4.1 核心思路

在 RIFE 混合之前，先把相邻 DA3 关键帧的静态区深度对齐。根据报告的数值形态，不应只拟合比例尺度，而应先使用一个低维修正场：

`delta_k(x, y) = c0_k + cx_k * x_norm + cy_k * y_norm`

如果简化实验表明平面项收益小，可退化成只有 `c0` 的常数修正。报告中常数模型已把 13–20 mm 主导差异降到约 2 mm 中位残差，这已经接近“两关键帧差 <4 mm 时，RIFE 切换只产生约 1 mm 变化”的安全区。

### 4.2 配准 mask

只用可靠静态像素拟合：

- RGB 前后向光流一致；
- photometric residual 较小；
- 远离两张深度的边界带；
- 排除夹爪、手臂和明显运动区；
- 用 Huber/Tukey 损失或 RANSAC 拟合，不让少量错配控制整帧。

### 4.3 防漂移

不应简单地把每一帧对齐到前一帧，否则会积累漂移。对所有关键帧的修正参数 `theta_k=[c0,cx,cy]` 做一次小型全局优化：

`pair consistency + lambda1 * temporal second difference + lambda2 * absolute correction`

其中：

- pair consistency 约束相邻关键帧在静态 mask 内一致；
- temporal second difference 避免修正参数拐弯或周期跳变；
- absolute correction 尽量保留 DA3 的米制锚定；
- 每隔一段时间固定一个锚点或限制修正均值为 0。

每个 episode 只优化每个关键帧 1–3 个参数，计算开销相对 DA3/RIFE 可忽略。

### 4.4 RIFE 混合的次要改动

关键帧对齐后仍可以保留 RIFE 的 warp，但对“低运动 + 低光度残差 + 前后向流一致”的像素，建议把 RIFE mask 替换为时间权重 `t/gap`；只在真实遮挡和运动区使用学习 mask。这是保险层，不是根治层。

## 5. P0-B：Metric Video Depth Anything 作为替代主线

Metric-VDA 直接在 32 帧时序窗口内预测每帧米制深度，不需要把两张深度用 RIFE mask 混合，因此从机制上避开了本报告的直接故障模式。

官方已发布 Metric-VDA Small/Base/Large；公开结果中，ScanNet TAE 由 Small 的 1.48 降至 Base 的 1.26 和 Large 的 1.09。官方 A100/FP16 参考延迟为 Small 7.5 ms、Large 14 ms，但这是 `1×32×518×518` 的官方环境数字，不能直接折算本机的端到端 FPS。

当前仓库已有：

- `third_party/video_depth_anything`；
- `src/depth_repair/vda.py`；
- `vits/vitb/vitl` 和 metric checkpoint 选项。

本地未发现 Metric-VDA checkpoint，所以目前是“接口已接好，权重和实测尚缺”。

### 必须特别验证的风险

1. Metric-VDA 的米制模型主要由 Virtual KITTI 和 IRS 训练，机器人手腕近景可能有域外尺度偏差。
2. 官方指标显示它的时序一致性强，但单帧米制精度未在 NYUv2/KITTI 上全面超过 UniDepthV2。
3. offline 实现用 32 帧窗口、10 帧重叠，metric 模式在窗口间明确设置 `scale=1, shift=0`，只对 8 帧做线性交叉融合。因此必须检查是否每 22 帧出现窗口边界周期，不能只看平均指标。
4. 官方 streaming 模式在 ScanNet 上 `d1` 从 0.926 降到 0.836，首轮应测 offline，不应先上 streaming。

## 6. P1：相对视频深度 + 稀疏米制锚定

如果 Metric-VDA 的时序稳定性好，但手腕近景的绝对尺度不好，可以改用相对 VDA 或 SCOPE 输出稳定几何，再用稀疏米制锚点拟合低频尺度。

对相对视差 `q_t`，使用 clip 共享而非逐帧独立的参数：

`1 / D_t = a_clip * q_t + b_clip`

锚点可来自：

- 已修正焦距换算的 DA3METRIC；
- 更慢但几何质量更高的 Nested；
- 未来可用的传感器深度或已知机器人几何。

这条路线比“Nested 深度图 + RIFE”稳定，因为米制锚点只控制 2 个低频参数，不再提供需要逐像素混合的两张深度场。

SCOPE 是 2026 年新发布的值得关注备选：它针对长视频生成共享参数的尺度不变点图，官方报告 ScanNet 的 temporal alignment error 相比既有方法降低 34.9%。但它不直接输出米制尺度，且本项目尚未有速度和近景手臂实测，因此暂列 P1。

## 7. VDPP + DA3Metric：可减闪烁，但官方权重不能直接保证米制深度

VDPP 的方向与本项目需求很接近：它只接收逐帧深度，用 16 帧时序窗口预测残差，在论文测试中后处理开销约 1.4–8.6 ms。若 DA3Metric 每帧推理、再由 VDPP 做时序修正，它有机会消除普通逐帧抖动，并完全绕开 RIFE mask 混合产生云斑的机制。

但官方 checkpoint 不能原样作为米制深度方案，原因有四点：

1. 官方 `run_video.py` 在送入 VDPP 前对每张深度图独立做 min-max 归一化，输出后又做一次全序列 min-max；这样会直接丢失 DA3Metric 的米制尺度和帧间绝对偏移。
2. VDPP 的空间训练损失是 affine-invariant，优化目标本身不要求输出保留输入的米制 scale/shift。
3. 它前级的 Differential Scaler 主要做逐帧乘性校正；本报告定位到的 DA3 误差却以 13–20 mm 加性偏移为主，并带 2–3 mm 空间倾斜。后续残差网络理论上能修正，但官方训练域主要是 DAv2/DPT 相对深度，没有 DA3Metric 上的证据。
4. VDPP 需要每个输出时刻都有一张初始深度。只给步长 5 的 DA3 关键帧无法直接生成中间 4 帧；若先用 RIFE 补齐再交给 VDPP，只是在事后修云斑，不能保证几何和米制值不被二次改变。

因此分两种判断：

- **只追求视觉不闪**：DA3Metric 每帧推理 + VDPP 很有希望，值得快速 A/B。
- **要求毫米值稳定可信且仍保持步长 5 的效率**：官方 VDPP checkpoint 不能直接解决，需要做 metric-aware 适配或仍以“关键帧低维配准 + RIFE”为主。

若要把 VDPP 发展成正式方案，推荐对 DA3Metric 输出进行微调：删除逐帧 min-max；用米制监督或稳定伪标签训练；增加绝对深度锚定损失；让网络显式预测 `scale + shift + 低频平面 + 局部残差`。这比直接让现成 VDPP 修改毫米图更符合当前误差结构。

## 8. 建议的两轮实验

### 第一轮：一天级的机制验证

后续各方案统一使用同一条固定测试 episode，避免样本差异掩盖方案差异：

- RGB：`/home/user/RoboCOIN-with-Depth/data/original/ai2_depth_preview/AI2_Alphabot_2_grind_ink_0_qced_hardlink_ep000000/videos/chunk-000/observation.images.cam_front_head_rgb/episode_000000.mp4`
- 深度：同目录 `observation.images.cam_front_head_depth/episode_000000.mp4`
- 规格：640×480、30 FPS、1059 帧（35.3 秒）。
- 选用原因：现有云斑统计在默认阈值下命中 154/1059 帧（14.54%），形成 120 个事件；直接深度差扫描的最强跳变落在帧 718，相邻帧绝对差 P95 为 160 mm，视觉和定量都比原茶具样本明显。

第一轮先测三个代表窗口 220–225、715–720、950–955，再跑完整 1059 帧 episode：

1. 修正 DA3METRIC 的 `focal/300`。
2. 对相邻关键帧分别测试：无对齐、常数对齐、平面对齐。
3. 原 RIFE mask 与“静态区时间权重 + 运动区 RIFE mask”对照。
4. 不需要重跑整库，直接重用已保存的关键帧和 RIFE 流。

成功条件：

- 静态区关键帧差 P95 降到 4 mm 以下或至少下降 70%；
- 0.4→0.6 跳变不再是其余帧的 3–15 倍；
- 运动边界不因平面修正而出现新的深度重影；
- 修正后的绝对深度中位数不发生不受控漂移。

### 第二轮：替代模型 A/B

仍固定使用上述 `grind_ink` 头部完整 episode，所有方案使用相同帧范围、输入视频和统计口径，比较：

1. 当前 DA3-RIFE；
2. DA3 + 关键帧低维配准 + RIFE；
3. Metric-VDA-Small offline；
4. Metric-VDA-Base offline；
5. DA3Metric 每帧 + 原始 VDPP checkpoint（只作为可行性对照）；
6. 如果 3/4 的绝对尺度不达标，再加“相对 VDA + 稀疏锚定”。

评价必须包含：

- **周期跳变**：按 `frame mod 5` 和 VDA 的 `frame mod 22` 分组统计相邻帧深度残差；
- **静态区抖动**：中位数、P95，并单独报告常数项与去常数后的空间起伏；
- **动态边界**：遮挡/反遮挡带内残差和盲检；
- **米制精度**：对独立 Nested 抽样帧的 MAE/RMSE/AbsRel，抽样帧不参与任何拟合；
- **点云稳定**：静态背景的 3D 抖动 P50/P95；
- **效率**：排除首次加载后的稳态 FPS、峰值显存、端到端耗时。

这一固定 episode 用于方案间可重复 A/B，不代表跨任务泛化结论；选出候选方案后，再单独补充 AI2/乐聚跨任务验证，且不得改写本固定基准结果。

## 9. 最终建议

1. **立即修正 DA3METRIC 的焦距换算**，并将原始 H5 内参传入处理链。
2. **先实现关键帧常数/平面配准小试验**。这与报告揭示的误差形态直接匹配，成本最低，也最容易证伪。
3. **同时下载并实测 Metric-VDA-Small/Base offline**。如果近景尺度达标，它应替换 DA3-RIFE 成为新主线。
4. **如果 Metric-VDA 只有尺度不达标，不回退到深度图插帧**，而是改用相对视频深度 + clip 共享的稀疏米制标定。
5. Nested 保留为质量锚点和离线评估参考，不再负责逐像素中间帧生成。
6. VDPP 先作为实验支线：先验证它能否降低 DA3Metric 的逐帧闪烁；只有经过 metric-aware 微调并通过绝对尺度测试后，才进入正式链路。

## 10. 主要参考

- DA3 官方仓库与 Metric 焦距换算：<https://github.com/ByteDance-Seed/Depth-Anything-3>
- Video Depth Anything 官方仓库、metric 模型与速度：<https://github.com/DepthAnything/Video-Depth-Anything>
- Metric-VDA-Small 权重（Apache-2.0）：<https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small>
- SCOPE 项目与官方实现：<https://scope3d.github.io/> 、<https://github.com/zhengzhang01/SCOPE>
- Align3R（动态视频全局几何对齐）：<https://igl-hkust.github.io/Align3R.github.io/>
- VDPP 官方仓库与推理代码：<https://github.com/injun-baek/VDPP>
- VDPP 论文：<https://arxiv.org/abs/2604.06665>
