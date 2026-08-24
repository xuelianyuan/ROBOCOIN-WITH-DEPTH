# 深度 Refine 目标与评估规范

> 飞书导入说明：本文按 Markdown 导入编写。独立数学表达式使用 `$$ ... $$`，行内表达式使用 `$ ... $`；请不要删除美元符号。若当前飞书导入器未自动渲染公式，可将公式块内容粘贴到飞书的“公式”组件中，公式本身使用 LaTeX 语法。

## 目的与边界

本项目从 RGB 通过 DA3 或 DA3-Nested 得到初始深度，再执行 Refine，使结果更接近同步、标定的真实 RGB-D 相机深度。目标不应只优化逐像素 RMSE，而应同时保证绝对尺度、物体形状、遮挡边缘和视频稳定性。

评估必须区分两种参考来源：

| 模式 | 参考深度 | 结论范围 |
|---|---|---|
| `reference_agreement` | DA3-Nested 的稀疏输出 | 只说明 Refine 与 Nested 参考的一致性，不能证明物理真值精度 |
| `rgbd_benchmark` | 同步、完成标定与对齐的 RGB-D 深度 | 可报告相对真实相机深度的尺度和几何精度 |

现有 `scripts/refine_depth.py` 属于 `reference_agreement`：在采样帧上学习原深度到 Nested 深度的映射，并报告 MAE、RMSE 与 Wasserstein-1。只有在真实 RGB-D 深度取代或补充 Nested 参考后，MAE、RMSE 等才应被称为对真实深度的误差。

## Refine 的四层目标

```text
DA3 / DA3-Nested 初始深度
        |
        +-- 1. Scale / Metric Refinement
        |      校准全局尺度、偏移和距离相关误差。
        |
        +-- 2. Spatial / Structural Refinement
        |      保持平面、曲面、局部深度变化与物体形状。
        |
        +-- 3. Edge / Boundary Refinement
        |      保持手、物体、桌面及遮挡处的深度不连续边界。
        |
        +-- 4. Temporal Refinement
               在相机或物体运动下抑制无意义的帧间跳变。
        |
        v
最终深度视频
```

每一层都应独立计量。尺度改善不能以模糊物体边界为代价；时序平滑也不能把真实运动错误地压平。

## 通用计算约定

- 以毫米保存和报告距离误差，计算时转为 `float64`；相对指标无单位，法线误差以度报告。
- 对每帧使用共同有效掩码：`finite(pred) & finite(gt) & (pred > 0) & (gt > 0)`，并记录 `valid_pixels` 和有效比例。
- 真实 RGB-D 必须先完成 RGB/Depth 内外参对齐、时间同步、深度尺度确认和无效/飞点掩码处理。不同分辨率时，先按相机模型重投影，再计算指标。
- 所有帧级结果先按有效像素数加权汇总为 episode 指标；数据集报告同时给出 episode 均值和中位数，避免长视频主导结果。
- 同时报告 `baseline`（Refine 前）和 `refined`（Refine 后），以及相对变化百分比。不得只报告 Refine 后数值。

设预测深度为 `D`，真实或参考深度为 `G`，共同有效像素集合为 `Omega`，像素数为 `N`。

## 指标定义

### 1. 尺度与绝对距离

| 指标 | 含义 | 单位/方向 |
|---|---|---|
| MAE | 平均绝对距离偏差。 | mm，越低越好 |
| RMSE | 对大误差更敏感。 | mm，越低越好 |
| AbsRel | 减少不同距离范围的影响。 | 无量纲，越低越好 |
| delta1 | 比例误差小于 1.25 的像素比例。 | 无量纲，越高越好 |
| Bias | 诊断系统性偏近或偏远。 | mm，绝对值越低越好 |

核心公式（飞书公式块）：

$$
\mathrm{MAE}=\frac{1}{N}\sum_{i\in\Omega}|D_i-G_i|
$$

$$
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i\in\Omega}(D_i-G_i)^2}
$$

$$
\mathrm{AbsRel}=\frac{1}{N}\sum_{i\in\Omega}\frac{|D_i-G_i|}{G_i}
$$

$$
\delta_1=\frac{1}{N}\sum_{i\in\Omega}\mathbf{1}\left[\max\left(\frac{D_i}{G_i},\frac{G_i}{D_i}\right)<1.25\right]
$$

$$
\mathrm{Bias}=\frac{1}{N}\sum_{i\in\Omega}(D_i-G_i)
$$

当前实现：MAE、RMSE、Bias（字段名 `mean_error_mm`）已实现；AbsRel 与 delta1 尚未实现。`wasserstein_mm` 是整体深度分布差异的补充诊断指标，不能替代逐像素几何误差。

### 2. 空间结构与局部几何

| 指标 | 用途 | 前提与解释 |
|---|---|---|
| Gradient Error | 比较水平、垂直深度梯度。 | 使用同一 Sobel 或中心差分算子，不应跨无效像素计算。 |
| Boundary Depth MAE (mm) | 从 `G` 的梯度生成边界带（例如阈值后膨胀 2 至 3 像素），仅在边界带内计算 MAE。 | 评价物体、手和桌面交界处的深度准确性。 |
| Boundary F1 | 将 `D`、`G` 的深度边缘二值化，在固定像素容差内匹配，报告 precision、recall、F1。 | 与 Boundary Depth MAE 配套：前者评价边界位置，后者评价边界上的距离。 |
| Normal Error | 比较表面法线夹角。 | 要求 `fx, fy, cx, cy`；应剔除深度突变和无效区域。 |
| 3D Point Error | 比较对应像素反投影后的三维点距离。 | 使用同一像素对应点，不以最近邻点云距离作为主指标。 |

梯度、法线和三维误差公式：

$$
E_{grad}=\frac{1}{N}\sum_{i\in\Omega}\left(|\nabla_xD_i-\nabla_xG_i|+|\nabla_yD_i-\nabla_yG_i|\right)
$$

$$
E_{normal}=\frac{1}{N}\sum_{i\in\Omega}\arccos\left(\operatorname{clip}(\mathbf{n}_{D,i}\cdot\mathbf{n}_{G,i},-1,1)\right)
$$

$$
E_{3D}=\frac{1}{N}\sum_{i\in\Omega}\left\|\mathbf{P}_{D,i}-\mathbf{P}_{G,i}\right\|_2
$$

结构层优先实现 Gradient Error 和 Boundary Depth MAE；有可靠内参后再启用 Normal Error 与 3D Point Error。Boundary Error 建议在报告中明确拆为 `boundary_depth_mae_mm` 和 `boundary_f1`，避免定义含糊。

### 3. 时序一致性

相邻帧的真实深度变化不能被当成闪烁。因此，应先用 RGB 光流、相机位姿或深度重投影，将 `t-1` 帧的深度 warp 到 `t` 帧，只在非遮挡且共同有效区域 `Omega_t` 上比较。

| 指标 | 含义 | 单位/方向 |
|---|---|---|
| t-MAE | 比较真实与预测的帧间深度变化。 | mm，越低越好 |
| t-RMSE | 帧间变化误差的均方根。 | mm，越低越好 |
| Flicker (mm) | 在对齐后的静态/可见区域上报告深度时间二阶差分的平均绝对值，或报告其相对真实值的误差。 | 衡量不由真实运动造成的闪烁，越低越好。 |

时序误差公式：

$$
\Delta D_t=D_t-\operatorname{warp}(D_{t-1}),\qquad \Delta G_t=G_t-\operatorname{warp}(G_{t-1})
$$

$$
t\text{-MAE}=\frac{1}{N_t}\sum_{i\in\Omega_t}|\Delta D_{t,i}-\Delta G_{t,i}|
$$

$$
t\text{-RMSE}=\sqrt{\frac{1}{N_t}\sum_{i\in\Omega_t}(\Delta D_{t,i}-\Delta G_{t,i})^2}
$$

$$
\mathrm{Flicker}=\frac{1}{N_t}\sum_{i\in\Omega_t}|D_{t+1,i}-2D_{t,i}+D_{t-1,i}|
$$

若暂时没有连续 RGB-D Ground Truth，可报告 `self_tmae_mm` 和 `self_flicker_mm` 作为无监督稳定性诊断，但必须标注为 self-consistency，不能等同于真实时序精度。当前稀疏 Nested 采样不足以可靠评价完整视频的时序指标；应对 Refine 后完整视频和连续 RGB-D 序列计算。

## 建议的报告结构

每个 `refine_parameters.json` 应保留现有映射信息，并扩展为以下稳定结构。字段值均应包含 `baseline` 与 `refined`；不可用的指标使用 `null` 并写明原因，而不是静默省略。

```json
{
  "evaluation_mode": "rgbd_benchmark",
  "reference": {
    "type": "synchronized_rgbd",
    "depth_unit": "mm",
    "camera_intrinsics_available": true,
    "temporal_alignment": "rgb_flow"
  },
  "metric_accuracy": {
    "mae_mm": {"baseline": 0.0, "refined": 0.0},
    "rmse_mm": {"baseline": 0.0, "refined": 0.0},
    "abs_rel": {"baseline": 0.0, "refined": 0.0},
    "delta1": {"baseline": 0.0, "refined": 0.0},
    "bias_mm": {"baseline": 0.0, "refined": 0.0}
  },
  "structure": {
    "gradient_error_mm_per_px": {"baseline": 0.0, "refined": 0.0},
    "boundary_depth_mae_mm": {"baseline": 0.0, "refined": 0.0},
    "boundary_f1": {"baseline": 0.0, "refined": 0.0},
    "normal_mae_deg": {"baseline": 0.0, "refined": 0.0},
    "point3d_mae_mm": {"baseline": 0.0, "refined": 0.0}
  },
  "temporal": {
    "t_mae_mm": {"baseline": 0.0, "refined": 0.0},
    "t_rmse_mm": {"baseline": 0.0, "refined": 0.0},
    "flicker_mm": {"baseline": 0.0, "refined": 0.0}
  }
}
```

保留原有 `mean_mm`、分位数和 `wasserstein_mm` 于 `distribution_diagnostics`，用于排查尺度映射是否改变整体分布；它们不应作为几何质量的唯一准入条件。

## 实施优先级与准入规则

1. 立即补充 `abs_rel`、`delta1`，并将现有报告标记为 `reference_agreement`。
2. 在真实 RGB-D 对齐后增加 Gradient Error、Boundary Depth MAE 和 Boundary F1；它们最直接对应手、物体与桌面的局部结构质量。
3. 读取 `camera_params.json` 或等价标定，增加 Normal Error 与逐像素对应的 3D Point Error。
4. 对完整输出视频增加经运动补偿的 t-MAE、t-RMSE 和 Flicker；不要仅在稀疏采样关键帧上宣称视频时序质量。

`improved` 不应继续只要求 MAE、RMSE、Wasserstein-1 同时下降。建议输出分层判定：

| 判定 | 条件 |
|---|---|
| `metric_improved` | MAE、RMSE、AbsRel 下降且 delta1 上升。 |
| `structure_preserved` | Gradient Error、Boundary Depth MAE 不上升，Boundary F1 不下降；有内参时 Normal/3D Error 不上升。 |
| `temporal_improved` | t-MAE、t-RMSE、Flicker 下降。 |
| `accepted` | 对应运行阶段所需的分层判定全部为真；不具备 GT 或内参时明确标为 `not_evaluated`。 |

这个规则避免“全局尺度变好但边缘变糊”或“视频更平滑但真实运动被抹掉”的假改善。
