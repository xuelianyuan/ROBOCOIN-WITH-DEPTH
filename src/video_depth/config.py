"""
config.py — DA3 Nested + RIFE 流水线配置与校验

负责把唯一默认 YAML 转为强类型配置，并集中检查采样、窗口、设备和编码参数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEPRECATED_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE"


@dataclass(slots=True)
class SamplingConfig:
    """关键帧采样参数。``stride`` 必须为正整数。"""

    stride: int = 5
    scene_change_threshold: float = 0.35


@dataclass(slots=True)
class DA3Config:
    """DA3 Nested 序列窗口参数，模型输出按米解释。"""

    model_id: str = DEFAULT_MODEL_ID
    process_res: int = 504
    window_size: int = 16
    overlap: int = 4
    align_scale: bool = True
    max_scale_correction: float = 0.05
    min_alignment_pixels: int = 100


@dataclass(slots=True)
class RIFEConfig:
    """RIFE 权重、padding 与低置信度阈值。"""

    checkpoint: str = "third_party/Practical-RIFE/train_log"
    pad_multiple: int = 64
    min_confidence: float = 0.05


@dataclass(slots=True)
class OutputConfig:
    """12-bit 深度视频与无损 NPZ 输出参数。"""

    codec: str = "libx265"
    pixel_format: str = "gray12le"
    crf: int = 6
    preset: str = "slow"
    max_depth_mm: int = 4095
    save_npz: bool = True


@dataclass(slots=True)
class VideoDepthConfig:
    """完整流水线配置。

    代码默认值仅用于程序化构造和旧调用兼容；普通运行的权威默认值来自
    ``configs/da3_nested_rife.yaml``。
    """

    strategy: str = "da3_nested_rife"
    device: str = "cuda:0"
    seed: int = 42
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    da3: DA3Config = field(default_factory=DA3Config)
    rife: RIFEConfig = field(default_factory=RIFEConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VideoDepthConfig":
        """加载 YAML 并校验，未知字段会产生可执行的错误。"""

        config_path = Path(path)
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"配置文件不存在: {config_path}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"配置根节点必须是 mapping: {config_path}")
        config = cls.from_dict(raw)
        config.validate()
        return config

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "VideoDepthConfig":
        """从 mapping 构造配置，主要供 CLI 覆盖和测试使用。"""

        allowed = {"strategy", "device", "seed", "sampling", "da3", "rife", "output"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"未知配置字段: {sorted(unknown)}")
        return cls(
            strategy=str(raw.get("strategy", "da3_nested_rife")),
            device=str(raw.get("device", "cuda:0")),
            seed=int(raw.get("seed", 42)),
            sampling=_section(SamplingConfig, raw.get("sampling", {}), "sampling"),
            da3=_section(DA3Config, raw.get("da3", {}), "da3"),
            rife=_section(RIFEConfig, raw.get("rife", {}), "rife"),
            output=_section(OutputConfig, raw.get("output", {}), "output"),
        )

    def validate(self) -> None:
        """检查所有跨字段约束；失败时抛出 ``ValueError``。"""

        if self.strategy != "da3_nested_rife":
            raise ValueError("strategy 必须为 da3_nested_rife")
        if self.sampling.stride <= 0:
            raise ValueError("sampling.stride 必须 > 0")
        if not 0.0 <= self.sampling.scene_change_threshold <= 1.0:
            raise ValueError("scene_change_threshold 必须位于 [0, 1]")
        if self.da3.model_id == DEPRECATED_MODEL_ID:
            raise ValueError("已弃用无 -1.1 的 DA3 Nested 权重，请使用刷新后的 -1.1 模型")
        if self.da3.process_res <= 0 or self.da3.window_size <= 0:
            raise ValueError("process_res 和 window_size 必须 > 0")
        if not 0 <= self.da3.overlap < self.da3.window_size:
            raise ValueError("da3.overlap 必须满足 0 <= overlap < window_size")
        if not 0.0 <= self.da3.max_scale_correction <= 0.5:
            raise ValueError("max_scale_correction 必须位于 [0, 0.5]")
        if self.da3.min_alignment_pixels < 1:
            raise ValueError("min_alignment_pixels 必须 >= 1")
        if self.rife.pad_multiple < 1:
            raise ValueError("rife.pad_multiple 必须 >= 1")
        if not 0.0 <= self.rife.min_confidence <= 1.0:
            raise ValueError("rife.min_confidence 必须位于 [0, 1]")
        if self.output.pixel_format != "gray12le":
            raise ValueError("当前深度编码只支持 gray12le")
        if self.output.max_depth_mm <= 0 or self.output.max_depth_mm > 4095:
            raise ValueError("max_depth_mm 必须位于 [1, 4095]")

    def to_dict(self) -> dict[str, Any]:
        """返回可安全写入 YAML/JSON 的配置字典。"""

        return asdict(self)


def _section(section_type, raw: Any, name: str):
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} 必须是 mapping")
    field_names = set(section_type.__dataclass_fields__)
    unknown = set(raw) - field_names
    if unknown:
        raise ValueError(f"{name} 包含未知字段: {sorted(unknown)}")
    return section_type(**raw)
