"""
rife_depth.py — 基于 RIFE RGB 运动的米制深度补帧

RIFE 只读取相邻 RGB 关键帧估计双向采样流；本文件把该运动作用于前后米制深度，
结合遮挡 mask、有效掩码和关键帧置信度融合。关键帧逐值原样保留。
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)
_RIFE_HOME = Path(__file__).resolve().parents[2] / "third_party" / "Practical-RIFE"


class RIFEDepthInterpolator:
    """以 RIFE 双向 RGB 运动补齐非关键帧深度。

    ``motion_predictor`` 是 CPU 测试注入点，签名为 ``(rgb0, rgb1, timestep)``，
    返回 ``flow (1,4,H,W)`` 与 mask logits ``(1,1,H,W)``。
    """

    def __init__(
        self,
        device: str,
        checkpoint: str | Path | None = None,
        pad_multiple: int = 64,
        min_confidence: float = 0.05,
        motion_predictor: Callable[[np.ndarray, np.ndarray, float], tuple[torch.Tensor, torch.Tensor]]
        | None = None,
    ) -> None:
        self.device = torch.device(device)
        checkpoint_path = Path(checkpoint or (_RIFE_HOME / "train_log"))
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
        self.checkpoint = checkpoint_path
        self.pad_multiple = pad_multiple
        self.min_confidence = min_confidence
        self._external_predictor = motion_predictor
        self._rife_model: Any | None = None

    @torch.inference_mode()
    def interpolate(
        self,
        rgb_frames: Sequence[np.ndarray],
        key_indices: Sequence[int],
        key_depth_m: np.ndarray,
        key_confidence: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """补齐全部非关键帧。

        Args:
            rgb_frames: 完整 ``uint8 RGB (T,H,W,3)`` 序列。
            key_indices: 严格递增且含首尾的关键帧索引。
            key_depth_m: ``float32 (K,H,W)`` 米制关键帧深度。
            key_confidence: 与深度同 shape 的置信度。
        Returns:
            完整 ``depth_m, confidence``，shape 均为 ``(T,H,W)``。
        """

        count = len(rgb_frames)
        _validate_inputs(rgb_frames, key_indices, key_depth_m, key_confidence)
        height, width = rgb_frames[0].shape[:2]
        depth = np.zeros((count, height, width), dtype=np.float32)
        confidence = np.zeros_like(depth)
        for local, frame_index in enumerate(key_indices):
            # 直接赋值而非重新采样，确保关键帧逐值一致。
            depth[frame_index] = key_depth_m[local]
            confidence[frame_index] = key_confidence[local]

        for pair_index, (index0, index1) in enumerate(zip(key_indices[:-1], key_indices[1:])):
            gap = index1 - index0
            if gap <= 1:
                continue
            d0 = self._padded_tensor(key_depth_m[pair_index], channels=False)
            d1 = self._padded_tensor(key_depth_m[pair_index + 1], channels=False)
            c0 = self._padded_tensor(key_confidence[pair_index], channels=False, zero_pad=True)
            c1 = self._padded_tensor(key_confidence[pair_index + 1], channels=False, zero_pad=True)
            v0 = self._padded_tensor(
                np.isfinite(key_depth_m[pair_index]) & (key_depth_m[pair_index] > 0),
                channels=False,
                zero_pad=True,
            )
            v1 = self._padded_tensor(
                np.isfinite(key_depth_m[pair_index + 1]) & (key_depth_m[pair_index + 1] > 0),
                channels=False,
                zero_pad=True,
            )

            for offset in range(1, gap):
                timestep = offset / gap
                try:
                    flow, mask_logits = self._predict_motion(
                        rgb_frames[index0], rgb_frames[index1], timestep
                    )
                    merged, merged_conf = self._blend_depth(
                        d0, d1, c0, c1, v0, v1, flow, mask_logits
                    )
                except RuntimeError as exc:
                    LOGGER.warning(
                        "RIFE 运动估计失败，使用像素线性安全回退: pair=(%d,%d) t=%.3f error=%s",
                        index0,
                        index1,
                        timestep,
                        exc,
                    )
                    merged, merged_conf = self._linear_fallback(d0, d1, c0, c1, v0, v1, timestep)
                depth[index0 + offset] = merged[0, 0, :height, :width].cpu().numpy()
                confidence[index0 + offset] = merged_conf[0, 0, :height, :width].cpu().numpy()

        return depth, confidence

    def _predict_motion(
        self, rgb0: np.ndarray, rgb1: np.ndarray, timestep: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._external_predictor is not None:
            flow, mask = self._external_predictor(rgb0, rgb1, timestep)
            return flow.to(self.device), mask.to(self.device)
        model = self._load_rife()
        image0 = self._padded_tensor(rgb0, channels=True) / 255.0
        image1 = self._padded_tensor(rgb1, channels=True) / 255.0
        images = torch.cat((image0, image1), dim=1)
        flow_list, mask_logits, _ = model.flownet(
            images, timestep, scale_list=[16, 8, 4, 2, 1]
        )
        return flow_list[-1], mask_logits

    def _load_rife(self) -> Any:
        """加载一次 RIFE，并把上游全局 device 显式绑定到所选设备。"""

        if self._rife_model is not None:
            return self._rife_model
        if not (self.checkpoint / "flownet.pkl").is_file():
            raise FileNotFoundError(f"RIFE 权重不存在: {self.checkpoint / 'flownet.pkl'}")
        home = str(_RIFE_HOME)
        if home not in sys.path:
            sys.path.insert(0, home)
        rife_module = importlib.import_module("train_log.RIFE_HDv3")
        warplayer_module = importlib.import_module("model.warplayer")
        rife_module.device = self.device
        warplayer_module.device = self.device
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                model = rife_module.Model()
                model.load_model(str(self.checkpoint), rank=-1)
        else:
            model = rife_module.Model()
            model.load_model(str(self.checkpoint), rank=-1)
        model.flownet.to(self.device)
        model.eval()
        self._rife_model = model
        return model

    def _padded_tensor(
        self, values: np.ndarray, *, channels: bool, zero_pad: bool = False
    ) -> torch.Tensor:
        array = np.asarray(values)
        if channels:
            tensor = torch.from_numpy(array).to(self.device, dtype=torch.float32)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            tensor = torch.from_numpy(array.astype(np.float32, copy=False)).to(self.device)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        height, width = tensor.shape[-2:]
        padded_h = ((height + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        padded_w = ((width + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        pad = (0, padded_w - width, 0, padded_h - height)
        if pad[1] or pad[3]:
            tensor = (
                F.pad(tensor, pad, mode="constant", value=0.0)
                if zero_pad
                else F.pad(tensor, pad, mode="replicate")
            )
        return tensor

    def _blend_depth(
        self,
        d0: torch.Tensor,
        d1: torch.Tensor,
        c0: torch.Tensor,
        c1: torch.Tensor,
        v0: torch.Tensor,
        v1: torch.Tensor,
        flow: torch.Tensor,
        mask_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if flow.shape[1] != 4:
            raise RuntimeError(f"RIFE flow 应有 4 通道，实际为 {tuple(flow.shape)}")
        warped_d0 = _warp(d0, flow[:, :2], padding_mode="border")
        warped_d1 = _warp(d1, flow[:, 2:4], padding_mode="border")
        warped_c0 = _warp(c0, flow[:, :2], padding_mode="zeros").clamp_min(0)
        warped_c1 = _warp(c1, flow[:, 2:4], padding_mode="zeros").clamp_min(0)
        valid0 = _warp(v0, flow[:, :2], padding_mode="zeros") > 0.999
        valid1 = _warp(v1, flow[:, 2:4], padding_mode="zeros") > 0.999
        mask = torch.sigmoid(mask_logits)
        weight0 = mask * warped_c0 * valid0
        weight1 = (1.0 - mask) * warped_c1 * valid1
        total = weight0 + weight1
        merged = torch.where(
            total > 1e-8,
            (warped_d0 * weight0 + warped_d1 * weight1) / total.clamp_min(1e-8),
            torch.zeros_like(total),
        )
        merged_conf = total.clamp(0.0, 1.0)
        merged = torch.where(merged_conf >= self.min_confidence, merged, torch.zeros_like(merged))
        return merged.float(), merged_conf.float()

    @staticmethod
    def _linear_fallback(
        d0: torch.Tensor,
        d1: torch.Tensor,
        c0: torch.Tensor,
        c1: torch.Tensor,
        v0: torch.Tensor,
        v1: torch.Tensor,
        timestep: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight0 = (1.0 - timestep) * c0 * (v0 > 0.5)
        weight1 = timestep * c1 * (v1 > 0.5)
        total = weight0 + weight1
        depth = torch.where(
            total > 1e-8,
            (d0 * weight0 + d1 * weight1) / total.clamp_min(1e-8),
            torch.zeros_like(total),
        )
        return depth.float(), (0.25 * total).clamp(0.0, 1.0).float()


def _warp(
    values: torch.Tensor, flow: torch.Tensor, *, padding_mode: str
) -> torch.Tensor:
    """按 RIFE 的反向像素流采样张量，保持线性深度值不变。"""

    _, _, height, width = values.shape
    horizontal = torch.linspace(-1.0, 1.0, width, device=values.device, dtype=values.dtype)
    vertical = torch.linspace(-1.0, 1.0, height, device=values.device, dtype=values.dtype)
    grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
    base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    norm_x = flow[:, 0] / max((width - 1.0) / 2.0, 1.0)
    norm_y = flow[:, 1] / max((height - 1.0) / 2.0, 1.0)
    grid = base + torch.stack((norm_x, norm_y), dim=-1)
    return F.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def _validate_inputs(
    rgb_frames: Sequence[np.ndarray],
    key_indices: Sequence[int],
    key_depth_m: np.ndarray,
    key_confidence: np.ndarray,
) -> None:
    if not rgb_frames:
        raise ValueError("RGB 帧序列为空")
    if list(key_indices) != sorted(set(key_indices)):
        raise ValueError("关键帧索引必须严格递增且无重复")
    if not key_indices or key_indices[0] != 0 or key_indices[-1] != len(rgb_frames) - 1:
        raise ValueError("关键帧索引必须包含首尾")
    expected = (len(key_indices), *rgb_frames[0].shape[:2])
    if key_depth_m.shape != expected or key_confidence.shape != expected:
        raise ValueError(f"关键帧深度/置信度 shape 必须为 {expected}")
    if key_depth_m.dtype != np.float32:
        raise ValueError("关键帧深度必须为 float32 米制值")
