"""DA3Metric 的 resize 内参校正及 flip 一致性推理适配器。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from depth_anything_3.api import DepthAnything3


class MetricPredictor:
    """读取本地模型；输出米制深度、翻转预测及非天空区域。"""

    def __init__(self, model_path: str, process_res: int, device: str) -> None:
        self.model = DepthAnything3.from_pretrained(model_path).to(device).eval()
        self.process_res, self.device = process_res, device
        self.processed_intrinsics: list | None = None
        self.output_keys: list[str] = []

    @torch.inference_mode()
    def predict(
        self,
        rgb: list[np.ndarray],
        intrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """输入原分辨率 K；利用官方预处理返回的 K' 做 focal/300 米制转换。"""
        n, (h, w) = len(rgb), rgb[0].shape[:2]
        # flip 后主点同样翻转；焦距保持为正，供 metric scaling 使用。
        flipped_k = intrinsic.copy()
        flipped_k[0, 2] = w - 1 - flipped_k[0, 2]
        images = rgb + [np.ascontiguousarray(frame[:, ::-1]) for frame in rgb]
        ks = np.stack([intrinsic] * n + [flipped_k] * n)
        imgs_cpu, _, processed_k = self.model.input_processor(
            images, intrinsics=ks, process_res=self.process_res,
            process_res_method="upper_bound_resize", num_workers=1)
        imgs, _, in_t = self.model._prepare_model_inputs(imgs_cpu, None, processed_k)
        raw = self.model.forward(imgs, None, in_t, export_feat_layers=[])
        self.output_keys = sorted(raw.keys())
        self.processed_intrinsics = processed_k[0].tolist()
        # DA3METRIC-LARGE 的 raw depth 是 canonical focal=300；API 不自动完成米制缩放。
        focal = (in_t[..., 0, 0] + in_t[..., 1, 1]) / 2
        depth = raw["depth"].float() * focal[..., None, None] / 300.0
        depth = F.interpolate(
            depth[0, :, None],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        non_sky = raw["sky"][0] < 0.3
        non_sky = F.interpolate(non_sky[:, None].float(), size=(h, w), mode="nearest")[:, 0].bool()
        depth, non_sky = depth.cpu().numpy(), non_sky.cpu().numpy()
        return depth[:n], depth[n:, :, ::-1].copy(), non_sky[:n] & non_sky[n:, :, ::-1]
