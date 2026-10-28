"""尺度恢复测试：异常值、正斜率、chunk 覆盖及重叠映射约定。"""
import numpy as np
import pytest

from src.depth_refine.methods.metric_vdpp import (
    chunk_ranges,
    fit_positive_affine,
    load_vdpp_checkpoint,
    reliable_anchor_mask,
    smooth_affines,
)


def test_huber_rejects_outliers() -> None:
    """10% 大异常值不能显著改变恢复尺度。"""
    x = np.linspace(0, 1, 10000)
    y = 1.7 * x + 0.2
    y[::10] += 10
    a, b, _ = fit_positive_affine(x, y)
    assert a == pytest.approx(1.7, abs=1e-3)
    assert b == pytest.approx(0.2, abs=1e-3)


def test_positive_slope_and_degenerate_input() -> None:
    """负相关不会翻转近远语义；常量输入禁止伪造拟合结果。"""
    x = np.linspace(0, 1, 1000)
    assert fit_positive_affine(x, -x)[0] > 0
    with pytest.raises(ValueError):
        fit_positive_affine(np.ones(1000), x)


def test_chunk_coverage() -> None:
    """末块包含新增帧，1059 帧完整覆盖。"""
    ranges = chunk_ranges(1059, 32, 8)
    assert ranges[0] == (0, 32)
    assert ranges[-1][1] == 1059
    assert all(b - a > 8 for a, b in ranges)
    assert set(i for a, b in ranges for i in range(a, b)) == set(range(1059))
    assert chunk_ranges(8, 32, 8) == [(0, 8)]


def test_overlap_affine_direction() -> None:
    """v0=2*v1+0.1 时，两组 metric 参数应恢复相同 q。"""
    result = smooth_affines([(1.0, 0.2), (2.0, 0.3)], [(2.0, 0.1)])
    np.testing.assert_allclose(result, [(1, 0.2), (2, 0.3)], atol=1e-7)
    assert np.all(result[:, 0] > 0)


def test_anchor_mask_rejects_sky_and_flip_mismatch() -> None:
    """首帧无需光流；天空和 flip 不一致像素不能成为尺度锚点。"""
    rgb = np.zeros((16, 16, 3), np.uint8)
    depth = np.full((16, 16), 2.0, np.float32)
    flipped = depth.copy()
    flipped[8, 8] = 3.0
    non_sky = np.ones((16, 16), bool)
    non_sky[4, 4] = False
    mask = reliable_anchor_mask(None, rgb, depth, flipped, non_sky)
    assert mask.dtype == bool
    assert mask.any()
    assert not mask[8, 8]
    assert not mask[4, 4]


def test_checkpoint_only_allows_legacy_shift_head(tmp_path) -> None:
    """官方旧 shift head 可忽略，其他未知参数仍必须报错。"""
    import torch

    model = torch.nn.Linear(2, 1)
    state = model.state_dict()
    state["shift_head.0.weight"] = torch.zeros(1)
    checkpoint = tmp_path / "compatible.pth"
    torch.save(state, checkpoint)
    assert load_vdpp_checkpoint(model, str(checkpoint)) == ("shift_head.0.weight",)

    state["unknown.weight"] = torch.zeros(1)
    checkpoint = tmp_path / "incompatible.pth"
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint 与模型不兼容"):
        load_vdpp_checkpoint(model, str(checkpoint))
