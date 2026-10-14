"""以合成深度和小视频验证候选检测、逐帧读取及时间区间口径。"""
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from src.depth_refine.evaluation.cloud_artifacts import (
    CloudConfig, cloud_segments, detect_cloud, summarize_segments,
)
from src.depth_refine.io_utils import iter_video_frames, probe_video
from src.depth_refine.pipeline.cloud_statistics import _aggregate, discover_depth_videos, run_cloud_statistics


def test_segments_keep_singletons_and_never_bridge_gaps():
    segments = cloud_segments([True, False, True, True, True, False, True], 30)
    assert [(s['start_frame'], s['end_frame']) for s in segments] == [(0, 0), (2, 4), (6, 6)]
    result = summarize_segments(segments, 7, 6)
    assert result['isolated_single_frames'] == 2
    assert result['consecutive_events'] == 1
    assert result['frames_in_consecutive_events'] == 3
    assert result['unknown_frames'] == 1
    assert result['suspected_frames'] == 5
    assert result['pattern'] == 'mixed'
    assert segments[-1]['duration_seconds'] == pytest.approx(1 / 30)
    assert segments[-1]['end_seconds_exclusive'] == pytest.approx(7 / 30)


def test_empty_and_all_positive_segments():
    assert cloud_segments([], 30) == []
    segments = cloud_segments([True] * 5, 10)
    result = summarize_segments(segments, 5, 5)
    assert result['events_at_least_5_frames'] == 1
    assert result['longest_event_seconds'] == 0.5
    assert result['pattern'] == 'segment_dominant'
    assert summarize_segments([], 0, 0)['consecutive_frame_fraction'] is None
    with pytest.raises(ValueError):
        cloud_segments([], 0)


def test_cloud_detector_distinguishes_plane_and_coherent_texture():
    y, x = np.mgrid[:240, :320].astype(np.float32)
    plane = 700 + x * 0.8 + y * 0.4
    cloud = plane + 12 * np.sin(x / 5) * np.cos(y / 5)
    config = CloudConfig()
    flat, _, _ = detect_cloud(plane, config)
    textured, _, mask = detect_cloud(cloud, config)
    assert flat['assessable'] and not flat['suspected']
    assert textured['suspected'] and mask.any()
    invalid, _, _ = detect_cloud(np.zeros_like(plane), config)
    assert not invalid['assessable'] and not invalid['suspected']
    constant, _, _ = detect_cloud(np.full_like(plane, 500), config)
    assert constant['assessable'] and not constant['suspected']
    step = np.where(x < 160, 500, 1500).astype(np.float32)
    assert not detect_cloud(step, config)[0]['suspected']


def _tiny_video(path: Path) -> list[np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [np.full((64, 96), v, dtype='<u2') for v in [400, 401, 700, 1000, 1200]]
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gray16le',
        '-s', '96x64', '-r', '10', '-i', '-', '-c:v', 'ffv1', '-threads', '1',
        '-pix_fmt', 'gray16le', str(path)], input=b''.join(f.tobytes() for f in frames), check=True)
    # 使用原生 gray16le 输入检查数值原样往返，不在测试中做位深转换。
    return frames


def test_native_depth_stream_and_early_close(tmp_path):
    path = tmp_path / 'tiny.mkv'
    frames = _tiny_video(path)
    info = probe_video(path)
    decoded = list(iter_video_frames(path, 'gray16le'))
    assert len(decoded) == len(frames)
    assert decoded[0].shape == (64, 96)
    assert np.array_equal(decoded[0], frames[0])
    scaled = list(iter_video_frames(path, 'gray16le', max_width=64))
    assert len(scaled) == 5 and scaled[0].shape == (43, 64)
    with pytest.raises(ValueError, match='mismatch'):
        list(iter_video_frames(path, 'gray12le'))
    iterator = iter_video_frames(path, 'gray16le', info=info)
    next(iterator)
    iterator.close()


def test_discovery_and_report_exclude_derived_videos(tmp_path):
    root = tmp_path / 'dataset'
    path = root / 'task' / 'videos' / 'chunk-000' / 'observation.images.any_camera_depth' / 'episode_000000.mp4'
    path.parent.mkdir(parents=True)
    # lossless x265 的微型 gray12le 视频检查逐帧标签数量及时间计算。
    frames = [np.full((64, 96), 500 + i, dtype='<u2') for i in range(5)]
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gray12le',
        '-s', '96x64', '-r', '10', '-i', '-', '-c:v', 'libx265', '-threads', '1',
        '-x265-params', 'lossless=1:pools=none:frame-threads=1:log-level=error',
        '-pix_fmt', 'gray12le', str(path)], input=b''.join(f.tobytes() for f in frames), check=True)
    derived = root / 'task' / 'depth_results' / 'refine' / 'videos' / 'chunk-000' / 'cam_depth' / 'episode_000000.mp4'
    derived.parent.mkdir(parents=True)
    derived.touch()
    assert len(discover_depth_videos([root, root])) == 1
    output = tmp_path / 'report'
    summary = run_cloud_statistics([root], output, CloudConfig(), workers=1)
    assert summary['overall']['completed_videos'] == 1
    assert summary['overall']['total_frames'] == 5
    assert summary['overall']['suspected_frames'] == 0
    assert (output / 'REPORT.md').exists()
    assert len(list(output.rglob('review.png'))) == 1
    from scripts.depth_refine.run_refine import _decode
    assert all(np.array_equal(a, b) for a, b in zip(_decode(path, 'gray12le'), frames))
    # 正例初始化 OpenCV 后仍可并行解码，覆盖线程池初始化后 fork 死锁的回归场景。
    y, x = np.mgrid[:240, :320].astype(np.float32)
    reference = tmp_path / 'reference.npy'
    np.save(reference, 700 + x * 0.8 + y * 0.4 + 12 * np.sin(x / 5) * np.cos(y / 5))
    parallel = run_cloud_statistics([root], tmp_path / 'parallel', CloudConfig(),
                                   references=[reference], workers=2)
    assert parallel['overall'] == summary['overall']
    with pytest.raises(FileExistsError):
        run_cloud_statistics([root], output, CloudConfig())
    resumed = run_cloud_statistics([root], output, CloudConfig(), resume=True)
    assert resumed['overall'] == summary['overall']
    with pytest.raises(ValueError, match='identical'):
        run_cloud_statistics([root], output, CloudConfig(coverage=0.03), resume=True)


def test_corrupt_video_is_reported_and_excluded(tmp_path):
    root = tmp_path / 'dataset'
    path = root / 'task' / 'videos' / 'chunk-000' / 'cam_depth' / 'episode_000000.mp4'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'not a video')
    summary = run_cloud_statistics([root], tmp_path / 'report', CloudConfig())
    assert summary['overall']['failed_videos'] == 1
    assert summary['overall']['completed_videos'] == 0
    assert summary['overall']['total_frames'] == 0
    assert summary['failures'][0]['error']


def test_task_and_affected_video_statistics_keep_denominators_separate():
    def report(task, flags):
        counts = summarize_segments(cloud_segments(flags, 30), len(flags), len(flags))
        return {'dataset': 'ds', 'task': task, 'status': 'completed', **counts,
                'sensitivity': {'default': counts}}

    reports = [report('task_a', [True, False, True, True]),
               report('task_a', [False] * 5), report('task_b', [False] * 3)]
    summary = _aggregate(reports)
    assert summary['tasks'] == 2
    assert summary['tasks_with_suspected_cloud'] == 1
    assert summary['task_suspected_fraction'] == 0.5
    assert summary['videos_with_suspected_cloud'] == 1
    assert summary['video_suspected_fraction'] == pytest.approx(1 / 3)
    assert summary['total_frames'] == 12
    assert summary['frames_in_videos_with_suspected_cloud'] == 4
    assert summary['suspected_frames'] == 3
    assert summary['suspected_fraction_in_affected_videos'] == 0.75
    assert summary['events'] == 2
    assert summary['isolated_single_frames'] == 1
    assert summary['consecutive_events'] == 1
    assert summary['frames_in_consecutive_events'] == 2
    # 同名任务在不同数据集是不同任务，失败视频不进入视频占比分母。
    reports.append({'dataset': 'other', 'task': 'task_a', 'status': 'failed'})
    summary = _aggregate(reports)
    assert summary['tasks'] == 3
    assert summary['video_suspected_fraction'] == pytest.approx(1 / 3)
    assert summary['failed_videos'] == 1
