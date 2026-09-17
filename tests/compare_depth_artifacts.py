#!/usr/bin/env python3
"""深度图斑块诊断：比较编码前、深度视频解码后和网页预览三个阶段的数据。

功能：
    1. 按帧号读取 gray12le 深度视频，可选读取同帧的编码前 NPY 和 H.264 预览。
    2. 将 NPY 转换到编码数值域，模拟裁剪和整数截断，分别计算量化、编码及总误差。
    3. 使用统一灰度范围显示深度图，支持局部裁剪，并比较预览与直接归一化图像。
    4. 输出对比 PNG、深度及误差 NPY、误差统计 JSON 和横截面深度 CSV。

限制：不运行模型；缺少编码前 NPY 时，不能区分模型输出与深度编码的影响。
深度误差的单位取决于编码数值域，预览误差则使用 0–255 显示值域。
依赖：numpy、Pillow、ffmpeg、ffprobe。命令示例见同目录 compare_depth_artifacts.md。
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ============================== 一、视频帧读取 ==============================

def decode(path, frame, pixel_format):
    """按从 0 开始的帧号解码，检查像素格式及字节数，避免隐式转换深度位数。"""
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,pix_fmt', '-of', 'json', str(path)
    ]))['streams'][0]
    if pixel_format == 'gray12le' and info.get('pix_fmt') != 'gray12le':
        raise ValueError(f"Expected gray12le depth, got {info.get('pix_fmt')}; refusing implicit bit-depth conversion")
    raw = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(path), '-vf', f'select=eq(n\\,{frame})',
        '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', pixel_format, '-'
    ])
    dtype = '<u2' if pixel_format == 'gray12le' else 'u1'
    channels = 3 if pixel_format == 'rgb24' else 1
    shape = (info['height'], info['width']) + ((3,) if channels == 3 else ())
    expected = info['height'] * info['width'] * channels * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise ValueError(f'Frame {frame} unavailable or incomplete in {path}')
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


# ============================== 二、误差统计 ==============================

def metrics(a, b, valid):
    """仅统计掩码内的像素；偏差 bias 的方向为 a 减 b。"""
    delta = (a.astype(float) - b.astype(float))[valid]
    if not delta.size:
        return {'valid_pixels': 0}
    return {'valid_pixels': int(delta.size), 'mae': float(np.abs(delta).mean()),
            'rmse': float(np.sqrt(np.mean(delta ** 2))),
            'p99_abs_error': float(np.percentile(np.abs(delta), 99)),
            'max_abs_error': float(np.abs(delta).max()), 'bias': float(delta.mean())}


# ============================== 三、诊断主流程 ==============================

def main():
    # ---------- 3.1 解析输入路径、帧号、数值域转换及显示参数 ----------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth-video', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=0, help='Zero-based source frame index')
    parser.add_argument('--raw-npy', type=Path, help='Pre-encode HxW or TxHxW depth')
    parser.add_argument('--raw-frame', type=int, help='NPY stack index; defaults to --frame')
    parser.add_argument('--raw-scale', type=float, default=1, help='Transform NPY to encoded numeric domain: raw * scale + offset')
    parser.add_argument('--raw-offset', type=float, default=0)
    parser.add_argument('--preview-video', type=Path, help='Inspector H.264 preview, not original depth MP4')
    parser.add_argument('--preview-frame', type=int, help='Defaults to --frame; set explicitly for sparse previews')
    parser.add_argument('--range', type=float, nargs=2, metavar=('MIN', 'MAX'), help='Inspector normalize_min/max; required with preview')
    parser.add_argument('--crop', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--output', type=Path, default=Path('outputs/depth_artifacts'))
    args = parser.parse_args()
    if args.frame < 0 or any(v is not None and v < 0 for v in (args.raw_frame, args.preview_frame)):
        parser.error('Frame indices must be nonnegative')
    if args.preview_video and args.range is None:
        parser.error('--preview-video requires the exact Inspector --range MIN MAX')
    # ---------- 3.2 读取深度视频并确定局部比较区域 ----------
    depth = decode(args.depth_video, args.frame, 'gray12le')
    h, w = depth.shape
    x, y, cw, ch = args.crop or (0, 0, w, h)
    if x < 0 or y < 0 or cw <= 0 or ch <= 0 or x + cw > w or y + ch > h:
        parser.error('--crop must be inside the depth image')
    crop = np.s_[y:y + ch, x:x + cw]
    arrays = {'B_decoded': depth.astype(float)}
    # ---------- 3.3 可选加载编码前数据，模拟编码器的量化步骤 ----------
    if args.raw_npy:
        raw = np.load(args.raw_npy, allow_pickle=False)
        if raw.ndim == 3:
            raw = raw[args.raw_frame if args.raw_frame is not None else args.frame]
        if raw.shape != depth.shape:
            raise ValueError(f'NPY shape {raw.shape} differs from depth {depth.shape}; no automatic resizing')
        raw = raw.astype(float) * args.raw_scale + args.raw_offset
        arrays = {'A_raw': raw, 'A_quantized': np.trunc(np.clip(raw, 0, 4095)), **arrays}
    # ---------- 3.4 确定统一灰度范围并初始化诊断报告 ----------
    valid_depth = (depth > 0) & (depth < 4095)
    if not valid_depth.any():
        raise ValueError('No valid decoded depth pixels')
    lo, hi = args.range or (float(depth[valid_depth].min()), float(depth[valid_depth].max()))
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        parser.error('Display range must be finite and MAX > MIN (use --range for a constant frame)')
    args.output.mkdir(parents=True, exist_ok=True)
    panels = []
    report = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'display_range': [lo, hi], 'crop': [x, y, cw, ch], 'metrics': {},
              'notes': ['Depth errors are in the encoded numeric domain, not necessarily millimeters.',
                        'A_quantized assumes clip(0,4095) then uint16 truncation, as in DepthVideoEncoder.',
                        'Without pre-encode NPY, model errors cannot be separated from source encoding errors.',
                        'Preview comparison includes 8-bit video compression and color conversion; browser GPU rendering is not tested.']}

    # ---------- 3.5 保存各阶段数据及固定范围的灰度图 ----------
    def add_panel(name, pixels):
        """保存单张图片，同时收集到最终对比图中。"""
        image = Image.fromarray(pixels).convert('RGB')
        image.save(args.output / f'{name}.png')
        panels.append((name, image))

    rendered = {}
    for name, data in arrays.items():
        valid = np.isfinite(data) & (data > 0) & (data < 4095)
        gray = np.zeros(depth.shape, np.uint8)
        gray[valid] = np.clip((data[valid] - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
        rendered[name] = gray
        np.save(args.output / f'{name}.npy', data[crop])
        add_panel(name, gray[crop])

    # ---------- 3.6 计算局部误差，保存带符号数组与绝对误差图 ----------
    def compare(name, a, b, valid):
        """误差图各自按最大绝对误差拉伸；白色对应数值写入报告。"""
        a, b, valid = a[crop], b[crop], valid[crop]
        report['metrics'][name] = metrics(a, b, valid)
        delta = a.astype(float) - b.astype(float)
        delta[~valid] = np.nan
        np.save(args.output / f'{name}_error.npy', delta)
        maximum = float(np.max(np.abs(delta[valid]))) if valid.any() else 0
        error = np.zeros(delta.shape, np.uint8)
        if maximum:
            error[valid] = np.clip(np.abs(delta[valid]) * 255 / maximum, 0, 255).astype(np.uint8)
        report['metrics'][name]['error_image_white_equals'] = maximum
        add_panel(f'{name}_abs_error', error)

    # ---------- 3.7 拆分量化误差、视频编码误差和总深度误差 ----------
    if args.raw_npy:
        valid = np.isfinite(arrays['A_raw']) & (arrays['A_raw'] > 0) & (arrays['A_raw'] < 4095) & valid_depth
        compare('quantization', arrays['A_quantized'], arrays['A_raw'], valid)
        compare('codec', arrays['B_decoded'], arrays['A_quantized'], valid)
        compare('total_depth', arrays['B_decoded'], arrays['A_raw'], valid)
        report['raw_clipped_or_invalid_pixels'] = int((~(np.isfinite(arrays['A_raw']) & (arrays['A_raw'] > 0) & (arrays['A_raw'] < 4095)))[crop].sum())
    # ---------- 3.8 比较网页预览与深度直接归一化后的 8 位图 ----------
    if args.preview_video:
        preview = decode(args.preview_video, args.preview_frame if args.preview_frame is not None else args.frame, 'rgb24')
        if preview.shape[:2] not in ((h, w), (h + h % 2, w + w % 2)):
            raise ValueError('Preview dimensions do not match depth, allowing only Inspector even-size padding')
        preview = preview[:h, :w, 0]  # 与 Inspector 直方图一致，取红色通道。
        add_panel('C_preview_red', preview[crop])
        compare('preview_vs_B_8bit', preview, rendered['B_decoded'], np.ones(depth.shape, bool))
    # ---------- 3.9 拼接对比图，导出横截面曲线及 JSON 报告 ----------
    panel_w, panel_h = max(cw, 280), ch + 30
    sheet = Image.new('RGB', (panel_w * min(3, len(panels)), panel_h * ((len(panels) + 2) // 3)), '#202020')
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(panels):
        px, py = (index % 3) * panel_w, (index // 3) * panel_h
        draw.text((px + 5, py + 8), name, fill='white')
        sheet.paste(image, (px, py + 30))
    sheet.save(args.output / 'comparison.png')
    # 导出裁剪区域中间一行的深度值，便于检查阶梯变化，无需额外绘图库。
    row = ch // 2
    names = list(arrays)
    np.savetxt(args.output / 'profile.csv', np.column_stack([np.arange(x, x + cw)] + [arrays[n][crop][row] for n in names]),
               delimiter=',', header=','.join(['x'] + names), comments='')
    report['profile_y'] = y + row
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Saved comparison.png, report.json, profile.csv and arrays to {args.output}")


# ============================== 四、命令行入口 ==============================

if __name__ == '__main__':
    main()
