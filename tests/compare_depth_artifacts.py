#!/usr/bin/env python3
"""Compare pre-encode depth, decoded gray12le depth and an optional browser preview.

Requires numpy, Pillow, ffmpeg and ffprobe. See compare_depth_artifacts.md.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def decode(path, frame, pixel_format):
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


def metrics(a, b, valid):
    delta = (a.astype(float) - b.astype(float))[valid]
    if not delta.size:
        return {'valid_pixels': 0}
    return {'valid_pixels': int(delta.size), 'mae': float(np.abs(delta).mean()),
            'rmse': float(np.sqrt(np.mean(delta ** 2))),
            'p99_abs_error': float(np.percentile(np.abs(delta), 99)),
            'max_abs_error': float(np.abs(delta).max()), 'bias': float(delta.mean())}


def main():
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
    depth = decode(args.depth_video, args.frame, 'gray12le')
    h, w = depth.shape
    x, y, cw, ch = args.crop or (0, 0, w, h)
    if x < 0 or y < 0 or cw <= 0 or ch <= 0 or x + cw > w or y + ch > h:
        parser.error('--crop must be inside the depth image')
    crop = np.s_[y:y + ch, x:x + cw]
    arrays = {'B_decoded': depth.astype(float)}
    if args.raw_npy:
        raw = np.load(args.raw_npy, allow_pickle=False)
        if raw.ndim == 3:
            raw = raw[args.raw_frame if args.raw_frame is not None else args.frame]
        if raw.shape != depth.shape:
            raise ValueError(f'NPY shape {raw.shape} differs from depth {depth.shape}; no automatic resizing')
        raw = raw.astype(float) * args.raw_scale + args.raw_offset
        arrays = {'A_raw': raw, 'A_quantized': np.trunc(np.clip(raw, 0, 4095)), **arrays}
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

    def add_panel(name, pixels):
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

    def compare(name, a, b, valid):
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

    if args.raw_npy:
        valid = np.isfinite(arrays['A_raw']) & (arrays['A_raw'] > 0) & (arrays['A_raw'] < 4095) & valid_depth
        compare('quantization', arrays['A_quantized'], arrays['A_raw'], valid)
        compare('codec', arrays['B_decoded'], arrays['A_quantized'], valid)
        compare('total_depth', arrays['B_decoded'], arrays['A_raw'], valid)
        report['raw_clipped_or_invalid_pixels'] = int((~(np.isfinite(arrays['A_raw']) & (arrays['A_raw'] > 0) & (arrays['A_raw'] < 4095)))[crop].sum())
    if args.preview_video:
        preview = decode(args.preview_video, args.preview_frame if args.preview_frame is not None else args.frame, 'rgb24')
        if preview.shape[:2] not in ((h, w), (h + h % 2, w + w % 2)):
            raise ValueError('Preview dimensions do not match depth, allowing only Inspector even-size padding')
        preview = preview[:h, :w, 0]  # Inspector histogram also samples the red channel.
        add_panel('C_preview_red', preview[crop])
        compare('preview_vs_B_8bit', preview, rendered['B_decoded'], np.ones(depth.shape, bool))
    panel_w, panel_h = max(cw, 280), ch + 30
    sheet = Image.new('RGB', (panel_w * min(3, len(panels)), panel_h * ((len(panels) + 2) // 3)), '#202020')
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(panels):
        px, py = (index % 3) * panel_w, (index // 3) * panel_h
        draw.text((px + 5, py + 8), name, fill='white')
        sheet.paste(image, (px, py + 30))
    sheet.save(args.output / 'comparison.png')
    # Numeric cross-section makes depth steps visible without a plotting dependency.
    row = ch // 2
    names = list(arrays)
    np.savetxt(args.output / 'profile.csv', np.column_stack([np.arange(x, x + cw)] + [arrays[n][crop][row] for n in names]),
               delimiter=',', header=','.join(['x'] + names), comments='')
    report['profile_y'] = y + row
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Saved comparison.png, report.json, profile.csv and arrays to {args.output}")


if __name__ == '__main__':
    main()
