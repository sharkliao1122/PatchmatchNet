"""Batch colorize millimeter depth PNGs; original images are never modified."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_depth(path):
    with Image.open(path) as im:
        a = np.array(im)
    if a.ndim != 2 or a.dtype.kind not in 'uif':
        raise ValueError(f'Expected single-channel numeric depth: {path}')
    return a


def colors(t):
    # Jet: near = red, far = blue.
    x = 1 - np.clip(t, 0, 1)
    rgb = np.stack([np.clip(1.5 - np.abs(4*x-k), 0, 1) for k in (3, 2, 1)], axis=-1)
    return np.rint(rgb * 255).astype(np.uint8)


def add_color_scale(image, lo, hi):
    """Overlay a labeled scale inside the image, preserving pixel dimensions."""
    panel = Image.new('RGB', (600, 84), (24, 24, 24))
    draw = ImageDraw.Draw(panel)
    draw.text((14, 7), 'Depth (mm)    Near: red / Far: blue    Invalid: black', fill='white')
    ramp = colors(np.linspace(0, 1, 560))
    panel.paste(Image.fromarray(np.repeat(ramp[None, :, :], 16, axis=0)), (20, 27))
    for f in (0, .25, .5, .75, 1):
        x = 20 + round(559*f)
        draw.line((x, 43, x, 47), fill='white')
        label = f'{lo+(hi-lo)*f:g}'
        box = draw.textbbox((0, 0), label)
        tx = max(4, min(596-(box[2]-box[0]), x-(box[2]-box[0])/2))
        draw.text((tx, 49), label, fill='white')
    draw.text((14, 66), 'Depth outside this range uses the nearest endpoint color.', fill='white')
    w, h = image.size
    scale = min(1, max(1, w-16)/600, max(1, h-16)/84)
    panel = panel.resize((max(1, round(600*scale)), max(1, round(84*scale))), Image.Resampling.LANCZOS)
    image.paste(panel, ((w-panel.width)//2, max(0, h-panel.height-8)))
    return image


def main():
    p = argparse.ArgumentParser(description='Depth PNG colorization (input unit: mm; zero is invalid).')
    p.add_argument('input', nargs='?', type=Path, default=Path(__file__).resolve().parent/'20260914_60_1_cm_output'/'depth_aligned')
    p.add_argument('-o', '--output', type=Path)
    p.add_argument('--min-mm', type=float, help='Fixed minimum depth in millimeters')
    p.add_argument('--max-mm', type=float, help='Fixed maximum depth in millimeters')
    p.add_argument('--no-legend', action='store_true', help='Disable the in-image depth scale')
    args = p.parse_args()
    files = sorted(args.input.glob('*.png'))
    if not files:
        p.error(f'No PNG files: {args.input}')
    out = args.output or args.input.with_name(args.input.name + '_color')
    if out.resolve() == args.input.resolve():
        p.error('Output must differ from input.')
    if out.exists() and any(out.iterdir()):
        p.error(f'Output is not empty; choose a new --output directory: {out}')
    lo, hi = args.min_mm, args.max_mm
    if lo is None or hi is None:
        # Bounded sample across the sequence, same range for every frame.
        samples = []
        for i in np.unique(np.linspace(0, len(files)-1, min(100, len(files))).astype(int)):
            a = read_depth(files[i])[::4, ::4]
            samples.append(a[np.isfinite(a) & (a > 0)])
        values = np.concatenate(samples)
        if not values.size:
            p.error('No valid depth samples; specify --min-mm and --max-mm.')
        low, high = np.percentile(values, [1, 99])
        lo = float(np.floor(low)) if lo is None else lo
        hi = float(np.ceil(high)) if hi is None else hi
        if args.min_mm is None and args.max_mm is None and hi <= lo:
            hi = lo + 1
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        p.error('Range must be finite and max-mm > min-mm.')
    out.mkdir(parents=True, exist_ok=True)
    print(f'{len(files)} images; shared range {lo:g}..{hi:g} mm; output: {out}', flush=True)
    for i, path in enumerate(files, 1):
        a = read_depth(path)
        valid = np.isfinite(a) & (a > 0)
        t = (np.where(valid, a, lo).astype(np.float32)-lo)/(hi-lo)
        rgb = colors(t)
        rgb[~valid] = 0
        result = Image.fromarray(rgb)
        if not args.no_legend:
            result = add_color_scale(result, lo, hi)
        result.save(out/path.name)
        if i == 1 or i % 100 == 0 or i == len(files):
            print(f'{i}/{len(files)}', flush=True)
    legend = Image.new('RGB', (640, 100), 'white')
    ramp = colors(np.linspace(0, 1, 600))
    legend.paste(Image.fromarray(np.repeat(ramp[None, :, :], 25, axis=0)), (20, 15))
    draw = ImageDraw.Draw(legend)
    for f in (0, .25, .5, .75, 1):
        draw.text((min(570, 20+int(600*f)), 45), f'{lo+(hi-lo)*f:g} mm', fill='black')
    draw.text((20, 70), 'Near: red | Far: blue | Invalid: black | Out of range: clipped', fill='black')
    legend.save(out/'color_scale.png')
    (out/'colorization.json').write_text(json.dumps(dict(input=str(args.input.resolve()), count=len(files), min_mm=lo, max_mm=hi, colormap='jet reversed (near red)', invalid='nonfinite or <=0: black (outside legend overlay)', legend_overlay=not args.no_legend, range_method='explicit bounds or sampled sequence 1st/99th percentiles', outside_range='clipped'), indent=2), encoding='utf-8')
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
