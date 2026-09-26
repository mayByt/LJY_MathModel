#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缩略图总览：把一批 PNG 按文件名排好，拼成若干页（每格带文件名与成图尺寸，并画出画布边框），
用于快速扫一遍整批图：留白是否过大、字号是否一致、配色语义是否统一、有没有明显的遮挡。
总览只用来发现问题，细节仍要逐张打开原图看。

用法：python contact_sheet.py 绘图/output/q1 [更多目录或文件…] [--match "Q1-0*"] [--cols 3]
      [--width 560] [--per-page 9] [--dpi 300] [--out 输出目录]
默认输出到系统临时目录下的 contact_sheets/，不写进成图目录；打印每页路径。需要 pillow。
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def cjk_font(size: int):
    """找一个能显示中文文件名的字体。"""
    try:
        from matplotlib import font_manager as fm
        for fam in ("SimSun", "Songti SC", "PingFang SC", "Heiti SC", "Noto Sans CJK SC",
                    "Source Han Sans SC", "Microsoft YaHei", "WenQuanYi Zen Hei"):
            try:
                return ImageFont.truetype(fm.findfont(fm.FontProperties(family=fam), fallback_to_default=False), size)
            except (ValueError, OSError):
                continue
    except ImportError:
        pass
    return ImageFont.load_default()


def collect(inputs, pattern):
    files = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files += [f for f in p.glob("*.png")]
        elif p.suffix.lower() == ".png":
            files.append(p)
    if pattern:
        files = [f for f in files if fnmatch.fnmatch(f.name, pattern) or fnmatch.fnmatch(f.stem, pattern)]
    return sorted(set(files), key=lambda f: natural_key(f.name))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--match", help="只取文件名匹配的图，如 'Q2-1*'")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--width", type=int, default=560, help="每格缩略图宽度（像素）")
    ap.add_argument("--per-page", type=int, default=9)
    ap.add_argument("--dpi", type=float, default=300, help="用于把像素折算成 mm")
    ap.add_argument("--out", help="输出目录（默认系统临时目录下的 contact_sheets/）")
    a = ap.parse_args()

    files = collect(a.inputs, a.match)
    if not files:
        raise SystemExit("没有找到 PNG")
    tag = Path(a.inputs[0]).name or "figs"
    out = Path(a.out) if a.out else Path(tempfile.gettempdir()) / "contact_sheets" / tag
    out.mkdir(parents=True, exist_ok=True)
    font = cjk_font(17)
    pad, label_h, max_h = 14, 26, int(a.width * 1.1)

    pages = [files[i:i + a.per_page] for i in range(0, len(files), a.per_page)]
    for pi, page in enumerate(pages, 1):
        thumbs = []
        for f in page:
            im = Image.open(f).convert("RGB")
            w, h = im.size
            size = f"{w / a.dpi * 25.4:.0f}×{h / a.dpi * 25.4:.0f} mm"
            scale = min(a.width / w, max_h / h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            thumbs.append((f.stem, size, im))
        rows = [thumbs[i:i + a.cols] for i in range(0, len(thumbs), a.cols)]
        row_h = [max(t[2].size[1] for t in r) + label_h + pad for r in rows]
        W = a.cols * (a.width + pad) + pad
        H = sum(row_h) + pad
        sheet = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(sheet)
        y = pad
        for r, rh in zip(rows, row_h):
            x = pad
            for name, size, im in r:
                sheet.paste(im, (x, y))
                draw.rectangle([x - 1, y - 1, x + im.size[0], y + im.size[1]], outline="#C8CED3")   # 画布边界
                label = f"{name}  ·  {size}"
                while draw.textlength(label, font=font) > a.width and len(name) > 4:
                    name = name[:-1]
                    label = f"{name}…  ·  {size}"
                draw.text((x, y + im.size[1] + 4), label, fill="#333333", font=font)
                x += a.width + pad
            y += rh
        path = out / f"contact_{tag}_{pi:02d}.png"
        sheet.save(path)
        print(path)


if __name__ == "__main__":
    main()
