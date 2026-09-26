#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交付前检查成图目录（递归）：
- 每张图的 PNG 与 PDF 成对；
- PNG 分辨率等于指定 dpi（默认 300），并给出按该 dpi 折算的成图尺寸（mm）；
- PDF 嵌入了字体、没有 Type 3 字体，用到的字体都在允许列表内（默认 Times New Roman、SimSun、STIX）；
  出现 DejaVu 等字体通常说明有字符回退到了默认字体；
- 每个目录的 stats.json 可以读取。

用法：python check_output.py 绘图/output [--dpi 300] [--allow TimesNewRoman SimSun STIX] [--list]
退出码：0 = 通过；1 = 有问题。只解析 matplotlib 生成的 PDF（对象未压缩），其他工具生成的 PDF 可能读不出字体。
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path


def png_info(path: Path):
    """读取 PNG 的像素尺寸与 pHYs 中的 dpi（不依赖 pillow）。"""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None, None
    pos, w, h, dpi = 8, None, None, None
    while pos + 8 <= len(data):
        n, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if typ == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif typ == b"pHYs":
            px, py, unit = struct.unpack(">IIB", body[:9])
            if unit == 1:
                dpi = (px * 0.0254, py * 0.0254)
        elif typ == b"IDAT":
            break
        pos += 12 + n
    return w, h, dpi


def pdf_fonts(data: bytes):
    names = sorted({m.decode("latin-1").split("+", 1)[-1]
                    for m in re.findall(rb"/BaseFont\s*/([^\s/\[\]<>()]+)", data)})
    type3 = bool(re.search(rb"/Subtype\s*/Type3", data))
    embedded = len(re.findall(rb"/FontFile[23]?\b", data))
    return names, type3, embedded


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="成图目录，如 绘图/output")
    ap.add_argument("--dpi", type=float, default=300)
    ap.add_argument("--allow", nargs="+", default=["TimesNewRoman", "SimSun", "STIX"],
                    help="允许出现在 PDF 中的字体名前缀（忽略大小写、空格与连字符）")
    ap.add_argument("--list", action="store_true", help="逐图列出尺寸与字体")
    a = ap.parse_args()

    root = Path(a.root)
    if not root.is_dir():
        sys.exit(f"目录不存在：{root}")
    allow = [norm(x) for x in a.allow]
    problems = []
    per_dir = defaultdict(lambda: defaultdict(set))
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in (".png", ".pdf"):
            per_dir[p.parent][p.stem].add(p.suffix.lower())

    total = 0
    for d, stems in sorted(per_dir.items()):
        rel = d.relative_to(root) if d != root else Path(".")
        n_ok = 0
        for stem, exts in sorted(stems.items()):
            total += 1
            issues = []
            if exts != {".png", ".pdf"}:
                issues.append("缺 " + ("PDF" if ".pdf" not in exts else "PNG"))
            size = ""
            if ".png" in exts:
                w, h, dpi = png_info(d / f"{stem}.png")
                if dpi is None:
                    issues.append("PNG 无 dpi 信息")
                elif abs(dpi[0] - a.dpi) > 1 or abs(dpi[1] - a.dpi) > 1:
                    issues.append(f"PNG {dpi[0]:.0f} dpi（应为 {a.dpi:.0f}）")
                if w and h:
                    size = f"{w / a.dpi * 25.4:.0f}×{h / a.dpi * 25.4:.0f} mm"
            fonts = []
            if ".pdf" in exts:
                fonts, type3, embedded = pdf_fonts((d / f"{stem}.pdf").read_bytes())
                if type3:
                    issues.append("PDF 含 Type 3 字体（设 pdf.fonttype = 42）")
                if fonts and not embedded:
                    issues.append("PDF 未嵌入字体")
                other = [f for f in fonts if not any(norm(f).startswith(x) for x in allow)]
                if other:
                    issues.append("PDF 出现其他字体：" + "、".join(other))
            if a.list:
                print(f"  {rel / stem}  {size}  {'、'.join(fonts)}")
            if issues:
                problems.append(f"{rel / stem}: " + "；".join(issues))
            else:
                n_ok += 1
        stats = d / "stats.json"
        note = ""
        if stats.exists():
            try:
                note = f"，stats.json {len(json.loads(stats.read_text(encoding='utf-8')))} 条"
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                problems.append(f"{rel / 'stats.json'}: 无法读取（{e}）")
        print(f"[{rel}] {len(stems)} 张图，{n_ok} 张通过{note}")

    print(f"\n共 {total} 张图。" + ("全部通过。" if not problems else f"{len(problems)} 张有问题："))
    for s in problems:
        print("  ✗ " + s)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
