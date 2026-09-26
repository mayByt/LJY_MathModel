#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配色工具：为填充色派生同色相深色描边，检查灰度亮度、与白底的对比度、色盲下的可分性，
并输出样条图与可直接粘进 style/palettes.py 的片段。

用法：
  python palette_tool.py "#F7A1C4" "#81D4FA" "#7FB3F0" "#B0BEC5" --names 甲 乙 丙 基线
  python palette_tool.py --file 色板.txt [--out 色板检查.png] [--dl 0.16 --ds 0.75]
色板文件里每行一个颜色，可在颜色后写名字（如 “#F7A1C4 甲组”）；只列出的颜色参与两两比较。

判读（CIEDE2000 色差 ΔE00）：≥ 20 类别清楚可分；10–20 需配合标记/线型；< 10 容易混淆。
灰度看 CIELAB 明度 L*：同一张图里要区分的类别，L* 最好相差 ≥ 15，黑白打印才分得开。
描边/线条与白底的对比度建议 ≥ 3:1（WCAG 图形对象标准）。
色盲模拟用 Machado et al. (2009) 严重度 1.0 的矩阵（红色盲 protan、绿色盲 deutan、蓝色盲 tritan）。
"""
from __future__ import annotations

import argparse
import itertools
import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from pathlib import Path

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602), (0.004733, 0.691367, 0.303900)),
}
VISION_CN = {"normal": "正常视觉", "protan": "红色盲", "deutan": "绿色盲", "tritan": "蓝色盲"}


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def rgb2hex(c):
    return "#" + "".join(f"{round(min(max(v, 0), 1) * 255):02X}" for v in c)


def derive_line(fill, dl=0.16, ds=0.75, l_min=0.28):
    """HLS 空间色相不变，亮度降低 dl、饱和度乘 ds（与模板 palettes.derive_line 一致）。"""
    h, l, s = rgb_to_hls(*hex2rgb(fill))
    return rgb2hex(hls_to_rgb(h, max(l - dl, l_min), s * ds))


def to_linear(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


def linear_to_lab(lin):
    r, g, b = lin
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    f = lambda t: t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29   # noqa: E731
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def lab(hexc, vision="normal"):
    lin = to_linear(hex2rgb(hexc))
    if vision != "normal":
        m = MACHADO[vision]
        lin = tuple(min(max(sum(m[i][j] * lin[j] for j in range(3)), 0.0), 1.0) for i in range(3))
    return linear_to_lab(lin)


def simulate_hex(hexc, vision):
    lin = to_linear(hex2rgb(hexc))
    m = MACHADO[vision]
    lin = [min(max(sum(m[i][j] * lin[j] for j in range(3)), 0.0), 1.0) for i in range(3)]
    return rgb2hex([12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055 for v in lin])


def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        dhp = dh if abs(dh) <= 180 else (dh - 360 if dh > 180 else dh + 360)
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    else:
        hbp = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hbp - 30)) + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6)) - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dtheta = 30 * math.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / math.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -math.sin(math.radians(2 * dtheta)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def contrast_white(hexc):
    r, g, b = to_linear(hex2rgb(hexc))
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 1.05 / (y + 0.05)


def parse(args):
    items = []
    if args.file:
        for line in Path(args.file).read_text(encoding="utf-8").splitlines():
            m = re.search(r"#?[0-9A-Fa-f]{6}", line)
            if m:
                name = line.replace(m.group(0), "").strip(" ,，\t") or None
                items.append(("#" + m.group(0).lstrip("#").upper(), name))
    for i, c in enumerate(args.colors):
        items.append(("#" + c.lstrip("#").upper(), None))
    names = args.names or []
    return [(c, n or (names[i] if i < len(names) else f"色{i + 1}")) for i, (c, n) in enumerate(items)]


def worst_pairs(cols, key, vision):
    out = []
    for (c1, n1), (c2, n2) in itertools.combinations(cols, 2):
        out.append((ciede2000(lab(key(c1), vision), lab(key(c2), vision)), n1, n2))
    return sorted(out)


def swatch_figure(rows, path):
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.patches import Rectangle
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    def exists(fam):
        try:
            fm.findfont(fm.FontProperties(family=fam), fallback_to_default=False)
            return True
        except ValueError:
            return False
    cjk = [f for f in ("SimSun", "Songti SC", "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei") if exists(f)][:1]
    plt.rcParams["font.family"] = [f for f in ("Times New Roman",) if exists(f)] + cjk + ["DejaVu Sans"]
    n = len(rows)
    bands = [("填充 + 描边", None), ("灰度", "gray"), ("绿色盲", "deutan"), ("红色盲", "protan")]
    fig, ax = plt.subplots(figsize=(max(4.0, 1.0 * n + 1.2), 3.8))
    ax.set_xlim(-1.3, n)
    ax.set_ylim(-0.2, len(bands) + 0.75)
    ax.set_axis_off()
    for j, (title, mode) in enumerate(bands):
        y = len(bands) - 1 - j
        ax.text(-0.15, y + 0.4, title, ha="right", va="center", fontsize=8)
        for i, (fill, line, name) in enumerate(rows):
            f, ln = fill, line
            if mode == "gray":
                f = ln = None
                gl, gl2 = lab(fill)[0] / 100, lab(line)[0] / 100
                f, ln = str(gl), str(gl2)
            elif mode:
                f, ln = simulate_hex(fill, mode), simulate_hex(line, mode)
            ax.add_patch(Rectangle((i + 0.08, y + 0.1), 0.84, 0.6, fc=f, ec=ln, lw=1.6))
            ax.plot([i + 0.2, i + 0.8], [y + 0.4, y + 0.4], color=ln, lw=1.6)
            if j == 0:
                ax.text(i + 0.5, y + 0.85, f"{name}\n填 {fill}\n描 {line}", ha="center", va="bottom",
                        fontsize=6.5, linespacing=1.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("colors", nargs="*", help="填充色，如 #F7A1C4")
    ap.add_argument("--names", nargs="*", help="与颜色一一对应的名字")
    ap.add_argument("--file", help="色板文件：每行一个颜色，可跟名字")
    ap.add_argument("--dl", type=float, default=0.16, help="描边色亮度降低量（HLS）")
    ap.add_argument("--ds", type=float, default=0.75, help="描边色饱和度倍数（HLS）")
    ap.add_argument("--out", help="样条图输出路径（png）")
    a = ap.parse_args()
    cols = parse(a)
    if not cols:
        raise SystemExit("请给出颜色")

    rows = [(c, derive_line(c, a.dl, a.ds), n) for c, n in cols]
    print(f"{'名字':<8}{'填充':<10}{'描边':<10}{'L*填充':>8}{'L*描边':>8}{'描边对比度':>10}")
    for fill, line, name in rows:
        print(f"{name:<8}{fill:<10}{line:<10}{lab(fill)[0]:>8.1f}{lab(line)[0]:>8.1f}{contrast_white(line):>9.2f}:1"
              + ("  ← 描边偏浅" if contrast_white(line) < 3 else ""))

    if len(rows) >= 2:
        fills = [(f, n) for f, _, n in rows]
        lines = [(ln, n) for _, ln, n in rows]
        print("\n两两色差最小的组合（ΔE00；≥20 清楚，10–20 需加标记/线型，<10 易混）：")
        for label, group in (("填充", fills), ("描边", lines)):
            for vision in ("normal", "deutan", "protan", "tritan"):
                d, n1, n2 = worst_pairs(group, lambda c: c, vision)[0]
                flag = "" if d >= 20 else ("  ← 需冗余编码" if d >= 10 else "  ← 易混淆")
                print(f"  {label} · {VISION_CN[vision]}：{n1}—{n2} {d:.1f}{flag}")
        gray = sorted((lab(f)[0], n) for f, n in fills)
        gaps = [(b[0] - a_[0], a_[1], b[1]) for a_, b in zip(gray, gray[1:])]
        g, n1, n2 = min(gaps)
        print(f"  灰度：明度最接近的是 {n1}—{n2}（ΔL* = {g:.1f}）" + ("  ← 黑白打印难以区分" if g < 15 else ""))

    print("\n# 粘进 style/palettes.py 的 register(...)：把键名换成项目的语义角色")
    print("fill={" + ", ".join(f'"{n}": "{f}"' for f, _, n in rows) + "},")
    print("line={" + ", ".join(f'"{n}": "{ln}"' for _, ln, n in rows) + "},")

    if a.out:
        swatch_figure(rows, a.out)
        print(f"\n样条图：{a.out}")


if __name__ == "__main__":
    main()
