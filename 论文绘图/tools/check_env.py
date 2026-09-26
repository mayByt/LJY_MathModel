#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘图环境检查：依赖、字体（中文宋体 + 英文 Times New Roman）、中英混排是否缺字、PDF 是否嵌入字体。

用法：python check_env.py [--cn SimSun] [--en "Times New Roman"]
退出码：0 = 可以开始画图；1 = 有必须先处理的问题（缺必需依赖、缺字体、缺字形）。
"""
from __future__ import annotations

import argparse
import importlib
import platform
import re
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

REQUIRED = {"matplotlib": "3.6", "numpy": None, "pandas": None, "scipy": None}
OPTIONAL = {
    "cmcrameri": "Crameri 科学色图",
    "PIL": "缩略图总览、读取 PNG（pillow）",
    "squarify": "矩形树图",
    "openpyxl": "读取 xlsx",
    "librosa": "语谱图",
    "cv2": "读取视频帧（opencv-python）",
}


def vtuple(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def check_packages() -> bool:
    ok = True
    print("== 依赖")
    for name, need in REQUIRED.items():
        try:
            v = getattr(importlib.import_module(name), "__version__", "?")
            bad = bool(need) and vtuple(v) < vtuple(need)
            print(f"  {'✗' if bad else '✓'} {name} {v}" + (f"（需要 ≥ {need}：中英逐字形回退从 3.6 起才支持）" if bad else ""))
            ok &= not bad
        except ImportError:
            print(f"  ✗ {name} 未安装")
            ok = False
    for name, use in OPTIONAL.items():
        try:
            v = getattr(importlib.import_module(name), "__version__", "")
            print(f"  · {name} {v}  {use}")
        except ImportError:
            print(f"  · {name} 未安装  {use}（用到时再装）")
    print(f"  · ffmpeg {'可用' if shutil.which('ffmpeg') else '未找到'}（解码音视频时才需要）")
    return ok


def find_font(family: str):
    from matplotlib import font_manager as fm
    try:
        return fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
    except ValueError:
        return None


def pdf_fonts(data: bytes):
    """从 matplotlib 生成的 PDF 中读出字体名（去掉子集前缀）、是否含 Type 3 字体、嵌入的字体文件数。"""
    names = sorted({m.decode("latin-1").split("+", 1)[-1]
                    for m in re.findall(rb"/BaseFont\s*/([^\s/\[\]<>()]+)", data)})
    type3 = bool(re.search(rb"/Subtype\s*/Type3", data))
    embedded = len(re.findall(rb"/FontFile[23]?\b", data))
    return names, type3, embedded


def advice(cache: str) -> str:
    sysname = platform.system()
    if sysname == "Darwin":
        return ("macOS 自带 Times New Roman，但没有 SimSun：把 simsun.ttc（Windows 的 C:\\Windows\\Fonts 或 Office 自带）"
                f"拷到 ~/Library/Fonts/，再删除 matplotlib 字体缓存：rm -rf '{cache}'")
    if sysname == "Linux":
        return ("把 simsun.ttc 与 times*.ttf 拷到 ~/.local/share/fonts/，运行 fc-cache -fv，"
                f"再删除 matplotlib 字体缓存：rm -rf '{cache}'")
    return f"确认已安装字体；仍找不到时删除 matplotlib 字体缓存目录 {cache} 后重试"


def check_fonts(en: str, cn: str) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import logging
    import matplotlib.pyplot as plt

    logging.getLogger("fontTools").setLevel(logging.ERROR)
    ok = True
    print("== 字体")
    for fam in (en, cn):
        path = find_font(fam)
        print(f"  {'✓' if path else '✗'} {fam}: {path or '未找到'}")
        ok &= bool(path)
    if not ok:
        print("  → " + advice(matplotlib.get_cachedir()))
        return False

    plt.rcParams.update({"font.family": [en, cn], "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(3.2, 1.4))
    ax.set_xlabel("时间 / s（Times New Roman 0123）")
    ax.set_ylabel("得分 φ ≥ 0.5")
    ax.text(0.05, 0.6, r"$\alpha_{t}^{2}$", transform=ax.transAxes)       # 公式单独成串
    ax.text(0.45, 0.6, "中文与英文 Mixed 95%", transform=ax.transAxes)
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        png, pdf = Path(d) / "t.png", Path(d) / "t.pdf"
        fig.savefig(png, dpi=150)
        fig.savefig(pdf)
        names, type3, embedded = pdf_fonts(pdf.read_bytes())
    plt.close(fig)
    missing = sorted({str(w.message) for w in caught if "Glyph" in str(w.message) or "missing from" in str(w.message)})
    print("== 中英混排测试")
    if missing:
        ok = False
        for m in missing[:5]:
            print(f"  ✗ 缺字：{m}")
    else:
        print("  ✓ 无缺字")
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())                  # noqa: E731
    need = {norm(en): False, norm(cn): False}
    for n in names:
        for k in need:
            if norm(n).startswith(k):
                need[k] = True
    print(f"  PDF 字体：{', '.join(names) or '（无）'}；嵌入字体文件 {embedded} 个；Type 3：{'有' if type3 else '无'}")
    if not all(need.values()) or type3 or not embedded:
        ok = False
        print("  ✗ PDF 未按预期嵌入两种字体（检查 pdf.fonttype = 42 与字体安装）")
    else:
        print("  ✓ PDF 已嵌入中英两种字体")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--en", default="Times New Roman", help="英文字体（默认 Times New Roman）")
    ap.add_argument("--cn", default="SimSun", help="中文字体（默认 SimSun 宋体）")
    a = ap.parse_args()
    print(f"Python {sys.version.split()[0]}  {platform.system()} {platform.machine()}")
    ok = check_packages()
    ok = check_fonts(a.en, a.cn) and ok
    print("\n结论：" + ("环境可用。" if ok else "有问题需要先处理（见上方 ✗）。"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
