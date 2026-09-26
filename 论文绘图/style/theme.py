# -*- coding: utf-8 -*-
"""论文插图统一样式：字体、字号、尺寸与配色。

- 中文宋体，英文与数字 Times New Roman，靠 matplotlib（≥3.6）的逐字形回退实现混排；公式用 STIX。
- 中文与 ``$...$`` 不能写在同一个字符串里（mathtext 不回退，中文会变成方框）。
  需要时拆成两个文本对象，或直接用 Unicode（φ、τ、≥、×、²）。
- 图内不放标题和说明文字，说明写进论文图注。
- 配色方案在 palettes.py 注册，第 2 步讨论定稿后修改 ACTIVE_SCHEME。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

MM = 1 / 25.4

# 字体。目标期刊要求无衬线体时（如 Nature、Cell 要求 Arial/Helvetica），改这里并把中文换成黑体。
FONTS = {"en": "Times New Roman", "cn": "SimSun", "math": "stix"}

# 版心宽度（mm）：A4 论文正文约 160 mm。投期刊时按期刊栏宽改。
WIDTHS = {"full": 160, "two_thirds": 107, "half": 78}

# 字号（pt），按最终印刷尺寸。图按实际尺寸建、插入论文时不缩放，各图字号才一致。
FONT_SIZES = {"label": 9, "tick": 8, "legend": 8, "annot": 7.5}


@dataclass(frozen=True)
class Palette:
    """一套配色。fill 是填充色（柱、面、点、置信带），line 是同色相加深的线条/描边色。

    角色键由项目自定并全篇固定，例如主类别 c1/c2/c3（可改名为 T/A/V）、
    方法 ours/alt/base、有符号量 neg/neu/pos。
    """
    name: str
    fill: dict
    line: dict
    accent: str                       # 该问主题强调色
    seq: list | str                   # 顺序色图：锚点列表、matplotlib 名称或 "cmc:名称"（cmcrameri）
    div: list | str                   # 发散色图（负 → 0 → 正），配合 TwoSlopeNorm(0) 使用
    ink: str = "#333333"              # 文字与坐标轴
    grid: str = "#E8ECEF"             # 网格线
    ref: str = "#A9B4BA"              # 参考线：零线、y = x、阈值线
    extra: dict = field(default_factory=dict)

    def pair(self, key: str) -> tuple[str, str]:
        """(填充色, 描边色)。"""
        return self.fill[key], self.line[key]

    @staticmethod
    def _cmap(spec, name):
        if isinstance(spec, str):
            if spec.startswith("cmc:"):
                try:
                    import cmcrameri.cm as cmc
                except ImportError as e:
                    raise ImportError("该配色用到 cmcrameri 色图，请先安装：pip install cmcrameri") from e
                return getattr(cmc, spec[4:])
            return mpl.colormaps[spec]
        return LinearSegmentedColormap.from_list(name, spec)

    def cmap_seq(self):
        return self._cmap(self.seq, f"{self.name}_seq")

    def cmap_div(self, soft=False):
        spec=self.extra.get("div_soft",self.div) if soft else self.div
        return self._cmap(spec, f"{self.name}_div")

    def cmap_seq_soft(self):
        return self._cmap(self.extra.get("seq_soft",self.seq), f"{self.name}_seq_soft")

    def combo(self, keys, kind: str = "fill") -> str:
        """组合类别的颜色：各单项颜色做 RGB 平均，保持同色系。keys 可为 "TA" 或 ["c1", "c2"]。"""
        src = self.fill if kind == "fill" else self.line
        rgbs = [to_rgb(src[k]) for k in keys]
        return to_hex(tuple(sum(c[i] for c in rgbs) / len(rgbs) for i in range(3)))


SCHEMES: dict[str, dict] = {}         # 由 palettes.py 注册
ACTIVE_SCHEME = "forest"
CURRENT: Palette | None = None        # 最近一次 use() 应用的配色（供公共图元取参考线颜色等）


def palette(question: str = "default", scheme: str | None = None) -> Palette:
    from style import palettes  # noqa: F401  导入即注册 SCHEMES

    spec = SCHEMES[scheme or ACTIVE_SCHEME]
    qs = spec["questions"]
    q = {**qs["default"], **qs.get(question, {})}
    opt = {k: spec[k] for k in ("ink", "grid", "ref") if k in spec}
    return Palette(name=f"{spec['name']}_{question}", fill=spec["fill"], line=spec["line"],
                   accent=q["accent"], seq=q["seq"], div=spec["div"], extra=spec.get("extra", {}), **opt)


def _rc(p: Palette, cycle: list) -> dict:
    fs = FONT_SIZES
    return {
        "font.family": [FONTS["en"], FONTS["cn"], "STIXGeneral", "STIX Two Text"],
        "mathtext.fontset": FONTS["math"],
        "axes.unicode_minus": True,
        "font.size": fs["label"],
        "axes.labelsize": fs["label"],
        "axes.titlesize": fs["label"],
        "xtick.labelsize": fs["tick"],
        "ytick.labelsize": fs["tick"],
        "legend.fontsize": fs["legend"],
        "legend.frameon": True,                      # 白色不透明底、无边框：网格线与数据不会穿过图例
        "legend.facecolor": "white",
        "legend.edgecolor": "none",
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.15,
        "axes.edgecolor": p.ink,
        "axes.labelcolor": p.ink,
        "text.color": p.ink,
        "xtick.color": p.ink,
        "ytick.color": p.ink,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": p.grid,
        "grid.linewidth": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
        "axes.labelpad": 3,
        "legend.labelspacing": .35,
        "patch.linewidth": 0.7,
        "hatch.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,                          # 嵌入 TrueType：PDF 文字可选中、可编辑
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.prop_cycle": mpl.cycler(color=[p.line[k] for k in cycle]),
    }


def use(question: str = "default", scheme: str | None = None) -> Palette:
    """应用全局样式并返回该问配色。"""
    global CURRENT
    p = palette(question, scheme)
    spec = SCHEMES[scheme or ACTIVE_SCHEME]
    cycle = spec.get("cycle") or list(p.line)[:3]
    mpl.rcParams.update(_rc(p, cycle))
    CURRENT = p
    return p
