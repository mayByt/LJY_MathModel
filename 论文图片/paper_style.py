"""全文统一绘图风格（论文终稿版）。

所有重绘图片均 `from paper_style import *` 后调用 `apply_style()`。
设计原则：
- 物理尺寸作图：单栏整宽 FULL_W=6.3in（约 16cm），在 LaTeX 中以 \\linewidth 插入，
  图中字号 ≈ 正文小五号，保证全文图文字号一致；半宽图用 HALF_W。
- 中文宋体（Noto Serif CJK SC）+ 西文 Times 度量（Liberation Serif）+ STIX 数学字体，
  与正文“宋体 + Times”一致。
- 统一配色：6 个类别色 + 1 套顺序色 + 1 套发散色，语义映射见 SEMANTIC。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# ---------------------------------------------------------------- 尺寸
FULL_W = 6.3          # 英寸，整宽
HALF_W = 3.1          # 英寸，半宽
PANEL_H = 2.35        # 英寸，单行子图推荐高度

# ---------------------------------------------------------------- 颜色
BLUE = "#2F5D8A"      # 主色：观测、主模型、指数成本
ORANGE = "#D98A3A"    # 次色：拟合/对照、幂函数成本
GREEN = "#4E9A78"     # 第三色：对数成本、第三类
RED = "#C0504D"       # 强调：阈值线、临界、警示
PURPLE = "#7B6BA8"    # 第五类
GRAY = "#8C8C8C"      # 参考、背景数据、估算值
INK = "#262626"       # 文字与坐标轴
GRID = "#E6E6E6"

CATEGORICAL = [BLUE, ORANGE, GREEN, RED, PURPLE, GRAY]


def lighten(color: str, amount: float = 0.6) -> tuple[float, float, float]:
    """向白色混合，amount=0 原色，1 白色。用于置信带、填充。"""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def darken(color: str, amount: float = 0.3) -> tuple[float, float, float]:
    r, g, b = to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


# 顺序色：浅 → 主蓝 → 深蓝；发散色：主蓝 ← 白 → 橙
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "paper_seq", ["#F4F7FB", "#C9D8E8", "#7FA3C6", BLUE, "#1B3552"]
)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "paper_div", ["#1B3552", BLUE, "#9DB7D1", "#F7F7F7", "#EDC39A", ORANGE, "#8A4E14"]
)
# 数值有序的离散序列（如三档预算、三档规模），从浅到深取
def seq_colors(n: int, lo: float = 0.35, hi: float = 0.95) -> list:
    if n == 1:
        return [SEQ_CMAP(hi)]
    return [SEQ_CMAP(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


# 语义映射：同一概念在全文用同一颜色
SEMANTIC = {
    # 质量成本函数
    "exp": BLUE, "指数": BLUE,
    "pow": ORANGE, "幂函数": ORANGE,
    "log": GREEN, "对数": GREEN,
    # 观测/拟合/参考
    "observed": BLUE, "fit": ORANGE, "reference": GRAY, "threshold": RED,
    # 算力增速情景
    "kappa_1.0": BLUE, "kappa_0.5": ORANGE, "kappa_0.25": GREEN,
    # 贡献分解
    "scale": BLUE, "tech": ORANGE,
    # 质量语义块
    "education": BLUE, "expression": ORANGE, "reasoning": GREEN,
    "noise": RED, "structure": PURPLE,
    # 映射证据等级
    "direct": BLUE, "near_direct": GREEN, "inferred": GRAY,
}
# 三档预算、三档实测规模：顺序色
BUDGET_COLORS = dict(zip(["1e19", "1e22", "1e24"], seq_colors(3)))
SCALE_COLORS = dict(zip(["1M", "60M", "1B"], seq_colors(3)))

# ---------------------------------------------------------------- 字体
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    str(Path(__file__).resolve().parent / "问题二/font_assets/NotoSerifCJKsc-Regular.otf"),
]


def _register_fonts() -> None:
    for path in _CJK_CANDIDATES:
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
            except Exception:  # noqa: BLE001
                pass


def apply_style() -> None:
    _register_fonts()
    mpl.rcParams.update({
        "font.family": ["Noto Serif CJK SC"],  # 中西文同一衬线字体，避免 mathtext 混排缺字
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.title_fontsize": 7.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Noto Serif CJK SC",
        "mathtext.it": "Liberation Serif:italic",
        "mathtext.bf": "Liberation Serif:bold",
        "mathtext.fallback": "stix",
        "axes.unicode_minus": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, text: str, x: float = -0.02, y: float = 1.02) -> None:
    """子图编号 (a)(b)…：左上角，加粗，与正文引用“图 x(a)”一致。"""
    ax.text(x, y, f"({text})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color=INK)


def save(fig, out_stem: str | Path) -> None:
    """同时输出矢量 PDF（进论文）和 PNG（预览）。out_stem 不含扩展名。"""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
