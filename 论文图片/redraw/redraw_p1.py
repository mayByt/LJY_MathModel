"""问题一论文终稿插图重绘（P1_F01–P1_F10）。

一次运行生成全部 10 张图：
    python redraw_p1.py            # 全部
    python redraw_p1.py 3 5        # 只画指定编号
数据只来自 论文图片/问题一/figure_data 与 问题一/results_p12_final_v2；
所有派生量均为确定性计算，公式写在对应函数注释中。
"""
import sys
sys.path.insert(0, "/home/chenxu/MayByte/LJY/论文图片")
from paper_style import *  # noqa: F401,F403
apply_style()

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import gaussian_kde

# ============================================================== 名称字典（全文统一）
DOMAIN_ZH = {
    # 17 个配比域
    "ubuntu_irc": "技术对话", "enron_emails": "商务邮件", "dm_mathematics": "数学推理",
    "nih_exporter": "科研项目", "hackernews": "科技论坛", "gutenberg_pg_19": "古典图书",
    "stackexchange": "技术问答", "pubmed_central": "生物医学全文", "wikipedia_en": "英文百科",
    "europarl": "议会文本", "uspto_backgrounds": "专利背景", "freelaw": "法律文本",
    "arxiv": "学术论文", "github": "代码", "pile_cc": "网页语料",
    "pubmed_abstracts": "生医摘要", "philpapers": "哲学论文",
}
QUALITY_ZH = {  # 7 个质量信号域
    "c4": "C4网页", "commoncrawl": "通用网页", "arxiv": "学术论文", "stackexchange": "技术问答",
    "book": "图书", "wikipedia": "百科", "github": "代码",
}
BLOCKS = ["education", "expression", "reasoning", "noise", "structure"]
BLOCK_ZH = {"education": "教育价值", "expression": "表达整洁", "reasoning": "推理信息",
            "noise": "噪声重复", "structure": "结构充分"}
BLOCK_COLOR = {b: SEMANTIC[b] for b in BLOCKS}
INDICATOR_ZH = {
    "rps_doc_num_sentences": "句数适度", "rps_doc_frac_chars_top_2gram": "低二元重复",
    "rps_doc_frac_chars_top_3gram": "低三元重复", "rps_lines_uppercase_letter_fraction": "大写适度",
    "rps_lines_ending_with_terminal_punctution_mark": "终止标点", "rps_doc_frac_no_alph_words": "字母词比例",
    "fineweb_edu": "教育价值", "rps_doc_word_count": "篇幅适度", "modernbert_reasoning": "推理性",
    "modernbert_professionalism": "专业性", "rps_doc_frac_unique_words": "独特词比例",
    "rps_doc_unigram_entropy": "词元熵", "dsir_math": "数学相似度", "dsir_books": "图书相似度",
    "dsir_wiki": "百科相似度", "fluency_en": "语言流畅", "rps_doc_mean_word_length": "词长适度",
    "qurater": "综合写作", "modernbert_readability": "可读性", "modernbert_cleanliness": "整洁度",
    "ad_en": "无广告", "rps_lines_numerical_chars_fraction": "数字适度",
}
MAP_ZH = {"direct": "直接映射", "near_direct": "近直接映射", "inferred": "推断映射"}
SCALE_ZH = {"test_1m": "1M", "test_60m": "60M", "test_1B": "1B"}

FD = Path("/home/chenxu/MayByte/LJY/论文图片/问题一/figure_data")
RD = Path("/home/chenxu/MayByte/LJY/问题一/results_p12_final_v2")
OUT = Path("/home/chenxu/MayByte/LJY/overleaf-paper/images/no1")


def rd(name):
    return pd.read_csv(FD / name)


def dz(x):
    return DOMAIN_ZH.get(x, x)


# ============================================================== F01
def fig01():
    """(a) 22 项同向化指标 Spearman 相关矩阵（按语义块、块内按权重排序）；(b) 总权重棒棒糖图。
    相关矩阵取 P1_F01_indicator_correlation.csv，与 P1_F01_indicator_weights.csv 同一指标序、
    同一套权重（旧稿亦用此文件）。"""
    corr = pd.read_csv(FD / "P1_F01_indicator_correlation.csv", index_col=0)
    w = rd("P1_F01_indicator_weights.csv")
    w["bi"] = w.block.map({b: i for i, b in enumerate(BLOCKS)})
    w = w.sort_values(["bi", "total_weight"], ascending=[True, False]).reset_index(drop=True)
    order = w.field.tolist()
    C = corr.loc[order, order].values
    n = len(order)

    fig = plt.figure(figsize=(FULL_W, 3.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.25], wspace=0.08,
                          left=0.125, right=0.985, top=0.93, bottom=0.14)
    ax = fig.add_subplot(gs[0])
    cax = ax.inset_axes([0.3, -0.075, 0.4, 0.028])
    axb = fig.add_subplot(gs[1], sharey=ax)

    im = ax.imshow(C, cmap=DIV_CMAP, norm=TwoSlopeNorm(0, -1, 1), aspect="auto",
                   extent=(-0.5, n - 0.5, n - 0.5, -0.5))
    ax.grid(False)
    ax.set_yticks(range(n))
    ax.set_yticklabels([INDICATOR_ZH[f] for f in order])
    ax.set_xticks([])
    # 块分隔线 + 顶部与左侧色带
    starts = [(g.index.min(), g.index.max()) for _, g in w.groupby("bi")]
    for (s, e), b in zip(starts, BLOCKS):
        for v in (s - 0.5,):
            if s > 0:
                ax.axhline(v, color="white", lw=1.2)
                ax.axvline(v, color="white", lw=1.2)
        ax.add_patch(Rectangle((s - 0.5, -1.35), e - s + 1, 0.7, color=BLOCK_COLOR[b],
                               clip_on=False, lw=0))
        ax.text((s + e) / 2, -1.55, BLOCK_ZH[b], ha="center", va="bottom", fontsize=7.5,
                color=BLOCK_COLOR[b])
        ax.add_patch(Rectangle((-1.35, s - 0.5), 0.7, e - s + 1, color=BLOCK_COLOR[b],
                               clip_on=False, lw=0))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(axis="y", pad=11, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    panel_label(ax, "a", x=-0.02, y=1.045)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cax.tick_params(labelsize=7, length=2)
    cax.text(-0.03, 0.5, "Spearman $r$", transform=cax.transAxes, ha="right", va="center", fontsize=7.5)

    # (b) 权重
    y = np.arange(n)
    for (s, e), b in zip(starts, BLOCKS):
        c = BLOCK_COLOR[b]
        sub = w.iloc[s:e + 1]
        axb.hlines(y[s:e + 1], 0, sub.total_weight, color=lighten(c, 0.35), lw=1.4)
        axb.scatter(sub.total_weight, y[s:e + 1], s=16, color=c, zorder=3, lw=0)
        tot = sub.total_weight.sum()
        axb.text(0.1, (s + e) / 2, f"$\\Sigma$={tot:.2f}", ha="right", va="center",
                 fontsize=7.5, color=c)
        if s > 0:
            axb.axhline(s - 0.5, color=GRID, lw=0.8)
    axb.axvline(1 / n, color=GRAY, ls=":", lw=0.8)
    axb.text(1 / n, -0.6, "等权 1/22", color=GRAY, fontsize=7, va="bottom", ha="center")
    axb.set_xlim(0, 0.1)
    axb.set_xticks([0, 0.03, 0.06, 0.09])
    axb.set_xlabel("综合权重 $w_j$")
    axb.tick_params(axis="y", left=False, labelleft=False)
    axb.grid(axis="y", visible=False)
    panel_label(axb, "b", x=-0.02, y=1.045)
    save(fig, OUT / "P1_F01_质量指标结构与权重")
    # 关键数值
    off = C[np.triu_indices(n, 1)]
    print("F01 max w", w.loc[w.total_weight.idxmax(), ["field", "total_weight"]].tolist(),
          "min w", w.loc[w.total_weight.idxmin(), ["field", "total_weight"]].tolist(),
          "max |r| offdiag", np.abs(off).max(), "median |r|", np.median(np.abs(off)))


# ============================================================== F02
def fig02():
    """(a) 7 个质量域分布：由 ≤500 个固定随机抽样点做高斯 KDE（确定性），
    叠加 P5–P95 区间、中位数及 Huber 稳健均值 q 与 95% CI；
    (b) 17 个配比域映射质量 q_m，误差线为映射敏感性区间。"""
    qd = rd("P1_F02_quality_domains.csv").sort_values("q").reset_index(drop=True)
    rq = rd("P1_F02_ridge_quantiles.csv").set_index("domain")
    pts = rd("P1_F02_ridge_point_sample.csv")
    md = rd("P1_F02_mixture_domain_quality.csv")

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(FULL_W, 3.1),
                                  gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.42))
    grid = np.linspace(0.15, 0.9, 400)
    H = 0.85
    for i, r in qd.iterrows():
        x = pts.loc[pts.domain == r.domain, "Q"].values
        kde = gaussian_kde(x)
        dens = kde(grid)
        dens = dens / dens.max() * H
        ax.fill_between(grid, i, i + dens, color=lighten(BLUE, 0.72), lw=0, zorder=2 + 0.01 * (10 - i))
        ax.plot(grid, i + dens, color=BLUE, lw=0.7, zorder=2.1 + 0.01 * (10 - i))
        p5, p50, p95 = rq.loc[r.domain, ["P05", "P50", "P95"]]
        ax.hlines(i - 0.08, p5, p95, color=INK, lw=0.9, zorder=4)
        ax.plot([p50], [i - 0.08], marker="|", ms=6, color=INK, mew=1.0, zorder=4)
        ax.errorbar(r.q, i + 0.12, xerr=[[r.q - r.ci_low], [r.ci_high - r.q]], fmt="D", ms=3.8,
                    color=ORANGE, mec="white", mew=0.4, capsize=1.5, lw=0.8, zorder=5)
        ax.text(0.905, i + 0.1, f"{r.q:.3f}", fontsize=7, color=darken(ORANGE, 0.2), va="bottom", ha="right")
    ax.set_yticks(range(len(qd)))
    ax.set_yticklabels([f"{QUALITY_ZH[d]}" for d in qd.domain])
    ax.set_ylim(-0.4, len(qd) - 0.05)
    ax.set_xlim(0.2, 0.91)
    ax.set_xlabel("综合质量分 $Q$")
    ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], color=ORANGE, marker="D", ls="", ms=3.8, label="稳健均值 $q$ (95% CI)"),
               Line2D([], [], color=INK, marker="|", lw=0.9, ms=6, label="P5–中位数–P95")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=2,
              columnspacing=0.8, handletextpad=0.4, borderaxespad=0)
    panel_label(ax, "a", y=1.06)

    # (b)
    tier = {"direct": 0, "near_direct": 1, "inferred": 2}
    md["t"] = md.mapping_type.map(tier)
    md["zh"] = md.domain.map(dz)
    md = md.sort_values(["t", "q", "zh"], ascending=[True, False, True]).reset_index(drop=True)
    ypos, yy, prev = [], 0, None
    for t in md.t:
        if prev is not None and t != prev:
            yy += 0.6 if t == 1 else 1.2
        ypos.append(yy)
        yy += 1
        prev = t
    md["y"] = ypos
    for _, r in md.iterrows():
        c = SEMANTIC[r.mapping_type]
        axb.hlines(r.y, r.sensitivity_ci_low, r.sensitivity_ci_high, color=lighten(c, 0.3), lw=1.6)
        axb.plot(r.q, r.y, "o", color=c, ms=4.2, mec="white", mew=0.4, zorder=3)
        if r.mapping_type == "near_direct":
            axb.text(r.sensitivity_ci_high + 0.006, r.y, f"← {QUALITY_ZH[r.quality_anchor]}",
                     fontsize=7, color=darken(c, 0.1), va="center")
    yi = md[md.mapping_type == "inferred"]
    r0 = yi.iloc[0]
    axb.text(r0.q, yi.y.min() - 0.75, f"{len(yi)} 域共用推断值 {r0.q:.3f}  [{r0.sensitivity_ci_low:.3f}, {r0.sensitivity_ci_high:.3f}]",
             fontsize=6.8, color=darken(GRAY, 0.3), ha="center", va="bottom")
    axb.set_yticks(md.y)
    axb.set_yticklabels(md.zh)
    axb.invert_yaxis()
    axb.set_xlim(0.40, 0.70)
    axb.set_xlabel("配比域质量 $q_m$")
    axb.grid(axis="y", visible=False)
    handles = [Line2D([], [], color=SEMANTIC[k], marker="o", ls="-", ms=4, lw=1.4, label=MAP_ZH[k])
               for k in ["direct", "near_direct", "inferred"]]
    axb.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=3,
               columnspacing=0.8, handletextpad=0.3, handlelength=1.4, borderaxespad=0)
    panel_label(axb, "b", y=1.06)
    save(fig, OUT / "P1_F02_领域质量景观与映射")
    inf = md[md.mapping_type == "inferred"].iloc[0]
    print("F02 inferred q", inf.q, inf.sensitivity_ci_low, inf.sensitivity_ci_high)


# ============================================================== F03
def fig03():
    """(a) Q–Q 图：扩展集余集分位数 vs A1 抽样集分位数（ecdf_quantiles 401 个分位点）；
    (b)(c) 分位数位移函数 Δ(τ)=Q_余集(τ)−Q_A1(τ) 及 bootstrap 95% CI（shift_function 原值）。"""
    eq = rd("P1_F03_ecdf_quantiles.csv")
    sf = rd("P1_F03_shift_function.csv")
    col = {"arxiv": BLUE, "github": GREEN}

    fig = plt.figure(figsize=(FULL_W, 2.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], wspace=0.32, hspace=0.18,
                          left=0.09, right=0.985, top=0.93, bottom=0.14)
    ax = fig.add_subplot(gs[:, 0])
    for d in ["github", "arxiv"]:
        g = eq[eq.domain == d].pivot(index="quantile", columns="group", values="Q")
        g = g.loc[(g.index >= 0.01) & (g.index <= 0.99)]  # 去掉极值分位（最小/最大值）仅为显示
        ax.plot(g["扩展集余集"], g["A1抽样集"], color=col[d], lw=1.2, label=f"{DOMAIN_ZH[d]}")
    lim = (0.25, 0.85)
    ax.plot(lim, lim, color=GRAY, ls="--", lw=0.8, zorder=0, label="$y=x$")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("扩展集余集分位数 $Q$")
    ax.set_ylabel("A1 抽样集分位数 $Q$")
    ax.legend(loc="upper left")
    # 标出 arXiv 高分位跳变段
    g = eq[eq.domain == "arxiv"].pivot(index="quantile", columns="group", values="Q")
    seg = g.loc[(g.index >= 0.88) & (g.index <= 0.915)]
    ax.annotate("学术论文 $\\tau\\approx0.9$\n高分段跳变", xy=(0.765, 0.70), xytext=(0.66, 0.45),
                fontsize=7, color=darken(BLUE, 0.1), ha="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=darken(BLUE, 0.1), mutation_scale=6))
    panel_label(ax, "a", y=1.04)

    axs = [fig.add_subplot(gs[0, 1])]
    axs.append(fig.add_subplot(gs[1, 1], sharex=axs[0]))
    for k, (d, a) in enumerate(zip(["arxiv", "github"], axs)):
        s = sf[sf.domain == d]
        a.fill_between(s["quantile"], s.ci_low * 100, s.ci_high * 100, color=lighten(col[d], 0.7), lw=0)
        a.plot(s["quantile"], s.difference * 100, "o-", color=col[d], ms=2.6, lw=1.0)
        a.axhline(0, color=GRAY, lw=0.7, ls="--")
        a.set_ylim(-6, 13)
        a.text(0.02, 0.95, DOMAIN_ZH[d], transform=a.transAxes, va="top", fontsize=8, color=col[d])
        panel_label(a, "bc"[k])
    s = sf[(sf.domain == "arxiv")]
    r = s.loc[s.difference.idxmax()]
    axs[0].annotate(f"$\\Delta$={r.difference:.3f}", xy=(r["quantile"], r.difference * 100),
                    xytext=(r["quantile"] - 0.2, r.difference * 100 + 1.5), fontsize=7, color=darken(BLUE, 0.1),
                    ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=darken(BLUE, 0.1)))
    plt.setp(axs[0].get_xticklabels(), visible=False)
    axs[1].set_xlabel("分位水平 $\\tau$")
    axs[1].set_xlim(0, 1)
    fig.text(0.47, 0.54, "分位数位移 $\\Delta(\\tau)$ / $10^{-2}$", rotation=90, va="center", ha="center",
             fontsize=8.5)
    save(fig, OUT / "P1_F03_扩展集代表性诊断")
    for d in ["arxiv", "github"]:
        s = sf[sf.domain == d]
        s2 = s[s["quantile"] < 0.875]
        print("F03", d, "max|Δ| τ<=0.85", s2.difference.abs().max(), "CI excl 0:",
              ((s.ci_low > 0) | (s.ci_high < 0)).sum())
    print(sf[sf["quantile"].round(2) == 0.9])


# ============================================================== F04
def fig04():
    """(a) 各质量域冲突率按冲突类型堆叠；“其他”= quality_domains.conflict_rate − 列出类型率之和；
    (b) 高分块→低分块冲突计数矩阵。"""
    ct = rd("P1_F04_conflict_type_matrix.csv")
    qd = rd("P1_F02_quality_domains.csv").set_index("domain")
    bp = rd("P1_F04_conflict_block_pairs.csv")
    rc = rd("P1_F04_robust_correction_summary.csv")

    cats = [("reasoning>structure", "推理信息>结构充分", SEMANTIC["reasoning"]),
            ("noise>structure", "噪声重复>结构充分", SEMANTIC["noise"]),
            ("expression>structure", "表达整洁>结构充分", SEMANTIC["expression"]),
            ("education>structure", "教育价值>结构充分", SEMANTIC["education"]),
            ("structure>*", "结构充分>其他块", SEMANTIC["structure"]),
            ("other", "其他组合", lighten(GRAY, 0.3))]
    ct["cat"] = ct.conflict_type.where(ct.conflict_type.isin([c[0] for c in cats[:4]]), "other")
    ct.loc[ct.conflict_type.str.startswith("structure>"), "cat"] = "structure>*"
    tab = ct.pivot_table(index="domain", columns="cat", values="rate", aggfunc="sum").fillna(0)
    tab["other"] = tab.get("other", 0) + (qd.loc[tab.index, "conflict_rate"] - tab.sum(1)).clip(lower=0)
    tab = tab.loc[qd.loc[tab.index, "conflict_rate"].sort_values().index]

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(FULL_W, 2.85),
                                  gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.55))
    left = np.zeros(len(tab))
    y = np.arange(len(tab))
    for key, lab, c in cats:
        v = tab[key].values * 100 if key in tab else np.zeros(len(tab))
        ax.barh(y, v, left=left, color=c, height=0.66, label=lab, lw=0)
        left += v
    for i, d in enumerate(tab.index):
        ax.text(qd.loc[d, "conflict_rate"] * 100 + 0.2, i, f"{qd.loc[d, 'conflict_rate'] * 100:.1f}%",
                va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{QUALITY_ZH[d]}" for d in tab.index])
    ax.set_xlabel("显著冲突率 / %")
    ax.set_xlim(0, 12.2)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.8, handlelength=1.0, handleheight=0.8, labelspacing=0.25,
              title="高分块>低分块", title_fontsize=6.8)
    panel_label(ax, "a")

    M = bp.pivot_table(index="high_block", columns="low_block", values="count", aggfunc="sum")
    M = M.reindex(index=BLOCKS, columns=BLOCKS)
    Mv = M.values.astype(float)
    im = axb.imshow(np.where(np.isnan(Mv), np.nan, Mv), cmap=SEQ_CMAP,
                    norm=LogNorm(vmin=1, vmax=np.nanmax(Mv)), aspect="equal")
    axb.grid(False)
    for i in range(5):
        for j in range(5):
            if i == j:
                axb.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color="white", hatch="////",
                                        ec=GRID, lw=0))
                continue
            v = Mv[i, j]
            if np.isnan(v):
                axb.text(j, i, "0", ha="center", va="center", fontsize=7, color=GRAY)
            else:
                axb.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7,
                         color="white" if v > 300 else INK)
    axb.set_xticks(range(5))
    axb.set_xticklabels([BLOCK_ZH[b] for b in BLOCKS], rotation=35, ha="right")
    axb.set_yticks(range(5))
    axb.set_yticklabels([BLOCK_ZH[b] for b in BLOCKS])
    for t, b in zip(axb.get_xticklabels(), BLOCKS):
        t.set_color(BLOCK_COLOR[b])
    for t, b in zip(axb.get_yticklabels(), BLOCKS):
        t.set_color(BLOCK_COLOR[b])
    axb.set_xlabel("低分语义块", labelpad=1)
    axb.set_ylabel("高分语义块")
    for sp in axb.spines.values():
        sp.set_visible(False)
    axb.tick_params(length=0)
    cb = fig.colorbar(im, ax=axb, fraction=0.046, pad=0.03)
    cb.set_label("冲突样本数", labelpad=1)
    cb.outline.set_linewidth(0.4)
    r = rc[rc.conflict.astype(str) == "True"].iloc[0]
    axb.text(0.5, -0.5, f"{int(r['count'])} 条冲突样本 Huber 校正量：均值 {r['mean']:.3f}，中位数 {r['median']:.3f}",
             transform=axb.transAxes, ha="center", va="top", fontsize=7.3, color=INK)
    panel_label(axb, "b", x=-0.33, y=1.02)
    save(fig, OUT / "P1_F04_质量冲突机制与稳健消解")
    print("F04 top pairs", bp.sort_values("count", ascending=False).head(4).values.tolist(),
          "total", bp["count"].sum(), "to structure share",
          bp[bp.low_block == "structure"]["count"].sum() / bp["count"].sum())


# ============================================================== F05
def fig05():
    """1M/60M/1B 折外预测 vs 实测 Loss 的 hexbin（prediction_used 列：60M/1B 为校准后预测）。
    图中 ρ、R² 为 13 个 loss 域上逐域指标的中位数（metric_summary），n 为点数。"""
    p = pd.read_csv(FD / "P1_F05_prediction_plot_data.csv.gz")
    ms = rd("P1_F05_metric_summary.csv").set_index("split")
    splits = ["test_1m", "test_60m", "test_1B"]
    fig = plt.figure(figsize=(FULL_W, 2.45))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.38,
                          left=0.075, right=0.93, top=0.92, bottom=0.17)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    cax = fig.add_subplot(gs[3])
    for k, (s, ax) in enumerate(zip(splits, axs)):
        g = p[p.split == s]
        x, yv = g.observed.values, g.prediction_used.values
        lo = np.floor(min(x.min(), np.quantile(yv, 0.002)) * 2) / 2
        hi = np.ceil(max(x.max(), np.quantile(yv, 0.998)) * 2) / 2
        inside = (yv >= lo) & (yv <= hi)
        hb = ax.hexbin(x[inside], yv[inside], gridsize=26, extent=(lo, hi, lo, hi), cmap=SEQ_CMAP,
                       mincnt=1, bins="log", vmin=1, vmax=200, lw=0.1, edgecolors="face")
        ax.plot([lo, hi], [lo, hi], color=GRAY, ls="--", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(False)
        m = ms.loc[s]
        txt = f"$\\tilde\\rho$ = {m.spearman:.3f}\n$\\tilde R^2$ = {m.r2:.3f}\n$n$ = {len(g)}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=7.2, linespacing=1.3)
        ax.text(0.96, 0.05, SCALE_ZH[s], transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, fontweight="bold", color=INK)
        ax.set_xlabel("实测 Loss")
        if k == 0:
            ax.set_ylabel("折外预测 Loss")
        nout = (~inside).sum()
        if nout:
            ax.text(0.96, 0.16, f"{nout} 点越界未显示", transform=ax.transAxes, ha="right", fontsize=6.5,
                    color=GRAY)
        panel_label(ax, "abc"[k])
        print("F05", s, "pooled spearman",
              pd.Series(x).corr(pd.Series(yv), method="spearman"), "n out", nout, lo, hi)
    fig.canvas.draw()
    p0 = axs[-1].get_position()
    cax.set_position([p0.x1 + 0.015, p0.y0, 0.012, p0.height])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("样本数（对数刻度）", labelpad=1)
    cb.outline.set_linewidth(0.4)
    save(fig, OUT / "P1_F05_配比模型跨尺度验证")


# ============================================================== F06
def fig06():
    """(a) 方向边际收益 + bootstrap 95% CI 森林图；
    (b) 下三角标准化交互 z_ij = β_ij / SE_ij，SE_ij=(ci_high−ci_low)/(2×1.96)（bootstrap 百分位区间折算），
    点标记 sign_probability ≥ 0.95 的稳定对。β<0 为互补（蓝），β>0 为替代（橙）。"""
    de = rd("P1_F06_directional_effect_bootstrap.csv").sort_values("benefit", ascending=False).reset_index(drop=True)
    it = rd("P1_F06_interaction_summary.csv")
    order = de.domain.tolist()
    n = len(order)
    fig = plt.figure(figsize=(FULL_W, 3.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.3, 0.035], wspace=0.06,
                          left=0.115, right=0.91, top=0.95, bottom=0.2)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharey=ax)
    cax = fig.add_subplot(gs[2])
    y = np.arange(n)
    for i, r in de.iterrows():
        if r.ci_low > 0:
            c = BLUE
        elif r.ci_high < 0:
            c = ORANGE
        else:
            c = GRAY
        ax.hlines(i, r.ci_low, r.ci_high, color=lighten(c, 0.3), lw=1.5)
        ax.plot(r.benefit, i, "o", color=c, ms=4, mec="white", mew=0.4, zorder=3)
    ax.axvline(0, color=GRAY, ls="--", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([dz(d) for d in order])
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("方向边际收益")
    ax.set_xlim(-4, 9.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], color=c, marker="o", ms=4, lw=1.3, label=l) for c, l in
               [(BLUE, "显著为正"), (GRAY, "区间含 0"), (ORANGE, "显著为负")]]
    ax.legend(handles=handles, loc="lower right", handlelength=1.2, labelspacing=0.3)
    panel_label(ax, "a", y=1.01)

    idx = {d: i for i, d in enumerate(order)}
    Z = np.full((n, n), np.nan)
    S = np.zeros((n, n), bool)
    for _, r in it.iterrows():
        i, j = idx[r.domain_i], idx[r.domain_j]
        i, j = max(i, j), min(i, j)
        se = (r.ci_high - r.ci_low) / (2 * 1.96)
        Z[i, j] = r.interaction / se
        S[i, j] = r.sign_probability >= 0.95
    vmax = 6
    im = axh.imshow(np.clip(Z[:, :n - 1], -vmax, vmax), cmap=DIV_CMAP, norm=TwoSlopeNorm(0, -vmax, vmax),
                    aspect="auto", extent=(-0.5, n - 1.5, n - 0.5, -0.5))
    ii, jj = np.where(S)
    axh.scatter(jj, ii, s=3.5, color="white", edgecolor=INK, lw=0.35, zorder=3)
    axh.grid(False)
    axh.set_xticks(range(n - 1))
    axh.set_xticklabels([dz(d) for d in order[:-1]], rotation=60, ha="right", rotation_mode="anchor")
    axh.tick_params(axis="y", labelleft=False)
    for sp in axh.spines.values():
        sp.set_visible(False)
    axh.tick_params(length=0)
    axh.set_xlim(-0.5, n - 1.5)
    # 行与 (a) 共用纵轴：第 i 行即 (a) 中同名领域
    axh.scatter([], [], s=6, color="white", edgecolor=INK, lw=0.4, label="符号稳定 $P\\geq0.95$")
    axh.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), handletextpad=0.2)
    panel_label(axh, "b", x=0.02, y=1.01)
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label("标准化交互 $z_{ij}$（负=互补，正=替代）", labelpad=2)
    cb.outline.set_linewidth(0.4)
    save(fig, OUT / "P1_F06_领域边际效应与交互网络")
    print("F06 stable pairs", S.sum(), "of", np.isfinite(Z).sum(),
          "stable complement", (S & (Z < 0)).sum(), "stable substitute", (S & (Z > 0)).sum())
    print(de[["domain", "benefit", "ci_low", "ci_high"]].round(3).to_string())


# ============================================================== F07
def ilr_pca_projector():
    """复现 P1_F07_ilr_pca_training_map：p→(p+1e-5) 归一化→clr→Helmert ILR→对 512 训练配方中心化 PCA。
    与原文件 PC1/PC2 相关系数 >0.99999；为消除微小差异，再用最小二乘仿射映射对齐到原坐标。"""
    from scipy.linalg import helmert
    d = np.load(RD / "problem2_bridge_input.npz")
    P = d["p_train"]
    H = helmert(17)

    def ilr(X):
        X = np.atleast_2d(X) + 1e-5
        X = X / X.sum(1, keepdims=True)
        L = np.log(X)
        return (L - L.mean(1, keepdims=True)) @ H.T

    Zt = ilr(P)
    mu = Zt.mean(0)
    _, _, Vt = np.linalg.svd(Zt - mu, full_matrices=False)
    pc = (Zt - mu) @ Vt[:2].T
    ref = rd("P1_F07_ilr_pca_training_map.csv")[["PC1", "PC2"]].values
    A = np.c_[pc, np.ones(len(pc))]
    coef, *_ = np.linalg.lstsq(A, ref, rcond=None)
    err = np.abs(A @ coef - ref).max()

    def proj(X):
        z = (ilr(X) - mu) @ Vt[:2].T
        return np.c_[z, np.ones(len(z))] @ coef
    return proj, err


def fig07():
    """(a) 训练均值配方 p_ref → 近优配方 p* 哑铃图，p* 的 bootstrap 95% 区间与逐域上界；
    (b) ILR–PCA 平面：512 个训练配方（颜色=响应面目标值）、200 个 bootstrap 近优解投影、p* 与 p_ref。"""
    ci = rd("P1_F07_composition_intervals.csv").sort_values(["p_star", "p_ref"]).reset_index(drop=True)
    tm = rd("P1_F07_ilr_pca_training_map.csv")
    bt = rd("P1_F07_optimal_bootstrap_projection.csv")
    fig = plt.figure(figsize=(FULL_W, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], wspace=0.28, left=0.115, right=0.985,
                          top=0.86, bottom=0.13)
    ax = fig.add_subplot(gs[0])
    n = len(ci)
    y = np.arange(n)
    for i, r in ci.iterrows():
        ax.hlines(i, r.ci_low * 100, r.ci_high * 100, color=lighten(ORANGE, 0.62), lw=3.2, zorder=1)
        ax.annotate("", xy=(r.p_star * 100, i), xytext=(r.p_ref * 100, i),
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.6), zorder=2)
        ax.plot(r.upper_bound * 100, i, marker="|", color=RED, ms=6, mew=1.0, zorder=2) \
            if r.upper_bound < 0.6 else None
    ax.plot(ci.p_ref * 100, y, "o", color=GRAY, ms=3.8, mec="white", mew=0.3, zorder=3, label="$p_{\\rm ref}$ 训练均值")
    ax.plot(ci.p_star * 100, y, "o", color=ORANGE, ms=4.2, mec=darken(ORANGE, 0.3), mew=0.4, zorder=4,
            label="$p^*$ 近优配比")
    ax.plot([], [], color=lighten(ORANGE, 0.62), lw=3.2, label="$p^*$ bootstrap 95% 区间")
    ax.plot([], [], marker="|", color=RED, ls="", ms=6, mew=1.0, label="份额上界 $u_i$")
    ax.set_yticks(y)
    ax.set_yticklabels([dz(d) for d in ci.domain])
    ax.set_xlabel("配比份额 / %")
    ax.set_xlim(-1, 60)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=7, labelspacing=0.25,
              handlelength=1.4, columnspacing=0.8, borderaxespad=0)
    panel_label(ax, "a", y=1.13)

    axb = fig.add_subplot(gs[1])
    sc = axb.scatter(tm.PC1, tm.PC2, c=tm.objective, cmap=SEQ_CMAP.reversed(), s=7, lw=0.2,
                     edgecolor="white", zorder=1)
    axb.scatter(bt.PC1, bt.PC2, marker="x", s=9, color=ORANGE, lw=0.6, alpha=0.8, zorder=2,
                label="bootstrap 近优解")
    proj, err = ilr_pca_projector()
    ps = proj(ci.set_index("domain").loc[pd.Index(np.load(RD / "problem2_bridge_input.npz")["domains"]), "p_star"].values)
    pr = proj(np.load(RD / "problem2_bridge_input.npz")["p_ref_point"])
    axb.plot(ps[0, 0], ps[0, 1], marker="*", ms=11, color=ORANGE, mec=INK, mew=0.6, ls="", zorder=4, label="$p^*$")
    axb.plot(pr[0, 0], pr[0, 1], marker="o", ms=5.5, color=GRAY, mec=INK, mew=0.6, ls="", zorder=4,
             label="$p_{\\rm ref}$")
    axb.set_xlabel("ILR–PCA 第 1 主成分")
    axb.set_ylabel("ILR–PCA 第 2 主成分")
    axb.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, fontsize=7, handletextpad=0.3,
               columnspacing=0.8, borderaxespad=0)
    cb = fig.colorbar(sc, ax=axb, fraction=0.05, pad=0.02)
    cb.set_label("响应面目标值（越低越好）", labelpad=2)
    cb.outline.set_linewidth(0.4)
    panel_label(axb, "b", y=1.13)
    save(fig, OUT / "P1_F07_组成空间近优配比与不确定性")
    print("F07 proj align err", err, "p* PC", ps, "p_ref PC", pr, "train obj min", tm.objective.min())
    at = ci[np.isclose(ci.p_star, ci.upper_bound, atol=1e-8)]
    print("F07 at upper bound:", at.domain.tolist(), "zero:", ci[ci.p_star == 0].domain.tolist())


# ============================================================== F08
def fig08():
    """17 域 2% 份额局部替换收益在 512 个参照配方上的分布：点=单个参照配方，
    粗线=10%–90% 分位，菱形=中位数，黑线=中位数 bootstrap 95% CI；右侧为有利占比。
    x 轴截断在 [−0.30, 0.35]，越界点画在边缘并注明数量。"""
    pt = pd.read_csv(FD / "P1_F08_local_replacement_points.csv.gz")
    sm = rd("P1_F08_local_replacement_summary.csv").sort_values("median_benefit", ascending=False).reset_index(drop=True)
    lo, hi = -0.20, 0.30
    fig, ax = plt.subplots(figsize=(FULL_W, 3.7))
    fig.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.115)
    rng = np.random.default_rng(2026)  # 仅用于点的纵向抖动显示
    for i, r in sm.iterrows():
        b = pt.loc[pt.domain == r.domain, "benefit"].values
        jit = rng.uniform(-0.28, 0.28, len(b))
        keep = (b >= lo) & (b <= hi)  # 越界点不画，只在边缘注明数量
        ax.scatter(b[keep], i + jit[keep], s=1.6, color=lighten(BLUE, 0.68), lw=0, zorder=1, rasterized=True)
        stable = r.bootstrap_median_ci_low > 0
        c = BLUE if stable else GRAY
        ax.hlines(i, r.q10_benefit, r.q90_benefit, color=lighten(c, 0.3), lw=2.2, zorder=2)
        ax.hlines(i, r.bootstrap_median_ci_low, r.bootstrap_median_ci_high, color=INK, lw=1.0, zorder=3)
        ax.plot(r.median_benefit, i, "D", color=c, ms=4, mec="white", mew=0.4, zorder=4)
        nl, nh = (b < lo).sum(), (b > hi).sum()
        if nl:
            ax.text(lo + 0.003, i - 0.3, f"←{nl}", fontsize=6.5, color=darken(GRAY, 0.2), va="center")
        if nh:
            ax.text(hi - 0.003, i - 0.3, f"{nh}→", fontsize=6.5, color=darken(GRAY, 0.2), va="center", ha="right")
        ax.text(1.015, i, f"{r.beneficial_fraction * 100:.1f}%", transform=ax.get_yaxis_transform(),
                va="center", fontsize=7.3, color=c)
    ax.text(1.015, -0.9, "有利占比", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5)
    ax.axvline(0, color=GRAY, ls="--", lw=0.7)
    ax.set_yticks(range(len(sm)))
    ax.set_yticklabels([dz(d) for d in sm.domain])
    ax.set_ylim(len(sm) - 0.5, -0.5)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("转入 2% 份额后 13 域平均 Loss 降低量（正值有利）")
    ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], color=lighten(BLUE, 0.6), marker="o", ls="", ms=2.5, label="单个参照配方"),
               Line2D([], [], color=lighten(BLUE, 0.3), lw=2.2, label="P10–P90"),
               Line2D([], [], color=BLUE, marker="D", ls="", ms=4, label="中位数（CI>0）"),
               Line2D([], [], color=GRAY, marker="D", ls="", ms=4, label="中位数（CI 含 0）"),
               Line2D([], [], color=INK, lw=1.0, label="中位数 95% CI")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=5, fontsize=7,
              handlelength=1.3, columnspacing=0.8, handletextpad=0.3, borderaxespad=0)
    save(fig, OUT / "P1_F08_领域局部替换效应")
    print(sm[["domain", "median_benefit", "bootstrap_median_ci_low", "bootstrap_median_ci_high",
              "beneficial_fraction"]].round(4).to_string())


# ============================================================== F09
def fig09():
    """(a) 5 类候选模型 NRMSE–中位 Spearman 散点与 Pareto 前沿；
    (b) 线性/二阶 Ridge 的 NRMSE 正则化路径，竖线为所选 α；阴影 = 二阶 Ridge NRMSE ≤ 1.01×最小值的 α 区间。"""
    mc = rd("P1_F09_model_comparison.csv")
    rp = rd("P1_F09_ridge_regularization_path.csv")
    sa = rd("P1_F09_selected_alpha.csv").set_index("model")
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(FULL_W, PANEL_H + 0.25),
                                  gridspec_kw=dict(wspace=0.32))
    fr = mc[mc.pareto_frontier].sort_values("normalized_rmse")
    ax.plot(fr.normalized_rmse, fr.median_spearman, color=BLUE, lw=0.9, ls="-", zorder=1)
    for _, r in mc.iterrows():
        c = BLUE if r.pareto_frontier else GRAY
        mk = "*" if r.model == "quadratic_ridge" else "o"
        ax.plot(r.normalized_rmse, r.median_spearman, mk, color=c, ms=10 if mk == "*" else 5,
                mec="white", mew=0.4, zorder=3)
        off = {"quadratic_ridge": (0.001, -0.006, "left"), "elastic_net": (0.0015, 0.0035, "left"),
               "linear_ridge": (0.0015, 0.0, "left"), "extra_trees": (-0.0015, 0.004, "right"),
               "random_forest": (-0.0015, -0.007, "right")}[r.model]
        ax.text(r.normalized_rmse + off[0], r.median_spearman + off[1], r.model_zh, fontsize=7.3,
                ha=off[2], va="center", color=darken(c, 0.15))
    ax.set_xlabel("IQR 标准化 NRMSE（越低越好）")
    ax.set_ylabel("逐域 Spearman 中位数")
    ax.set_xlim(0.388, 0.450)
    ax.set_ylim(0.800, 0.862)
    handles = [Line2D([], [], color=BLUE, marker="o", ls="-", ms=4, lw=0.9, label="Pareto 前沿"),
               Line2D([], [], color=GRAY, marker="o", ls="", ms=4, label="被支配"),
               Line2D([], [], color=BLUE, marker="*", ls="", ms=8, label="选定模型")]
    ax.legend(handles=handles, loc="center right")
    panel_label(ax, "a")

    cols = {"quadratic_ridge": BLUE, "linear_ridge": ORANGE}
    q = rp[rp.model == "quadratic_ridge"].sort_values("alpha")
    thr = q.normalized_rmse.min() * 1.01
    ok = q[q.normalized_rmse <= thr]
    axb.axvspan(ok.alpha.min(), ok.alpha.max(), color=lighten(BLUE, 0.85), lw=0, zorder=0)
    for m in ["linear_ridge", "quadratic_ridge"]:
        g = rp[rp.model == m].sort_values("alpha")
        axb.plot(g.alpha, g.normalized_rmse, color=cols[m], lw=1.2, label=g.model_zh.iloc[0])
        s = sa.loc[m]
        axb.plot(s.alpha, s.normalized_rmse, "o", color=cols[m], ms=4.5, mec="white", mew=0.5, zorder=4)
        axb.axvline(s.alpha, color=cols[m], ls=":", lw=0.8)
    s = sa.loc["quadratic_ridge"]
    axb.annotate(f"$\\alpha^*$={s.alpha:.4f}\nNRMSE={s.normalized_rmse:.3f}", xy=(s.alpha, s.normalized_rmse),
                 xytext=(1.2e-5, 0.425), fontsize=7, color=darken(BLUE, 0.1),
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=darken(BLUE, 0.1)))
    axb.set_xscale("log")
    axb.set_xlabel("正则化强度 $\\alpha$")
    axb.set_ylabel("IQR 标准化 NRMSE")
    axb.set_ylim(0.39, 0.46)
    axb.set_xlim(1e-6, 1e4)
    h, l = axb.get_legend_handles_labels()
    h.append(Patch(color=lighten(BLUE, 0.85), label="二阶 NRMSE≤1.01×最小"))
    axb.legend(handles=h, loc="upper left")
    panel_label(axb, "b")
    save(fig, OUT / "P1_F09_响应面选择与正则化诊断")
    print("F09 1% band", ok.alpha.min(), ok.alpha.max(), "max nrmse on path",
          rp.groupby("model").normalized_rmse.max().to_dict())


# ============================================================== F10
def fig10():
    """(a) 各扰动情景的目标代价（%，基准响应面下）vs 配比 L1 变化（百分点）；
    (b) 情景 × 领域的份额变化 Δp（百分点），行按基准份额排序。"""
    sc = rd("P1_F10_sensitivity_scenarios.csv")
    cp = rd("P1_F10_sensitivity_compositions.csv")
    gcol = {"support": BLUE, "upper": ORANGE, "alpha": GREEN}
    gzh = {"support": "支持半径分位", "upper": "份额上界分位", "alpha": "Ridge 强度"}
    fig = plt.figure(figsize=(FULL_W, 3.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.82, 0.035], wspace=0.05, left=0.075, right=0.915,
                          top=0.93, bottom=0.2)
    ax = fig.add_subplot(gs[0])
    ax.set_position(ax.get_position().translated(-0.0, 0))
    ax.plot(0, 0, "o", color=INK, ms=4.5, zorder=4)
    ax.text(-0.8, -0.12, "基准", fontsize=7.3, va="top", ha="left")
    lab_off = {"support_q95": (-2, 0.12, "right"), "support_q975": (1.5, 0.12, "left"),
               "upper_q95": (-1.5, -0.3, "right"), "upper_q975": (1.8, 0.1, "left"),
               "alpha_half": (1.5, 0.22, "left"), "alpha_double": (1.5, 0.45, "left")}
    for _, r in sc[sc.group != "baseline"].iterrows():
        c = gcol[r.group]
        ax.plot([0, r.l1_shift_pp], [0, r.objective_penalty_pct], color=lighten(c, 0.5), lw=0.8, zorder=1)
        ax.plot(r.l1_shift_pp, r.objective_penalty_pct, "o", color=c, ms=5, mec="white", mew=0.4, zorder=3)
        if r.group == "alpha":
            continue
        dx, dy, ha = lab_off[r.scenario]
        ax.text(r.l1_shift_pp + dx, r.objective_penalty_pct + dy, r.label, fontsize=7, ha=ha, color=darken(c, 0.15))
    al = sc[sc.group == "alpha"].set_index("scenario")
    ax.annotate(f"$0.5\\alpha^*$ / $2\\alpha^*$：L1 {al.l1_shift_pp.min():.1f}/{al.l1_shift_pp.max():.1f}，"
                f"\n代价 {al.objective_penalty_pct.min():.3f}%/{al.objective_penalty_pct.max():.3f}%",
                xy=(al.l1_shift_pp.mean(), 0.03), xytext=(14, 0.45), fontsize=7, color=darken(GREEN, 0.15),
                arrowprops=dict(arrowstyle="-", lw=0.5, color=darken(GREEN, 0.15)))
    ax.set_xlabel("配比 L1 变化 / 百分点")
    ax.set_ylabel("目标代价 / %")
    ax.set_xlim(-3, 68)
    ax.set_ylim(-0.25, 3.9)
    handles = [Line2D([], [], color=gcol[g], marker="o", ls="", ms=4.5, label=gzh[g]) for g in gcol]
    ax.legend(handles=handles, loc="upper left")
    panel_label(ax, "a")

    axh = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])
    pa = ax.get_position()
    ax.set_position([pa.x0, pa.y0, pa.width * 0.8, pa.height])  # 为 (b) 左侧领域名留白
    scen = ["support_q975", "support_q95", "upper_q975", "upper_q95", "alpha_half", "alpha_double"]
    base = cp[cp.scenario == "baseline"].set_index("domain").share.sort_values(ascending=False)
    D = cp.pivot(index="domain", columns="scenario", values="delta_pp").loc[base.index, scen]
    vmax = np.ceil(np.abs(D.values).max())
    im = axh.imshow(D.values, cmap=DIV_CMAP, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    axh.grid(False)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            v = D.values[i, j]
            if abs(v) >= 1:
                axh.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.3,
                         color="white" if abs(v) > 10 else INK)
    lab = sc.set_index("scenario").label
    axh.set_xticks(range(len(scen)))
    axh.set_xticklabels([lab[s].replace("支持半径 ", "半径 ").replace("份额上界 ", "上界 ") for s in scen],
                        rotation=35, ha="right", rotation_mode="anchor")
    for t, s in zip(axh.get_xticklabels(), scen):
        t.set_color(darken(gcol[sc.set_index("scenario").group[s]], 0.1))
    axh.set_yticks(range(D.shape[0]))
    axh.set_yticklabels([f"{dz(d)} ({base[d] * 100:.1f})" for d in D.index], fontsize=7)
    axh.tick_params(length=0)
    for sp in axh.spines.values():
        sp.set_visible(False)
    for j in (1.5, 3.5):
        axh.axvline(j, color="white", lw=1.5)
    panel_label(axh, "b", x=-0.33)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("份额变化 $\\Delta p$ / 百分点", labelpad=1)
    cb.outline.set_linewidth(0.4)
    save(fig, OUT / "P1_F10_近优配比参数敏感性")
    print(sc[["scenario", "objective_penalty_pct", "l1_shift_pp", "active_upper_bounds"]].to_string())


FIGS = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07, 8: fig08, 9: fig09, 10: fig10}

if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]] or list(FIGS)
    for i in ids:
        FIGS[i]()
