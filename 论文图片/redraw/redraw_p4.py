"""问题四（技术演进分析与前沿预测）终稿插图重绘。

运行：python redraw_p4.py  → 输出到 overleaf-paper/images/no4/
数据：论文图片/问题四/figure_data/*.csv 与 问题四/results_v2/*。
派生量均为确定性计算：
- 逐任务复现“精确”判据：|重算 - 官方| <= 1e-6（与 problem4_v2.py 的 official_exact 计数一致，1736/1786、737/1782）。
- one-SE 阈值 = min(CV-MAE) + 该模型 CV-MAE 的 SE。
- 规模贡献占比 = scale_score / total_score（逐 bootstrap 抽样）。
- F06 趋势线 = x_C0 + 月斜率 × (月序 - 截止月序)，窗口为截止前 24 个月（参数来自 compute_growth_summary.json）。
"""
import sys
sys.path.insert(0, "/home/chenxu/MayByte/LJY/论文图片")
from paper_style import *  # noqa: F401,F403
apply_style()

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FD = Path("/home/chenxu/MayByte/LJY/论文图片/问题四/figure_data")
RS = Path("/home/chenxu/MayByte/LJY/问题四/results_v2")
OUT = Path("/home/chenxu/MayByte/LJY/overleaf-paper/images/no4")

KAPPA_COL = {1.0: BLUE, 0.5: ORANGE, 0.25: GREEN}
KAPPA_LAB = {1.0: r"$\kappa$=1（100%）", 0.5: r"$\kappa$=0.5（50%）", 0.25: r"$\kappa$=0.25（25%）"}
TIER_COL = {"high_supported": BLUE, "expanded_supported": GREEN, "out_of_bridge_range": GRAY}
TIER_LAB = {"high_supported": "高支持", "expanded_supported": "扩展支持", "out_of_bridge_range": "范围外"}
BUD_COL = dict(zip([1e19, 1e22, 1e24], seq_colors(3)))
BUD_LAB = {1e19: r"$10^{19}$", 1e22: r"$10^{22}$", 1e24: r"$10^{24}$"}
COST_MK = {"exponential": "o", "power": "s", "logarithmic": "^"}
COST_LAB = {"exponential": "指数", "power": "幂函数", "logarithmic": "对数"}


# ------------------------------------------------------------------ F01
def f01():
    fun = pd.read_csv(FD / "F01_data_funnel.csv")
    t = pd.read_csv(RS / "task_level_aggregates.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.45),
                             gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.42))
    ax = axes[0]
    y = np.arange(len(fun))[::-1]
    cols = [GRAY, GRAY, BLUE, BLUE, ORANGE]
    ax.barh(y, fun["count"], color=[lighten(c, 0.15) for c in cols], height=0.62, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlim(1, 3e5)
    labels = ["C1/C2 原始记录", "版本去重后", "C4 接受匹配", "确认性前沿样本", "posttrained 样本"]
    ax.set_yticks(y, labels)
    for yi, n in zip(y, fun["count"]):
        ax.text(n * 1.25, yi, f"{n:,}", va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("记录数（对数轴）")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "a", x=-0.62)

    for ax, key, off, name, lab in [(axes[1], "bbh", "official_bbh", "BBH", "b"),
                                    (axes[2], "math", "official_math", "MATH Lvl 5", "c")]:
        rec = "bbh_normalized_equal_pct" if key == "bbh" else "math_weighted_pct"
        d = t.dropna(subset=[f"{key}_abs_diff"])
        ex = d[f"{key}_abs_diff"] <= 1e-6
        lim = max(d[off].max(), d[rec].max()) * 1.04
        ax.plot([0, lim], [0, lim], color=GRAY, lw=0.7, ls="--", zorder=1)
        ax.scatter(d.loc[~ex, off], d.loc[~ex, rec], s=5, color=ORANGE, alpha=0.55, lw=0, label="未精确复现", zorder=2)
        ax.scatter(d.loc[ex, off], d.loc[ex, rec], s=5, color=BLUE, alpha=0.55, lw=0, label="精确复现", zorder=3)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"官方 {name} 分数")
        ax.set_ylabel("逐任务重算分数")
        rate = ex.mean()
        verdict = "≥95%，采用" if rate >= 0.95 else "<95%，仅审计"
        ax.text(0.04, 0.97, f"精确 {ex.sum()}/{len(d)}\n= {rate:.1%}\n{verdict}", transform=ax.transAxes,
                va="top", fontsize=7.5, color=BLUE if rate >= 0.95 else RED)
        panel_label(ax, lab, x=-0.2)
    axes[1].legend(loc="lower right", markerscale=2.5, handletextpad=0.2, borderaxespad=0.1)
    save(fig, OUT / "P4_F01_数据筛选与逐任务复现审计")


# ------------------------------------------------------------------ F02
def f02():
    sel = pd.read_csv(FD / "F02_average_model_selection.csv")
    cvp = pd.read_csv(FD / "F02_average_cv_predictions.csv")
    tgt = pd.read_csv(FD / "F02_selected_target_metrics.csv")
    mat = pd.read_csv(FD / "A01_bridge_cv_mae_matrix.csv").set_index("target_name")
    allcv = pd.read_csv(RS / "bridge_cv_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.55),
                             gridspec_kw=dict(width_ratios=[1, 1.15, 1], wspace=0.5))
    # (a)
    ax = axes[0]
    names = {"constant": "常数", "linear": "单调线性", "spline": "单调样条", "isotonic": "保序"}
    sel = sel.set_index("model").loc[["constant", "linear", "spline", "isotonic"]]
    best = sel.cv_mae.idxmin()
    thr = sel.loc[best, "cv_mae"] + sel.loc[best, "cv_mae_se"]
    yy = np.arange(len(sel))[::-1]
    ax.axvspan(0, thr, color=lighten(BLUE, 0.9), zorder=0)
    ax.axvline(thr, color=RED, ls="--", lw=0.9)
    for yi, (m, r) in zip(yy, sel.iterrows()):
        c = ORANGE if r.selected else GRAY
        ax.errorbar(r.cv_mae, yi, xerr=r.cv_mae_se, fmt="o", color=c, ms=5 if r.selected else 4,
                    capsize=2, lw=1.2, mfc=c if r.selected else "white")
        ax.text(r.cv_mae, yi + 0.26, f"{r.cv_mae:.2f}", ha="center", fontsize=7, color=c)
    ax.set_yticks(yy, [names[m] for m in sel.index])
    ax.set_ylim(-0.6, len(sel) - 0.4)
    ax.set_xlim(7, 14.5)
    ax.text(thr + 0.12, -0.45, f"one-SE 阈值 {thr:.2f}", color=RED, fontsize=7, va="bottom")
    ax.set_xlabel("留来源族 CV-MAE / 分")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "a", x=-0.3)

    # (b)
    ax = axes[1]
    hi = cvp.weight >= 1.0
    lim = (0, 50)
    ax.plot(lim, lim, color=GRAY, ls="--", lw=0.7, zorder=1)
    for mask, c, mk, lab in [(~hi, GREEN, "o", "中可比（权重 0.25）"), (hi, BLUE, "D", "高可比（权重 1）")]:
        d = cvp[mask]
        ax.vlines(d.actual, d.p10, d.p90, color=lighten(c, 0.6), lw=0.6, zorder=2)
        ax.scatter(d.actual, d.prediction, s=12 if mk == "o" else 14, marker=mk, color=c,
                   edgecolor="white", lw=0.3, zorder=3, label=lab)
    mae = np.average(np.abs(cvp.actual - cvp.prediction))
    ax.set_xlim(lim); ax.set_ylim(-5, 50)
    ax.set_xlabel("观测 Average / 分")
    ax.set_ylabel("留族预测 Average / 分")
    r = sel.loc["linear"]
    ax.text(0.03, 0.97, f"单调线性：CV-MAE {r.cv_mae:.1f}\n80%区间覆盖 {r.coverage80:.0%}", transform=ax.transAxes,
            va="top", fontsize=7.5)
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    panel_label(ax, "b", x=-0.18)

    # (c)
    ax = axes[2]
    order = tgt.sort_values("cv_mae").target_name.tolist()
    yy = np.arange(len(order))
    for yi, tn in zip(yy, order):
        r = tgt[tgt.target_name == tn].iloc[0]
        base = mat.loc[tn, "constant"]
        c = RED if r.bridge_weak else ORANGE
        ax.plot([base, r.cv_mae], [yi, yi], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
        ax.scatter(base, yi, s=22, facecolor="white", edgecolor=GRAY, lw=0.9, zorder=2)
        ax.errorbar(r.cv_mae, yi, xerr=r.cv_mae_se, fmt="o", color=c, ms=4.2, capsize=1.8, lw=1, zorder=3)
        tag = "（弱）" if r.bridge_weak else f"（{names[r.model]}）"
        ax.text(max(base, r.cv_mae) + r.cv_mae_se + 0.4, yi, f"{r.cv_mae:.1f}", va="center", fontsize=7, color=c)
    ax.set_yticks(yy, [tn.replace("MMLU_PRO", "MMLU-PRO") for tn in order])
    for lbl, tn in zip(ax.get_yticklabels(), order):
        if tgt[tgt.target_name == tn].bridge_weak.iloc[0]:
            lbl.set_color(RED)
    ax.set_xlim(0, 27)
    ax.set_xlabel("选中模型 CV-MAE / 分")
    ax.grid(axis="y", visible=False)
    h = [Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY, label="常数基线"),
         Line2D([], [], marker="o", ls="", color=ORANGE, label="选中桥接"),
         Line2D([], [], marker="o", ls="", color=RED, label="弱桥接")]
    ax.legend(handles=h, loc="upper left", handletextpad=0.05, borderaxespad=0.1, fontsize=6.5, handlelength=1.0,
              labelspacing=0.25)
    panel_label(ax, "c", x=-0.3)
    save(fig, OUT / "P4_F02_Loss_Benchmark桥接选择与留族验证")


# ------------------------------------------------------------------ F03
def f03():
    d = pd.read_csv(FD / "F03_problem3_average_mapping.csv")
    fig, ax = plt.subplots(figsize=(FULL_W, 2.6))
    x0, x1 = 1.55, 3.65
    ax.axvspan(1.65, 2.84, color=lighten(GREEN, 0.88), zorder=0, lw=0)
    ax.axvspan(2.0933, 2.5978, color=lighten(BLUE, 0.8), zorder=0, lw=0)
    ax.axvspan(2.84, x1, color=lighten(GRAY, 0.85), zorder=0, lw=0)
    ax.axvspan(x0, 1.65, color=lighten(GRAY, 0.85), zorder=0, lw=0)
    ytxt = 33
    ax.text((2.0933 + 2.5978) / 2, ytxt, "高可比桥接范围\n2.093–2.598", ha="center", va="top", fontsize=7, color=BLUE)
    ax.text(1.66, ytxt, "扩展范围\n1.65–2.84", ha="left", va="top", fontsize=7, color=darken(GREEN, 0.1))
    ax.text(3.25, ytxt, "桥接范围外（外推）", ha="center", va="top", fontsize=7, color=darken(GRAY, 0.2))
    for (b, ct), g in d.groupby(["budget_FLOPs", "quality_cost_type"]):
        ax.vlines(g.predicted_loss, g.p10, g.p90, color=BUD_COL[b], lw=0.7, alpha=0.7, zorder=2)
        ax.scatter(g.predicted_loss, g["median"], marker=COST_MK[ct], s=16, color=BUD_COL[b],
                   edgecolor="white", lw=0.3, zorder=3)
    pos = {1e24: (1.99, 22.5), 1e22: (2.29, 15.5), 1e19: (3.10, 9.0)}
    for b, g in d.groupby("budget_FLOPs"):
        tier = TIER_LAB[g.bridge_support.iloc[0]]
        ax.text(*pos[b], f"{BUD_LAB[b]} FLOPs\n中位 {g['median'].min():.1f}–{g['median'].max():.1f}（{tier}）",
                fontsize=7, color=darken(BUD_COL[b], 0.1) if b != 1e19 else darken(BUD_COL[b], 0.4))
    ax.set_xlim(x0, x1); ax.set_ylim(0, 34)
    ax.set_xlabel("问题三最优方案预测 Loss")
    ax.set_ylabel("映射 Average / 分")
    ax.grid(axis="x", visible=False)
    h = [Line2D([], [], marker="o", ls="", color=BUD_COL[b], label=f"预算 {BUD_LAB[b]}") for b in BUD_COL]
    h += [Line2D([], [], marker=COST_MK[c], ls="", mfc="white", mec=INK, label=COST_LAB[c]) for c in COST_MK]
    h += [Line2D([], [], color=GRAY, lw=0.8, label="80% 联合区间")]
    ax.legend(handles=h, loc="upper right", bbox_to_anchor=(1.0, 0.86), ncol=2, fontsize=7, columnspacing=0.8,
              handletextpad=0.2, labelspacing=0.3)
    save(fig, OUT / "P4_F03_问题三Loss到综合能力映射")


# ------------------------------------------------------------------ F04
def f04():
    p = pd.read_csv(RS / "frontier_training_panel.csv")
    p = p[p.frontier_eligible.astype(str) == "True"] if "frontier_eligible" in p else p
    p["mon"] = pd.to_datetime(p.submission_date).dt.to_period("M")
    months = pd.period_range("2024-06", "2025-01", freq="M")
    midx = {m: i for i, m in enumerate(months)}
    p["mi"] = p.mon.map(midx)
    cur = pd.read_csv(FD / "F04_dynamic_frontier_curves.csv")
    fig, ax = plt.subplots(figsize=(FULL_W, 2.8))
    sc = ax.scatter(p.log10_compute, p["Average ⬆️"], c=p.mi, cmap=SEQ_CMAP, vmin=-2.5, vmax=len(months) - 1,
                    s=22, edgecolor=INK, lw=0.35, zorder=3)
    c0 = cur[cur.month == "2024-06"]; c1 = cur[cur.month == "2025-01"]
    ax.plot(c0.log10_compute, c0.score, color=lighten(ORANGE, 0.35), lw=3.2, zorder=2,
            label="0.90 前沿，2024-06", solid_capstyle="butt")
    ax.plot(c1.log10_compute, c1.score, color=darken(ORANGE, 0.35), lw=1.1, ls=(0, (4, 2)), zorder=2.5,
            label="0.90 前沿，2025-01")
    diff = np.max(np.abs(c1.score.values - c0.score.values))
    ax.text(22.05, 43, f"两期前沿逐点最大差 {diff:.4f} 分\n（曲线重合 → 非规模项≈0）", fontsize=7.5,
            color=darken(ORANGE, 0.3), va="top")
    lab = {"Qwen/Qwen2.5-72B": (-8, 6, "right"), "Qwen/Qwen2-72B": (6, -8, "left"),
           "microsoft/phi-4": (-6, 6, "right"), "google/gemma-2-27b": (6, -8, "left")}
    for m, (dx, dy, ha) in lab.items():
        r = p[p.Model == m].iloc[0]
        ax.annotate(m.split("/")[1], (r.log10_compute, r["Average ⬆️"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, ha=ha, color=INK,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=GRAY))
    ax.set_xlabel(r"训练算力 $\log_{10}C$ / FLOPs")
    ax.set_ylabel("Average 综合能力 / 分")
    ax.set_xlim(21.5, 25.2); ax.set_ylim(0, 57)
    ax.legend(loc="upper left", handlelength=2.4)
    cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.04, aspect=25)
    cb.set_ticks([0, 2, 4, 6, 7]); cb.set_ticklabels([str(months[i]) for i in [0, 2, 4, 6, 7]])
    cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.4)
    cb.set_label("提交月份", fontsize=8)
    save(fig, OUT / "P4_F04_训练算力与动力学能力前沿")


# ------------------------------------------------------------------ F05
def f05():
    dec = pd.read_csv(RS / "contribution_decomposition.csv").set_index("estimate")
    dr = pd.read_csv(FD / "F05_contribution_draws.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.35),
                             gridspec_kw=dict(width_ratios=[1.1, 1, 1], wspace=0.45))
    ax = axes[0]
    items = [("总变化", "total_score", GRAY), ("规模\n贡献", "scale_score", BLUE), ("非规模\n残余", "tech_score", ORANGE)]
    for i, (nm, col, c) in enumerate(items):
        ax.bar(i, dec.loc["central", col], color=lighten(c, 0.45), width=0.6, edgecolor="none")
        ax.vlines(i, dec.loc["p025", col], dec.loc["p975", col], color=c, lw=1.0)
        ax.vlines(i, dec.loc["p10", col], dec.loc["p90", col], color=c, lw=3.2)
        ax.plot(i, dec.loc["central", col], "o", color="white", mec=c, ms=4, zorder=4)
        v = dec.loc["central", col]
        txt = f"{v:.3f}" if abs(v) > 1e-3 else f"{v:.4f}"
        ax.text(i + 0.1, dec.loc["p975", col] + 0.15 if abs(v) > 1e-3 else 0.5, txt, ha="center", fontsize=7.5, color=c)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(3), [x[0] for x in items])
    ax.set_ylabel("能力变化 / 分")
    ax.set_ylim(-0.5, 7.6)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], color=INK, lw=3.2, label="80%"), Line2D([], [], color=INK, lw=1, label="95%")],
              loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=7, handlelength=1.2)
    panel_label(ax, "a", x=-0.2)

    ax = axes[1]
    bins = np.linspace(dr.scale_score.min(), dr.scale_score.max(), 30)
    ax.hist(dr.scale_score, bins=bins, color=lighten(BLUE, 0.35), edgecolor="white", lw=0.3)
    ax.axvline(dec.loc["central", "scale_score"], color=BLUE, lw=1.1)
    ax.set_xlabel("规模贡献 bootstrap 抽样 / 分")
    ax.set_ylabel("频数")
    ax.text(0.03, 0.97, f"中心 {dec.loc['central','scale_score']:.2f}\n95%: [{dec.loc['p025','scale_score']:.2f}, "
            f"{dec.loc['p975','scale_score']:.2f}]\n$P$(>0) = {(dr.scale_score > 0).mean():.0%}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.set_ylim(0, 250)
    ax.grid(axis="x", visible=False)
    panel_label(ax, "b", x=-0.22)

    ax = axes[2]
    share = 100 * dr.scale_score / dr.total_score  # 规模贡献占总变化比例
    q = np.percentile(share, [2.5, 50, 97.5])
    ax.hist(share, bins=np.linspace(q[0] - 0.1, q[2] + 0.1, 30), color=lighten(BLUE, 0.35), edgecolor="white", lw=0.3)
    ax.axvline(100, color=RED, ls="--", lw=0.9)
    ax.set_ylim(0, 200)
    ax.set_xlabel("规模贡献占比 / %")
    ax.set_ylabel("频数")
    ax.text(0.03, 0.97, f"中位 {q[1]:.2f}%\n95%: [{q[0]:.2f}, {q[2]:.2f}]%", transform=ax.transAxes,
            va="top", fontsize=7)
    ax.grid(axis="x", visible=False)
    panel_label(ax, "c", x=-0.22)
    save(fig, OUT / "P4_F05_规模与非规模贡献联合分布")
    return q


# ------------------------------------------------------------------ F06
def f06():
    mo = pd.read_csv(RS / "compute_growth_monthly.csv")
    sm = json.loads((RS / "compute_growth_summary.json").read_text())
    sp = pd.read_csv(FD / "F06_compute_scenario_paths.csv", parse_dates=["date"])
    mo["date"] = pd.to_datetime(mo.month + "-01")
    fig, ax = plt.subplots(figsize=(FULL_W, 2.7))
    end_i = mo.month_index.max()
    win = mo[mo.month_index >= end_i - 23]
    tr = sm["endpoint_log10_compute"] + sm["monthly_log10_slope"] * (win.month_index - end_i)
    ax.scatter(mo.date, mo.log10_compute_q90, s=8 + 2.2 * mo.model_families, color=BLUE, edgecolor="white",
               lw=0.4, zorder=3, label="月度 0.90 分位算力（点面积∝家族数）")
    ax.plot(win.date, tr, color=INK, lw=1.3, zorder=2, label="24 个月稳健趋势")
    cut = mo.date.max()
    ax.axvline(cut, color=GRAY, ls="--", lw=0.8)
    for k in [0.25, 0.5, 1.0]:
        g = sp[sp.scenario_factor == k]
        ax.fill_between(g.date, g.p10, g.p90, color=KAPPA_COL[k], alpha=0.14, lw=0)
        ax.plot(g.date, g.center, color=KAPPA_COL[k], lw=1.4)
        ax.text(g.date.iloc[-1] + pd.Timedelta(days=12), g.center.iloc[-1],
                f"{KAPPA_LAB[k]}  {g.center.iloc[-1]:.2f}", va="center", fontsize=7.5, color=KAPPA_COL[k])
    ax.plot(cut, sm["endpoint_log10_compute"], "o", color=ORANGE, mec=INK, mew=0.5, ms=5, zorder=5)
    ax.annotate(f"$x_{{C0}}$ = {sm['endpoint_log10_compute']:.3f}\n$r_C$ = {sm['annual_log10_slope']:.3f} /年"
                f"（{10**sm['annual_log10_slope']:.2f}×/年）",
                (cut, sm["endpoint_log10_compute"]), xytext=(22, -62), textcoords="offset points", fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=GRAY))
    ax.text(cut + pd.Timedelta(days=10), 22.75, "截止 2025-03", color=GRAY, fontsize=7)
    ax.set_xlim(pd.Timestamp("2023-01-15"), pd.Timestamp("2028-04-01"))
    ax.set_ylim(22.6, 27.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("C4 发布时间")
    ax.set_ylabel(r"训练算力 $\log_{10}C$ / FLOPs")
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(color=GRAY, alpha=0.25, lw=0)); l.append("情景 80% 区间")
    ax.legend(h, l, loc="upper left", fontsize=7)
    save(fig, OUT / "P4_F06_训练算力趋势与放缓情景")
    return sm


# ------------------------------------------------------------------ F07
def f07():
    f = pd.read_csv(FD / "F07_forecast_summary_q090.csv")
    cur = pd.read_csv(FD / "F04_dynamic_frontier_curves.csv")
    p = pd.read_csv(RS / "frontier_training_panel.csv")
    obs_max = p["Average ⬆️"].max()
    obs_mod = p.loc[p["Average ⬆️"].idxmax(), "Model"].split("/")[1]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.28))
    ax = axes[0]
    off = {1.0: -2.6, 0.5: 0, 0.25: 2.6}
    ax.axhline(obs_max, color=GRAY, ls="--", lw=0.8)
    ax.text(1.0, 22, f"灰虚线：截止期观测最高 {obs_max:.1f}（{obs_mod}）", fontsize=7, color=darken(GRAY, 0.2))
    for k in [1.0, 0.5, 0.25]:
        g = f[f.scenario_factor == k].sort_values("horizon_months")
        x = g.horizon_months + off[k]
        c = KAPPA_COL[k]
        ax.vlines(x, g.p025, g.p975, color=c, lw=0.9)
        ax.vlines(x, g.p10, g.p90, color=c, lw=3.5, alpha=0.55)
        ax.plot(x, g.forecast_score, "o", color=c, mec="white", mew=0.6, ms=5.5, label=KAPPA_LAB[k], zorder=4)
        for xi, v, top in zip(x, g.forecast_score, g.p975):
            ax.text(xi, top + 1.2, f"{v:.1f}", fontsize=7, color=c, ha="center", va="bottom")
    ax.set_xticks([12, 24], ["12 个月\n(2026-03)", "24 个月\n(2027-03)"])
    ax.set_xlim(0, 30)
    ax.set_ylim(18, 104)
    ax.set_ylabel(r"0.90 能力前沿 Average / 分")
    ax.set_xlabel("预测步长（截止 2025-03 起）")
    ax.grid(axis="x", visible=False)
    panel_label(ax, "a", x=-0.14)

    ax = axes[1]
    c1 = cur[cur.month == "2025-01"]
    xr = (p.log10_compute.min(), p.log10_compute.max())
    ax.axvspan(xr[0], xr[1], color=lighten(GRAY, 0.88), lw=0, zorder=0)
    ax.plot(c1.log10_compute, c1.score, color=ORANGE, lw=1.3, label="2025-01 前沿（样本范围内）")
    ax.axvline(f.compute_endpoint_log10.iloc[0], color=GRAY, ls=":", lw=0.8)
    for _, r in f.iterrows():
        c = KAPPA_COL[r.scenario_factor]
        mk = "o" if r.horizon_months == 12 else "s"
        ax.vlines(r.forecast_log10_compute, r.p10, r.p90, color=c, lw=2.2, alpha=0.5)
        ax.plot(r.forecast_log10_compute, r.forecast_score, mk, color=c, mec="white", mew=0.5, ms=5, zorder=4)
    ax.text(xr[0] + 0.05, 95, "观测算力范围", fontsize=7, color=darken(GRAY, 0.2), va="top")
    ax.text(f.compute_endpoint_log10.iloc[0] + 0.03, 28, "$x_{C0}$", fontsize=7.5, color=GRAY)
    h = [Line2D([], [], color=ORANGE, lw=1.3, label="2025-01 前沿"),
         Line2D([], [], marker="o", ls="", color=INK, label="12 个月"),
         Line2D([], [], marker="s", ls="", color=INK, label="24 个月")]
    ax.legend(handles=h, loc="lower right", fontsize=7)
    hk = [Line2D([], [], marker="o", ls="", color=KAPPA_COL[k], label=KAPPA_LAB[k]) for k in [1.0, 0.5, 0.25]]
    hk += [Line2D([], [], color=GRAY, lw=3.5, alpha=0.6, label="80% 区间"), Line2D([], [], color=GRAY, lw=0.9, label="95% 区间（仅 a）")]
    from matplotlib.legend import Legend
    ax.add_artist(Legend(ax, hk, [x.get_label() for x in hk], loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=7,
                         handletextpad=0.3, frameon=False))
    ax.set_xlabel(r"预测训练算力 $\log_{10}C$ / FLOPs")
    ax.set_ylabel(r"0.90 能力前沿 Average / 分")
    ax.set_xlim(21.5, 26.9); ax.set_ylim(0, 100)
    panel_label(ax, "b", x=-0.14)
    save(fig, OUT / "P4_F07_算力放缓下的未来能力前沿")
    return f


# ------------------------------------------------------------------ F08
def f08():
    bk = pd.read_csv(FD / "F08_backtest_skill.csv")
    w = pd.read_csv(FD / "F08_average_uncertainty_widths.csv")
    m3 = pd.read_csv(FD / "F03_problem3_average_mapping.csv")[["scenario_index", "bridge_support"]]
    w = w.merge(m3, left_on="scenario_id", right_on="scenario_index")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.55), gridspec_kw=dict(width_ratios=[1, 1.05], wspace=0.35))
    ax = axes[0]
    rows = [("M-C", 3), ("M-C", 6), ("M-time", 3), ("M-time", 6)]
    yy = np.arange(len(rows))[::-1]
    cmap = {"M-C": BLUE, "M-time": GRAY}
    rng = np.random.default_rng(0)
    for yi, (m, h) in zip(yy, rows):
        g = bk[(bk.model == m) & (bk.horizon_months == h)]
        c = cmap[m]
        ax.scatter(g.skill, yi + rng.uniform(-0.12, 0.12, len(g)), s=12, color=lighten(c, 0.35), lw=0, zorder=2)
        mu = g.skill.mean()
        ax.plot(mu, yi, "D", color=c, mec="white", ms=6, zorder=3)
        ax.text(mu, yi + 0.3, f"{mu:+.2f}", ha="center", fontsize=7, color=c)
    ax.axvline(0, color=RED, ls="--", lw=0.8)
    ax.set_yticks(yy, [f"{m}，{h} 个月" for m, h in rows])
    ax.set_xlabel("相对 M-scale 的技能分数（弹球损失）")
    ax.set_xlim(-9.8, 1.5)
    ax.set_ylim(-0.6, 3.7)
    ax.grid(axis="y", visible=False)
    ax.text(-9.6, 3.55, "点：4 个滚动截止期；菱形：均值", fontsize=7, color=darken(GRAY, 0.2), va="top")
    panel_label(ax, "a", x=-0.36)

    ax = axes[1]
    comps = [("upstream_only", "上游 Loss"), ("bridge_only", "桥接模型"), ("joint", "联合传播")]
    for i, (cp, nm) in enumerate(comps):
        g = w[w.component == cp]
        for tier, c in TIER_COL.items():
            gg = g[g.bridge_support == tier]
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(gg)), gg.width80, s=11, color=c, alpha=0.8, lw=0,
                       zorder=3, label=TIER_LAB[tier] if i == 0 else None)
        q1, med, q3 = np.percentile(g.width80, [25, 50, 75])
        ax.hlines(med, i - 0.3, i + 0.3, color=INK, lw=1.2, zorder=4)
        ax.text(i, 13.1, f"中位 {med:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(3), [c[1] for c in comps])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.6, 14)
    ax.set_ylabel("Average 80% 区间宽度 / 分")
    ax.grid(axis="x", visible=False)
    ax.legend(title="桥接支持", loc="center left", bbox_to_anchor=(0.0, 0.45), fontsize=7, handletextpad=0.1, markerscale=1.3)
    panel_label(ax, "b", x=-0.14)
    save(fig, OUT / "P4_F08_滚动回测与桥接不确定性")
    return bk, w


# ------------------------------------------------------------------ A02
def a02():
    s = pd.read_csv(FD / "A02_frontier_sensitivity_used.csv")
    lab = {("epsilon", "0.1"): r"$\epsilon$ = 0.1", ("epsilon", "0.5"): r"$\epsilon$ = 0.5（基准）",
           ("epsilon", "1.0"): r"$\epsilon$ = 1.0",
           ("strict_unrestricted", "strict"): "仅宽松许可证子样本", ("compute_evidence", "reported_operation"): "仅报告/操作计数算力",
           ("time_axis", "publication_date"): "时间轴：发布日", ("ability_construct", "six_domain_rank_normal"): "能力分：六域秩正态",
           ("month_block_length", "1"): "月份块长 = 1", ("month_block_length", "3"): "月份块长 = 3"}
    s["name"] = [lab[(a, str(b))] for a, b in zip(s.sensitivity, s.level)]
    s = s.iloc[::-1].reset_index(drop=True)
    base = s[s.name.str.contains("基准")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    yy = np.arange(len(s))
    for ax, col, lo, hi, c, xl, scl in [(axes[0], "scale_score", "scale_p025", "scale_p975", BLUE, "规模贡献 / 分", 1),
                                        (axes[1], "tech_score", "tech_p025", "tech_p975", ORANGE,
                                         r"非规模残余 / $10^{-3}$ 分", 1e3)]:
        ax.axvline(base[col] * scl, color=GRAY, ls=":", lw=0.8)
        if col == "tech_score":
            ax.axvline(0, color=RED, ls="--", lw=0.8)
        for yi, r in s.iterrows():
            if pd.notna(r[lo]):
                ax.hlines(yi, r[lo] * scl, r[hi] * scl, color=c, lw=1.1)
            isb = "基准" in r["name"]
            ax.plot(r[col] * scl, yi, "D" if isb else "o", color=c, mec="white", mew=0.5, ms=6 if isb else 5, zorder=3)
            if col == "scale_score":
                ax.text(max(r[col], r[hi] if pd.notna(r[hi]) else r[col]) + 0.25, yi, f"{r[col]:.2f}",
                        va="center", fontsize=7, color=c)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(yy, s.name)
    axes[0].set_xlim(0, 13)
    axes[1].set_xlim(-35, 8)
    axes[1].text(0.02, 0.02, "横线：月份块 bootstrap 95% 区间", transform=axes[1].transAxes, fontsize=7, color=darken(GRAY, 0.2))
    panel_label(axes[0], "a", x=-0.02); panel_label(axes[1], "b", x=-0.02)
    save(fig, OUT / "P4_A02_贡献分解规格敏感性")
    return s


if __name__ == "__main__":
    f01(); f02(); f03(); f04()
    q = f05(); print("share q", q)
    f06(); f07(); f08(); a02()
    print("done")
