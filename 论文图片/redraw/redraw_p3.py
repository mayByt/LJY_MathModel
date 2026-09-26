"""问题三全部图件重绘（论文终稿版）。

数据：论文图片/问题三/figure_data/*.csv(.gz) 与 问题三/results_v1/transition_points.csv。
所有派生量均为确定性计算：
  - 100·ΔL = 100·(L*(L_ctx) − L*(2048))  （数据已给出 delta_loss_x100）
  - C_attn/C_train = η·L_ctx/6          （数据已给出 attention_train_ratio）
  - N*/N*(2048), D*/D*(2048)            （数据已给出 N_ratio, D_ratio）
运行：python redraw_p3.py
"""
import sys
sys.path.insert(0, "/home/chenxu/MayByte/LJY/论文图片")
from paper_style import *  # noqa: F401,F403
apply_style()

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

FD = Path("/home/chenxu/MayByte/LJY/论文图片/问题三/figure_data")
RD = Path("/home/chenxu/MayByte/LJY/问题三/results_v1")
OUT = Path("/home/chenxu/MayByte/LJY/overleaf-paper/images/no3")

COSTS = ["exponential", "power", "logarithmic"]
COST_CN = {"exponential": "指数成本", "power": "幂函数成本", "logarithmic": "对数成本"}
COST_C = {"exponential": BLUE, "power": ORANGE, "logarithmic": GREEN}
BUDGETS = [1e19, 1e22, 1e24]
BUD_C = dict(zip(BUDGETS, seq_colors(3)))
BUD_LAB = {1e19: r"$10^{19}$", 1e22: r"$10^{22}$", 1e24: r"$10^{24}$"}
CTX = [2048, 4096, 8192, 32768, 131072]
ETA = 2e-4
L_CRIT = 6 / ETA  # 30000
Q0 = 0.5389559587975633

EVENT_CN = {
    "quality_investment_start": "质量投入启动",
    "quality_saturation": "质量达到饱和",
    "Q_increment_slope_break": "质量增量斜率断点",
    "N_B_slope_break": r"$N^*$ 斜率断点",
    "D_B_slope_break": r"$D^*$ 斜率断点",
}


def pow10_fmt(x, _pos=None):
    return rf"$10^{{{int(round(x))}}}$"


def ctx_fmt(x, _pos=None):
    return {2048: "2K", 4096: "4K", 8192: "8K", 32768: "32K", 131072: "128K"}.get(int(round(x)), "")


def primary_transitions(ctx=4096):
    t = pd.read_csv(RD / "transition_points.csv")
    t = t[(t.context_length == ctx) & (t.stable) & (t.transition_class == "primary")]
    return t


# ============================================================ F01
def fig01():
    d = pd.read_csv(FD / "P3_F01_budget_share_composition.csv")
    tr = primary_transitions()
    c_train, c_qual, c_attn = lighten(BLUE, 0.5), lighten(PURPLE, 0.25), lighten(GRAY, 0.6)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, PANEL_H + 0.25), sharey=True)
    for i, (ax, ct) in enumerate(zip(axes, COSTS)):
        g = d[d.quality_cost_type == ct].sort_values("log10_budget")
        x = g.log10_budget.values
        y1 = g.share_train_pct.values
        y2 = y1 + g.share_quality_pct.values
        ax.fill_between(x, 0, y1, color=c_train, lw=0)
        ax.fill_between(x, y1, y2, color=c_qual, lw=0)
        ax.fill_between(x, y2, 100, color=c_attn, lw=0)
        ax.plot(x, y1, color=darken(c_train, 0.25), lw=0.7)
        ax.plot(x, y2, color=darken(c_qual, 0.25), lw=0.7)
        # 质量提升份额峰值
        k = g.share_quality_pct.values.argmax()
        ax.plot(x[k], y2[k], "s", ms=3.8, mfc="white", mec=INK, mew=0.8, zorder=5, clip_on=False)
        ha = "left" if x[k] < 21 else "right"
        ax.annotate(f"提质峰值 {g.share_quality_pct.values[k]:.1f}%", (x[k], (y1[k] + y2[k]) / 2),
                    xytext=(x[k] + 0.18, (y1[k] + y2[k]) / 2), fontsize=7.5, color=INK,
                    ha=ha, va="center")
        for _, r in tr[tr.quality_cost_type == ct].iterrows():
            xm = np.log10(r.budget_mid)
            ax.axvline(xm, color=RED, ls="--", lw=0.9)
            ax.text(xm + 0.1, 3, EVENT_CN[r.event_type].replace("质量", ""), rotation=90,
                    fontsize=7, color=RED, ha="left", va="bottom")
        ax.set_xlim(19, 24)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(FixedLocator([19, 20, 21, 22, 23, 24]))
        ax.xaxis.set_major_formatter(FuncFormatter(pow10_fmt))
        ax.set_xlabel("算力预算 $C$ / FLOPs")
        ax.set_title(COST_CN[ct], color=COST_C[ct])
        ax.grid(False)
        panel_label(ax, "abc"[i])
    axes[0].set_ylabel("预算份额 / %")
    handles = [Patch(color=c_train, label=r"基础训练 $6ND$"),
               Patch(color=c_qual, label=r"质量提升 $D[g(Q)-g(Q_0)]_+$"),
               Patch(color=c_attn, label=r"长上下文注意力 $\eta NDL_{\mathrm{ctx}}$"),
               Line2D([], [], color=RED, ls="--", lw=0.9, label="主要结构转移")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, OUT / "P3_F01_预算约束下的边际资源竞争机制")


# ============================================================ F02
def fig02():
    p = pd.read_csv(FD / "P3_F02_context_loss_bar3d.csv")
    r = pd.read_csv(FD / "P3_F02_context_cost_ratio.csv")
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 2 * PANEL_H + 0.2))
    axes = axes.ravel()
    ymax = p.delta_loss_x100.max() * 1.12
    for i, ct in enumerate(COSTS):
        ax = axes[i]
        for b in BUDGETS:
            g = p[(p.quality_cost_type == ct) & (p.budget_FLOPs == b)].sort_values("context_length")
            ax.plot(g.context_length, g.delta_loss_x100, "-o", color=BUD_C[b], ms=3.5,
                    label=rf"$C={BUD_LAB[b][1:-1]}$")
            v = g.delta_loss_x100.values[-1]
            ax.annotate(f"{v:.1f}", (131072, v), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, va="center", color=darken(BUD_C[b], 0.35))
        ax.axvline(L_CRIT, color=RED, ls="--", lw=0.9)
        ax.set_title(COST_CN[ct], color=COST_C[ct])
        ax.set_ylim(-1, ymax)
        ax.set_ylabel(r"最优损失惩罚 $100\,\Delta L^*$")
        panel_label(ax, "abc"[i])
    axes[0].text(L_CRIT * 1.07, ymax * 0.93, r"$L_{\mathrm{ctx}}^{\mathrm{crit}}=30000$",
                 color=RED, fontsize=7.5, ha="left", va="top")
    axes[0].legend(loc="upper left", title="算力预算 / FLOPs")
    # (d) 注意力/基础训练开销比
    ax = axes[3]
    ax.plot(r.context_length, r.attention_train_ratio, color=GRAY, lw=1.3)
    ax.axhline(1, color=GRAY, ls=":", lw=0.8)
    ax.axvline(L_CRIT, color=RED, ls="--", lw=0.9)
    ax.plot([L_CRIT], [1], "o", color=RED, ms=4, zorder=5)
    for L in CTX:
        ax.plot(L, ETA * L / 6, "o", mfc="white", mec=GRAY, ms=3.5, mew=0.9, zorder=4)
    ax.annotate(r"$\eta L_{\mathrm{ctx}}/6=1$", (L_CRIT, 1), xytext=(-8, 10),
                textcoords="offset points", ha="right", fontsize=7.5, color=RED)
    ax.set_yscale("log")
    ax.set_ylabel(r"$C_{\mathrm{attn}}/C_{\mathrm{train}}=\eta L_{\mathrm{ctx}}/6$")
    ax.set_title("注意力开销相对基础训练开销")
    panel_label(ax, "d")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlim(1500, 180000)
        ax.xaxis.set_major_locator(FixedLocator(CTX))
        ax.xaxis.set_major_formatter(FuncFormatter(ctx_fmt))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"上下文长度 $L_{\mathrm{ctx}}$ / token")
    fig.tight_layout()
    save(fig, OUT / "P3_F02_上下文成本临界与最优损失惩罚")


# ============================================================ F03
def fig03():
    d = pd.read_csv(FD / "P3_F03_optimization_landscape.csv.gz")
    paths = pd.read_csv(FD / "P3_F07_optimal_configuration_paths.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, PANEL_H + 0.35), sharey=True)
    for i, (ax, ct) in enumerate(zip(axes, COSTS)):
        g = d[d.quality_cost_type == ct]
        piv = g.pivot_table(index="Q_A", columns="log10_N_B", values="predicted_loss")
        X, Y = np.meshgrid(piv.columns.values, piv.index.values)
        Z = piv.values
        lmin = np.nanmin(Z)
        ex = (Z - lmin) * 100  # 相对本面板最优的超额 Loss ×100
        levels = [0, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 10]
        cf = ax.contourf(X, Y, ex, levels=levels, cmap=SEQ_CMAP.reversed(), extend="max")
        ax.contour(X, Y, ex, levels=levels[1:], colors=[INK], linewidths=0.3, alpha=0.4)
        opt = paths[(paths.quality_cost_type == ct) & (np.isclose(paths.budget_FLOPs, 1e19))].iloc[0]
        ax.plot(np.log10(opt.N_B), opt.Q_A, marker="*", ms=10, color=RED, mec="white", mew=0.7,
                zorder=6, clip_on=False)
        ax.annotate(rf"$N^*$={opt.N_B:.2f}B, $D^*$={opt.D_B:.2f}B" + "\n" + rf"$Q_A^*$={opt.Q_A:.3f}",
                    (np.log10(opt.N_B), opt.Q_A), xytext=(0.97, 0.05 if opt.Q_A > 0.9 else 0.95),
                    textcoords="axes fraction", ha="right", va="bottom" if opt.Q_A > 0.9 else "top",
                    fontsize=7, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.set_title(COST_CN[ct], color=COST_C[ct])
        ax.set_xlabel(r"$\log_{10}(N/\mathrm{B})$")
        ax.grid(False)
        panel_label(ax, "abc"[i])
    axes[0].set_ylabel(r"质量水平 $Q_A$")
    axes[0].set_yticks([Q0, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].set_yticklabels([r"$Q_0$", "0.6", "0.7", "0.8", "0.9", "1.0"])
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes([0.915, 0.2, 0.015, 0.68])
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label(r"超额损失 $100(L-L^*)$")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    cb.ax.tick_params(labelsize=7)
    save(fig, OUT / "P3_F03_三类质量成本下的低预算优化景观")


# ============================================================ F04
def fig04():
    d = pd.read_csv(FD / "P3_F04_context_response_profiles.csv")
    ls = {1e19: "-", 1e22: "--", 1e24: ":"}
    mk = {1e19: "o", 1e22: "s", 1e24: "^"}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, PANEL_H + 0.3))
    cols = [("N_ratio", r"$N^*/N^*(2048)$"), ("D_ratio", r"$D^*/D^*(2048)$"), ("Q_A", r"最优质量 $Q_A^*$")]
    for i, (ax, (c, lab)) in enumerate(zip(axes, cols)):
        for ct in COSTS:
            for b in BUDGETS:
                g = d[(d.quality_cost_type == ct) & (d.budget_FLOPs == b)].sort_values("context_length")
                ax.plot(g.context_length, g[c], ls=ls[b], marker=mk[b], color=COST_C[ct], ms=3.2,
                        mfc="white" if b == 1e22 else COST_C[ct], mew=0.8, lw=1.1)
        ax.axvline(L_CRIT, color=RED, ls="--", lw=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xlim(1700, 160000)
        ax.xaxis.set_major_locator(FixedLocator(CTX))
        ax.xaxis.set_major_formatter(FuncFormatter(ctx_fmt))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"上下文长度 $L_{\mathrm{ctx}}$")
        ax.set_ylabel(lab)
        panel_label(ax, "abc"[i])
    axes[0].set_ylim(0.3, 1.05)
    axes[1].set_ylim(0.25, 1.05)
    axes[2].axhline(Q0, color=GRAY, ls=":", lw=0.8)
    axes[2].text(1800, Q0 + 0.01, r"$Q_0$", color=GRAY, fontsize=7.5)
    axes[2].text(L_CRIT * 1.08, 0.9, r"$L_{\mathrm{ctx}}^{\mathrm{crit}}$", color=RED, fontsize=7.5)
    h1 = [Line2D([], [], color=COST_C[c], lw=1.4, label=COST_CN[c]) for c in COSTS]
    h2 = [Line2D([], [], color=INK, ls=ls[b], marker=mk[b], ms=3.2, mfc="white" if b == 1e22 else INK,
                 label=rf"$C={BUD_LAB[b][1:-1]}$") for b in BUDGETS]
    fig.legend(handles=h1 + h2, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, OUT / "P3_F04_上下文长度对最优配置的影响")


# ============================================================ F05
def fig05():
    iv = pd.read_csv(FD / "P3_F05_stable_transition_intervals.csv")
    bs = pd.read_csv(FD / "P3_F05_transition_joint_bootstrap.csv")
    order = [("power", "quality_investment_start"), ("power", "Q_increment_slope_break"),
             ("power", "quality_saturation"), ("power", "N_B_slope_break"), ("power", "D_B_slope_break"),
             ("exponential", "Q_increment_slope_break"), ("exponential", "quality_saturation"),
             ("exponential", "D_B_slope_break")]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.9))
    rng = np.random.default_rng(0)  # 仅用于散点竖向抖动的显示位置
    ylabels = []
    for k, (ct, ev) in enumerate(order):
        y = len(order) - 1 - k
        r = iv[(iv.quality_cost_type == ct) & (iv.event_type == ev)].iloc[0]
        col = COST_C[ct]
        # bootstrap 抽样（仅联合 bootstrap 覆盖的事件）
        vals = pd.concat([bs[(bs.quality_cost_type == ct) & (bs.x_event == ev)].x_log10_budget,
                          bs[(bs.quality_cost_type == ct) & (bs.y_event == ev)].y_log10_budget])
        if len(vals):
            ax.scatter(vals + rng.uniform(-0.02, 0.02, len(vals)), y + rng.uniform(-0.28, -0.12, len(vals)),
                       s=3, color=lighten(col, 0.35), lw=0, alpha=0.8, zorder=2)
        ax.plot([r.q025, r.q975], [y, y], color=lighten(col, 0.3), lw=1.2, zorder=3)
        ax.plot([r.q25, r.q75], [y, y], color=col, lw=3.2, solid_capstyle="butt", zorder=3)
        filled = r.transition_class == "primary"
        ax.plot(r["median"], y, "o", ms=4.5, color=col, mfc=col if filled else "white", mew=1, zorder=5)
        ax.plot(r.center_log10, y, "D", ms=3.6, mfc="none", mec=INK, mew=0.8, zorder=6)
        ax.text(r.q975 + 0.04, y, f"$10^{{{r.center_log10:.2f}}}$", va="center", fontsize=7, color=col)
        ylabels.append(("主  " if filled else "次  ") + EVENT_CN[ev].replace("$", "$"))
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels(ylabels)
    for tl, (ct, _) in zip(ax.get_yticklabels(), order):
        tl.set_color(COST_C[ct])
    ax.axhline(2.5, color=GRID, lw=0.8)
    ax.set_xlim(18.95, 20.95)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(19, 21.01, 0.25)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_xlabel(r"转移预算 $\log_{10}(C/\mathrm{FLOPs})$")
    ax.grid(axis="y", visible=False)
    h = [Line2D([], [], color=BLUE, lw=3, label="指数成本"),
         Line2D([], [], color=ORANGE, lw=3, label="幂函数成本"),
         Line2D([], [], color=GRAY, lw=1.2, label="95% 区间"),
         Line2D([], [], color=GRAY, lw=3.2, label="四分位区间"),
         Line2D([], [], ls="", marker="o", color=GRAY, label="bootstrap 中位数（实心=主转移）"),
         Line2D([], [], ls="", marker="D", mfc="none", mec=INK, label="中心参数结果")]
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.1))
    fig.tight_layout()
    save(fig, OUT / "P3_F05_稳定结构转移的区间估计")


# ============================================================ F06
def fig06():
    d = pd.read_csv(FD / "P3_F06_sensitivity_4096.csv")
    dims = [("quality_mapping_slope", "质量映射斜率 ×0.5 / ×1.5"),
            ("cost_shape", "成本形状 ×0.8 / ×1.2"),
            ("cost_amplitude", "成本幅度 ×0.8 / ×1.2"),
            ("attention_eta", r"注意力系数 $\eta$ ×0.8 / ×1.2"),
            ("quality_empirical_cap", r"质量经验上限 $Q\leq0.88$"),
            ("support", "严格联合支持域")]
    off = {"exponential": 0.22, "power": 0.0, "logarithmic": -0.22}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.9), sharey=True)
    for i, (ax, b) in enumerate(zip(axes, BUDGETS)):
        g = d[np.isclose(d.budget_FLOPs, b)]
        ax.axvspan(-100, 0, color=lighten(BLUE, 0.9), lw=0)
        ax.axvspan(0, 100, color=lighten(ORANGE, 0.88), lw=0)
        ax.axvline(0, color=INK, lw=0.6)
        for k, (dim, _) in enumerate(dims):
            y = len(dims) - 1 - k
            for ct in COSTS:
                s = g[(g.dimension == dim) & (g.quality_cost_type == ct)].copy()
                s["setting"] = s.setting.astype(str)
                col = COST_C[ct]
                yy = y + off[ct]
                if len(s) == 2:
                    s = s.sort_values("setting", key=lambda v: v.astype(float))
                    lo, hi = s.loss_change_x100.values
                    ax.plot([lo, hi], [yy, yy], color=lighten(col, 0.45), lw=0.9, zorder=2)
                    ax.plot(lo, yy, "o", ms=4, mfc="white", mec=col, mew=1, zorder=4)
                    ax.plot(hi, yy, "o", ms=4, color=col, zorder=4)
                else:
                    ax.plot(s.loss_change_x100.values, [yy], "D", ms=3.8, color=col, zorder=4)
        ax.set_xscale("symlog", linthresh=1, linscale=1.0)
        ax.set_xlim(-8, 30)
        ax.xaxis.set_major_locator(FixedLocator([-5, -1, 0, 1, 5, 20]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
        ax.set_title(rf"$C={BUD_LAB[b][1:-1]}$ FLOPs")
        ax.set_xlabel(r"最优损失变化 $100\,\Delta L^*$")
        for k in range(len(dims) - 1):
            ax.axhline(k + 0.5, color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.5, len(dims) - 0.5)
        panel_label(ax, "abc"[i])
    axes[0].set_yticks(range(len(dims))[::-1])
    axes[0].set_yticklabels([lab for _, lab in dims])
    axes[0].text(-0.6, -0.42, "改善", color=BLUE, fontsize=7, ha="right")
    axes[0].text(0.6, -0.42, "恶化", color=darken(ORANGE, 0.2), fontsize=7, ha="left")
    h = [Line2D([], [], ls="", marker="o", color=COST_C[c], label=COST_CN[c]) for c in COSTS]
    h += [Line2D([], [], ls="", marker="o", mfc="white", mec=INK, label="较低设置"),
          Line2D([], [], ls="", marker="o", color=INK, label="较高设置"),
          Line2D([], [], ls="", marker="D", color=INK, ms=3.8, label="单侧检查")]
    fig.legend(handles=h, loc="upper center", ncol=6, bbox_to_anchor=(0.55, 1.06))
    fig.tight_layout()
    save(fig, OUT / "P3_F06_关键假设的敏感性分析")


# ============================================================ F07
def fig07():
    d = pd.read_csv(FD / "P3_F07_optimal_configuration_paths.csv")
    tr = primary_transitions()
    p_start = np.log10(tr[(tr.quality_cost_type == "power") & (tr.event_type == "quality_investment_start")].budget_mid.iloc[0])
    p_sat = np.log10(tr[(tr.quality_cost_type == "power") & (tr.event_type == "quality_saturation")].budget_mid.iloc[0])
    e_sat = np.log10(tr[(tr.quality_cost_type == "exponential") & (tr.event_type == "quality_saturation")].budget_mid.iloc[0])
    fig = plt.figure(figsize=(FULL_W, 4.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.26)
    axq = fig.add_subplot(gs[0, :])
    axn = fig.add_subplot(gs[1, 0], sharex=axq)
    axd = fig.add_subplot(gs[1, 1], sharex=axq)
    stage_c = [lighten(ORANGE, 0.93), lighten(ORANGE, 0.8), lighten(GRAY, 0.9)]
    bounds = [19, p_start, p_sat, 24]
    for ax in (axq, axn, axd):
        for j in range(3):
            ax.axvspan(bounds[j], bounds[j + 1], color=stage_c[j], lw=0, zorder=0)
        for xv, c in [(p_start, ORANGE), (p_sat, ORANGE), (e_sat, BLUE)]:
            ax.axvline(xv, color=c, ls=":", lw=0.9, zorder=1)
    for ct in COSTS:
        g = d[d.quality_cost_type == ct].sort_values("log10_budget")
        axq.plot(g.log10_budget, g.Q_A, color=COST_C[ct], lw=1.5, label=COST_CN[ct], zorder=3)
        axn.plot(g.log10_budget, g.N_B, color=COST_C[ct], lw=1.3, zorder=3)
        axd.plot(g.log10_budget, g.D_B, color=COST_C[ct], lw=1.3, zorder=3)
    axq.axhline(Q0, color=GRAY, ls="--", lw=0.8)
    axq.text(23.95, Q0 - 0.03, r"$Q_0=0.539$", color=GRAY, fontsize=7.5, ha="right")
    axq.set_ylabel(r"最优质量 $Q_A^*$")
    axq.set_ylim(0.45, 1.04)
    axq.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # 阶段标注（幂函数成本）
    ytxt = 0.78
    axq.annotate("未启动", (19.065, 0.5), xytext=(19.16, 0.475), fontsize=7, color=darken(ORANGE, 0.3), va="center", arrowprops=dict(arrowstyle="-", color=darken(ORANGE, 0.3), lw=0.6))
    axq.text(p_sat - 0.12, 0.475, "竞争", ha="center", va="center", fontsize=7.5, color=darken(ORANGE, 0.3))
    axq.text(22.2, 0.75, "饱和后规模主导（$Q_A^*$ 触及上限，增量预算全部流向 $N,D$）",
             ha="center", fontsize=7.5, color=INK)
    axq.annotate(rf"幂函数：启动 $10^{{{p_start:.2f}}}$，饱和 $10^{{{p_sat:.2f}}}$",
                 (p_sat, 0.93), xytext=(p_sat + 0.35, 0.9), fontsize=7, color=darken(ORANGE, 0.25),
                 arrowprops=dict(arrowstyle="-", color=darken(ORANGE, 0.25), lw=0.6))
    axq.annotate(rf"指数：饱和 $10^{{{e_sat:.2f}}}$", (e_sat, 0.67), xytext=(e_sat + 0.35, 0.64),
                 fontsize=7, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    axq.legend(loc="lower right", ncol=3, bbox_to_anchor=(1.0, 0.1))
    for ax, lab in [(axn, r"最优参数量 $N^*$ / 十亿"), (axd, r"最优训练 token $D^*$ / 十亿")]:
        ax.set_yscale("log")
        ax.set_ylabel(lab)
        ax.set_xlabel("算力预算 $C$ / FLOPs")
    for ax in (axq, axn, axd):
        ax.set_xlim(19, 24)
        ax.xaxis.set_major_locator(FixedLocator([19, 20, 21, 22, 23, 24]))
        ax.xaxis.set_major_formatter(FuncFormatter(pow10_fmt))
        ax.grid(axis="x", visible=False)
    # N*, D* 在 1e24 的数值
    for ax, c in [(axn, "N_B"), (axd, "D_B")]:
        v = d[(d.quality_cost_type == "exponential")].sort_values("log10_budget")[c].values[-1]
        ax.annotate(f"{v:.1f}B" if v < 100 else f"{v:.0f}B", (24, v), xytext=(-3, 4), textcoords="offset points", ha="right",
                    fontsize=7, color=INK)
    panel_label(axq, "a", x=-0.01)
    panel_label(axn, "b")
    panel_label(axd, "c")
    save(fig, OUT / "P3_F07_最优配置路径与结构转移")


# ============================================================ F08
def fig08():
    s = pd.read_csv(FD / "P3_F08_loss_surface.csv.gz")
    g = pd.read_csv(FD / "P3_F08_grid_candidates.csv")
    t = pd.read_csv(FD / "P3_F08_multistart_trajectory.csv")
    v = pd.read_csv(FD / "P3_F08_verification_summary.csv").iloc[0]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(FULL_W, PANEL_H + 0.5),
                                 gridspec_kw=dict(width_ratios=[1.15, 1]))
    piv = s.pivot_table(index="Q_A", columns="log10_N_B", values="loss_excess_x1e3")
    X, Y = np.meshgrid(piv.columns.values, piv.index.values)
    levels = [0, 0.5, 1, 2, 3, 4, 6, 8, 10, 14, 20, 29]
    cf = ax.contourf(X, Y, piv.values, levels=levels, cmap=SEQ_CMAP.reversed())
    ax.contour(X, Y, piv.values, levels=levels[1:], colors=[INK], linewidths=0.3, alpha=0.45)
    st = g[g.top20_start]
    ax.plot(st.log10_N_B, st.Q_A, "s", ms=4, mfc="white", mec=ORANGE, mew=0.9, zorder=5, label="网格候选起点（20 个）")
    for k, tr in t.groupby("start_rank"):
        tr = tr.sort_values("iteration")
        ax.plot(tr.log10_N_B, tr.Q_A, "-", color=ORANGE, lw=0.7, alpha=0.9, zorder=4)
        ax.plot(tr.log10_N_B.values[1:-1], tr.Q_A.values[1:-1], ".", color=ORANGE, ms=2.2, zorder=4)
    ax.plot(np.log10(v.N_B), v.Q_A, "*", ms=11, color=RED, mec="white", mew=0.7, zorder=6, label="最优解")
    ax.plot([], [], "-", color=ORANGE, lw=0.8, label="SLSQP 迭代轨迹")
    ax.set_xlabel(r"$\log_{10}(N/\mathrm{B})$")
    ax.set_ylabel(r"质量水平 $Q_A$")
    ax.set_xlim(piv.columns.min(), piv.columns.max())
    ax.set_ylim(piv.index.min(), piv.index.max())
    ax.grid(False)
    ax.legend(loc="lower right", fontsize=7, facecolor="white", framealpha=0.85, frameon=True, edgecolor="none")
    cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r"超额损失 $10^3(L-L^*)$")
    cb.ax.tick_params(labelsize=7)
    panel_label(ax, "a")
    # (b) 相对 Loss 差距
    floor = 1e-16
    ranks = sorted(t.start_rank.unique())
    cols = [SEQ_CMAP(0.35 + 0.6 * i / (len(ranks) - 1)) for i in range(len(ranks))]
    for c, k in zip(cols, ranks):
        tr = t[t.start_rank == k].sort_values("iteration")
        bx.plot(tr.iteration, np.maximum(tr.relative_loss_gap, floor), "-o", color=c, ms=2, lw=0.8)
    bx.set_yscale("log")
    bx.set_ylim(floor * 0.5, 1)
    bx.axhline(1e-8, color=RED, ls="--", lw=0.8)
    bx.text(t.iteration.max(), 1.4e-8, r"$10^{-8}$", color=RED, fontsize=7.5, ha="right")
    bx.set_xlabel("SLSQP 迭代次数")
    bx.set_ylabel(r"相对损失差距 $(L-L^*)/L^*$")
    bx.xaxis.set_major_locator(FixedLocator(range(0, 12, 2)))
    bx.text(0.97, 0.97, rf"20/20 起点收敛，$L^*$={v.loss_2d:.5f}" + "\n" + r"与细网格、三维复核相对差 = 0",
            transform=bx.transAxes, ha="right", va="top", fontsize=7)
    bx.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    panel_label(bx, "b")
    fig.tight_layout()
    save(fig, OUT / "P3_F08_多起点寻优过程与收敛验证")


if __name__ == "__main__":
    which = sys.argv[1:] or ["01", "02", "03", "04", "05", "06", "07", "08"]
    for w in which:
        globals()[f"fig{w}"]()
        print("done", w)
