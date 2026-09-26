"""问题二（跨维度数据融合与广义标度律）终稿插图重绘。

运行：python redraw_p2.py  → 输出到 overleaf-paper/images/no2/
数据：论文图片/问题二/figure_data/ 与 问题二/results_p12_final_v2/
派生量（均为确定性计算）：
  * 经典/广义标度律曲线：L = E + A n^-α + B d^-β + c_Q (1-Q)^ν，n=N/1e9, d=D/1e9（s_Q=1）
  * 数据坍缩：y = L_obs − E − A n^-α，应落在 B d^-β 上
  * 配比效应相对强度：θ(n) = slope / (intercept − E)（composite 行），E 取拟合值及 1.5/1.9 敏感性
  * 等 Loss 前沿曲线：由上式对 D 解析反解；等价资源增幅：brentq 求根
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
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
from scipy.optimize import brentq
from scipy.stats import spearmanr

FD = Path("/home/chenxu/MayByte/LJY/论文图片/问题二/figure_data")
RES = Path("/home/chenxu/MayByte/LJY/问题二/results_p12_final_v2")
OUT = Path("/home/chenxu/MayByte/LJY/overleaf-paper/images/no2")

E, A, B, ALPHA, BETA = 1.6897975628980855, 0.3539803206859227, 1.2403055835550174, 0.33997658188127156, 0.27987812854272615
CQ, NUQ = 0.3620399283468315, 0.9902880682591864
QREF = 0.5389559587975633
MODEL_CN = {"GQ1": "有效样本量", "GQ2": "加性质量惩罚", "GQ3": "指数变化", "GQ4": "双通道质量"}
MODEL_COL = {"GQ1": GRAY, "GQ2": GREEN, "GQ3": BLUE, "GQ4": ORANGE}


def classic(n, d):
    return E + A * np.power(n, -ALPHA) + B * np.power(d, -BETA)


def general(n, d, q):
    return classic(n, d) + CQ * np.power(np.clip(1 - q, 0, None), NUQ)


def d_on_isoloss(n, L, q):
    """解析反解等 Loss 线上的 D（十亿 Token）；不可达处返回 nan。"""
    rem = L - E - A * np.power(n, -ALPHA) - CQ * (1 - q) ** NUQ
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(rem > 0, np.power(np.where(rem > 0, rem, 1) / B, -1 / BETA), np.nan)
    return d


def fmt_N(nb):
    return f"{nb*1000:.0f}M" if nb < 1 else f"{nb:.1f}B"


# =====================================================================  F01
def fig01():
    df = pd.read_csv(FD / "F01_classic_scaling.csv")
    rs = pd.read_csv(FD / "F01_residual_summary.csv").sort_values("N_params_B")
    Ns = np.sort(df.N_params_B.unique())
    cols = dict(zip(Ns, seq_colors(8, lo=0.3, hi=1.0)))

    fig = plt.figure(figsize=(FULL_W, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.85], wspace=0.42)
    ax1, ax3, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # (a) 观测 + M0 拟合
    dgrid = np.logspace(np.log10(df.D_tokens_B.min()), np.log10(df.D_tokens_B.max()), 200)
    for N in Ns:
        s = df[df.N_params_B == N]
        ax1.plot(dgrid, classic(N, dgrid), color=ORANGE, lw=0.6, zorder=2)
        ax1.scatter(s.D_tokens_B, s.val_loss, s=3, color=cols[N], lw=0, alpha=0.9, zorder=3)
    ax1.set_xscale("log")
    ax1.set_xlabel("训练数据量 $D$ / 十亿 Token")
    ax1.set_ylabel("验证损失 $L$")
    ax1.legend(handles=[Line2D([], [], ls="", marker="o", ms=3, color=BLUE, label="观测检查点（按 $N$ 着色）"),
                        Line2D([], [], color=ORANGE, lw=0.9, label="M0 拟合")],
               loc="upper right", handlelength=1.2)
    ax1.text(0.03, 0.04, "8 种规模 × 147 点", transform=ax1.transAxes, fontsize=7.5, color=GRAY)
    panel_label(ax1, "a")

    # (c→中间) 数据坍缩
    y = df.val_loss - E - A * np.power(df.N_params_B, -ALPHA)
    for N in Ns:
        m = df.N_params_B == N
        ax3.scatter(df.D_tokens_B[m], y[m], s=3, color=cols[N], lw=0, alpha=0.8, zorder=2)
    ax3.plot(dgrid, B * dgrid ** (-BETA), color=ORANGE, lw=1.0, zorder=3,
             label=r"$B\,d^{-\beta}$，$\beta=0.280$")
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set_yticks([0.3, 0.5, 1.0, 2.0]); ax3.set_yticklabels(["0.3", "0.5", "1.0", "2.0"])
    ax3.yaxis.set_minor_formatter(plt.NullFormatter())
    ax3.set_xlabel("训练数据量 $D$ / 十亿 Token")
    ax3.set_ylabel(r"$L-E-A\,n^{-\alpha}$")
    ax3.legend(loc="upper right", handlelength=1.2)
    ax3.text(0.03, 0.04, "8 条曲线坍缩为一条", transform=ax3.transAxes, fontsize=7.5, color=GRAY)
    panel_label(ax3, "b")

    # (c) 残差区间
    for i, r in enumerate(rs.itertuples()):
        c = cols[r.N_params_B]
        ax2.plot([r.q025, r.q975], [i, i], color=c, lw=0.9, solid_capstyle="butt")
        ax2.plot([r.q25, r.q75], [i, i], color=c, lw=4.0, solid_capstyle="butt")
        ax2.plot(r.median, i, "o", ms=3.2, mfc="white", mec=INK, mew=0.6, zorder=4)
        ax2.text(4.9, i, f"{r.RMSE_x1e4:.2f}", va="center", ha="left", fontsize=7, color=INK)
    ax2.text(4.9, len(rs) - 0.35, "RMSE", ha="left", va="bottom", fontsize=7, color=GRAY)
    ax2.axvline(0, color=GRAY, lw=0.6, ls="--")
    ax2.set_yticks(range(len(rs)))
    ax2.set_yticklabels([fmt_N(v) for v in rs.N_params_B])
    ax2.set_xlim(-4.6, 4.6)
    ax2.set_ylim(-0.6, len(rs) - 0.2)
    ax2.set_xlabel(r"残差 / $10^{-4}$")
    ax2.set_ylabel("参数量 $N$")
    ax2.grid(axis="y", visible=False)
    panel_label(ax2, "c")
    save(fig, OUT / "P2R_F01_经典标度律拟合与数据坍缩")


# =====================================================================  F02
def fig02():
    sel = pd.read_csv(FD / "F02_quality_model_selection.csv").set_index("model")
    pr = pd.read_csv(FD / "F03_GQ2_leave_cell_predictions.csv")
    fig = plt.figure(figsize=(FULL_W, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.9, 1.15], wspace=0.55)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))

    splits = ["leave_N", "leave_D", "leave_Q", "leave_cell"]
    xl = ["留 $N$", "留 $D$", "留 $Q$", "留 $(N,D)$"]
    x = np.arange(4)
    for m in ["GQ1", "GQ4", "GQ3", "GQ2"]:
        v = [sel.loc[m, f"nRMSE_{s}"] for s in splits]
        ax1.plot(x, v, "-o", color=MODEL_COL[m], lw=1.6 if m == "GQ2" else 1.0,
                 ms=4 if m == "GQ2" else 3, zorder=5 if m == "GQ2" else 3,
                 label=f"{m} {MODEL_CN[m]}")
    ax1.set_xticks(x); ax1.set_xticklabels(xl)
    ax1.set_ylabel("成块留出 nRMSE")
    ax1.set_ylim(0.13, 0.335); ax1.set_yticks([0.15, 0.20, 0.25])
    h, l = ax1.get_legend_handles_labels()
    order = [3, 2, 1, 0]
    ax1.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=2,
               fontsize=6.5, handlelength=1.0, columnspacing=0.5, handletextpad=0.3, bbox_to_anchor=(0.5, 1.03))
    panel_label(ax1, "a")

    # (b) 等权 RMSE
    ms = ["GQ1", "GQ2", "GQ3", "GQ4"]
    yy = np.arange(4)[::-1]
    for yi, m in zip(yy, ms):
        ax2.barh(yi, sel.loc[m, "equal_weight_RMSE"], xerr=sel.loc[m, "standard_error"], height=0.6,
                 color=MODEL_COL[m], ecolor=INK, error_kw=dict(lw=0.7, capsize=2))
        ax2.text(sel.loc[m, "equal_weight_RMSE"] + sel.loc[m, "standard_error"] + 0.003, yi,
                 f"{sel.loc[m, 'equal_weight_RMSE']:.3f}", va="center", fontsize=7)
    one_se = sel.loc["GQ2", "equal_weight_RMSE"] + sel.loc["GQ2", "standard_error"]
    ax2.axvline(one_se, color=RED, ls="--", lw=0.9)
    ax2.text(one_se, 3.62, "one-SE", color=RED, fontsize=7, ha="center", va="bottom")
    ax2.set_yticks(yy); ax2.set_yticklabels([f"{MODEL_CN[m]}\n({m})" for m in ms], fontsize=7)
    ax2.set_xlim(0, 0.15); ax2.set_ylim(-0.6, 3.95)
    ax2.set_xlabel("四类切分等权 RMSE")
    ax2.grid(axis="y", visible=False)
    panel_label(ax2, "b")

    # (c) 预测-观测
    norm = Normalize(0.1, 1.0)
    sc = ax3.scatter(pr.val_loss, pr.predicted, c=pr.Q_score, cmap=SEQ_CMAP, norm=norm, s=6,
                     lw=0.2, edgecolors=darken(BLUE, 0.3))
    lo, hi = 1.95, 3.8
    ax3.plot([lo, hi], [lo, hi], color=GRAY, ls="--", lw=0.8)
    ax3.set_xlim(lo, hi); ax3.set_ylim(lo, hi); ax3.set_aspect("equal")
    ax3.set_xlabel("观测损失 $L$"); ax3.set_ylabel("GQ2 留单元预测")
    rmse = np.sqrt(np.mean(pr.residual ** 2)); rho = spearmanr(pr.val_loss, pr.predicted)[0]
    ax3.text(0.96, 0.04, f"$n$={len(pr)}\nRMSE={rmse:.3f}\nSpearman={rho:.3f}", transform=ax3.transAxes,
             va="bottom", ha="right", fontsize=7)
    cb = fig.colorbar(sc, ax=ax3, fraction=0.05, pad=0.03)
    cb.set_label("质量分数 $Q$"); cb.outline.set_linewidth(0.4)
    panel_label(ax3, "c")
    save(fig, OUT / "P2R_F02_质量模型成块交叉验证选择")
    return rmse, rho


# =====================================================================  F04
def fig04():
    s = pd.read_csv(FD / "F04_B7_B8_response_summary.csv")
    fl = pd.read_csv(FD / "F04_B8_floor_rate_by_N.csv")
    fig, axs = plt.subplots(2, 2, figsize=(FULL_W, 4.3))
    plt.subplots_adjust(wspace=0.3, hspace=0.45)
    groups = [
        (axs[0, 0], [("B7 主质量网格", BLUE, "B7（45 个 $(N,D)$ 单元）")]),
        (axs[0, 1], [("B8 校准区·小规模", lighten(ORANGE, 0.25), "小规模 $N\\leq1$B"),
                     ("B8 校准区·大规模", darken(ORANGE, 0.25), "大规模 2.8–12B")]),
        (axs[1, 0], [("B8 外推区·近端", lighten(PURPLE, 0.25), "近端 20–70B"),
                     ("B8 外推区·远端", darken(PURPLE, 0.25), "远端 120–700B")]),
    ]
    titles = ["B7：质量提高、损失下降", "B8 校准区：响应方向反转", "B8 外推区"]
    for (ax, gl), lab, tt in zip(groups, "abc", titles):
        for g, c, name in gl:
            d = s[s.group == g].sort_values("Q_score")
            ax.fill_between(d.Q_score, d.low, d.high, color=c, alpha=0.22, lw=0)
            ax.plot(d.Q_score, d["median"], "-o", color=c, ms=2.5, label=name)
        if lab != "a":
            ax.axhline(0.5, color=RED, ls="--", lw=0.8)
            ax.text(1.0, 0.53, "截断地板 $L=0.5$", color=RED, fontsize=7, ha="right", va="bottom")
            ax.set_ylim(0.35, 3.0)
        else:
            ax.set_ylim(2.0, 3.35)
            d = s[s.group == "B7 主质量网格"]
            a0, a1 = d.iloc[0]["median"], d.iloc[-1]["median"]
            ax.annotate(f"中位数 {a0:.2f}→{a1:.2f}", xy=(1.0, a1), xytext=(0.45, 2.1),
                        fontsize=7, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.6))
        ax.set_xlabel("质量分数 $Q$"); ax.set_ylabel("验证损失 $L$")
        ax.set_title(tt)
        ax.legend(loc="upper left", handlelength=1.2)
        panel_label(ax, lab)
    ax = axs[1, 1]
    for dt, c, name in [("calibrated", ORANGE, "B8 校准区"), ("extrapolated", PURPLE, "B8 外推区")]:
        d = fl[fl.data_type == dt]
        ax.plot(d.N_params_B, d.floor_rate_pct, "-o", color=c, ms=3, label=name)
    ax.set_xscale("log")
    last = fl.iloc[-1]
    ax.annotate(f"{last.floor_rate_pct:.1f}%", xy=(last.N_params_B, last.floor_rate_pct),
                xytext=(-4, 5), textcoords="offset points", ha="right", fontsize=7, color=PURPLE)
    ax.set_xlabel("参数量 $N$ / 十亿"); ax.set_ylabel("触及截断地板比例 / %")
    ax.set_title("地板记录比例随规模上升")
    ax.set_ylim(-2, 45)
    ax.legend(loc="upper left")
    panel_label(ax, "d")
    save(fig, OUT / "P2R_F04_B7与B8质量响应压力测试")


# =====================================================================  F05
def fig05():
    mt = pd.read_csv(RES / "mixture_transfer_parameters.csv")
    scales = ["1M", "60M", "1B"]
    nval = {"1M": 1e6, "60M": 6e7, "1B": 1e9}
    cols = dict(zip(scales, seq_colors(3)))
    dom = mt[mt.role == "domain"]
    comp = mt[mt.role == "composite"].set_index("scale").loc[scales]

    theta = comp.slope / (comp.intercept - E)
    th_lo = comp.slope / (comp.intercept - 1.5)   # E=1.5 → 分母更大 → θ 更小
    th_hi = comp.slope / (comp.intercept - 1.9)
    th_mean = theta.mean()

    fig = plt.figure(figsize=(FULL_W, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.55, wspace=0.62)
    ax1 = fig.add_subplot(gs[:, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, 1])

    piv = dom.pivot(index="loss_domain", columns="scale", values="slope")[scales].sort_values("1M")
    y = np.arange(len(piv))
    for i, (d, r) in enumerate(piv.iterrows()):
        ax1.plot([r.min(), r.max()], [i, i], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
    for sc in scales:
        ax1.scatter(piv[sc], y, s=16, color=cols[sc], edgecolors=INK, linewidths=0.3, zorder=3, label=sc)
    ax1.axvline(0, color=GRAY, lw=0.6, ls="--")
    DOMAIN_CN = {"ubuntu_irc": "技术对话", "enron_emails": "商务邮件", "dm_mathematics": "数学推理",
                 "nih_exporter": "科研项目", "hackernews": "科技论坛", "gutenberg_pg_19": "古典图书",
                 "stackexchange": "技术问答", "pubmed_central": "生物医学全文", "wikipedia_en": "英文百科",
                 "europarl": "议会文本", "uspto_backgrounds": "专利背景", "freelaw": "法律文本",
                 "arxiv": "学术论文", "github": "代码", "pile_cc": "网页语料",
                 "pubmed_abstracts": "生医摘要", "philpapers": "哲学论文"}
    ax1.set_yticks(y); ax1.set_yticklabels([DOMAIN_CN.get(d, d) for d in piv.index], fontsize=7.5)
    ax1.set_xlabel("损失对 $R(p)$ 的斜率")
    ax1.grid(axis="y", visible=False)
    ax1.legend(title="实测规模", loc="lower right", handletextpad=0.2)
    panel_label(ax1, "a")

    x = np.array([nval[s] for s in scales])
    ax2.plot(x, comp.slope, color=GRAY, lw=0.9, zorder=1)
    for xi, sc in zip(x, scales):
        ax2.scatter(xi, comp.loc[sc, "slope"], s=26, color=cols[sc], edgecolors=INK, linewidths=0.4, zorder=3)
        ax2.annotate(f"{comp.loc[sc, 'slope']:.3f}", (xi, comp.loc[sc, "slope"]), xytext=(5, 3),
                     textcoords="offset points", fontsize=7)
    ax2.set_xscale("log"); ax2.set_ylim(0, 0.34); ax2.set_xlim(4e5, 3e9)
    ax2.set_ylabel("绝对效应 $s(n)$")
    ax2.set_xticks(x); ax2.set_xticklabels(scales)
    ax2.text(0.04, 0.08, f"1B/1M = {comp.loc['1B','slope']/comp.loc['1M','slope']*100:.0f}%",
             transform=ax2.transAxes, fontsize=7, color=INK)
    panel_label(ax2, "b")

    ax3.fill_between(x, th_lo, th_hi, color=lighten(ORANGE, 0.65), lw=0, label=r"$E\in[1.5,1.9]$")
    ax3.plot(x, theta, color=GRAY, lw=0.9, zorder=1)
    for xi, sc in zip(x, scales):
        ax3.scatter(xi, theta[sc], s=26, color=cols[sc], edgecolors=INK, linewidths=0.4, zorder=3)
        ax3.annotate(f"{theta[sc]:.3f}", (xi, theta[sc]), xytext=(5, -9), textcoords="offset points", fontsize=7)
    ax3.axhline(th_mean, color=ORANGE, ls="--", lw=0.9, label=rf"均值 $\bar\theta$={th_mean:.3f}")
    ax3.set_xscale("log"); ax3.set_ylim(0, 0.16); ax3.set_xlim(4e5, 3e9)
    ax3.set_xticks(x); ax3.set_xticklabels(scales)
    ax3.set_xlabel("模型规模 $n$")
    ax3.set_ylabel(r"相对效应 $\theta(n)$")
    ax3.legend(loc="lower right", ncol=2, fontsize=6.8, handlelength=1.2, columnspacing=0.8)
    panel_label(ax3, "c")
    save(fig, OUT / "P2R_F05_配比效应跨尺度迁移")
    return pd.DataFrame({"slope": comp.slope, "intercept": comp.intercept, "theta_E_fit": theta,
                         "theta_E1.5": th_lo, "theta_E1.9": th_hi}), th_mean


# =====================================================================  F06
def fig06():
    ls = pd.read_csv(FD / "F06_generalized_landscape.csv.gz")
    fr = pd.read_csv(FD / "F06_frontier_nodes.csv")
    Nu = np.sort(ls.N_params_B.unique()); Du = np.sort(ls.D_tokens_B.unique())
    Z = ls.pivot(index="D_tokens_B", columns="N_params_B", values="prediction").loc[Du, Nu].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.85))
    plt.subplots_adjust(wspace=0.5)
    levels = np.arange(1.95, 3.45, 0.1)
    cf = ax1.contourf(Nu, Du, Z, levels=levels, cmap=SEQ_CMAP, extend="both")
    cs = ax1.contour(Nu, Du, Z, levels=[2.0, 2.2, 2.5], colors="white", linewidths=0.9)
    ax1.clabel(cs, fmt=lambda v: f"L={v:.1f}", fontsize=6.5, inline_spacing=2)
    ax1.add_patch(Rectangle((0.07, 10), 12 - 0.07, 600 - 10, fill=False, ec=ORANGE, lw=1.1, ls="--"))
    ax1.text(0.085, 650, "B1/B7 观测支持域", color=darken(ORANGE, 0.2), fontsize=7, va="bottom")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("参数量 $N$ / 十亿"); ax1.set_ylabel("训练数据量 $D$ / 十亿 Token")
    cb = fig.colorbar(cf, ax=ax1, pad=0.02, fraction=0.05, ticks=[2.0, 2.4, 2.8, 3.2])
    cb.set_label("预测损失 $L$"); cb.outline.set_linewidth(0.4)
    panel_label(ax1, "a")

    qs = [0.4, QREF, 0.8]
    qc = dict(zip(qs, seq_colors(3)))
    lst = {2.0: "-", 2.2: "--", 2.5: ":"}
    ng = np.logspace(np.log10(0.07), 3, 400)
    for q in qs:
        for L in [2.0, 2.2, 2.5]:
            d = d_on_isoloss(ng, L, q)
            d = np.where((d >= 10) & (d <= 36000), d, np.nan)
            ax2.plot(ng, d, color=qc[q], ls=lst[L], lw=1.2)
            nd = fr[(np.isclose(fr.Q, q, atol=1e-4)) & (np.isclose(fr.target_loss, L))]
            ax2.scatter(nd.N_params_B, nd.D_tokens_B, s=9, color=qc[q], edgecolors="white", linewidths=0.3, zorder=3)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(0.07, 1000); ax2.set_ylim(10, 36000)
    ax2.set_xlabel("参数量 $N$ / 十亿"); ax2.set_ylabel("训练数据量 $D$ / 十亿 Token")
    h1 = [Line2D([], [], color=qc[q], lw=1.4, label=f"$Q$={q:.3f}") for q in qs]
    h2 = [Line2D([], [], color=GRAY, ls=lst[L], lw=1.1, label=f"$L$={L:.1f}") for L in lst]
    leg = ax2.legend(handles=h1, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7,
                     handlelength=1.4, columnspacing=0.8)
    ax2.add_artist(leg)
    ax2.legend(handles=h2, loc="lower left", fontsize=7, handlelength=1.8)
    panel_label(ax2, "b")
    save(fig, OUT / "P2R_F06_广义标度律地形与等损失前沿")
    # 前沿移动量：L=2.2、N=25.8B 时各质量所需 D
    return {q: float(d_on_isoloss(np.array([25.84059]), 2.2, q)[0]) for q in qs}


# =====================================================================  F07
def fig07():
    sp = pd.read_csv(FD / "F07_elasticity_scale_path.csv")
    ss = pd.read_csv(FD / "F07_elasticity_scenario_summary.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.5), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.35))
    base = sp[sp.s_Q == 1.0].sort_values("N_params_B")
    lo = sp[sp.s_Q == 0.5].sort_values("N_params_B"); hi = sp[sp.s_Q == 1.5].sort_values("N_params_B")
    for key, c, name in [("epsilon_N_total", BLUE, r"$\varepsilon_N$ 参数"), ("epsilon_D_total", ORANGE, r"$\varepsilon_D$ Token"),
                         ("epsilon_Q_total", GREEN, r"$\varepsilon_Q$ 质量")]:
        ax1.fill_between(base.N_params_B, np.minimum(lo[key].values, hi[key].values),
                         np.maximum(lo[key].values, hi[key].values), color=lighten(c, 0.7), lw=0)
        ax1.plot(base.N_params_B, base[key], color=c, lw=1.4, label=name)
    diff = base.epsilon_Q_total.values - base.epsilon_N_total.values
    k = np.where(np.diff(np.sign(diff)) != 0)[0]
    ncross = None
    if len(k):
        i = k[0]
        ncross = float(np.exp(np.interp(0, [diff[i], diff[i + 1]], np.log(base.N_params_B.values[i:i + 2]))))
        ycross = float(np.interp(np.log(ncross), np.log(base.N_params_B.values), base.epsilon_Q_total.values))
        ax1.axvline(ncross, color=RED, ls="--", lw=0.8)
        ax1.annotate(f"$N$≈{ncross:.2f}B\n质量弹性超过参数", xy=(ncross, ycross), xytext=(0.9, 0.108),
                     fontsize=7, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.6))
    ax1.set_xscale("log")
    ax1.set_xlabel("参数量 $N$ / 十亿（$D$=150B，$Q=Q_{ref}$）")
    ax1.set_ylabel(r"标准弹性 $\varepsilon$")
    h, l = ax1.get_legend_handles_labels()
    h.append(Patch(color=lighten(GREEN, 0.7))); l.append("$s_Q$∈[0.5,1.5] 情景带")
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=7.5)
    panel_label(ax1, "a")

    yy = np.arange(3)[::-1]
    for yi, r in zip(yy, ss.itertuples()):
        left = 0
        for v, c in [(r.mean_share_N, BLUE), (r.mean_share_D, ORANGE), (r.mean_share_Q, GREEN)]:
            ax2.barh(yi, v * 100, left=left, color=c, height=0.62, edgecolor="white", lw=0.5)
            ax2.text(left + v * 50, yi, f"{v*100:.1f}", ha="center", va="center", color="white", fontsize=7)
            left += v * 100
        ax2.text(101.5, yi, f"{r.dominant_Q}/{r.grid_states}", va="center", fontsize=7, color=darken(GREEN, 0.2))
    ax2.text(101.5, 2.52, "质量主导", fontsize=6.8, color=darken(GREEN, 0.2), va="bottom")
    ax2.set_yticks(yy); ax2.set_yticklabels([f"$s_Q$={v}" for v in ss.s_Q])
    ax2.set_xlim(0, 100); ax2.set_ylim(-0.5, 2.9)
    ax2.set_xlabel("27 个状态平均弹性份额 / %")
    ax2.grid(visible=False)
    panel_label(ax2, "b")
    save(fig, OUT / "P2R_F07_资源弹性组成状态图")
    return ncross


# =====================================================================  F08
def fig08():
    eq = pd.read_csv(FD / "F08_equivalence_sensitivity.csv")
    N0, D0, Q0, dQ = 1.0, 150.0, QREF, 0.10
    L0 = general(N0, D0, Q0); Lt = general(N0, D0, Q0 + dQ)
    xN = brentq(lambda x: general(N0 * (1 + x), D0, Q0) - Lt, 0, 5)
    xD = brentq(lambda x: general(N0, D0 * (1 + x), Q0) - Lt, 0, 5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.5), gridspec_kw=dict(wspace=0.32))
    x = np.linspace(0, 1.0, 300)
    ax1.plot(x * 100, general(N0 * (1 + x), D0, Q0), color=BLUE, label="仅增加参数量 $N$")
    ax1.plot(x * 100, general(N0, D0 * (1 + x), Q0), color=ORANGE, label="仅增加 Token 量 $D$")
    ax1.axhline(Lt, color=GREEN, ls="--", lw=1.0, label=r"提质 $\Delta Q$=0.10 的目标")
    for xv, c, t in [(xN, BLUE, "+"), (xD, ORANGE, "+")]:
        ax1.plot(xv * 100, Lt, "o", color=c, ms=4.5, mec="white", mew=0.5, zorder=4)
        ax1.plot([xv * 100] * 2, [Lt, 2.4725], color=c, ls=":", lw=0.8)
        ax1.text(xv * 100, Lt + 0.0012, f"+{xv*100:.1f}%", color=c, fontsize=7.5, ha="right" if c == BLUE else "left",
                 va="bottom", fontweight="bold")
    ax1.set_xlim(0, 100); ax1.set_ylim(2.4725, 2.519)
    ax1.set_xlabel("单项资源增幅 / %"); ax1.set_ylabel("预测损失 $L$")
    ax1.text(0.97, 0.66, f"基准 $N$=1B，$D$=150B，$Q$={Q0:.3f}", transform=ax1.transAxes, ha="right", fontsize=7, color=GRAY)
    ax1.legend(loc="upper right", fontsize=7)
    panel_label(ax1, "a")

    e = eq[np.isclose(eq.delta_Q, 0.1)]
    qa = sorted(e.Q_A.unique())
    lsty = {qa[0]: ":", qa[1]: "-", qa[2]: "--"}
    for q in qa:
        d = e[np.isclose(e.Q_A, q)].sort_values("s_Q")
        lw = 1.5 if np.isclose(q, QREF) else 1.0
        for col, c, mk in [("equiv_N_pct", BLUE, "o"), ("equiv_D_pct", ORANGE, "s")]:
            ax2.plot(d.s_Q, d[col], ls=lsty[q], color=c, lw=lw, marker=mk, ms=3.5 if lw > 1 else 2.8,
                     alpha=1 if lw > 1 else 0.75)
        hit = d[d.hits_quality_ceiling]
        ax2.scatter(hit.s_Q, hit.equiv_D_pct, s=40, facecolors="none", edgecolors=RED, lw=0.8, zorder=5)
        ax2.scatter(hit.s_Q, hit.equiv_N_pct, s=40, facecolors="none", edgecolors=RED, lw=0.8, zorder=5)
    ax2.annotate("触及 $Q$=1 截断", xy=(1.5, 37.1), xytext=(1.08, 90), fontsize=7, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.6))
    ax2.set_xticks([0.5, 1.0, 1.5]); ax2.set_xlim(0.4, 1.6); ax2.set_ylim(0, 110)
    ax2.set_xlabel("质量映射强度 $s_Q$"); ax2.set_ylabel(r"等价资源增幅 / %（$\Delta Q$=0.10）")
    h = [Line2D([], [], color=BLUE, marker="o", ms=3, label="参数 $N$"),
         Line2D([], [], color=ORANGE, marker="s", ms=3, label="Token $D$")] + \
        [Line2D([], [], color=GRAY, ls=lsty[q], label=f"$Q_A$={q:.3f}" + ("（参考）" if np.isclose(q, QREF) else "")) for q in qa]
    ax2.legend(handles=h, loc="upper left", fontsize=6.8, ncol=1, handlelength=1.8)
    panel_label(ax2, "b")
    save(fig, OUT / "P2R_F08_质量提升的有限资源等价量")
    return L0, Lt, xN, xD


# =====================================================================  F10
def fig10():
    ex = pd.read_csv(FD / "F10_external_observed_predicted.csv")
    fig, axs = plt.subplots(2, 2, figsize=(FULL_W, 4.6))
    plt.subplots_adjust(wspace=0.3, hspace=0.42)
    stats = {}
    names = {"B2": "B2", "B4": "B4", "B5": "B5", "B10": "B10（估算值）"}
    for ax, src, lab in zip(axs.flat, ["B2", "B4", "B5", "B10"], "abcd"):
        d = ex[ex.source_table == src]
        c = GRAY if src == "B10" else BLUE
        ax.scatter(d.val_loss, d.predicted, s=5 if len(d) > 200 else 9, color=c, alpha=0.55 if len(d) > 200 else 0.8, lw=0)
        lo = min(d.val_loss.min(), d.predicted.min()); hi = max(d.val_loss.max(), d.predicted.max())
        pad = 0.05 * (hi - lo); lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=ORANGE, ls="--", lw=0.9)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        rmse = np.sqrt(np.mean((d.predicted - d.val_loss) ** 2)); rho = spearmanr(d.val_loss, d.predicted)[0]
        stats[src] = (len(d), rmse, rho)
        ax.text(0.97, 0.04, f"$n$={len(d)}\nRMSE={rmse:.3f}\nSpearman={rho:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7)
        ax.set_title(names[src])
        ax.set_xlabel("观测损失"); ax.set_ylabel("经典标度律预测")
        panel_label(ax, lab)
    save(fig, OUT / "P2R_F10_经典标度律外部证据验证")
    return stats


if __name__ == "__main__":
    fig01()
    print("F02 GQ2 leave-cell RMSE, rho:", fig02())
    fig04()
    th, thm = fig05()
    print(th.to_string()); print("mean theta:", thm)
    print("F06 D needed at N=25.8B,L=2.2:", fig06())
    print("F07 crossover N:", fig07())
    print("F08 L0, Lt, xN, xD:", fig08())
    print("F10:", fig10())
