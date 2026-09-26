"""Read frozen results and create a palette discussion board; never rerun models."""
from pathlib import Path
import hashlib
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "论文绘图"
CFG = json.loads((OUT / "config/style.json").read_text())
SRC = json.loads((OUT / "config/data_sources.json").read_text())
COL = CFG["palette"]
LINE = CFG["line"]
for family in ("Times New Roman", "SimSun"):
    fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"], "font.size": 9,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "axes.unicode_minus": False, "text.color": "#202020",
    "axes.labelcolor": "#202020", "figure.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False,
})
FILES = {
    "q1": ROOT / SRC["q1"] / "quality_domain.csv",
    "q2": ROOT / SRC["q2"] / "quality_model_comparison.csv",
    "q3": ROOT / SRC["q3"] / "optimal_allocations.csv",
    "q4": ROOT / SRC["q4"] / "frontier_forecast.csv",
}
data = {k: pd.read_csv(v) for k, v in FILES.items()}
stats = {"purpose": "真实冻结结果的配色讨论样张，不是正式入稿图", "sources": {
    k: {"file": str(v.relative_to(ROOT)), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()}
    for k, v in FILES.items()
}}
fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.4))
fig.subplots_adjust(left=.105, right=.975, bottom=.095, top=.9, wspace=.37, hspace=.64)

# Quality anchor rows have empirical intervals; mapped domains are kept out.
a = data["q1"].query("mapping_type == 'quality_anchor'").sort_values("q")
ax = axes[0, 0]
y = np.arange(len(a))
ax.errorbar(a.q, y, xerr=np.array([a.q-a.ci_low, a.ci_high-a.q]),
            fmt="o", color=LINE["forest_green"], mfc=COL["leaf_green"],
            ms=6, capsize=2.5, lw=1.2, label="95%条件区间")
ax.set(yticks=y, yticklabels=a.domain, xlabel="领域质量 Q", xlim=(.40,.65))
ax.set_ylim(-.6,len(a)-.4)
ax.legend(loc="lower right",bbox_to_anchor=(1,1.025))
ax.text(0,1.15,"a  问题一",transform=ax.transAxes,fontsize=11)
stats["q1"] = {"selection": "quality_anchor", "rows": a.to_dict("records")}

# Selection score is equal-weight over split families, NOT all folds.
b = data["q2"].groupby(["model", "split_type"]).RMSE.mean().unstack()
b = b.assign(score=b.mean(axis=1)).sort_values("score")
ax = axes[0, 1]
splits = [c for c in b if c != "score"]
marks = ["o", "s", "^", "D"]
split_color={"leave_D":"forest_green", "leave_N":"blue_gray", "leave_Q":"leaf_green", "leave_cell":"indigo", "leave_ND":"indigo"}
short = {"leave_N":"留 N", "leave_D":"留 D", "leave_Q":"留 Q", "leave_ND":"留 (N,D)", "leave_cell":"留 (N,D)"}
for j,s in enumerate(splits):
    key=split_color[s]
    ax.scatter(b[s],np.arange(len(b))+(j-1.5)*.13,c=COL[key],
               edgecolors=LINE[key], marker=marks[j],s=34,
               label=short.get(s,s),zorder=3)
for i,(_,row) in enumerate(b.iterrows()):
    ax.plot([row[splits].min(),row[splits].max()],[i,i],color="#D9D9D9",lw=1,zorder=0)
ax.set(yticks=np.arange(len(b)),yticklabels=b.index,xlabel="分块验证 RMSE")
ax.invert_yaxis()
ax.legend(ncol=2,loc="lower right",bbox_to_anchor=(1,1.025),columnspacing=1.1,handletextpad=.3)
ax.text(0,1.15,"b  问题二",transform=ax.transAxes,fontsize=11)
stats["q2"] = {"aggregation": "先按model/split_type取折均值，再按四类split等权；点为各split均值", "records": b.reset_index().to_dict("records")}

c = data["q3"].query("context_length == 4096 and quality_cost_type == 'exponential' and support_mode == 'operational_extended' and solver_success == True").sort_values("budget_FLOPs")
ax = axes[1, 0]
x = np.arange(len(c)); bottom=np.zeros(len(c))
for field,key,label in [("share_train","indigo","训练"),("share_quality","leaf_green","质量"),("share_attention","ochre","注意力")]:
    vals=c[field].to_numpy()
    ax.bar(x,vals,bottom=bottom,color=COL[key],edgecolor="white",linewidth=.7,width=.56,label=label)
    for xx,v,bb in zip(x,vals,bottom):
        if v>.065:ax.text(xx,bb+v/2,f"{v:.0%}",ha="center",va="center",fontsize=8.5,color="white" if key=="indigo" else "#202020")
    bottom += vals
assert np.allclose(bottom,1,atol=1e-6)
ax.set(xticks=x,xticklabels=[rf"$10^{{{int(np.log10(v))}}}$" for v in c.budget_FLOPs],ylim=(0,1.02),ylabel="预算份额",xlabel="算力预算 / FLOPs")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.legend(ncol=3,loc="lower right",bbox_to_anchor=(1,1.025),columnspacing=1.0,handlelength=1.2)
ax.text(0,1.15,"c  问题三",transform=ax.transAxes,fontsize=11)
stats["q3"]={"conditions":{"context_length":4096,"quality_cost_type":"exponential","support_mode":"operational_extended"},"records":c[["budget_FLOPs","share_train","share_quality","share_attention","Q_A","support_status"]].to_dict("records")}

d=data["q4"].query("tau == 0.9 and horizon_months == 24").sort_values("scenario_factor",ascending=False)
ax=axes[1,1]
for i,(_,row) in enumerate(d.iterrows()):
    key=["indigo","blue_gray","forest_green"][i]
    ax.plot([row.p025,row.p975],[i,i],color=COL[key],lw=1.2)
    ax.plot([row.p10,row.p90],[i,i],color=COL[key],lw=6,solid_capstyle="butt")
    ax.scatter(row.forecast_score,i,color=LINE[key],marker=marks[i],s=40,zorder=5)
ax.set(yticks=np.arange(len(d)),yticklabels=["维持增速","增速减半","增速四分之一"],xlabel="综合能力 / 分",xlim=(30,100),ylim=(-.65,2.65))
ax.invert_yaxis()
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],color=LINE["slate_teal"],marker="o",lw=0,label="中心预测"),Line2D([0],[0],color=COL["slate_teal"],lw=6,label="80%"),Line2D([0],[0],color=COL["slate_teal"],lw=1.2,label="95%")],ncol=3,loc="lower right",bbox_to_anchor=(1,1.025),handlelength=1.4,columnspacing=.9)
ax.text(0,1.15,"d  问题四",transform=ax.transAxes,fontsize=11)
stats["q4"]={"forecast_origin":"2025-03-13","horizon_months":24,"target_month":"2027-03","tau":.9,"interval":"经验预测区间","rows":d.to_dict("records")}
for ax in axes.flat:
    ax.tick_params(direction="out",width=.7,length=3)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fig.savefig(OUT/"previews/柔绿森林_四问真实数据样张.png",dpi=300)
    missing=[str(w.message) for w in caught if "Glyph" in str(w.message)]
    if missing:raise RuntimeError(missing)
plt.close(fig)
(OUT/"previews/preview_stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2,allow_nan=False))
print("四问讨论样张已生成；无缺字告警。")
