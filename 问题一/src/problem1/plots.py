from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import setup_matplotlib
from .data_quality import BLOCKS, QUALITY_FIELDS


def generate_figures(
    quality: dict[str, Any],
    mapping: dict[str, Any],
    mixture: dict[str, Any],
    figure_dir: Path,
) -> list[str]:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    setup_matplotlib()
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    scored = quality["scored"]
    sample = scored.sample(min(12000, len(scored)), random_state=2026)
    corr = sample[QUALITY_FIELDS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"label": "Spearman 相关"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    fig.savefig(figure_dir / "F1_quality_indicator_correlation.pdf")
    plt.close(fig)
    paths.append("F1_quality_indicator_correlation.pdf")

    combined = mapping["combined_quality"].copy()
    combined["display_domain"] = combined["domain"] + np.where(
        combined["mapping_type"] == "quality_anchor", " [质量域]", " [配比域]"
    )
    combined = combined.sort_values("q")
    y = np.arange(len(combined))
    fig, ax = plt.subplots(figsize=(8.5, max(6, 0.25 * len(combined))))
    ax.errorbar(
        combined["q"],
        y,
        xerr=np.vstack([combined["q"] - combined["ci_low"], combined["ci_high"] - combined["q"]]),
        fmt="o",
        markersize=3.5,
        capsize=2,
        color="#2F5597",
        ecolor="#7F8FA6",
    )
    ax.set_yticks(y, combined["display_domain"])
    ax.set_xlabel("综合质量分 Q")
    ax.set_ylabel("领域")
    ax.set_xlim(0, 1)
    fig.savefig(figure_dir / "F2_domain_quality_forest.pdf")
    plt.close(fig)
    paths.append("F2_domain_quality_forest.pdf")

    conflict = (
        scored.groupby(["domain", "conflict_type"]).size().rename("count").reset_index()
    )
    conflict = conflict[conflict["conflict_type"] != "none"]
    if not conflict.empty:
        total = scored.groupby("domain").size().rename("total")
        conflict = conflict.join(total, on="domain")
        conflict["rate"] = conflict["count"] / conflict["total"]
        pivot = conflict.pivot(index="domain", columns="conflict_type", values="rate").fillna(0)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        sns.heatmap(pivot, cmap="mako", ax=ax, cbar_kws={"label": "冲突率"})
        ax.set_xlabel("最大块 > 最小块")
        ax.set_ylabel("质量域")
        fig.savefig(figure_dir / "F3_quality_conflict_heatmap.pdf")
        plt.close(fig)
        paths.append("F3_quality_conflict_heatmap.pdf")

    coef = mixture["coefficients"]
    inter = coef[coef["term_type"] == "interaction"].copy()
    inter["stable_coef"] = np.where(inter["sign_probability"] >= 0.8, inter["coefficient"], 0.0)
    agg = inter.groupby(["domain_i", "domain_j"])["stable_coef"].mean()
    domains = mixture["audit"]["mixture_domains"]
    matrix = pd.DataFrame(0.0, index=domains, columns=domains)
    for (i, j), value in agg.items():
        matrix.loc[i, j] = matrix.loc[j, i] = value
    scale = np.nanquantile(np.abs(matrix.to_numpy()), 0.95)
    scale = max(float(scale), 1e-8)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-scale, vmax=scale, square=True, ax=ax, cbar_kws={"label": "稳定交互系数均值"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    fig.savefig(figure_dir / "F4_mixture_interactions.pdf")
    plt.close(fig)
    paths.append("F4_mixture_interactions.pdf")

    pred = mixture["predictions"]
    test = pred[pred["split"].isin(["test_1m", "test_60m", "test_1B"])].copy()
    test["display_prediction"] = np.where(
        test["predicted_calibrated"].notna(), test["predicted_calibrated"], test["predicted"]
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, split in zip(axes, ["test_1m", "test_60m", "test_1B"]):
        sub = test[test["split"] == split]
        ax.scatter(sub["observed"], sub["display_prediction"], s=6, alpha=0.25, color="#2F5597")
        lo = min(sub["observed"].min(), sub["display_prediction"].min())
        hi = max(sub["observed"].max(), sub["display_prediction"].max())
        ax.plot([lo, hi], [lo, hi], "--", color="#C00000", linewidth=1)
        ax.set_title(split)
        ax.set_xlabel("真实 Loss")
        ax.set_ylabel("预测 Loss")
    fig.tight_layout()
    fig.savefig(figure_dir / "F5_mixture_prediction.pdf")
    plt.close(fig)
    paths.append("F5_mixture_prediction.pdf")

    optimal = mixture["optimal"].sort_values("p_star", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(optimal["domain"], optimal["p_star"], color="#4C78A8")
    low = np.minimum(optimal["ci_low"].to_numpy(float), optimal["p_star"].to_numpy(float))
    high = np.maximum(optimal["ci_high"].to_numpy(float), optimal["p_star"].to_numpy(float))
    ax.errorbar(
        optimal["p_star"],
        np.arange(len(optimal)),
        xerr=np.vstack([optimal["p_star"].to_numpy(float) - low, high - optimal["p_star"].to_numpy(float)]),
        fmt="none",
        ecolor="#333333",
        capsize=2,
    )
    near_optimal = mixture["summary"].get("optimization_diagnostics", {}).get("near_optimal_only", False)
    ax.set_xlabel("参考近优配比" if near_optimal else "最优配比")
    ax.set_ylabel("训练领域")
    fig.savefig(figure_dir / "F6_optimal_mixture.pdf")
    plt.close(fig)
    paths.append("F6_optimal_mixture.pdf")

    return paths
