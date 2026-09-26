from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_results_report(
    report_path: Path,
    cfg: dict[str, Any],
    quality: dict[str, Any],
    mapping: dict[str, Any],
    mixture: dict[str, Any],
    acceptance: dict[str, Any],
    figures: list[str],
) -> None:
    qd = quality["domain"].sort_values("q", ascending=False)
    metrics = mixture["metrics"]
    summary_rows = []
    for split in ["test_1m", "test_60m", "test_1B", "est_10b", "est_70b"]:
        sub = metrics[metrics["split"] == split]
        summary_rows.append(
            {
                "split": split,
                "Spearman中位数": sub["spearman"].median(),
                "Kendall中位数": sub["kendall"].median(),
                "R2中位数": sub["r2"].median(),
                "归一化Regret中位数": sub["normalized_regret"].median(),
            }
        )
    metric_summary = pd.DataFrame(summary_rows)
    optimal = mixture["optimal"].sort_values("p_star", ascending=False)
    opt_diag = mixture["summary"]["optimization_diagnostics"]
    optimization_heading = "## 近优可行配比" if opt_diag["near_optimal_only"] else "## 最优配比"
    output_root = cfg.get("output_root", "results")
    figure_root = cfg.get("figure_root", "figures")

    lines = [
        "# 问题一计算结果",
        "",
        "> 本报告由问题一程序从结果 CSV/JSON 自动生成，数值不在文档中手工重算。",
        "",
        "## 运行环境与配置",
        "",
        f"- 随机种子：`{cfg['seed']}`",
        f"- 质量 bootstrap：`{cfg['bootstrap_reps']}` 次",
        f"- 映射 bootstrap：`{cfg['mapping_bootstrap_reps']}` 次",
        f"- 交互 bootstrap：`{cfg['interaction_bootstrap_reps']}` 次",
        f"- 优化 bootstrap：`{cfg['optimization_bootstrap_reps']}` 次",
        "",
        "## 数据读取与预处理",
        "",
        f"去重质量参考宇宙共 **{quality['audit']['deduplicated_reference_rows']:,}** 条。",
        f"综合质量范围为 `{quality['summary']['quality_range'][0]:.6f}`–`{quality['summary']['quality_range'][1]:.6f}`，"
        f"显著冲突率为 `{quality['summary']['conflict_rate']:.4%}`。",
        "",
        "## 七个质量域",
        "",
        qd.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 17 域映射",
        "",
        f"映射温度 `tau={mapping['metrics']['tau']:.6f}`；已知域留一 MAE="
        f"`{mapping['metrics']['loo_mae']:.6f}`，Spearman=`{mapping['metrics']['loo_spearman']:.6f}`。",
        "",
        mapping["mapping"].sort_values("q", ascending=False).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 配比模型验证",
        "",
        metric_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"内部 CV 冻结的模型角色：`{mixture['summary']['model_role']}`。",
        "",
        optimization_heading,
        "",
        (
            "该表中的 `p_star` 是当前多起点搜索得到的最佳参考候选；由于支持域边界被激活或多起点结果未完全一致，"
            "应结合 bootstrap 区间解释为近优可行配比族，不宣称唯一全局最优。"
            if opt_diag["near_optimal_only"]
            else "多起点和随机可行点检查均支持当前最优候选。"
        ),
        "",
        optimal.to_markdown(index=False, floatfmt=".6f"),
        "",
        "约束回代：",
        "",
        f"- 配比和：`{mixture['summary']['p_star_sum']:.12f}`",
        f"- 最大上界违反量：`{mixture['summary']['p_star_max_upper_violation']:.3e}`",
        f"- 支持距离/阈值：`{mixture['summary']['p_star_support_distance']:.6f}` / `{mixture['summary']['support_radius']:.6f}`",
        f"- 最优 5 个可行起点目标相对离散度：`{opt_diag['multistart_relative_dispersion']:.6e}`",
        f"- 随机可行点检查：候选目标 `{opt_diag['p_star_objective']:.6f}`，"
        f"随机最佳 `{opt_diag['random_best_objective']:.6f}`（接受 `{opt_diag['random_accepted']}` 点）",
        f"- 支持域边界是否激活：`{opt_diag['support_boundary_active']}`",
        "",
        "## 质量与配比边际收益关联",
        "",
        f"Spearman=`{mixture['summary']['quality_effect_association']['spearman_q_vs_benefit']:.6f}`，"
        f"Kendall=`{mixture['summary']['quality_effect_association']['kendall_q_vs_benefit']:.6f}`。该结果只解释关联。",
        "",
        "## 验收",
        "",
        f"总体状态：**{acceptance['overall']}**；PASS={acceptance['summary']['PASS']}，"
        f"WARN={acceptance['summary']['WARN']}，FAIL={acceptance['summary']['FAIL']}。",
        "",
        f"完整规则见 `{output_root}/acceptance_report.csv` 和 `{output_root}/acceptance_report.json`。",
        "",
        "## 数据驱动图表",
        "",
        *[f"- `{figure_root}/{name}`" for name in figures],
        "",
        "## 可复现运行方式",
        "",
        "```bash",
        "python problem1.py all --config <config>",
        "```",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
