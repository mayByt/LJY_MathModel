from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import json_dump


def _rule(rule_id: str, name: str, status: str, observed: Any, threshold: str, action: str = "") -> dict[str, Any]:
    return {
        "id": rule_id,
        "name": name,
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "action": action,
    }


def build_acceptance_report(
    cfg: dict[str, Any],
    quality: dict[str, Any],
    mapping: dict[str, Any],
    mixture: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    rules = []
    qa = quality["audit"]
    expected_rows = {"A1": 51230, "A2": 17523, "A3": 203752}
    rows_ok = qa["rows"] == expected_rows
    rules.append(_rule("D01", "质量文件行数", "PASS" if rows_ok else "FAIL", qa["rows"], str(expected_rows)))
    fields_ok = qa["field_counts"] == {"A1": 27, "A2": 24, "A3": 24}
    rules.append(_rule("D02", "字段数", "PASS" if fields_ok else "FAIL", qa["field_counts"], "27/24/24"))
    overlap = qa["overlap_found"]
    overlap_ok = overlap.get("arxiv") == 1419 and overlap.get("github") == 10000
    rules.append(_rule("D04", "A1 与扩展集重叠", "PASS" if overlap_ok else "FAIL", overlap, "1419/10000"))
    mismatch = qa["overlap_quality_mismatch_count"]
    rules.append(_rule("D05", "重复 ID 质量字段冲突", "PASS" if mismatch == 0 else "FAIL", mismatch, "0"))
    known_nf = (
        qa["nonfinite"]["A1"]["records"] == 18
        and qa["nonfinite"]["A1"]["elements"] == 108
        and qa["nonfinite"]["A2"]["elements"] == 0
        and qa["nonfinite"]["A3"]["records"] == 1
        and qa["nonfinite"]["A3"]["elements"] == 6
    )
    rules.append(_rule("D06", "非有限值基准", "PASS" if known_nf else "FAIL", qa["nonfinite"], "A1=18/108,A2=0,A3=1/6"))
    qmin, qmax = quality["summary"]["quality_range"]
    rules.append(_rule("D07", "质量范围", "PASS" if 0 <= qmin <= qmax <= 1 else "FAIL", [qmin, qmax], "[0,1]"))
    weight_sums = quality["weights"].groupby("block")["conditional_weight"].sum()
    weight_ok = bool(np.allclose(weight_sums, 1.0, atol=1e-12))
    rules.append(_rule("D08", "块内权重和", "PASS" if weight_ok else "FAIL", weight_sums.to_dict(), "误差<=1e-12"))
    mix_audit = mixture["audit"]["splits"]
    simplex_error = max(v["closed_sum_max_deviation"] for v in mix_audit.values())
    rules.append(_rule("D09", "单纯形闭合", "PASS" if simplex_error <= 1e-12 else "FAIL", simplex_error, "<=1e-12"))
    conv = quality["summary"]["huber_convergence_rate"]
    conv_status = "PASS" if conv >= 0.9999 else ("WARN" if conv >= 0.999 else "FAIL")
    rules.append(_rule("Q06", "Huber 收敛率", conv_status, conv, ">=0.9999"))

    map_metrics = mapping["metrics"]
    mae_status = "PASS" if map_metrics["loo_mae"] <= 0.10 else "WARN"
    rules.append(_rule("MAP01", "已知域留一 MAE", mae_status, map_metrics["loo_mae"], "<=0.10", "WARN 时 inferred 仅作探索性"))
    rho = map_metrics["loo_spearman"]
    rho_status = "PASS" if np.isfinite(rho) and rho >= 0.70 else "WARN"
    rules.append(_rule("MAP02", "已知域留一 Spearman", rho_status, rho, ">=0.70", "WARN 时不对 inferred 域作强排序"))
    wsum = mapping["weights"].groupby("mixture_domain")["weight"].sum()
    rules.append(_rule("MAP03", "映射权重合法", "PASS" if np.allclose(wsum, 1, atol=1e-12) else "FAIL", wsum.to_dict(), "和误差<=1e-12"))

    metrics = mixture["metrics"]
    def median(split: str, field: str) -> float:
        return float(metrics.loc[metrics["split"] == split, field].median())

    m01_obs = {"spearman": median("test_1m", "spearman"), "kendall": median("test_1m", "kendall")}
    m01_ok = m01_obs["spearman"] >= 0.80 and m01_obs["kendall"] >= 0.60
    rules.append(_rule("M01", "1M 留出排序", "PASS" if m01_ok else "WARN", m01_obs, "Spearman>=0.80,Kendall>=0.60"))
    m02 = median("test_1m", "r2")
    rules.append(_rule("M02", "1M 留出 R2", "PASS" if m02 >= 0.50 else "WARN", m02, ">=0.50"))
    m03 = median("test_60m", "spearman")
    rules.append(_rule("M03", "60M 跨规模排序", "PASS" if m03 >= 0.80 else "WARN", m03, ">=0.80"))
    m04 = median("test_1B", "spearman")
    rules.append(_rule("M04", "1B 独立配比排序", "PASS" if m04 >= 0.75 else "WARN", m04, ">=0.75"))
    m05 = float(metrics[metrics["split"].isin(["test_1m", "test_60m", "test_1B"])]["normalized_regret"].median())
    rules.append(_rule("M05", "归一化 Regret", "PASS" if m05 <= 0.25 else "WARN", m05, "<=0.25"))
    gap = mixture["summary"]["ridge_vs_best_blackbox_relative_nrmse_gap"]
    rules.append(_rule("M07", "Ridge 与黑箱内部 CV 差距", "PASS" if gap <= 0.05 else "WARN", gap, "<=5%"))

    ms = mixture["summary"]
    opt_diag = ms["optimization_diagnostics"]
    rules.append(
        _rule(
            "O01",
            "多起点一致性",
            "PASS" if opt_diag["multistart_pass"] else "WARN",
            opt_diag["multistart_relative_dispersion"],
            "最优5个可行目标相对离散度<=1e-6",
            "WARN 时按近优可行集报告，不宣称唯一全局最优",
        )
    )
    rules.append(
        _rule(
            "O02",
            "随机可行点反证检查",
            "PASS" if opt_diag["random_check_pass"] else "FAIL",
            {
                "p_star": opt_diag["p_star_objective"],
                "random_best": opt_diag["random_best_objective"],
                "accepted": opt_diag["random_accepted"],
            },
            "候选目标不劣于10000个随机可行点",
        )
    )
    constraints_ok = (
        abs(ms["p_star_sum"] - 1.0) <= 1e-10
        and ms["p_star_min"] >= -1e-12
        and ms["p_star_max_upper_violation"] <= 1e-10
        and ms["p_star_support_distance"] <= ms["support_radius"] + 1e-8
    )
    rules.append(_rule("O03", "最优配比约束回代", "PASS" if constraints_ok else "FAIL", ms, "全部约束满足"))
    opt_rate = ms["optimization_success_rate"]
    rules.append(_rule("O04", "优化 bootstrap 成功率", "PASS" if opt_rate >= 0.80 else "WARN", opt_rate, ">=0.80"))

    q01 = quality["summary"]["sensitivity_min_domain_kendall"]
    q02 = quality["summary"]["sensitivity_min_median_extreme_overlap"]
    rules.append(_rule("Q01", "域排名敏感性", "PASS" if q01 >= 0.80 else "WARN", q01, "Kendall>=0.80"))
    rules.append(_rule("Q02", "样本极端组重合率", "PASS" if q02 >= 0.70 else "WARN", q02, ">=0.70"))

    counts = pd.Series([r["status"] for r in rules]).value_counts().to_dict()
    report = {
        "summary": {"PASS": counts.get("PASS", 0), "WARN": counts.get("WARN", 0), "FAIL": counts.get("FAIL", 0)},
        "rules": rules,
        "overall": "FAIL" if counts.get("FAIL", 0) else ("WARN" if counts.get("WARN", 0) else "PASS"),
    }
    json_dump(output_dir / "acceptance_report.json", report)
    pd.DataFrame(rules).to_csv(output_dir / "acceptance_report.csv", index=False)
    return report
