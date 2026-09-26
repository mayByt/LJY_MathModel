from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import json_dump
from .contracts import ScalingParameters
from .cost import cost_spec_from_config, quality_cost_increment, quality_cost_value, total_costs
from .objective import effective_quality, predict_loss, quality_upper_bound


def _rule(rules: list[dict[str, Any]], rid: str, name: str, status: str,
          observed: Any, threshold: str, action: str = "") -> None:
    rules.append({"id": rid, "name": name, "status": status,
                  "observed": observed, "threshold": threshold, "action": action})


def build_acceptance(audit: dict, center: pd.DataFrame, paths: pd.DataFrame,
                     transitions: pd.DataFrame, bootstrap: pd.DataFrame,
                     transition_bootstrap: pd.DataFrame, intervals: pd.DataFrame,
                     domain_tokens: pd.DataFrame, sensitivity: pd.DataFrame,
                     params: ScalingParameters, p_ref: np.ndarray, cfg: dict,
                     output_dir) -> dict:
    rules: list[dict[str, Any]] = []
    c7 = audit["c7"]
    state = audit["upstream_interface_state"]
    _rule(rules, "D01", "上游冻结状态", "PASS" if state.get("P12_FREEZE_READY") and state.get("FIXED_P_READY") and audit["upstream_acceptance"].get("FAIL", 1) == 0 else "FAIL", state, "P12/FIXED_P true and FAIL=0")
    _rule(rules, "D02", "上游hash", "PASS" if not audit["hash_mismatches"] else "FAIL", audit["hash_mismatches"], "empty")
    d03 = c7["shape"] == [45, 7] and c7["schema_valid"] and c7["duplicate_models"] == 0
    _rule(rules, "D03", "C7 schema", "PASS" if d03 else "FAIL", {k: c7[k] for k in ["shape", "schema_valid", "duplicate_models"]}, "45x7/fixed/no duplicates")
    _rule(rules, "D04", "C7上下文", "PASS" if c7["context_gate"] == "PASS" else "WARN", c7["verified_contexts"], str(cfg["contexts"]))
    _rule(rules, "D05", "C7错误隔离", "PASS" if audit["raw_A_or_B_read_by_problem3"] is False and c7["non_context_fields_used_by_problem3"] == [] else "FAIL", c7["non_context_fields_used_by_problem3"], "empty")
    p_error = abs(float(p_ref.sum()) - 1.0)
    _rule(rules, "D06", "固定配比单纯形", "PASS" if len(p_ref) == 17 and np.min(p_ref) >= 0 and p_error <= 1e-12 else "FAIL", {"n": len(p_ref), "min": float(np.min(p_ref)), "sum_error": p_error}, "17/nonnegative/error<=1e-12")

    probes = [(0.1, 10.0, params.Q0), (1.0, 100.0, .7), (10.0, 300.0, .9)]
    m01 = max(abs(float(predict_loss(n, d, q, params)) - (params.E + params.A*n**(-params.alpha) + params.B*d**(-params.beta) + params.c_Q*(1-q)**params.nu_Q)) for n,d,q in probes)
    _rule(rules, "M01", "Loss重建", "PASS" if m01 <= 1e-10 else "FAIL", m01, "<=1e-10")
    zero = max(abs(float(quality_cost_increment(params.Q0, params.Q0, cost_spec_from_config(cfg, name)))) for name in cfg["quality_cost_types"])
    _rule(rules, "M02", "质量零增量", "PASS" if zero <= 1e-6 else "FAIL", zero, "<=1e-6 FLOPs/token")
    qs = np.linspace(params.Q0, 1.0, 200)
    monotone = all(np.isfinite(quality_cost_value(qs, cost_spec_from_config(cfg, name))).all() and (np.diff(quality_cost_value(qs, cost_spec_from_config(cfg, name))) > 0).all() for name in cfg["quality_cost_types"])
    _rule(rules, "M03", "成本单调性", "PASS" if monotone else "FAIL", monotone, "finite and strictly increasing")
    sample = total_costs(2.0, 30.0, params.Q0, params.Q0, 4096, float(cfg["eta"]), cost_spec_from_config(cfg, "exponential"))
    expected = 6 * 2e9 * 30e9
    _rule(rules, "M04", "绝对单位", "PASS" if abs(sample["train"] / expected - 1) <= 1e-15 else "FAIL", sample["train"], str(expected))
    critical = 6 / float(cfg["eta"])
    ratios = {str(x): float(cfg["eta"]) * x / 6 for x in cfg["contexts"]}
    _rule(rules, "M05", "上下文临界", "PASS" if abs(critical - 30000) <= 1e-12 else "FAIL", {"critical": critical, "ratios": ratios}, "30000 and exact ratios")
    mapping_ok = all(abs(float(effective_quality(params.Q0, params.Q0, s)) - params.Q0) <= 1e-15 and quality_upper_bound(params.Q0, s, float(cfg["quality_eps"])) <= 1.0 for s in cfg["quality_mapping_slopes"])
    _rule(rules, "M06", "质量映射", "PASS" if mapping_ok else "FAIL", mapping_ok, "anchor preserved and <=1")

    op = center[center["support_mode"].eq("operational_extended")].copy()
    max_violation = float(op["max_normalized_violation"].max())
    _rule(rules, "O01", "约束可行", "PASS" if max_violation <= float(cfg["acceptance"]["max_normalized_violation"]) else "FAIL", max_violation, "<=1e-8")
    min_cost = float(op[["cost_train", "cost_quality", "cost_attention"]].min().min())
    _rule(rules, "O02", "成本非负", "PASS" if min_cost >= -1e-8 else "FAIL", min_cost, ">=0")
    slack = float(op["budget_relative_slack"].abs().max())
    _rule(rules, "O03", "预算使用", "PASS" if slack <= float(cfg["acceptance"]["budget_relative_slack"]) else "FAIL", slack, "<=1e-6")
    diff23 = float(op["verification_2d3d_relative_loss_difference"].max())
    _rule(rules, "O04", "二维三维一致", "PASS" if diff23 <= float(cfg["acceptance"]["two_three_loss_relative_difference"]) else "FAIL", diff23, "<=1e-6")
    fine = float(op["fine_grid_relative_loss_difference"].max())
    _rule(rules, "O05", "网格复核", "PASS" if fine <= float(cfg["acceptance"]["fine_grid_loss_relative_difference"]) else "FAIL", fine, "<=1e-4")
    repeat = float(op["repeat_relative_loss_difference"].max())
    _rule(rules, "O06", "可重复性", "PASS" if repeat <= 1e-10 else "FAIL", repeat, "<=1e-10")
    mono_budget = True
    for _, group in paths.groupby(["quality_cost_type", "context_length"]):
        mono_budget &= bool((np.diff(group.sort_values("budget_FLOPs")["predicted_loss"]) <= 1e-9).all())
    _rule(rules, "O07", "预算单调值函数", "PASS" if mono_budget else "FAIL", mono_budget, "non-increasing")
    context_ok = True
    for _, group in op.groupby(["quality_cost_type", "budget_FLOPs"]):
        context_ok &= bool((np.diff(group.sort_values("context_length")["predicted_loss"]) >= -1e-8).all())
    _rule(rules, "O08", "上下文有序性", "PASS" if context_ok else "FAIL", context_ok, "longer context not lower")
    baseline_gap = float((op["predicted_loss"] - op["baseline_Q0_loss"]).max())
    _rule(rules, "O09", "基线支配", "PASS" if baseline_gap <= 1e-9 else "FAIL", baseline_gap, "<=1e-9")
    token_err = float(domain_tokens.groupby("scenario_id").agg(total=("D_i_abs", "sum"), target=("D_abs", "first")).eval("abs(total-target)/target").max())
    _rule(rules, "O10", "领域Token", "PASS" if token_err <= 1e-12 else "FAIL", token_err, "<=1e-12")
    joint_ok = op["p_source"].eq("p_ref").all() and op["lambda0"].eq(0).all()
    _rule(rules, "O11", "联合配比门禁", "PASS" if joint_ok else "FAIL", {"p_sources": op["p_source"].unique().tolist(), "lambda": op["lambda0"].unique().tolist()}, "p_ref/lambda0=0")

    anchor_valid = bootstrap.groupby("draw_id")["solver_success"].all()
    u01 = float(anchor_valid.mean()) if len(anchor_valid) else 0.0
    _rule(rules, "U01", "三档bootstrap", "PASS" if u01 >= float(cfg["acceptance"]["bootstrap_valid_rate"]) else "FAIL", u01, ">=0.95")
    summaries = transition_bootstrap[transition_bootstrap["record_type"].eq("draw_summary")]
    u02 = float(summaries["draw_valid"].mean()) if len(summaries) else 0.0
    _rule(rules, "U02", "转移路径抽样", "PASS" if u02 >= float(cfg["acceptance"]["bootstrap_valid_rate"]) else "FAIL", u02, ">=0.95")
    qcols = [c for c in intervals if c.endswith("_q025")]
    ordered = all((intervals[lo] <= intervals[lo.replace("_q025", "_median")]).all() and (intervals[lo.replace("_q025", "_median")] <= intervals[lo.replace("_q025", "_q975")]).all() for lo in qcols)
    _rule(rules, "U03", "区间顺序", "PASS" if ordered else "FAIL", ordered, "lower<=median<=upper")

    primary = transitions[transitions["transition_class"].eq("primary")]
    t01 = primary.empty or bool((primary["log10_interval_width"] <= float(cfg["acceptance"]["transition_log10_width"]) + 1e-12).all())
    _rule(rules, "T01", "主要转移", "PASS" if t01 else "FAIL", {"count": len(primary), "max_width": None if primary.empty else float(primary["log10_interval_width"].max())}, "changed active set and width<=0.01")
    secondary = transitions[transitions["transition_class"].eq("secondary")]
    t02 = secondary.empty or bool(((secondary["bic_improvement"] >= 10) & ((secondary["slope_before"] - secondary["slope_after"]).abs() >= .15)).all())
    _rule(rules, "T02", "次要断点", "PASS" if t02 else "FAIL", {"count": len(secondary)}, "BIC>=10 and slope diff>=.15")
    t03 = transitions["stable"].eq(transitions["stability_probability"] >= float(cfg["acceptance"]["transition_stability_probability"])).all() if len(transitions) else True
    _rule(rules, "T03", "稳定标签", "PASS" if t03 else "FAIL", t03, "probability>=0.70 iff stable")
    t04 = not transitions["event_type"].astype(str).str.contains("attention", case=False).any() if len(transitions) else True
    _rule(rules, "T04", "场景预算区分", "PASS" if t04 else "FAIL", t04, "attention critical not a budget transition")

    frame = pd.DataFrame(rules)
    frame.to_csv(output_dir / "problem3_acceptance_report.csv", index=False)
    hard_failures = frame.loc[frame["status"].eq("FAIL"), "id"].tolist()
    central_ids = {f"D{i:02d}" for i in range(1, 7)} | {f"M{i:02d}" for i in range(1, 7)} | {f"O{i:02d}" for i in range(1, 12)} | {"T01", "T02", "T04"}
    uncertainty_ids = {"U01", "U02", "U03", "T03"}
    central_ready = not any(x in central_ids for x in hard_failures)
    uncertainty_ready = not any(x in uncertainty_ids for x in hard_failures)
    state_out = {"PROBLEM3_CENTRAL_READY": central_ready,
                 "PROBLEM3_UNCERTAINTY_READY": uncertainty_ready,
                 "PROBLEM3_READY": central_ready and uncertainty_ready and not hard_failures}
    summary = {status: int(frame["status"].eq(status).sum()) for status in ["PASS", "WARN", "FAIL"]}
    report = {"summary": summary, "overall": "FAIL" if hard_failures else ("WARN" if summary["WARN"] else "PASS"),
              "interface_state": state_out, "hard_gate_failures": hard_failures,
              "rules": rules, "sensitivity_rows": int(len(sensitivity))}
    json_dump(output_dir / "problem3_acceptance_report.json", report)
    return report
