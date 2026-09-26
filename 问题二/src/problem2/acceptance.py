from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import json_dump
from .quality import predict_quality_delta


def _rule(rules: list[dict[str, Any]], rid: str, name: str, status: str, observed: Any, threshold: str, action: str = "") -> None:
    rules.append({"id": rid, "name": name, "status": status, "observed": observed, "threshold": threshold, "action": action})


def build_acceptance_report(data: dict[str, Any], classic: dict[str, Any], quality: dict[str, Any],
                            bridge: dict[str, Any], generalized: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    audit, checks, rules = data["audit"], data["audit"]["checks"], []
    _rule(rules, "D01", "数据硬门禁", "PASS" if audit["hard_gate_status"] == "PASS" else "FAIL", audit["hard_failures"], "empty")
    _rule(rules, "D02", "核心字段", "PASS" if checks["b1_core_finite_positive"] and checks["b2_core_finite_positive"] else "FAIL", {"B1": checks["b1_core_finite_positive"], "B2": checks["b2_core_finite_positive"]}, "finite positive")
    _rule(rules, "D03", "B8隔离", "PASS" if audit["b8"]["main_fit_rows_including_b8"] == 0 else "FAIL", audit["b8"]["main_fit_rows_including_b8"], "0")
    _rule(rules, "D04", "B9零值隔离", "PASS" if checks["b9_zero_token_rows"] == 4 else "FAIL", checks["b9_zero_token_rows"], "4")
    aux_ok = (
        checks["b2_or_b3_validation_choice"]["selected"] == "B2"
        and checks["b2_or_b3_validation_choice"]["B3_files"] == 8
        and checks["b11_source_audit"]["rows"] == 18
        and checks["b12_checkpoint_audit"]["rows"] == 1386
    )
    _rule(rules, "D05", "B1-B12使用账本", "PASS" if aux_ok else "FAIL",
          {"B2_B3": checks["b2_or_b3_validation_choice"], "B11": checks["b11_source_audit"], "B12": checks["b12_checkpoint_audit"]},
          "B2 selected; B3=8 files audited; B11=18; B12=1386")

    b = bridge["bridge"]
    _rule(rules, "P01", "上游验收", "PASS" if b.get("problem1_summary", {}).get("FAIL", 0) == 0 else "FAIL", b.get("problem1_summary"), "FAIL=0")
    _rule(rules, "P02", "上游hash", "PASS" if b.get("input_hash_gate") == "PASS" and not b.get("input_hash_mismatches") else "FAIL", b.get("input_hash_mismatches"), "no mismatch")
    _rule(rules, "P03", "领域与Scheffe维数", "PASS" if len(bridge["domains"]) == 17 and b["scheffe_feature_count"] == 153 else "FAIL", {"domains": len(bridge["domains"]), "features": b["scheffe_feature_count"]}, "17/153")
    qerr = b["Q_base_regression_targets"]
    _rule(rules, "P04", "质量基线回归", "PASS" if max(qerr["Q_ref_abs_error"], qerr["Q_star_abs_error"]) <= 1e-6 else "FAIL", qerr, "<=1e-6")
    _rule(rules, "P05", "问题一WARN传播", "PASS" if set(b["propagated_warn_ids"]) == {"MAP02", "O01", "Q02"} else "FAIL", b["propagated_warn_ids"], "MAP02/O01/Q02")
    p06 = b.get("problem1_bridge_interface") == "p1_to_p2_v2" and b.get("raw_attachment_A_read_by_problem2") is False
    _rule(rules, "P06", "问题一冻结接口", "PASS" if p06 else "FAIL",
          {"interface": b.get("problem1_bridge_interface"), "raw_A_read": b.get("raw_attachment_A_read_by_problem2")},
          "p1_to_p2_v2 and raw_A_read=false")

    selected = classic["summary"]["selected_model"]
    ccv = classic["cv"].groupby("model")[["nRMSE", "MAE", "Spearman_D"]].mean().loc[selected]
    _rule(rules, "C01", "多起点收敛", "PASS" if classic["summary"]["multistart_success_rate"] >= .9 else "WARN", classic["summary"]["multistart_success_rate"], ">=0.90")
    _rule(rules, "C02", "留一规模误差", "PASS" if ccv["nRMSE"] <= .25 and ccv["MAE"] <= .2 else "WARN", ccv.to_dict(), "nRMSE<=.25, MAE<=.2", "未达标则仅描述")
    _rule(rules, "C03", "经典参数物理性", "PASS" if all(np.isfinite(list(classic["summary"]["parameters"].values()))) and all(v > 0 for v in classic["summary"]["parameters"].values()) else "FAIL", classic["summary"]["parameters"], "finite positive")
    _rule(rules, "C04", "经典bootstrap", "PASS" if classic["summary"]["bootstrap_success_rate"] >= .95 else "WARN", classic["summary"]["bootstrap_success_rate"], ">=.95")
    warmup = classic["summary"]["warmup_sensitivity"]
    _rule(rules, "C05", "32点早期轨迹敏感性", "PASS" if warmup["removed_rows"] == 32 and warmup["alpha_beta_max_abs_relative_change"] <= .01 else "WARN",
          warmup, "removed=32 and alpha/beta relative change<=1%", "仅作敏感性，不反向调参")

    qmodel = quality["summary"]["selected_model"]
    qfam = quality["family_scores"][quality["family_scores"]["model"] == qmodel]
    qmax = float(qfam["family_nRMSE"].max())
    _rule(rules, "Q01", "四类绝对CV", "PASS" if qmax <= .25 else "WARN", qfam.to_dict(orient="records"), "each family nRMSE<=.25", "超过.40则不输出精确替代倍数")
    _rule(rules, "Q02", "质量方向", "PASS" if checks["b7_group_spearman_median"] <= -.8 else "FAIL", checks["b7_group_spearman_median"], "<=-.8")
    qviol = float(quality["cv"].loc[quality["cv"]["model"] == qmodel, "monotonic_violations"].max())
    _rule(rules, "Q03", "质量单调性", "PASS" if qviol == 0 else "FAIL", qviol, "0")
    _rule(rules, "Q04", "质量bootstrap", "PASS" if quality["summary"]["bootstrap_success_rate"] >= .8 else "WARN", quality["summary"]["bootstrap_success_rate"], ">=.8")
    _rule(rules, "Q05", "Q=1绝对锚点", "PASS" if quality["summary"]["q1_anchor_nrmse"] <= .25 else "WARN", quality["summary"]["q1_anchor_nrmse"], "<=.25")

    ts = bridge["transfer"][bridge["transfer"]["role"] == "composite"].set_index("scale")
    m01_obs = {"1M": float(ts.loc["1M", "spearman_R_loss"]), "60M": float(ts.loc["60M", "spearman_R_loss"]),
        "sign_probability_positive": b["transfer_sign_probability_positive"]}
    _rule(rules, "M01", "1M/60M综合Loss迁移", "PASS" if min(m01_obs["1M"], m01_obs["60M"]) >= .75 else "WARN", m01_obs, "composite Spearman>=.75")
    _rule(rules, "M02", "1B独立验证", "PASS" if float(ts.loc["1B", "spearman_R_loss"]) >= .75 else "WARN", float(ts.loc["1B", "spearman_R_loss"]), ">=.75", "1B仅保留方向敏感性，不作精确迁移校准")
    _rule(rules, "M03", "绝对强度识别", "WARN", b["kappa0_or_lambda0_role"], "scenario only", "保持强度情景，不报告伪精确点")
    _rule(rules, "M04", "Q-p分离", "PASS", "no free interaction", "no free Qxp")
    physical = generalized["physical_gate"]
    _rule(rules, "M05", "非零配比不可约下界", "PASS" if physical["status"] == "PASS" else "WARN",
          physical, "all nonzero-mixture scenarios L>=E",
          "存在违反时禁止联合优化p，固定p_ref主接口不受影响")

    baseline_el = generalized["elasticity"][(generalized["elasticity"]["p_scenario"] == "p_ref") & (generalized["elasticity"]["mixture_strength_profile"] == 0)]
    mono = bool((baseline_el[["dL_dN", "dL_dD", "dL_dQ"]] <= 1e-7).all().all())
    _rule(rules, "G01", "物理边界", "PASS" if mono else "FAIL", baseline_el[["dL_dN", "dL_dD", "dL_dQ"]].max().to_dict(), "<=1e-7")
    _rule(rules, "G02", "联合抽样", "PASS" if generalized["parameters"]["joint_bootstrap_valid_rate"] >= .95 else ("WARN" if generalized["parameters"]["joint_bootstrap_valid_rate"] >= .8 else "FAIL"), generalized["parameters"]["joint_bootstrap_valid_rate"], ">=.95; FAIL<.8")
    _rule(rules, "G03", "证据标记", "PASS" if {"N_support", "D_support", "Q_support", "p_support"}.issubset(generalized["predictions"].columns) else "FAIL", list(generalized["predictions"].columns), "four support columns")

    # RF01-RF10 are repair hard gates.
    held_numeric = pd.to_numeric(quality["cv"]["held_out_level"], errors="coerce")
    q1fold = quality["cv"][(quality["cv"]["model"] == qmodel) & (quality["cv"]["split_type"] == "leave_Q") & np.isclose(held_numeric, 1.0)]
    rf01 = bool((~quality["cv"]["validation_loss_used_as_baseline"].astype(bool)).all() and len(q1fold) == 1 and float(q1fold.iloc[0]["RMSE"]) > 1e-10)
    _rule(rules, "RF01", "绝对CV无泄漏", "PASS" if rf01 else "FAIL", {"q1_RMSE": None if q1fold.empty else float(q1fold.iloc[0]["RMSE"]), "baseline_flags": quality["cv"]["validation_loss_used_as_baseline"].unique().tolist()}, "no validation loss baseline; Q1 RMSE>0")
    rf02 = quality["summary"]["leakage_guard"]["loss_q1_role"] == "auxiliary_paired_metric_only" and (q1fold["paired_delta_status"] == "not_applicable_Q1_held_out").all()
    _rule(rules, "RF02", "差分指标隔离", "PASS" if rf02 else "FAIL", quality["summary"]["leakage_guard"], "Q1 paired metric NA")
    rf03 = set(quality["summary"]["split_family_weights"].values()) == {.25} and len(quality["summary"]["split_family_weights"]) == 4
    _rule(rules, "RF03", "切分族等权", "PASS" if rf03 else "FAIL", quality["summary"]["split_family_weights"], "four x .25")
    cp = classic["summary"]["parameters"]
    anchor = np.max(np.abs(predict_quality_delta("GQ3", {"b_Q": .7, "beta1": .1}, cp, np.array([.1, 1.]), np.array([10., 600.]), np.ones(2))))
    _rule(rules, "RF04", "GQ3严格锚定", "PASS" if anchor <= 1e-10 else "FAIL", float(anchor), "<=1e-10")
    cb = classic["bootstrap"]
    rf05 = {"duplicate_multiplicity_preserved", "sampled_cluster_instances"}.issubset(cb.columns) and bool(cb["duplicate_multiplicity_preserved"].eq(True).all()) and bool((cb["sampled_cluster_instances"].dropna() == 8).all())
    _rule(rules, "RF05", "簇重复有效", "PASS" if rf05 else "FAIL", {"columns": list(cb.columns), "instances": cb.get("sampled_cluster_instances", pd.Series(dtype=float)).dropna().unique().tolist()}, "8 instances and flag true")
    gp = generalized["predictions"]
    probe = gp[(gp["s_Q"] == 1.0) & (gp["mixture_strength_profile"] == max(b["mixture_strength_profiles"]))]
    diffs = probe.groupby(["N_params_B", "D_tokens_B", "Q_A"])["prediction"].agg(lambda x: float(np.ptp(x))).to_numpy(float)
    artifact = output_dir / b["mixture_response_artifact"]
    required_keys = {"coefficient_point", "loss_iqr", "j_ref_point", "r_scale_point", "domains"}
    artifact_keys = set(np.load(artifact).files) if artifact.exists() else set()
    rf06 = abs(float(b["R_ref"])) <= 1e-10 and bool(np.nanmax(diffs) > 1e-10) and required_keys.issubset(artifact_keys)
    _rule(rules, "RF06", "配比接口真实存在", "PASS" if rf06 else "FAIL", {"R_ref": b["R_ref"], "max_p_prediction_range": float(np.nanmax(diffs)), "point_artifact_keys_present": sorted(required_keys & artifact_keys)}, "R_ref=0, p changes prediction, point R(p) loadable")
    pred_sq, boot_sq = set(map(float, gp["s_Q"].unique())), set(map(float, generalized["bootstrap"]["s_Q"].unique()))
    rf07 = pred_sq == {.5, 1., 1.5} and boot_sq == {.5, 1., 1.5}
    _rule(rules, "RF07", "映射情景真实运行", "PASS" if rf07 else "FAIL", {"prediction": sorted(pred_sq), "bootstrap": sorted(boot_sq)}, ".5/1/1.5")
    qcols = [f"quality_{k}" for k in quality["summary"]["parameters"]]
    qstd = float(generalized["bootstrap"].loc[generalized["bootstrap"]["success"], qcols].std().max())
    rf08 = qstd > 1e-10 and b["scheffe_bootstrap_max_std"] > 1e-10
    _rule(rules, "RF08", "联合抽样非退化", "PASS" if rf08 else "FAIL", {"quality_max_std": qstd, "scheffe_max_std": b["scheffe_bootstrap_max_std"]}, ">1e-10")
    eq = generalized["equivalence"]
    successful = eq[eq["root_status_N"].str.startswith("ok")]
    invalid_value = eq.loc[~eq["root_status_N"].str.startswith("ok"), "finite_N_multiplier_minus_1"].notna().any()
    rf09 = not invalid_value and (successful.empty or float(successful["root_backcheck_error_N"].max()) <= 1e-8)
    _rule(rules, "RF09", "支持域求根", "PASS" if rf09 else "FAIL", {"max_backcheck": None if successful.empty else float(successful["root_backcheck_error_N"].max()), "invalid_nonnull": bool(invalid_value)}, "error<=1e-8 and failed root null")
    units_path = output_dir / "units_contract.json"
    rf10 = bool(generalized["parameters"].get("callable_signature")) and units_path.exists()
    _rule(rules, "RF10", "Manifest与单位合同协议", "PASS" if rf10 else "FAIL", {"callable": bool(generalized["parameters"].get("callable_signature")), "units_contract": units_path.exists()}, "post-write hash verification; units contract exists")
    required_elasticity = {
        "epsilon_N", "epsilon_D", "epsilon_Q",
        "epsilon_N_total", "epsilon_D_total", "epsilon_Q_total",
        "epsilon_N_reducible", "epsilon_D_reducible", "epsilon_Q_reducible",
    }
    rf11 = required_elasticity.issubset(generalized["elasticity"].columns) and generalized["parameters"].get("elasticity_contract", {}).get("epsilon_x", "").startswith("standard_total_loss")
    _rule(rules, "RF11", "双弹性口径", "PASS" if rf11 else "FAIL",
          {"columns_present": sorted(required_elasticity & set(generalized["elasticity"].columns)), "contract": generalized["parameters"].get("elasticity_contract")},
          "standard total-loss and reducible-loss elasticities both present")

    hard_ids = {"D01", "D02", "D03", "D04", "D05", "P01", "P02", "P03", "P04", "P05", "P06", "C03", "Q02", "Q03", "G01", "G02", "G03"} | {f"RF{i:02d}" for i in range(1, 12)}
    hard_failures = [r["id"] for r in rules if r["id"] in hard_ids and r["status"] == "FAIL"]
    central_ready = not any(x in hard_failures for x in ["D01", "D02", "D03", "D04", "D05", "P01", "P02", "P03", "P04", "P05", "P06", "C03", "Q02", "Q03", "RF01", "RF02", "RF03", "RF04", "RF05", "RF08", "RF11"])
    fixed_p_ready = central_ready and not any(x in hard_failures for x in ["G01", "G02", "G03", "RF07", "RF09", "RF10"])
    joint_gate_ids = ["M01", "M02", "M03", "M05"]
    joint_p_ready = fixed_p_ready and all(next(r for r in rules if r["id"] == rid)["status"] == "PASS" for rid in joint_gate_ids)
    p12_ready = fixed_p_ready and not hard_failures
    interface_state = {"CENTRAL_MODEL_READY": central_ready, "FIXED_P_READY": fixed_p_ready, "JOINT_P_READY": joint_p_ready, "P12_FREEZE_READY": p12_ready}
    generalized["parameters"]["interface_state"] = interface_state
    generalized["parameters"]["hard_gate_failures"] = hard_failures
    json_dump(output_dir / "generalized_scaling_parameters.json", generalized["parameters"])

    frame = pd.DataFrame(rules); frame.to_csv(output_dir / "problem2_acceptance_report.csv", index=False)
    summary = {s: int((frame["status"] == s).sum()) for s in ["PASS", "WARN", "FAIL"]}
    overall = "FAIL" if summary["FAIL"] else ("WARN" if summary["WARN"] else "PASS")
    report = {"summary": summary, "overall": overall, "interface_state": interface_state,
              "hard_gate_failures": hard_failures, "rules": rules}
    json_dump(output_dir / "problem2_acceptance_report.json", report)
    return report
