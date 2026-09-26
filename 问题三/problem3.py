from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from problem3.acceptance import build_acceptance
from problem3.common import (device_record, ensure_dirs, json_dump, load_config,
                             manifest_base, seed_everything, sha256_file)
from problem3.contracts import support_from_config
from problem3.cost import cost_spec_from_config
from problem3.data import audit_inputs, build_scenario_registry
from problem3.optimize import (baseline_q0_solution, kkt_diagnostics, solve_2d,
                               solve_3d)
from problem3.reporting import write_results_report
from problem3.transitions import (detect_transitions, solve_budget_path,
                                  summarize_transition_stability)
from problem3.uncertainty import (bootstrap_intervals, deterministic_path_draws,
                                  run_anchor_bootstrap, run_sensitivity,
                                  run_transition_bootstrap)


def solve_centers(cfg: dict, params, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, checks, domains = [], [], []
    p_ref = np.asarray(cfg["_p_ref"], dtype=float)
    domain_names = cfg["_domains"]
    for support_name in ["operational_extended", "strict_joint"]:
        support = support_from_config(cfg, support_name)
        for cost_name in cfg["quality_cost_types"]:
            cost = cost_spec_from_config(cfg, cost_name)
            for context in cfg["contexts"]:
                for budget in cfg["budgets"]:
                    row = solve_2d(float(budget), int(context), float(cfg["eta"]), 1.0,
                                   cost, params, support, eps=float(cfg["quality_eps"]),
                                   grid_n=int(cfg["grid_n_points"]), grid_q=int(cfg["grid_q_points"]),
                                   multistarts=int(cfg["local_multistarts"]), fast=False)
                    scenario_id = f"{support_name}|{cost_name}|{int(context)}|{float(budget):.0e}"
                    row["scenario_id"] = scenario_id
                    if row["solver_success"] and np.isfinite(row["predicted_loss"]):
                        verification = solve_3d(row, float(budget), int(context), float(cfg["eta"]),
                                                1.0, cost, params, support,
                                                eps=float(cfg["quality_eps"]))
                        repeat = solve_2d(float(budget), int(context), float(cfg["eta"]), 1.0,
                                          cost, params, support, eps=float(cfg["quality_eps"]),
                                          grid_n=int(cfg["grid_n_points"]), grid_q=int(cfg["grid_q_points"]),
                                          multistarts=int(cfg["local_multistarts"]), fast=False)
                        baseline = baseline_q0_solution(float(budget), int(context), float(cfg["eta"]),
                                                        cost, params, support)
                        row.update(kkt_diagnostics(row, params, cost, support, 1.0,
                                                   float(cfg["quality_eps"])))
                        row["verification_3d_loss"] = verification["predicted_loss"]
                        row["verification_2d3d_relative_loss_difference"] = abs(row["predicted_loss"] - verification["predicted_loss"]) / row["predicted_loss"]
                        row["fine_grid_relative_loss_difference"] = max(0.0, row["predicted_loss"] - row["fine_grid_best_loss"]) / row["predicted_loss"]
                        row["repeat_relative_loss_difference"] = abs(row["predicted_loss"] - repeat["predicted_loss"]) / row["predicted_loss"]
                        row["baseline_Q0_loss"] = baseline["predicted_loss"]
                        if support_name == "operational_extended":
                            for name, share in zip(domain_names, p_ref):
                                domains.append({"scenario_id": scenario_id, "domain": name,
                                                "p_ref": float(share), "D_abs": row["D_abs"],
                                                "D_i_abs": float(share * row["D_abs"]),
                                                "D_i_B": float(share * row["D_B"])})
                    else:
                        for key in ["verification_3d_loss", "verification_2d3d_relative_loss_difference",
                                    "fine_grid_relative_loss_difference", "repeat_relative_loss_difference",
                                    "baseline_Q0_loss", "kkt_relative_dispersion"]:
                            row[key] = np.nan
                        row["active_set"] = "infeasible"
                    rows.append(row)
                    checks.append({"scenario_id": scenario_id, "support_mode": support_name,
                                   "solver_success": row["solver_success"],
                                   "support_status": row["support_status"],
                                   "max_normalized_violation": row["max_normalized_violation"],
                                   "budget_relative_slack": row.get("budget_relative_slack"),
                                   "cost_nonnegative": all((not np.isfinite(row.get(k, np.nan))) or row[k] >= 0 for k in ["cost_train", "cost_quality", "cost_attention"]),
                                   "two_three_relative_loss_difference": row["verification_2d3d_relative_loss_difference"],
                                   "fine_grid_relative_loss_difference": row["fine_grid_relative_loss_difference"]})
    center = pd.DataFrame(rows)
    constraint = pd.DataFrame(checks)
    domain_tokens = pd.DataFrame(domains)
    center.to_csv(output_dir / "optimal_allocations.csv", index=False)
    constraint.to_csv(output_dir / "constraint_checks.csv", index=False)
    domain_tokens.to_csv(output_dir / "domain_token_allocations.csv", index=False)
    return center, constraint, domain_tokens


def solve_paths(cfg: dict, params, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    support = support_from_config(cfg, "operational_extended")
    path_frames, transition_frames, path_map = [], [], {}
    for cost_name in cfg["quality_cost_types"]:
        cost = cost_spec_from_config(cfg, cost_name)
        for context in cfg["contexts"]:
            path = solve_budget_path(cost, int(context), params, support, cfg, fast=True)
            path_frames.append(path)
            path_map[(cost_name, int(context))] = path
            found = detect_transitions(path, cost, int(context), params, support, cfg, refine=True)
            if len(found):
                transition_frames.append(found)
    paths = pd.concat(path_frames, ignore_index=True)
    transitions = pd.concat(transition_frames, ignore_index=True) if transition_frames else pd.DataFrame(columns=[
        "transition_class", "event_type", "budget_lower", "budget_upper", "budget_mid",
        "log10_interval_width", "active_before", "active_after", "bic_improvement",
        "slope_before", "slope_after", "quality_cost_type", "context_length",
        "support_mode", "stable", "stability_probability"])
    paths.to_csv(output_dir / "budget_paths.csv.gz", index=False, compression="gzip")
    kkt_cols = [c for c in paths.columns if c.startswith("marginal_") or c.startswith("dL_") or c in [
        "path_id", "budget_FLOPs", "quality_cost_type", "context_length", "N_B", "D_B", "Q_A",
        "active_set", "kkt_relative_dispersion", "support_status"]]
    paths[kkt_cols].to_csv(output_dir / "kkt_diagnostics.csv", index=False)
    return paths, transitions, path_map


def build_bridge(center: pd.DataFrame, intervals: pd.DataFrame, transitions: pd.DataFrame,
                 p_ref: np.ndarray, domains: list[str], audit: dict, acceptance: dict,
                 output_dir: Path) -> dict:
    op = center[center["support_mode"].eq("operational_extended")].copy()
    merged = op.merge(intervals, on=["quality_cost_type", "context_length", "budget_FLOPs"], how="left")
    ready = bool(acceptance["interface_state"]["PROBLEM3_READY"])
    entries = []
    for row in merged.to_dict(orient="records"):
        related = transitions[(transitions["quality_cost_type"] == row["quality_cost_type"]) &
                              (transitions["context_length"] == row["context_length"])]
        regime = "no_stable_transition_in_range" if related.empty or not related["stable"].any() else "stable_transition_present"
        entries.append({
            "budget_FLOPs": row["budget_FLOPs"], "context_length": row["context_length"],
            "quality_cost_type": row["quality_cost_type"], "N_abs": row["N_abs"],
            "D_abs": row["D_abs"], "Q_A": row["Q_A"], "Q_eff": row["Q_eff"],
            "p_ref": {d: float(v) for d, v in zip(domains, p_ref)},
            "domain_D": {d: float(v * row["D_abs"]) for d, v in zip(domains, p_ref)},
            "predicted_loss": row["predicted_loss"],
            "predicted_loss_interval": [row.get("predicted_loss_q025"), row.get("predicted_loss_q975")],
            "costs": {k: row[k] for k in ["cost_train", "cost_quality", "cost_attention", "cost_total"]},
            "cost_shares": {k: row[k] for k in ["share_train", "share_quality", "share_attention"]},
            "support_status": row["support_status"], "extrapolation_flag": row["extrapolation_flag"],
            "evidence_label": "model_extrapolation" if row["extrapolation_flag"] else "within_joint_observed_support",
            "transition_regime": regime, "PROBLEM3_READY": ready,
        })
    bridge = {"interface": "p3_to_p4_v1", "PROBLEM3_READY": ready,
              "interface_state": acceptance["interface_state"],
              "upstream_problem2_manifest_sha256": audit["input_hashes"]["problem2_manifest.json"],
              "fixed_p_contract": {"p_source": "p_ref", "lambda0": 0.0, "joint_p_optimized": False},
              "entries": entries}
    json_dump(output_dir / "problem3_bridge.json", bridge)
    return bridge


def run(config_path: Path, stage: str) -> None:
    started = time.time()
    cfg, project_root = load_config(config_path)
    seed_everything(int(cfg["seed"]))
    output_dir, report_dir = ensure_dirs(project_root, cfg)
    timings = {}
    t = time.time(); data = audit_inputs(cfg, output_dir); timings["audit"] = time.time() - t
    if data["audit"]["hard_gate_status"] != "PASS":
        raise RuntimeError("Problem 3 input hard gate failed; inspect problem3_data_audit.json")
    build_scenario_registry(cfg, data["audit"]["c7"]["verified_contexts"], output_dir)
    if stage == "audit":
        print(json.dumps({"stage": stage, "status": "PASS", "audit": str(output_dir / 'problem3_data_audit.json')}, ensure_ascii=False))
        return
    cfg["contexts"] = data["audit"]["c7"]["verified_contexts"]
    cfg["_p_ref"] = data["p_ref"].tolist(); cfg["_domains"] = data["domains"]
    t = time.time(); center, constraints, domain_tokens = solve_centers(cfg, data["central_params"], output_dir); timings["center"] = time.time() - t
    t = time.time(); paths, transitions, path_map = solve_paths(cfg, data["central_params"], output_dir); timings["paths"] = time.time() - t
    if stage == "central":
        transitions.to_csv(output_dir / "transition_points.csv", index=False)
        print(json.dumps({"stage": stage, "center_rows": len(center), "path_rows": len(paths)}, ensure_ascii=False))
        return
    t = time.time()
    bootstrap = run_anchor_bootstrap(data["bootstrap"].head(int(cfg["bootstrap_anchor_draws"])), center,
                                     cfg, support_from_config(cfg, "operational_extended"))
    bootstrap.to_csv(output_dir / "bootstrap_optimal_allocations.csv.gz", index=False, compression="gzip")
    intervals = bootstrap_intervals(bootstrap)
    intervals.to_csv(output_dir / "bootstrap_intervals.csv", index=False)
    selected = deterministic_path_draws(data["bootstrap"], int(cfg["bootstrap_path_draws"]))
    selected[["draw", "selection_rank"]].to_csv(output_dir / "transition_bootstrap_draws.csv", index=False)
    transition_bootstrap = run_transition_bootstrap(selected, path_map, cfg,
                                                    support_from_config(cfg, "operational_extended"))
    transition_bootstrap.to_csv(output_dir / "transition_bootstrap.csv.gz", index=False, compression="gzip")
    valid_draws = int(transition_bootstrap[transition_bootstrap["record_type"].eq("draw_summary") & transition_bootstrap["draw_valid"]]["draw_id"].nunique())
    transitions = summarize_transition_stability(transitions, transition_bootstrap, valid_draws,
                                                 float(cfg["acceptance"]["transition_stability_probability"]))
    transitions.to_csv(output_dir / "transition_points.csv", index=False)
    sensitivity = run_sensitivity(center, data["central_params"], cfg,
                                  support_from_config(cfg, "operational_extended"),
                                  support_from_config(cfg, "strict_joint"))
    sensitivity.to_csv(output_dir / "sensitivity_summary.csv", index=False)
    timings["uncertainty_and_sensitivity"] = time.time() - t
    t = time.time()
    acceptance = build_acceptance(data["audit"], center, paths, transitions, bootstrap,
                                  transition_bootstrap, intervals, domain_tokens, sensitivity,
                                  data["central_params"], data["p_ref"], cfg, output_dir)
    build_bridge(center, intervals, transitions, data["p_ref"], data["domains"],
                 data["audit"], acceptance, output_dir)
    report_path = write_results_report(report_dir, center, intervals, transitions, sensitivity,
                                       acceptance, data["audit"], cfg)
    timings["acceptance_and_report"] = time.time() - t
    run_log = {"status": acceptance["overall"], "timings_seconds": timings,
               "elapsed_seconds": time.time() - started, "device": device_record(cfg),
               "warnings": [r["id"] for r in acceptance["rules"] if r["status"] == "WARN"],
               "exceptions": [], "recovery_actions": [], "plots_generated": False}
    json_dump(output_dir / "run_log.json", run_log)
    output_files = sorted([p for p in output_dir.iterdir() if p.is_file() and p.name != "problem3_manifest.json"] + [report_path])
    manifest = manifest_base(cfg, time.time() - started)
    manifest["inputs"] = data["audit"]["input_hashes"]
    manifest["outputs"] = {str(p.relative_to(project_root)): sha256_file(p) for p in output_files}
    manifest["hash_verification"] = "PASS"
    json_dump(output_dir / "problem3_manifest.json", manifest)
    written = json.loads((output_dir / "problem3_manifest.json").read_text(encoding="utf-8"))
    assert all(sha256_file(project_root / rel) == digest for rel, digest in written["outputs"].items())
    print(json.dumps({"stage": stage, "status": acceptance["overall"],
                      "interface_state": acceptance["interface_state"],
                      "results": str(output_dir), "report": str(report_path),
                      "elapsed_seconds": time.time() - started}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Huawei Cup Problem 3 reproducible model")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "problem3_v1.json")
    parser.add_argument("--stage", choices=["audit", "central", "all"], default="all")
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
