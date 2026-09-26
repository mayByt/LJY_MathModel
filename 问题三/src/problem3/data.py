from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import json_dump, sha256_file
from .contracts import params_from_payload


CONSUMED_P2_FILES = [
    "generalized_scaling_parameters.json",
    "generalized_scaling_bootstrap.csv.gz",
    "problem1_bridge.json",
    "units_contract.json",
    "problem2_acceptance_report.json",
    "problem2_manifest.json",
    "quality_parameter_equivalence.csv",
    "elasticity_grid.csv",
]


EXPECTED_C7_COLUMNS = [
    "model_name", "n_layers", "n_heads", "d_model", "vocab_size",
    "max_position_embeddings", "training_data_TB",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_c7(c7_path: Path, cfg: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(c7_path)
    schema_ok = frame.columns.tolist() == EXPECTED_C7_COLUMNS
    duplicate = frame["model_name"].duplicated(keep=False) if "model_name" in frame else pd.Series(True, index=frame.index)
    context = pd.to_numeric(frame.get("max_position_embeddings"), errors="coerce")
    positive_integer = context.notna() & (context > 0) & np.isclose(context, np.round(context))
    heads = pd.to_numeric(frame.get("n_heads"), errors="coerce")
    width = pd.to_numeric(frame.get("d_model"), errors="coerce")
    ratio = width / heads
    architecture_ok = heads.gt(0) & width.gt(0) & np.isclose(ratio, np.round(ratio))

    sources = {int(k): str(v) for k, v in cfg["official_context_sources"].items()}
    official_status = context.map(lambda x: "verified_by_representative" if pd.notna(x) and int(x) in sources else "unverified")
    official_source = context.map(lambda x: sources.get(int(x), "") if pd.notna(x) else "")
    use = schema_ok & ~duplicate & positive_integer & official_status.eq("verified_by_representative")
    reason = np.where(~positive_integer, "invalid_context", np.where(duplicate, "duplicate_model", np.where(
        official_status.ne("verified_by_representative"), "context_not_independently_verified", "")))

    audit_frame = frame.copy()
    audit_frame["schema_valid"] = bool(schema_ok)
    audit_frame["duplicate_model"] = duplicate.astype(bool)
    audit_frame["positive_integer_context"] = positive_integer.astype(bool)
    audit_frame["architecture_consistency_flag"] = np.where(architecture_ok, "PASS", "WARN_noninteger_head_width")
    audit_frame["official_context_status"] = official_status
    audit_frame["official_source"] = official_source
    audit_frame["p3_use_flag"] = use.astype(bool)
    audit_frame["exclusion_reason"] = reason
    audit_frame.to_csv(output_dir / "c7_context_audit.csv", index=False)

    valid_contexts = sorted({int(x) for x in context[use]})
    counts = {str(int(k)): int(v) for k, v in context.value_counts().sort_index().items()}
    summary = {
        "path": str(c7_path), "sha256": sha256_file(c7_path), "shape": list(frame.shape),
        "columns": frame.columns.tolist(), "schema_valid": bool(schema_ok),
        "missing_cells": int(frame.isna().sum().sum()), "duplicate_models": int(duplicate.sum()),
        "architecture_warning_rows": int((~architecture_ok).sum()),
        "architecture_warning_models": frame.loc[~architecture_ok, "model_name"].astype(str).tolist(),
        "context_counts": counts, "verified_contexts": valid_contexts,
        "expected_contexts": list(map(int, cfg["contexts"])),
        "context_gate": "PASS" if valid_contexts == list(map(int, cfg["contexts"])) else "FAIL",
        "non_context_fields_used_by_problem3": [],
    }
    return audit_frame, summary


def audit_inputs(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    p2_root = Path(cfg["problem2_root"])
    missing = [name for name in CONSUMED_P2_FILES if not (p2_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing problem2 frozen inputs: {missing}")

    generalized = _read_json(p2_root / "generalized_scaling_parameters.json")
    bridge = _read_json(p2_root / "problem1_bridge.json")
    units = _read_json(p2_root / "units_contract.json")
    acceptance = _read_json(p2_root / "problem2_acceptance_report.json")
    manifest = _read_json(p2_root / "problem2_manifest.json")
    bootstrap = pd.read_csv(p2_root / "generalized_scaling_bootstrap.csv.gz")

    input_hashes = {name: sha256_file(p2_root / name) for name in CONSUMED_P2_FILES}
    hash_mismatches: dict[str, dict[str, str | None]] = {}
    for name in CONSUMED_P2_FILES:
        if name == "problem2_manifest.json":
            continue
        expected = manifest.get("outputs", {}).get(name)
        actual = input_hashes[name]
        if expected != actual:
            hash_mismatches[name] = {"expected": expected, "actual": actual}

    state = generalized.get("interface_state", {})
    hard_gate = (
        acceptance.get("summary", {}).get("FAIL", 1) == 0
        and state.get("P12_FREEZE_READY") is True
        and state.get("FIXED_P_READY") is True
        and generalized.get("classic_model") == "M0"
        and generalized.get("quality_model") == "GQ2"
        and not hash_mismatches
    )
    required_boot_cols = {
        "draw", "success", "classic_E", "classic_A", "classic_B", "classic_alpha",
        "classic_beta", "quality_c_Q", "quality_nu_Q", "Q_ref_draw",
    }
    boot_finite = bootstrap[list(required_boot_cols - {"draw", "success"})].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    boot_valid = bootstrap["success"].astype(bool) & boot_finite
    positive_cols = ["classic_E", "classic_A", "classic_B", "classic_alpha", "classic_beta", "quality_c_Q", "quality_nu_Q"]
    boot_valid &= (bootstrap[positive_cols] > 0).all(axis=1)

    p_ref = np.array([bridge["p_ref"][name] for name in bridge["domains"]], dtype=float)
    c7_path = Path(cfg["input_root"]) / "C_efficiency_evolution" / "model_architecture_metadata.csv"
    _, c7 = _audit_c7(c7_path, cfg, output_dir)

    audit = {
        "hard_gate_status": "PASS" if hard_gate and c7["context_gate"] == "PASS" else "FAIL",
        "upstream_interface_state": state,
        "upstream_acceptance": acceptance.get("summary", {}),
        "upstream_hard_gate_failures": acceptance.get("hard_gate_failures", []),
        "input_hashes": input_hashes,
        "hash_mismatches": hash_mismatches,
        "problem2_manifest_hash_verification": manifest.get("hash_verification"),
        "model_contract": {"classic_model": generalized.get("classic_model"), "quality_model": generalized.get("quality_model")},
        "bootstrap": {
            "rows": int(len(bootstrap)), "required_columns_present": required_boot_cols.issubset(bootstrap.columns),
            "valid_rows": int(boot_valid.sum()), "valid_rate": float(boot_valid.mean()),
        },
        "p_ref": {"domains": len(p_ref), "min": float(p_ref.min()), "sum": float(p_ref.sum()),
                  "simplex_error": abs(float(p_ref.sum()) - 1.0)},
        "Q_ref": float(bridge["Q_ref"]),
        "units": units,
        "c7": c7,
        "warnings_propagated": {
            "problem1": bridge.get("propagated_warn_ids", []),
            "problem2": [r["id"] for r in acceptance.get("rules", []) if r.get("status") == "WARN"],
        },
        "raw_A_or_B_read_by_problem3": False,
        "C7_only_direct_attachment": True,
    }
    json_dump(output_dir / "problem3_data_audit.json", audit)
    return {
        "audit": audit, "generalized": generalized, "bridge": bridge, "units": units,
        "problem2_acceptance": acceptance, "problem2_manifest": manifest,
        "bootstrap": bootstrap.loc[boot_valid].copy(), "p_ref": p_ref,
        "domains": list(bridge["domains"]), "central_params": params_from_payload(generalized),
    }


def build_scenario_registry(cfg: dict[str, Any], verified_contexts: list[int], output_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for cost in cfg["quality_cost_types"]:
        for context in verified_contexts:
            for budget in cfg["budgets"]:
                records.append({"role": "main_anchor", "quality_cost_type": cost, "context_length": context,
                                "budget_FLOPs": budget, "s_Q": cfg["main_quality_mapping_slope"],
                                "support_mode": "operational_extended", "eta": cfg["eta"]})
                records.append({"role": "strict_anchor", "quality_cost_type": cost, "context_length": context,
                                "budget_FLOPs": budget, "s_Q": cfg["main_quality_mapping_slope"],
                                "support_mode": "strict_joint", "eta": cfg["eta"]})
        for context in verified_contexts:
            records.append({"role": "continuous_path", "quality_cost_type": cost, "context_length": context,
                            "budget_FLOPs": "61_log_points", "s_Q": cfg["main_quality_mapping_slope"],
                            "support_mode": "operational_extended", "eta": cfg["eta"]})
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "scenario_registry.csv", index=False)
    return frame
