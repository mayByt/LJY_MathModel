from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import json_dump, sha256_file


B_FILES = {
    "B1": "B_scaling_laws/pythia_training_log_existing.csv",
    "B2": "B_scaling_laws/cerebras_training_log.csv",
    "B4": "B_scaling_laws/scaling_baseline.csv",
    "B5": "B_scaling_laws/published_scaling_data.csv",
    "B6": "B_scaling_laws/supplementary_NQ_experiment.csv",
    "B7": "B_scaling_laws/supplementary_NQ_experiment_expanded.csv",
    "B8": "B_scaling_laws/supplementary_NQ_experiment_large.csv",
    "B9": "B_scaling_laws/supplementary_large_models.csv",
    "B10": "B_scaling_laws/supplementary_large_baseline.csv",
    "B11": "B_scaling_laws/open_model_family_metadata.csv",
    "B12": "B_scaling_laws/pythia_checkpoint_index.csv",
}

B3_RELATIVE_GLOB = "B_scaling_laws/training_trajectories/*.csv"


def read_b_tables(input_root: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for key, rel in B_FILES.items():
        tables[key] = pd.read_csv(input_root / rel)
    b3_parts = []
    for path in sorted(input_root.glob(B3_RELATIVE_GLOB)):
        part = pd.read_csv(path)
        part["source_file"] = path.name
        b3_parts.append(part)
    if not b3_parts:
        raise FileNotFoundError(f"no B3 trajectories matched {B3_RELATIVE_GLOB}")
    tables["B3"] = pd.concat(b3_parts, ignore_index=True)
    return tables


def _finite_positive(df: pd.DataFrame, columns: list[str]) -> bool:
    for col in columns:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.isna().any() or not np.isfinite(x).all() or (x <= 0).any():
            return False
    return True


def audit_data(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    input_root = Path(cfg["input_root"])
    tables = read_b_tables(input_root)
    audit: dict[str, Any] = {"files": {}, "checks": {}, "b8": {}}
    invalid_rows: list[dict[str, Any]] = []
    for key, rel in B_FILES.items():
        path = input_root / rel
        df = tables[key]
        audit["files"][key] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "size_bytes": path.stat().st_size,
        }
    b3_paths = sorted(input_root.glob(B3_RELATIVE_GLOB))
    b3_hashes = {path.name: sha256_file(path) for path in b3_paths}
    b3_digest = hashlib.sha256("".join(f"{k}:{v};" for k, v in sorted(b3_hashes.items())).encode("utf-8")).hexdigest()
    audit["files"]["B3"] = {
        "path": B3_RELATIVE_GLOB,
        "sha256": b3_digest,
        "shape": list(tables["B3"].shape),
        "columns": list(tables["B3"].columns),
        "size_bytes": int(sum(path.stat().st_size for path in b3_paths)),
        "members": b3_hashes,
    }

    b1 = tables["B1"]
    b2 = tables["B2"]
    b3 = tables["B3"]
    b6 = tables["B6"]
    b7 = tables["B7"]
    b8 = tables["B8"]
    b9 = tables["B9"]
    b11 = tables["B11"]
    b12 = tables["B12"]
    ratio = b1["C_FLOPs_1e21"].to_numpy(float) * 1000.0 / (6.0 * b1["N_params_B"].to_numpy(float) * b1["D_tokens_B"].to_numpy(float))
    audit["checks"]["b1_rows"] = int(len(b1))
    audit["checks"]["b1_n_levels"] = int(b1["N_params_B"].nunique())
    audit["checks"]["b1_rows_per_n"] = {str(k): int(v) for k, v in b1.groupby("N_params_B").size().items()}
    audit["checks"]["b1_core_finite_positive"] = _finite_positive(b1, ["N_params_B", "D_tokens_B", "C_FLOPs_1e21", "val_loss"])
    audit["checks"]["b1_compute_ratio_median"] = float(np.median(ratio))
    audit["checks"]["b1_compute_ratio_p05"] = float(np.quantile(ratio, 0.05))
    audit["checks"]["b1_compute_ratio_p95"] = float(np.quantile(ratio, 0.95))
    audit["checks"]["b2_core_finite_positive"] = _finite_positive(b2, ["N_params_B", "D_tokens_B", "C_FLOPs_1e21", "val_loss"])
    audit["checks"]["b2_aux_all_missing"] = {
        col: bool(b2[col].isna().all()) for col in ["gpu_days", "step_time_ms", "grad_norm_avg"] if col in b2
    }
    audit["checks"]["b2_or_b3_validation_choice"] = {
        "selected": "B2",
        "B2_rows": int(len(b2)),
        "B3_rows": int(len(b3)),
        "B3_files": int(b3["source_file"].nunique()),
        "B3_role": "audited_interpolation_not_counted_as_independent_evidence",
    }
    audit["checks"]["b11_source_audit"] = {
        "rows": int(len(b11)),
        "families": int(b11["family"].nunique()),
        "missing_error": int(b11["error"].isna().sum()),
        "source_url_nonempty": int(b11["source_url"].fillna("").str.len().gt(0).sum()),
    }
    audit["checks"]["b12_checkpoint_audit"] = {
        "rows": int(len(b12)),
        "model_repos": int(b12["model_repo"].nunique()),
        "duplicate_repo_step": int(b12.duplicated(["model_repo", "step"]).sum()),
        "step_min": int(b12["step"].min()),
        "step_max": int(b12["step"].max()),
    }
    audit["usage_ledger"] = {
        "B1": "classic_fit_cv_bootstrap",
        "B2": "selected_family_out_validation",
        "B3": "audited_only_B2_selected_under_B2_or_B3_rule",
        "B4": "external_family_validation",
        "B5": "published_validation",
        "B6": "strict_subset_integrity_check_only",
        "B7": "quality_fit_blocked_cv",
        "B8": "direction_and_floor_stress_test_only",
        "B9": "extended_N_D_support_boundary",
        "B10": "estimated_large_model_scale_check",
        "B11": "family_source_provenance_audit",
        "B12": "checkpoint_index_integrity_audit",
    }

    b7_key = ["N_params_B", "D_tokens_B", "Q_score"]
    merged = b6.merge(b7, on=b7_key, how="left", suffixes=("_b6", "_b7"), indicator=True)
    shared = merged[merged["_merge"] == "both"]
    max_loss_diff = float(np.max(np.abs(shared["val_loss_b6"].to_numpy(float) - shared["val_loss_b7"].to_numpy(float)))) if len(shared) else float("nan")
    audit["checks"]["b6_in_b7_matches"] = int((merged["_merge"] == "both").sum())
    audit["checks"]["b6_in_b7_max_loss_diff"] = max_loss_diff
    audit["checks"]["b7_duplicate_keys"] = int(b7.duplicated(b7_key).sum())
    audit["checks"]["b7_nd_cells"] = int(b7.groupby(["N_params_B", "D_tokens_B"]).ngroups)
    audit["checks"]["b7_q_per_cell_min"] = int(b7.groupby(["N_params_B", "D_tokens_B"])["Q_score"].nunique().min())
    audit["checks"]["b7_q_per_cell_max"] = int(b7.groupby(["N_params_B", "D_tokens_B"])["Q_score"].nunique().max())

    group_stats = []
    for (n, d), g in b7.groupby(["N_params_B", "D_tokens_B"]):
        s = spearmanr(g["Q_score"], g["val_loss"]).statistic
        group_stats.append(float(s))
    audit["checks"]["b7_group_spearman_median"] = float(np.nanmedian(group_stats))

    b8_group_stats = []
    increased = 0
    steps = 0
    for (_, _), g in b8.groupby(["N_params_B", "D_tokens_B"]):
        gg = g.sort_values("Q_score")
        b8_group_stats.append(float(spearmanr(gg["Q_score"], gg["val_loss"]).statistic))
        diffs = np.diff(gg["val_loss"].to_numpy(float))
        increased += int((diffs > 0).sum())
        steps += int(len(diffs))
    audit["b8"] = {
        "rows_by_type": {str(k): int(v) for k, v in b8.groupby("data_type").size().items()},
        "q_range": [float(b8["Q_score"].min()), float(b8["Q_score"].max())],
        "loss_range": [float(b8["val_loss"].min()), float(b8["val_loss"].max())],
        "floor_count_loss_0_5": int(np.isclose(b8["val_loss"], 0.5).sum()),
        "group_spearman_median": float(np.nanmedian(b8_group_stats)),
        "adjacent_increases": increased,
        "adjacent_steps": steps,
        "main_fit_rows_including_b8": 0,
    }
    audit["checks"]["b9_zero_token_rows"] = int((pd.to_numeric(b9["D_tokens_B"], errors="coerce") == 0).sum())
    positive_b9 = b9[(pd.to_numeric(b9["N_params_B"], errors="coerce") > 0) & (pd.to_numeric(b9["D_tokens_B"], errors="coerce") > 0)]
    audit["checks"]["support_observed"] = {
        "N_params_B": [float(b1["N_params_B"].min()), float(b1["N_params_B"].max())],
        "D_tokens_B": [float(b7["D_tokens_B"].min()), float(b7["D_tokens_B"].max())],
        "Q": [float(b7["Q_score"].min()), float(b7["Q_score"].max())],
    }
    audit["checks"]["support_extended_B9"] = {
        "N_params_B": [float(positive_b9["N_params_B"].min()), float(positive_b9["N_params_B"].max())],
        "D_tokens_B": [float(positive_b9["D_tokens_B"].min()), float(positive_b9["D_tokens_B"].max())],
    }

    for key in ["B1", "B2", "B4", "B5", "B6", "B7", "B8", "B10"]:
        df = tables[key]
        for col in [c for c in ["N_params_B", "D_tokens_B", "Q_score", "val_loss"] if c in df.columns]:
            x = pd.to_numeric(df[col], errors="coerce")
            bad = x.isna() | ~np.isfinite(x) | (x < 0)
            for idx in df.index[bad]:
                invalid_rows.append({"source": key, "row": int(idx), "field": col, "value": str(df.loc[idx, col]), "reason": "non-finite or negative", "action": "audit"})
    pd.DataFrame(invalid_rows, columns=["source", "row", "field", "value", "reason", "action"]).to_csv(output_dir / "invalid_rows.csv", index=False)
    hard_failures: list[str] = []
    if len(b1) != 1176 or list(b7.shape) != [450, 5]:
        hard_failures.append("unexpected_core_shape")
    if not audit["checks"]["b1_core_finite_positive"] or not audit["checks"]["b2_core_finite_positive"]:
        hard_failures.append("nonfinite_or_nonpositive_core")
    if not (abs(audit["checks"]["b1_compute_ratio_median"] - 1.0) <= 0.01 and 0.98 <= audit["checks"]["b1_compute_ratio_p05"] <= 1.02 and 0.98 <= audit["checks"]["b1_compute_ratio_p95"] <= 1.02):
        hard_failures.append("compute_unit_mismatch")
    if not (audit["checks"]["b6_in_b7_matches"] == 360 and audit["checks"]["b6_in_b7_max_loss_diff"] <= 1e-12 and audit["checks"]["b7_duplicate_keys"] == 0):
        hard_failures.append("b6_b7_key_integrity")
    if not (audit["checks"]["b7_nd_cells"] == 45 and audit["checks"]["b7_q_per_cell_min"] == 10 and audit["checks"]["b7_q_per_cell_max"] == 10):
        hard_failures.append("b7_grid_incomplete")
    if audit["checks"]["b2_or_b3_validation_choice"]["B3_files"] != 8:
        hard_failures.append("b3_trajectory_file_count")
    if audit["checks"]["b11_source_audit"]["rows"] != 18 or audit["checks"]["b12_checkpoint_audit"]["rows"] != 1386:
        hard_failures.append("auxiliary_metadata_shape")
    audit["hard_failures"] = hard_failures
    audit["hard_gate_status"] = "PASS" if not hard_failures else "FAIL"
    json_dump(output_dir / "scaling_data_audit.json", audit)
    return {"tables": tables, "audit": audit, "invalid_rows": invalid_rows}


def b7_splits(b7: pd.DataFrame) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = []
    for n in sorted(b7["N_params_B"].unique()):
        mask = b7["N_params_B"] == n
        splits.append({"split_type": "leave_N", "held_out": float(n), "mask": mask.to_numpy()})
    for d in sorted(b7["D_tokens_B"].unique()):
        mask = b7["D_tokens_B"] == d
        splits.append({"split_type": "leave_D", "held_out": float(d), "mask": mask.to_numpy()})
    for n, d in sorted(b7.groupby(["N_params_B", "D_tokens_B"]).groups):
        mask = (b7["N_params_B"] == n) & (b7["D_tokens_B"] == d)
        splits.append({"split_type": "leave_cell", "held_out": f"N{n}_D{d}", "mask": mask.to_numpy()})
    for q in sorted(b7["Q_score"].unique()):
        mask = b7["Q_score"] == q
        kind = "endpoint" if q in (b7["Q_score"].min(), b7["Q_score"].max()) else "interior"
        splits.append({"split_type": "leave_Q", "held_out": float(q), "q_position": kind, "mask": mask.to_numpy()})
    return splits
