from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .contracts import ScalingParameters, SupportBounds, params_from_bootstrap
from .cost import CostSpec, cost_spec_from_config, quality_cost_increment, total_costs
from .objective import predict_loss, quality_upper_bound
from .optimize import classify_support, solve_2d
from .transitions import budget_grid, detect_transitions


PARAMETER_COLUMNS = [
    "classic_E", "classic_A", "classic_B", "classic_alpha", "classic_beta",
    "quality_c_Q", "quality_nu_Q", "Q_ref_draw",
]


def deterministic_path_draws(bootstrap: pd.DataFrame, count: int) -> pd.DataFrame:
    values = bootstrap[PARAMETER_COLUMNS].to_numpy(dtype=float)
    scale = np.std(values, axis=0, ddof=1)
    scale[scale == 0] = 1.0
    z = (values - np.mean(values, axis=0)) / scale
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    score = z @ vt[0]
    ordered = np.argsort(score)
    positions = np.linspace(0, len(ordered) - 1, min(count, len(ordered))).round().astype(int)
    selected = bootstrap.iloc[ordered[positions]].copy()
    selected["selection_rank"] = np.arange(len(selected))
    return selected


def run_anchor_bootstrap(bootstrap: pd.DataFrame, center: pd.DataFrame, cfg: dict,
                         support: SupportBounds) -> pd.DataFrame:
    rows: list[dict] = []
    center_key = {(r.quality_cost_type, int(r.context_length), float(r.budget_FLOPs)): r
                  for r in center.itertuples() if r.support_mode == support.name}
    for boot in bootstrap.itertuples(index=False):
        params = params_from_bootstrap(boot._asdict())
        for key, reference in center_key.items():
            cost_name, context, budget = key
            cost = cost_spec_from_config(cfg, cost_name)
            row = solve_2d(budget, context, float(cfg["eta"]), 1.0, cost, params, support,
                           eps=float(cfg["quality_eps"]),
                           starts=[(float(reference.N_B), max(params.Q0, float(reference.Q_A)))],
                           fast=True)
            row["draw_id"] = int(getattr(boot, "draw"))
            rows.append(row)
    return pd.DataFrame(rows)


def _local_lattice_path(reference: pd.DataFrame, params: ScalingParameters, cost: CostSpec,
                        context: int, support: SupportBounds, cfg: dict) -> pd.DataFrame:
    """Fast deterministic re-optimization used only for transition bootstrap paths.

    It evaluates a registered local lattice around every central continuation point,
    plus all relevant boundaries. Anchor uncertainty is still solved with SLSQP.
    """
    q_hi = quality_upper_bound(params.Q0, 1.0, float(cfg["quality_eps"]))
    n_ratios = np.array([0.65, 0.78, 0.90, 0.97, 1.0, 1.03, 1.11, 1.28, 1.55])
    q_offsets = np.array([-0.10, -0.05, -0.02, -0.005, 0.0, 0.005, 0.02, 0.05, 0.10])
    rows = []
    for ref in reference.sort_values("budget_FLOPs").itertuples():
        budget = float(ref.budget_FLOPs)
        n_values = np.unique(np.clip(np.r_[float(ref.N_B) * n_ratios, support.n_min, support.n_max],
                                     support.n_min, support.n_max))
        shifted_q = float(ref.Q_A) + (params.Q0 - float(ref.Q0))
        q_values = np.unique(np.clip(np.r_[shifted_q + q_offsets, params.Q0, q_hi], params.Q0, q_hi))
        nn, qq = np.meshgrid(n_values, q_values, indexing="ij")
        delta = quality_cost_increment(qq, params.Q0, cost) / 1e9
        dd_budget = (budget / 1e18) / ((6.0 + float(cfg["eta"]) * context) * nn + delta)
        dd = np.minimum(support.d_max, dd_budget)
        feasible = dd_budget >= support.d_min
        values = predict_loss(nn, dd, qq, params, 1.0)
        values = np.where(feasible, values, np.inf)
        if not np.isfinite(values).any():
            rows.append({"budget_FLOPs": budget, "solver_success": False,
                         "active_set": "infeasible", "N_B": np.nan, "D_B": np.nan,
                         "Q_A": np.nan, "Q0": params.Q0})
            continue
        i, j = np.unravel_index(np.argmin(values), values.shape)
        n, d, q = float(nn[i, j]), float(dd[i, j]), float(qq[i, j])
        tol = 2e-6
        active = []
        if q <= params.Q0 + tol: active.append("Q_lower")
        if q >= q_hi - tol: active.append("Q_upper")
        if n <= support.n_min * (1 + tol): active.append("N_lower")
        if n >= support.n_max * (1 - tol): active.append("N_upper")
        if d <= support.d_min * (1 + tol): active.append("D_lower")
        if d >= support.d_max * (1 - tol): active.append("D_upper")
        costs = total_costs(n, d, q, params.Q0, context, float(cfg["eta"]), cost)
        if abs(costs["total"] / budget - 1.0) <= 1e-6: active.append("budget")
        rows.append({"budget_FLOPs": budget, "context_length": context,
                     "quality_cost_type": cost.name, "N_B": n, "D_B": d, "Q_A": q,
                     "Q0": params.Q0, "predicted_loss": float(values[i, j]),
                     "active_set": "|".join(active) if active else "interior",
                     "solver_success": True, "support_status": classify_support(n, d),
                     "path_solver": "registered_local_lattice"})
    return pd.DataFrame(rows)


def run_transition_bootstrap(selected: pd.DataFrame, center_paths: dict[tuple[str, int], pd.DataFrame],
                             cfg: dict, support: SupportBounds) -> pd.DataFrame:
    records: list[dict] = []
    for boot in selected.itertuples(index=False):
        draw_id = int(getattr(boot, "draw"))
        params = params_from_bootstrap(boot._asdict())
        draw_valid = True
        for (cost_name, context), reference in center_paths.items():
            cost = cost_spec_from_config(cfg, cost_name)
            path = _local_lattice_path(reference, params, cost, context, support, cfg)
            valid = bool(path["solver_success"].all())
            draw_valid &= valid
            transitions = detect_transitions(path, cost, context, params, support, cfg, refine=False)
            for event in transitions.itertuples(index=False):
                record = event._asdict()
                record.update({"record_type": "event", "draw_id": draw_id,
                               "draw_valid": valid, "path_solver": "registered_local_lattice"})
                records.append(record)
        records.append({"record_type": "draw_summary", "draw_id": draw_id,
                        "draw_valid": draw_valid, "path_solver": "registered_local_lattice"})
    return pd.DataFrame(records)


def bootstrap_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["solver_success"] & np.isfinite(frame["predicted_loss"])].copy()
    keys = ["quality_cost_type", "context_length", "budget_FLOPs"]
    columns = ["N_B", "D_B", "Q_A", "Q_eff", "predicted_loss",
               "share_train", "share_quality", "share_attention"]
    records = []
    for key, group in valid.groupby(keys):
        row = dict(zip(keys, key))
        row["valid_draws"] = len(group)
        for column in columns:
            q = group[column].quantile([0.025, 0.5, 0.975])
            row[f"{column}_q025"] = float(q.loc[0.025])
            row[f"{column}_median"] = float(q.loc[0.5])
            row[f"{column}_q975"] = float(q.loc[0.975])
        records.append(row)
    return pd.DataFrame(records)


def run_sensitivity(center: pd.DataFrame, params: ScalingParameters, cfg: dict,
                    operational: SupportBounds, strict: SupportBounds) -> pd.DataFrame:
    base = center[center["support_mode"].eq("operational_extended")]
    records: list[dict] = []

    def evaluate(label: str, value: str, row, cost: CostSpec, *, slope: float = 1.0,
                 eta: float | None = None, q_cap: float = 1.0,
                 support: SupportBounds = operational) -> None:
        solved = solve_2d(float(row.budget_FLOPs), int(row.context_length),
                          float(cfg["eta"] if eta is None else eta), slope, cost, params, support,
                          eps=float(cfg["quality_eps"]), q_cap=q_cap,
                          starts=[(float(row.N_B), float(row.Q_A))], fast=True)
        records.append({"dimension": label, "setting": value,
                        "quality_cost_type": row.quality_cost_type,
                        "context_length": int(row.context_length), "budget_FLOPs": float(row.budget_FLOPs),
                        "support_mode": support.name, "solver_success": solved["solver_success"],
                        "N_B": solved["N_B"], "D_B": solved["D_B"], "Q_A": solved["Q_A"],
                        "predicted_loss": solved["predicted_loss"],
                        "loss_change": solved["predicted_loss"] - float(row.predicted_loss)
                        if np.isfinite(solved["predicted_loss"]) else np.nan})

    for row in base.itertuples():
        cost = cost_spec_from_config(cfg, row.quality_cost_type)
        for slope in [0.5, 1.5]:
            evaluate("quality_mapping_slope", str(slope), row, cost, slope=slope)
        evaluate("quality_empirical_cap", str(cfg["quality_empirical_cap"]), row, cost,
                 q_cap=float(cfg["quality_empirical_cap"]))
        evaluate("support", "strict_joint", row, cost, support=strict)
        if int(row.context_length) == 4096:
            for scale in [0.8, 1.2]:
                evaluate("cost_amplitude", str(scale), row,
                         cost_spec_from_config(cfg, row.quality_cost_type, amplitude_scale=scale))
                evaluate("cost_shape", str(scale), row,
                         cost_spec_from_config(cfg, row.quality_cost_type, shape_scale=scale))
                evaluate("attention_eta", str(scale), row, cost,
                         eta=float(cfg["eta"]) * scale)
    return pd.DataFrame(records)
