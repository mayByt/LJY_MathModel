from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import ScalingParameters, SupportBounds
from .cost import CostSpec
from .optimize import kkt_diagnostics, solve_2d


def budget_grid(cfg: dict) -> np.ndarray:
    points = np.geomspace(float(cfg["budget_path_min"]), float(cfg["budget_path_max"]),
                         int(cfg["budget_path_points"]))
    return np.unique(np.r_[points, np.asarray(cfg["budgets"], dtype=float)])


def solve_budget_path(cost: CostSpec, context: int, params: ScalingParameters,
                      support: SupportBounds, cfg: dict, *, fast: bool = False,
                      reference_path: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    previous: tuple[float, float] | None = None
    ref_by_budget = {}
    if reference_path is not None:
        ref_by_budget = {float(r.budget_FLOPs): (float(r.N_B), float(r.Q_A))
                         for r in reference_path.itertuples()}
    for budget in budget_grid(cfg):
        starts: list[tuple[float, float]] = []
        if previous is not None:
            starts.append(previous)
        if float(budget) in ref_by_budget:
            starts.append(ref_by_budget[float(budget)])
        row = solve_2d(float(budget), context, float(cfg["eta"]),
                       float(cfg["main_quality_mapping_slope"]), cost, params, support,
                       eps=float(cfg["quality_eps"]),
                       grid_n=int(cfg["grid_n_points"]), grid_q=int(cfg["grid_q_points"]),
                       multistarts=int(cfg["local_multistarts"]), starts=starts or None,
                       fast=fast)
        if np.isfinite(row.get("N_B", np.nan)):
            previous = (float(row["N_B"]), float(row["Q_A"]))
            row.update(kkt_diagnostics(row, params, cost, support,
                                       float(cfg["main_quality_mapping_slope"]),
                                       float(cfg["quality_eps"])))
        else:
            row["active_set"] = "infeasible"
            row["kkt_relative_dispersion"] = np.nan
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("budget_FLOPs").reset_index(drop=True)
    frame["path_id"] = f"{cost.name}|{context}"
    return frame


def _event_type(left: str, right: str) -> str:
    a, b = set(left.split("|")), set(right.split("|"))
    if "Q_lower" in a and "Q_lower" not in b:
        return "quality_investment_start"
    if "Q_upper" not in a and "Q_upper" in b:
        return "quality_saturation"
    for name in ["N_lower", "N_upper", "D_lower", "D_upper"]:
        if (name in a) != (name in b):
            return f"{name}_change"
    if "infeasible" in a and "infeasible" not in b:
        return "strict_support_becomes_feasible"
    return "active_set_change"


def refine_primary_transition(left_row: pd.Series, right_row: pd.Series, cost: CostSpec,
                              context: int, params: ScalingParameters, support: SupportBounds,
                              cfg: dict) -> dict:
    lo, hi = float(left_row["budget_FLOPs"]), float(right_row["budget_FLOPs"])
    left_active, right_active = str(left_row["active_set"]), str(right_row["active_set"])
    while math.log10(hi) - math.log10(lo) > float(cfg["acceptance"]["transition_log10_width"]):
        mid = math.sqrt(lo * hi)
        row = solve_2d(mid, context, float(cfg["eta"]), 1.0, cost, params, support,
                       eps=float(cfg["quality_eps"]),
                       starts=[(float(left_row["N_B"]), float(left_row["Q_A"])),
                               (float(right_row["N_B"]), float(right_row["Q_A"]))], fast=True)
        active = kkt_diagnostics(row, params, cost, support, 1.0,
                                 float(cfg["quality_eps"])).get("active_set", "infeasible")
        if active == left_active:
            lo = mid
        else:
            hi = mid
            right_active = str(active)
    return {
        "transition_class": "primary", "event_type": _event_type(left_active, right_active),
        "budget_lower": lo, "budget_upper": hi,
        "budget_mid": math.sqrt(lo * hi),
        "log10_interval_width": math.log10(hi) - math.log10(lo),
        "active_before": left_active, "active_after": right_active,
        "bic_improvement": np.nan, "slope_before": np.nan, "slope_after": np.nan,
    }


def _bic(y: np.ndarray, fitted: np.ndarray, k: int) -> float:
    n = len(y)
    rss = max(float(np.sum((y - fitted) ** 2)), 1e-300)
    return n * math.log(rss / n) + k * math.log(n)


def secondary_breakpoint(path: pd.DataFrame, variable: str, cfg: dict) -> dict | None:
    if variable == "Q_increment":
        clean = path[np.isfinite(path["Q_A"]) & np.isfinite(path["Q0"])].copy()
        clean = clean[clean["Q_A"] > clean["Q0"] + 1e-7]
        y = np.log(clean["Q_A"].to_numpy() - clean["Q0"].to_numpy())
    else:
        clean = path[np.isfinite(path[variable])].copy()
        y = np.log(clean[variable].to_numpy())
    x = np.log(clean["budget_FLOPs"].to_numpy())
    if len(x) < 12:
        return None
    base = np.polyfit(x, y, 1)
    base_bic = _bic(y, np.polyval(base, x), 2)
    best = None
    for split in range(5, len(x) - 5):
        p1, p2 = np.polyfit(x[:split], y[:split], 1), np.polyfit(x[split:], y[split:], 1)
        fitted = np.r_[np.polyval(p1, x[:split]), np.polyval(p2, x[split:])]
        improvement = base_bic - _bic(y, fitted, 4)
        slope_difference = abs(float(p1[0] - p2[0]))
        if best is None or improvement > best[0]:
            best = (improvement, slope_difference, split, p1[0], p2[0])
    if best is None:
        return None
    improvement, slope_difference, split, before, after = best
    if improvement < float(cfg["acceptance"]["secondary_bic_improvement"]) or \
            slope_difference < float(cfg["acceptance"]["secondary_slope_difference"]):
        return None
    budget = float(clean.iloc[split]["budget_FLOPs"])
    return {
        "transition_class": "secondary", "event_type": f"{variable}_slope_break",
        "budget_lower": float(clean.iloc[split - 1]["budget_FLOPs"]),
        "budget_upper": budget, "budget_mid": math.sqrt(float(clean.iloc[split - 1]["budget_FLOPs"]) * budget),
        "log10_interval_width": math.log10(budget) - math.log10(float(clean.iloc[split - 1]["budget_FLOPs"])),
        "active_before": str(clean.iloc[split - 1]["active_set"]),
        "active_after": str(clean.iloc[split]["active_set"]),
        "bic_improvement": float(improvement), "slope_before": float(before),
        "slope_after": float(after),
    }


def detect_transitions(path: pd.DataFrame, cost: CostSpec, context: int,
                       params: ScalingParameters, support: SupportBounds, cfg: dict,
                       refine: bool = True) -> pd.DataFrame:
    records: list[dict] = []
    ordered = path.sort_values("budget_FLOPs").reset_index(drop=True)
    for idx in range(1, len(ordered)):
        left, right = ordered.iloc[idx - 1], ordered.iloc[idx]
        if str(left["active_set"]) != str(right["active_set"]):
            if refine:
                event = refine_primary_transition(left, right, cost, context, params, support, cfg)
            else:
                lo, hi = float(left["budget_FLOPs"]), float(right["budget_FLOPs"])
                event = {
                    "transition_class": "primary", "event_type": _event_type(str(left["active_set"]), str(right["active_set"])),
                    "budget_lower": lo, "budget_upper": hi, "budget_mid": math.sqrt(lo * hi),
                    "log10_interval_width": math.log10(hi) - math.log10(lo),
                    "active_before": str(left["active_set"]), "active_after": str(right["active_set"]),
                    "bic_improvement": np.nan, "slope_before": np.nan, "slope_after": np.nan,
                }
            records.append(event)
    for variable in ["N_B", "D_B", "Q_increment"]:
        candidate = secondary_breakpoint(ordered, variable, cfg)
        if candidate is not None:
            records.append(candidate)
    for row in records:
        row.update({"quality_cost_type": cost.name, "context_length": int(context),
                    "support_mode": support.name, "stable": False,
                    "stability_probability": np.nan})
    return pd.DataFrame(records)


def summarize_transition_stability(center: pd.DataFrame, bootstrap: pd.DataFrame,
                                   valid_draws: int, threshold: float) -> pd.DataFrame:
    if center.empty:
        return center
    result = center.copy()
    if bootstrap.empty or valid_draws <= 0:
        result["stability_probability"] = 0.0
        result["stable"] = False
        return result
    events = bootstrap[bootstrap["record_type"].eq("event")]
    counts = events.groupby(["quality_cost_type", "context_length", "event_type"])["draw_id"].nunique()
    probabilities = []
    for row in result.itertuples():
        key = (row.quality_cost_type, row.context_length, row.event_type)
        probabilities.append(float(counts.get(key, 0)) / valid_draws)
    result["stability_probability"] = probabilities
    result["stable"] = result["stability_probability"] >= threshold
    return result
