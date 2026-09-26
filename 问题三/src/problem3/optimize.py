from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from .contracts import ScalingParameters, SupportBounds
from .cost import CostSpec, quality_cost_derivative, quality_cost_increment, total_costs
from .objective import loss_derivatives_absolute, predict_loss, quality_upper_bound


def _budget_d_billion(budget: float, n_b: float, q: float, q0: float, context: int,
                      eta: float, spec: CostSpec) -> float:
    denominator = (6.0 + eta * context) * n_b + float(quality_cost_increment(q, q0, spec)) / 1e9
    return (budget / 1e18) / denominator


def classify_support(n_b: float, d_b: float) -> str:
    n_joint = 0.070542 <= n_b <= 11.965825
    d_joint = 10.0 <= d_b <= 299.893
    if n_joint and d_joint:
        return "joint_observed"
    if n_joint != d_joint:
        if 11.965825 < n_b < 100.0:
            return "bridge_gap_extrapolated"
        return "one_component_extrapolated"
    if 11.965825 < n_b < 100.0:
        return "bridge_gap_extrapolated"
    return "boundary_supported_extrapolated"


def _record_solution(n_b: float, d_b: float, q: float, budget: float, context: int, eta: float,
                     slope: float, spec: CostSpec, params: ScalingParameters, support: SupportBounds,
                     solver: str, success: bool, message: str, grid_loss: float | None = None,
                     fine_grid_loss: float | None = None) -> dict:
    loss = float(predict_loss(n_b, d_b, q, params, slope))
    q_eff = float(params.Q0 + slope * (q - params.Q0))
    costs = total_costs(n_b, d_b, q, params.Q0, context, eta, spec)
    slack = budget - costs["total"]
    violation = max(0.0, -slack / budget, (support.n_min - n_b) / max(support.n_min, 1e-30),
                    (n_b - support.n_max) / max(support.n_max, 1e-30),
                    (support.d_min - d_b) / max(support.d_min, 1e-30),
                    (d_b - support.d_max) / max(support.d_max, 1e-30),
                    params.Q0 - q, q - 1.0)
    return {
        "budget_FLOPs": float(budget), "context_length": int(context), "eta": float(eta),
        "quality_cost_type": spec.name, "quality_cost_amplitude": spec.amplitude,
        "quality_cost_shape": spec.shape, "s_Q": float(slope), "support_mode": support.name,
        "N_B": float(n_b), "D_B": float(d_b), "N_abs": float(n_b * 1e9),
        "D_abs": float(d_b * 1e9), "Q_A": float(q), "Q_eff": q_eff,
        "Q0": float(params.Q0), "predicted_loss": loss,
        "cost_train": costs["train"], "cost_quality": costs["quality"],
        "cost_attention": costs["attention"], "cost_total": costs["total"],
        "share_train": costs["train"] / budget, "share_quality": costs["quality"] / budget,
        "share_attention": costs["attention"] / budget, "budget_slack": slack,
        "budget_relative_slack": slack / budget, "max_normalized_violation": float(violation),
        "support_status": classify_support(n_b, d_b) if support.name == "operational_extended" else "strict_joint",
        "extrapolation_flag": support.name == "operational_extended" and classify_support(n_b, d_b) != "joint_observed",
        "solver": solver, "solver_success": bool(success), "solver_message": str(message),
        "grid_best_loss": grid_loss, "fine_grid_best_loss": fine_grid_loss,
        "p_source": "p_ref", "lambda0": 0.0,
        **{f"param_{k}": v for k, v in asdict(params).items()},
    }


def infeasible_record(budget: float, context: int, eta: float, slope: float, spec: CostSpec,
                      params: ScalingParameters, support: SupportBounds, reason: str) -> dict:
    row = _record_solution(support.n_min, support.d_min, params.Q0, budget, context, eta, slope,
                           spec, params, support, "none", False, reason)
    for key in ["N_B", "D_B", "N_abs", "D_abs", "Q_A", "Q_eff", "predicted_loss",
                "cost_train", "cost_quality", "cost_attention", "cost_total", "share_train",
                "share_quality", "share_attention", "budget_slack", "budget_relative_slack"]:
        row[key] = np.nan
    row["support_status"] = "infeasible_under_strict_support" if support.name == "strict_joint" else "infeasible"
    row["extrapolation_flag"] = False
    return row


def solve_2d(budget: float, context: int, eta: float, slope: float, spec: CostSpec,
             params: ScalingParameters, support: SupportBounds, eps: float = 1e-6,
             q_cap: float = 1.0, grid_n: int = 45, grid_q: int = 35,
             multistarts: int = 20, starts: Iterable[tuple[float, float]] | None = None,
             fast: bool = False) -> dict:
    q_hi = quality_upper_bound(params.Q0, slope, eps, q_cap)
    min_cost = total_costs(support.n_min, support.d_min, params.Q0, params.Q0, context, eta, spec)["total"]
    if min_cost > budget * (1.0 + 1e-12):
        return infeasible_record(budget, context, eta, slope, spec, params, support,
                                 "minimum support point exceeds budget")

    log_bounds = (np.log(support.n_min), np.log(support.n_max))

    def recover(x: np.ndarray) -> tuple[float, float, float, float]:
        n = float(np.exp(x[0])); q = float(x[1])
        d_budget = _budget_d_billion(budget, n, q, params.Q0, context, eta, spec)
        return n, min(support.d_max, d_budget), q, d_budget

    def objective(x: np.ndarray) -> float:
        n, d, q, d_budget = recover(x)
        if d_budget < support.d_min:
            return 1e6 + 1e5 * (support.d_min - d_budget) / support.d_min
        return float(predict_loss(n, d, q, params, slope))

    def feasible_constraint(x: np.ndarray) -> float:
        return recover(x)[3] - support.d_min

    if fast and starts:
        # Warm-started path/bootstrap solves need only a compact guard grid.
        n_points, q_points = 7, 5
    else:
        n_points = max(17, grid_n // 2) if fast else grid_n
        q_points = max(13, grid_q // 2) if fast else grid_q
    n_grid = np.geomspace(support.n_min, support.n_max, n_points)
    q_grid = np.linspace(params.Q0, q_hi, q_points)
    candidates: list[tuple[float, float, float]] = []
    for n in n_grid:
        for q in q_grid:
            d_budget = _budget_d_billion(budget, float(n), float(q), params.Q0, context, eta, spec)
            if d_budget + 1e-12 < support.d_min:
                continue
            d = min(support.d_max, d_budget)
            candidates.append((float(predict_loss(n, d, q, params, slope)), float(n), float(q)))
    if not candidates:
        return infeasible_record(budget, context, eta, slope, spec, params, support, "no feasible grid point")
    candidates.sort(key=lambda value: value[0])
    initial = [(np.log(n), q) for _, n, q in candidates[:max(3, min(multistarts, len(candidates)))]]
    if starts:
        for n, q in starts:
            if support.n_min <= n <= support.n_max and params.Q0 <= q <= q_hi:
                initial.insert(0, (np.log(n), q))
    # De-duplicate while preserving order.
    unique: list[tuple[float, float]] = []
    seen = set()
    for item in initial:
        key = (round(item[0], 10), round(item[1], 10))
        if key not in seen:
            unique.append(item); seen.add(key)
    if fast:
        unique = unique[:3]

    best = None
    for x0 in unique:
        result = minimize(objective, np.asarray(x0), method="SLSQP",
                          bounds=[log_bounds, (params.Q0, q_hi)],
                          constraints=[{"type": "ineq", "fun": feasible_constraint}],
                          options={"ftol": 1e-12, "maxiter": 300, "disp": False})
        n, d, q, d_budget = recover(result.x)
        if d_budget + 1e-8 < support.d_min:
            continue
        value = float(predict_loss(n, d, q, params, slope))
        if best is None or value < best[0]:
            best = (value, n, d, q, result)
    if best is None:
        return infeasible_record(budget, context, eta, slope, spec, params, support, "all local refinements infeasible")

    _, n, d, q, result = best
    # Local fine grid is an independent numerical guard around the optimizer.
    n_lo = max(support.n_min, n / 1.025); n_hi = min(support.n_max, n * 1.025)
    q_span = max((q_hi - params.Q0) * 0.025, 2e-5)
    q_lo = max(params.Q0, q - q_span); q_hi_local = min(q_hi, q + q_span)
    fine_best = float(best[0]) if fast else np.inf
    if not fast:
        for nn in np.geomspace(n_lo, n_hi, 31):
            for qq in np.linspace(q_lo, q_hi_local, 31):
                db = _budget_d_billion(budget, float(nn), float(qq), params.Q0, context, eta, spec)
                if db + 1e-12 >= support.d_min:
                    fine_best = min(fine_best, float(predict_loss(nn, min(support.d_max, db), qq, params, slope)))
    return _record_solution(n, d, q, budget, context, eta, slope, spec, params, support,
                            "SLSQP-2D", bool(result.success), result.message,
                            grid_loss=candidates[0][0], fine_grid_loss=float(fine_best))


def solve_3d(reference: dict, budget: float, context: int, eta: float, slope: float,
             spec: CostSpec, params: ScalingParameters, support: SupportBounds,
             eps: float = 1e-6, q_cap: float = 1.0) -> dict:
    if not reference.get("solver_success", False):
        return dict(reference)
    q_hi = quality_upper_bound(params.Q0, slope, eps, q_cap)

    def unpack(x: np.ndarray) -> tuple[float, float, float]:
        return float(np.exp(x[0])), float(np.exp(x[1])), float(x[2])

    def objective(x: np.ndarray) -> float:
        n, d, q = unpack(x)
        return float(predict_loss(n, d, q, params, slope))

    def constraint(x: np.ndarray) -> float:
        n, d, q = unpack(x)
        return (budget - total_costs(n, d, q, params.Q0, context, eta, spec)["total"]) / budget

    starts = [
        np.array([np.log(reference["N_B"]), np.log(reference["D_B"]), reference["Q_A"]]),
        np.array([np.log(np.sqrt(support.n_min * support.n_max)),
                  np.log(np.sqrt(support.d_min * support.d_max)), params.Q0]),
    ]
    best = None
    bounds = [(np.log(support.n_min), np.log(support.n_max)),
              (np.log(support.d_min), np.log(support.d_max)), (params.Q0, q_hi)]
    for x0 in starts:
        if constraint(x0) < 0:
            x0 = starts[0]
        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                          constraints=[{"type": "ineq", "fun": constraint}],
                          options={"ftol": 1e-12, "maxiter": 500, "disp": False})
        n, d, q = unpack(result.x)
        if constraint(result.x) < -1e-8:
            continue
        value = objective(result.x)
        if best is None or value < best[0]:
            best = (value, n, d, q, result)
    if best is None:
        return infeasible_record(budget, context, eta, slope, spec, params, support, "3D verification failed")
    _, n, d, q, result = best
    return _record_solution(n, d, q, budget, context, eta, slope, spec, params, support,
                            "SLSQP-3D", bool(result.success), result.message)


def baseline_q0_solution(budget: float, context: int, eta: float, spec: CostSpec,
                         params: ScalingParameters, support: SupportBounds) -> dict:
    return solve_2d(budget, context, eta, 1.0, spec, params, support, q_cap=params.Q0,
                    grid_n=61, grid_q=1, multistarts=8, fast=True)


def kkt_diagnostics(row: dict, params: ScalingParameters, spec: CostSpec,
                    support: SupportBounds, slope: float, eps: float = 1e-6) -> dict:
    if not row.get("solver_success", False):
        return {"solver_success": False}
    n, d, q = float(row["N_B"]), float(row["D_B"]), float(row["Q_A"])
    deriv = loss_derivatives_absolute(n, d, q, params, slope)
    n_abs, d_abs = n * 1e9, d * 1e9
    delta_g = float(quality_cost_increment(q, params.Q0, spec))
    c_n = (6.0 + row["eta"] * row["context_length"]) * d_abs
    c_d = (6.0 + row["eta"] * row["context_length"]) * n_abs + delta_g
    c_q = d_abs * float(quality_cost_derivative(q, spec))
    ratios = {
        "marginal_N": -deriv["dL_dN_abs"] / c_n,
        "marginal_D": -deriv["dL_dD_abs"] / c_d,
        "marginal_Q": -deriv["dL_dQ"] / c_q,
    }
    q_hi = quality_upper_bound(params.Q0, slope, eps)
    tol = 2e-6
    active = []
    if q <= params.Q0 + tol: active.append("Q_lower")
    if q >= q_hi - tol: active.append("Q_upper")
    if n <= support.n_min * (1 + tol): active.append("N_lower")
    if n >= support.n_max * (1 - tol): active.append("N_upper")
    if d <= support.d_min * (1 + tol): active.append("D_lower")
    if d >= support.d_max * (1 - tol): active.append("D_upper")
    if abs(row["budget_relative_slack"]) <= 1e-6: active.append("budget")
    interior_values = [ratios["marginal_N"], ratios["marginal_D"]]
    if "Q_lower" not in active and "Q_upper" not in active:
        interior_values.append(ratios["marginal_Q"])
    mean = float(np.mean(interior_values)) if interior_values else np.nan
    residual = float(np.max(np.abs(np.asarray(interior_values) - mean)) / max(abs(mean), 1e-300)) if interior_values else np.nan
    return {**ratios, **deriv, "kkt_relative_dispersion": residual,
            "active_set": "|".join(active) if active else "interior", "solver_success": True}
