from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .classic import predict as classic_predict
from .common import json_dump
from .problem1_bridge import scheffe_features
from .quality import predict_quality_delta


def h_quality(q: np.ndarray | float, q_ref: float, slope: float, eps: float) -> np.ndarray:
    return np.clip(q_ref + slope * (np.asarray(q, float) - q_ref), eps, 1.0)


def load_mixture_response_artifact(path: str | Path) -> tuple[list[str], Callable[[np.ndarray], np.ndarray]]:
    """Load the frozen point-estimate R(p) without training data or fitted Python objects."""
    with np.load(Path(path)) as z:
        domains = [str(x) for x in z["domains"]]
        coef = z["coefficient_point"].astype(float)
        iqr = z["loss_iqr"].astype(float)
        j_ref = float(z["j_ref_point"])
        r_scale = float(z["r_scale_point"])

    def response(p: np.ndarray) -> np.ndarray:
        x, _ = scheffe_features(np.atleast_2d(np.asarray(p, float)), domains)
        j = np.mean((x @ coef.T) / iqr[None, :], axis=1)
        return (j - j_ref) / r_scale

    return domains, response


def generalized_predict(
    classic_model: str, classic_params: dict[str, float], quality_summary: dict[str, Any],
    n: np.ndarray, d: np.ndarray, q_a: np.ndarray, p: np.ndarray,
    q_ref: float, s_q: float, lambda0: float, delta_p: float, n0: float,
    mixture_response: Callable[[np.ndarray], np.ndarray], eps: float = 1e-6,
) -> np.ndarray:
    """Callable repaired model; p and both scenario controls affect the numerical path."""
    n, d, q_a = np.asarray(n, float), np.asarray(d, float), np.asarray(q_a, float)
    p = np.atleast_2d(np.asarray(p, float))
    if p.shape[0] not in {1, n.size}:
        raise ValueError("p must have one row or match n")
    q_eff = h_quality(q_a, q_ref, s_q, eps)
    base = classic_predict(classic_model, classic_params, n, d)
    qterm = predict_quality_delta(quality_summary["selected_model"], quality_summary["parameters"], classic_params, n, d, q_eff)
    rval = np.asarray(mixture_response(p), float)
    if rval.size == 1:
        rval = np.repeat(rval, n.size)
    return base + qterm + float(lambda0) * (n / float(n0)) ** float(delta_p) * rval


def _support_scalar(value: float, observed: tuple[float, float], extended: tuple[float, float] | None = None) -> str:
    if observed[0] <= value <= observed[1]:
        return "observed_or_interpolated"
    if extended is not None and extended[0] <= value <= extended[1]:
        return "extrapolated_supported_by_boundary_only"
    return "outside_supported_boundary"


def _p_support(p: np.ndarray, bridge: dict[str, Any]) -> str:
    p = np.asarray(p, float)
    if np.any(p < -1e-10) or abs(float(p.sum()) - 1.0) > 1e-8:
        return "invalid_simplex"
    upper = np.array(list(bridge["bridge"]["mixture_support"]["upper"].values()))
    if np.any(p > upper + 1e-10):
        return "outside_component_upper_bound"
    scale = np.array(list(bridge["bridge"]["mixture_support"]["scale"].values()))
    dist = np.min(np.sqrt(np.sum(((bridge["p_train"] - p) / scale) ** 2, axis=1)))
    return "within_registered_support" if dist <= float(bridge["bridge"]["mixture_support"]["radius"]) + 1e-10 else "outside_registered_support"


def _derivatives(classic_model: str, cp: dict[str, float], qs: dict[str, Any], n: float, d: float,
                 q: float, q_ref: float, s_q: float, lambda0: float, delta_p: float,
                 n0: float, rval: float, eps: float) -> dict[str, float | str]:
    if classic_model == "M0":
        dn = -cp["alpha"] * cp["A"] * n ** (-cp["alpha"] - 1)
        dd = -cp["beta"] * cp["B"] * d ** (-cp["beta"] - 1)
    else:
        dn = dd = np.nan
    mix_dn = 0.0 if lambda0 == 0 or delta_p == 0 else lambda0 * rval * delta_p * (n / n0) ** delta_p / n
    dn = float(dn + mix_dn) if np.isfinite(dn) else np.nan
    if qs["selected_model"] == "GQ2":
        c, nu = qs["parameters"]["c_Q"], qs["parameters"]["nu_Q"]
        qeff = float(h_quality(q, q_ref, s_q, eps))
        active = eps < q_ref + s_q * (q - q_ref) < 1.0
        dq = -c * nu * (1.0 - qeff) ** (nu - 1.0) * s_q if active else 0.0
        role = "analytic"
    else:
        dq, role = np.nan, "finite_difference_required"
    return {"dL_dN_analytic": dn, "dL_dD_analytic": float(dd), "dL_dQ_analytic": float(dq), "analytic_role": role}


def _finite_diff(predictor: Callable[[float, float, float], float], n: float, d: float, q: float, eps: float) -> tuple[float, float, float]:
    h = 1e-5
    dn = (predictor(n * (1 + h), d, q) - predictor(n * (1 - h), d, q)) / (2 * h * n)
    dd = (predictor(n, d * (1 + h), q) - predictor(n, d * (1 - h), q)) / (2 * h * d)
    qlo, qhi = max(eps, q - h), min(1.0, q + h)
    dq = (predictor(n, d, qhi) - predictor(n, d, qlo)) / (qhi - qlo)
    return dn, dd, dq


def _equivalent_root(predictor: Callable[[float, float, float], float], axis: str, n: float, d: float,
                     q: float, dq: float, observed_max: float, extended_max: float) -> tuple[float | None, str, float | None]:
    q2 = min(1.0, q + dq)
    if q2 <= q:
        return None, "target_q_out_of_support", None
    target, base = predictor(n, d, q2), predictor(n, d, q)
    if target >= base:
        return None, "quality_improvement_not_lower_loss", None
    origin = n if axis == "N" else d
    def f(value: float) -> float:
        return predictor(value, d, q) - target if axis == "N" else predictor(n, value, q) - target
    if origin >= extended_max or f(extended_max) > 0:
        return None, "no_root_within_extended_support", None
    root = float(brentq(f, origin, extended_max, xtol=1e-11, rtol=1e-11))
    status = "ok_observed" if root <= observed_max else "ok_extrapolated_supported_by_boundary_only"
    return root / origin - 1.0, status, abs(f(root))
def run_generalized(classic: dict[str, Any], quality: dict[str, Any], bridge: dict[str, Any],
                    data: dict[str, Any], cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    selected, qs, b7 = classic["selected"], quality["summary"], quality["b7"]
    cp = selected.params; q_ref = float(bridge["bridge"]["Q_ref"]); eps = float(cfg.get("quality_eps", 1e-6))
    n0 = float(np.median(b7["N_params_B"])); loss_iqr = float(np.subtract(*np.quantile(b7["val_loss"], [.75, .25])))
    profiles = list(map(float, bridge["bridge"]["mixture_strength_profiles"]))
    lambdas = [loss_iqr * x for x in profiles]
    delta_p = float(bridge["bridge"]["delta_p"])
    supports = data["audit"]["checks"]
    obs_n = tuple(supports["support_observed"]["N_params_B"]); obs_d = tuple(supports["support_observed"]["D_tokens_B"])
    obs_q = tuple(supports["support_observed"]["Q"]); ext_n = tuple(supports["support_extended_B9"]["N_params_B"])
    ext_d = tuple(supports["support_extended_B9"]["D_tokens_B"])

    p_cases = [("p_ref", bridge["p_ref"]), ("p_star_sensitivity", bridge["p_star"])]
    rows, elastic_rows = [], []
    for n in np.quantile(b7["N_params_B"], [.1, .5, .9]):
        for d in np.quantile(b7["D_tokens_B"], [.1, .5, .9]):
            for q in [.4, q_ref, .8]:
                for p_name, p in p_cases:
                    rval = float(bridge["R_of"](p)[0]); pstate = _p_support(p, bridge)
                    for s_q in cfg.get("quality_mapping_slopes", [.5, 1, 1.5]):
                        for profile, lambda0 in zip(profiles, lambdas):
                            pred_fun = lambda nn, dd, qq, p=p, sq=float(s_q), lam=lambda0: float(generalized_predict(
                                selected.model, cp, qs, np.array([nn]), np.array([dd]), np.array([qq]), p,
                                q_ref, sq, lam, delta_p, n0, bridge["R_of"], eps)[0])
                            pred = pred_fun(float(n), float(d), float(q))
                            rec = {"N_params_B": float(n), "D_tokens_B": float(d), "Q_A": float(q),
                                "p_scenario": p_name, "s_Q": float(s_q), "mixture_strength_profile": profile,
                                "lambda0_B_loss_units": lambda0, "R_p": rval, "prediction": pred,
                                "irreducible_loss_E": float(cp.get("E", np.nan)),
                                "physical_lower_bound_status": "PASS" if pred >= float(cp.get("E", -np.inf)) - 1e-12 else "VIOLATION",
                                "N_support": _support_scalar(float(n), obs_n, ext_n),
                                "D_support": _support_scalar(float(d), obs_d, ext_d),
                                "Q_support": _support_scalar(float(q), obs_q), "p_support": pstate,
                                "support_flag": "in_support" if pstate == "within_registered_support" else "sensitivity_or_outside",
                                "evidence_type": "modeled_scenario"}
                            rows.append(rec)
                            dn, ddv, dqv = _finite_diff(pred_fun, float(n), float(d), float(q), eps)
                            deriv = _derivatives(selected.model, cp, qs, float(n), float(d), float(q), q_ref,
                                float(s_q), lambda0, delta_p, n0, rval, eps)
                            total_denom = max(pred, 1e-12)
                            reducible_denom_raw = pred - cp.get("E", 0.0)
                            reducible_valid = reducible_denom_raw > 1e-12
                            reducible_denom = reducible_denom_raw if reducible_valid else np.nan
                            epsilon_total = {
                                "N": -float(n) * dn / total_denom,
                                "D": -float(d) * ddv / total_denom,
                                "Q": -float(q) * dqv / total_denom,
                            }
                            epsilon_reducible = {
                                "N": -float(n) * dn / reducible_denom if reducible_valid else np.nan,
                                "D": -float(d) * ddv / reducible_denom if reducible_valid else np.nan,
                                "Q": -float(q) * dqv / reducible_denom if reducible_valid else np.nan,
                            }
                            elastic_rows.append({**rec, **deriv, "dL_dN": dn, "dL_dD": ddv, "dL_dQ": dqv,
                                "epsilon_N": epsilon_total["N"], "epsilon_D": epsilon_total["D"], "epsilon_Q": epsilon_total["Q"],
                                "epsilon_N_total": epsilon_total["N"], "epsilon_D_total": epsilon_total["D"], "epsilon_Q_total": epsilon_total["Q"],
                                "epsilon_N_reducible": epsilon_reducible["N"], "epsilon_D_reducible": epsilon_reducible["D"], "epsilon_Q_reducible": epsilon_reducible["Q"],
                                "reducible_elasticity_status": "defined" if reducible_valid else "undefined_L_le_E",
                                "max_analytic_fd_abs_error": float(np.nanmax(np.abs(np.array([deriv["dL_dN_analytic"], deriv["dL_dD_analytic"], deriv["dL_dQ_analytic"]], float) - np.array([dn, ddv, dqv]))))})
    pred_df, elast = pd.DataFrame(rows), pd.DataFrame(elastic_rows)
    pred_df.to_csv(output_dir / "generalized_scaling_predictions.csv.gz", index=False, compression="gzip")
    elast.to_csv(output_dir / "elasticity_grid.csv", index=False)
    nonzero_mix = pred_df[pred_df["mixture_strength_profile"] > 0].copy()
    violation_mask = nonzero_mix["prediction"] < float(cp.get("E", -np.inf)) - 1e-12
    physical_by_scenario = (
        nonzero_mix.groupby(["p_scenario", "mixture_strength_profile"], as_index=False)
        .agg(rows=("prediction", "size"), min_prediction=("prediction", "min"),
             below_E_rows=("physical_lower_bound_status", lambda x: int((x == "VIOLATION").sum())))
    )
    physical_by_scenario.to_csv(output_dir / "mixture_physical_gate.csv", index=False)
    physical_gate = {
        "E": float(cp.get("E", np.nan)),
        "nonzero_scenario_rows": int(len(nonzero_mix)),
        "below_E_rows": int(violation_mask.sum()),
        "min_nonzero_scenario_prediction": float(nonzero_mix["prediction"].min()) if len(nonzero_mix) else None,
        "status": "PASS" if not bool(violation_mask.any()) else "BLOCK_JOINT_P",
        "action": "joint_p_disabled_if_any_prediction_below_E",
    }
    json_dump(output_dir / "mixture_physical_gate.json", physical_gate)

    eq_rows = []
    nmid, dmid = float(np.median(b7["N_params_B"])), float(np.median(b7["D_tokens_B"]))
    for s_q in cfg.get("quality_mapping_slopes", [.5, 1, 1.5]):
        pred_fun = lambda nn, dd, qq, sq=float(s_q): float(generalized_predict(selected.model, cp, qs,
            np.array([nn]), np.array([dd]), np.array([qq]), bridge["p_ref"], q_ref, sq, 0.0,
            delta_p, n0, bridge["R_of"], eps)[0])
        for q0 in [.4, q_ref, .8]:
            for delta_q in [.05, .1]:
                rn, sn, en = _equivalent_root(pred_fun, "N", nmid, dmid, q0, delta_q, obs_n[1], ext_n[1])
                rd, sd, ed = _equivalent_root(pred_fun, "D", nmid, dmid, q0, delta_q, obs_d[1], ext_d[1])
                _, _, dqd = _finite_diff(pred_fun, nmid, dmid, q0, eps)
                dlogn = nmid * _finite_diff(pred_fun, nmid, dmid, q0, eps)[0]
                local = -dqd * delta_q / max(-dlogn, 1e-12)
                eq_rows.append({"scenario": "p_ref_lambda0", "s_Q": float(s_q), "Q_A": q0, "delta_Q": delta_q,
                    "local_N_log_equiv": local, "finite_N_multiplier_minus_1": rn,
                    "finite_D_multiplier_minus_1": rd, "root_status_N": sn, "root_status_D": sd,
                    "root_backcheck_error_N": en, "root_backcheck_error_D": ed})
    equiv = pd.DataFrame(eq_rows); equiv.to_csv(output_dir / "quality_parameter_equivalence.csv", index=False)
    units_contract = {
        "N_params_B_unit": "1e9 parameters",
        "D_tokens_B_unit": "1e9 tokens",
        "N_abs_multiplier": 1e9,
        "D_abs_multiplier": 1e9,
        "FLOPs_formula_absolute": "6 * N_abs * D_abs",
        "FLOPs_to_C_FLOPs_1e21": "6 * N_params_B * D_tokens_B / 1000",
        "N0_params_B": n0,
        "N0_abs": n0 * 1e9,
        "internal_scaling_units": "billions for N and D; convert before cost optimization",
        "problem3_contract": "Use N_abs and D_abs for 6ND, quality investment, and attention-cost terms.",
    }
    json_dump(output_dir / "units_contract.json", units_contract)
    # Joint outer draws: independently resampled classic, quality, mapping, Scheffe and transfer components.
    cboot = classic["bootstrap"]; cboot = cboot[cboot["success"].astype(bool)].reset_index(drop=True)
    qboot = quality["bootstrap"]; qboot = qboot[qboot["success"].astype(bool)].reset_index(drop=True)
    tboot = bridge["transfer_bootstrap"]; tboot = tboot[tboot["valid"].astype(bool)].reset_index(drop=True)
    reps = int(cfg.get("bootstrap_reps", 500)); s_values = list(map(float, cfg.get("quality_mapping_slopes", [.5, 1, 1.5])))
    joint_rows = []
    for b in range(reps):
        try:
            cr, qr, tr = cboot.iloc[b % len(cboot)], qboot.iloc[b % len(qboot)], tboot.iloc[b % len(tboot)]
            cpd = {k: float(cr[k]) for k in cp}
            qpd = {k: float(qr[k]) for k in qs["parameters"]}
            qsd = {**qs, "parameters": qpd}
            sq = s_values[b % len(s_values)]; profile = profiles[b % len(profiles)]; lambda0 = loss_iqr * profile
            map_id = b % len(bridge["mapping_draws"]); scheffe_id = b % len(bridge["coef_draws"])
            near_id = b % len(bridge["near_optimal_p"])
            p_source = "p_ref_draw" if b % 2 == 0 else "near_optimal_sensitivity"
            pdraw = bridge["p_ref_draws"][scheffe_id].astype(float) if b % 2 == 0 else bridge["near_optimal_p"][near_id]
            qrefd = float(bridge["p_ref_draws"][scheffe_id] @ bridge["mapping_draws"][map_id])
            qbasep = float(pdraw @ bridge["mapping_draws"][map_id])
            rfun = lambda pp, sid=scheffe_id: bridge["R_of"](pp, sid)
            rval = float(rfun(pdraw)[0]); deltad = float(tr["delta_p"])
            pred_fun = lambda nn, dd, qq: float(generalized_predict(selected.model, cpd, qsd,
                np.array([nn]), np.array([dd]), np.array([qq]), pdraw, qrefd, sq, lambda0,
                deltad, n0, rfun, eps)[0])
            qeval = max(.4, min(.8, qrefd + .1)); pred = pred_fun(nmid, dmid, qeval)
            dn, ddv, dqv = _finite_diff(pred_fun, nmid, dmid, qeval, eps)
            rn, sn, en = _equivalent_root(pred_fun, "N", nmid, dmid, qeval, .05, obs_n[1], ext_n[1])
            rd, sd, ed = _equivalent_root(pred_fun, "D", nmid, dmid, qeval, .05, obs_d[1], ext_d[1])
            rec = {"draw": b, "success": True, "failure_stage": "", "reason": "",
                "classic_draw_id": int(cr["draw"]), "quality_draw_id": int(qr["draw"]),
                "mapping_draw_id": map_id, "scheffe_draw_id": scheffe_id,
                "mixture_draw_id": int(tr["draw"]), "near_optimal_draw_id": near_id,
                "s_Q": sq, "mixture_strength_scenario": profile, "lambda0_B_loss_units": lambda0,
                "p_source": p_source, "Q_ref_draw": qrefd, "Q_base_p_draw": qbasep, "R_p_draw": rval,
                "delta_p_draw": deltad, "prediction": pred, "below_irreducible_E": bool(pred < cpd.get("E", -np.inf) - 1e-12),
                "dL_dN": dn, "dL_dD": ddv, "dL_dQ": dqv,
                "finite_N_multiplier_minus_1": rn, "finite_D_multiplier_minus_1": rd,
                "root_status_N": sn, "root_status_D": sd, "root_backcheck_error_N": en, "root_backcheck_error_D": ed}
            rec.update({f"classic_{k}": v for k, v in cpd.items()}); rec.update({f"quality_{k}": v for k, v in qpd.items()})
            joint_rows.append(rec)
        except Exception as exc:  # pragma: no cover - retained as an auditable failed draw.
            joint_rows.append({"draw": b, "success": False, "failure_stage": "joint_prediction", "reason": exc.__class__.__name__,
                "classic_draw_id": b, "quality_draw_id": b, "mapping_draw_id": b, "scheffe_draw_id": b,
                "mixture_draw_id": b, "near_optimal_draw_id": b, "s_Q": s_values[b % len(s_values)],
                "mixture_strength_scenario": profiles[b % len(profiles)]})
    boot = pd.DataFrame(joint_rows)
    boot.to_csv(output_dir / "generalized_scaling_bootstrap.csv.gz", index=False, compression="gzip")

    params = {"classic_model": selected.model, "classic_parameters": cp,
        "quality_model": qs["selected_model"], "quality_parameters": qs["parameters"],
        "quality_fit_mode": qs["fit_mode"], "mixture_reference": "p_ref",
        "Q_ref": q_ref, "N0_params_B": n0, "delta_p": delta_p,
        "quality_mapping_slopes": s_values,
        "mixture_strength_profiles": [{"profile": p, "lambda0_B_loss_units": l,
            "conversion": "profile_multiplier_times_B7_loss_IQR", "identified": False} for p, l in zip(profiles, lambdas)],
        "final_formula_role": "L0_plus_quality_hs_plus_scale_dependent_Rp",
        "callable_signature": "generalized_predict(..., q_a, p, q_ref, s_q, lambda0, delta_p, n0, mixture_response)",
        "mixture_response_artifact": "problem1_mixture_response_bootstrap.npz",
        "mixture_response_loader": "problem2.generalized.load_mixture_response_artifact",
        "units_contract": "units_contract.json",
        "mixture_strength_role": bridge["bridge"]["kappa0_or_lambda0_role"],
        "elasticity_contract": {
            "epsilon_x": "standard_total_loss=-x/L*dL_dx",
            "epsilon_x_total": "standard_total_loss=-x/L*dL_dx",
            "epsilon_x_reducible": "mechanism_sensitivity=-x/(L-E)*dL_dx; undefined when L<=E",
        },
        "mixture_physical_gate": physical_gate,
        "joint_bootstrap_valid_rate": float(boot["success"].mean()), "interface_state": "PENDING_ACCEPTANCE",
        "support": {"N_observed": obs_n, "N_extended_B9": ext_n, "D_observed": obs_d,
            "D_extended_B9": ext_d, "Q": obs_q, "p": bridge["bridge"]["mixture_support"]}}
    json_dump(output_dir / "generalized_scaling_parameters.json", params)
    return {"parameters": params, "predictions": pred_df, "elasticity": elast, "equivalence": equiv, "bootstrap": boot, "physical_gate": physical_gate, "physical_by_scenario": physical_by_scenario}
