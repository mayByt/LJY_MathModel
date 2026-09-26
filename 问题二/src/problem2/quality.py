from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .classic import FitResult, predict as classic_predict
from .common import json_dump, mae, rmse, spearman
from .data import b7_splits


def _quality_table(b7: pd.DataFrame, classic_model: str, classic_params: dict[str, float]) -> pd.DataFrame:
    qmax = b7["Q_score"].max()
    anchor = b7[np.isclose(b7["Q_score"], qmax)][["N_params_B", "D_tokens_B", "val_loss"]].rename(columns={"val_loss": "loss_q1"})
    out = b7.merge(anchor, on=["N_params_B", "D_tokens_B"], how="left")
    out["delta_loss"] = out["val_loss"] - out["loss_q1"]
    out["classic_prediction"] = classic_predict(
        classic_model,
        classic_params,
        out["N_params_B"].to_numpy(float),
        out["D_tokens_B"].to_numpy(float),
    )
    out["residual_to_classic"] = out["val_loss"] - out["classic_prediction"]
    return out


def _quality_delta(model: str, theta: np.ndarray, n: np.ndarray, d: np.ndarray, q: np.ndarray, classic_params: dict[str, float]) -> np.ndarray:
    q = np.maximum(np.asarray(q, float), 1e-9)
    n = np.asarray(n, float)
    d = np.asarray(d, float)
    A = classic_params["A"]
    B = classic_params["B"]
    alpha = classic_params["alpha"]
    beta = classic_params["beta"]
    if model == "GQ1":
        gamma = np.exp(theta[0])
        return B * d ** (-beta) * (q ** (-gamma) - 1.0)
    if model == "GQ2":
        c = np.exp(theta[0])
        nu = np.exp(theta[1])
        return c * (1.0 - q) ** nu
    if model == "GQ3":
        bq = np.exp(theta[0])
        beta1 = theta[1]
        term = B * np.exp(bq * (1.0 - q)) * d ** (-(beta + beta1 * (1.0 - q)))
        anchor = B * d ** (-beta)
        return term - anchor
    if model == "GQ4":
        gamma_n = np.exp(theta[0])
        gamma_d = np.exp(theta[1])
        return A * n ** (-alpha) * (q ** (-alpha * gamma_n) - 1.0) + B * d ** (-beta) * (q ** (-beta * gamma_d) - 1.0)
    raise ValueError(model)


def _initials(model: str, starts: int, rng: np.random.Generator) -> list[np.ndarray]:
    values = []
    if model in {"GQ1"}:
        values.append(np.log([1.0]))
        values.extend(np.log(rng.uniform(0.05, 5.0, size=(starts - 1, 1))))
    elif model == "GQ2":
        values.append(np.log([0.5, 1.0]))
        for _ in range(starts - 1):
            values.append(np.log([rng.uniform(0.01, 5.0), rng.uniform(0.1, 5.0)]))
    elif model == "GQ3":
        values.append(np.array([np.log(0.5), 0.0]))
        for _ in range(starts - 1):
            values.append(np.array([np.log(rng.uniform(0.01, 3.0)), rng.uniform(-0.5, 0.5)]))
    elif model == "GQ4":
        values.append(np.log([0.2, 1.0]))
        for _ in range(starts - 1):
            values.append(np.log([rng.uniform(0.001, 3.0), rng.uniform(0.001, 5.0)]))
    return [np.asarray(v, float) for v in values]


def _param_dict(model: str, theta: np.ndarray) -> dict[str, float]:
    if model == "GQ1":
        return {"gamma": float(np.exp(theta[0]))}
    if model == "GQ2":
        return {"c_Q": float(np.exp(theta[0])), "nu_Q": float(np.exp(theta[1]))}
    if model == "GQ3":
        return {"b_Q": float(np.exp(theta[0])), "beta1": float(theta[1])}
    if model == "GQ4":
        return {"gamma_N": float(np.exp(theta[0])), "gamma_D": float(np.exp(theta[1]))}
    raise ValueError(model)


def fit_quality(df: pd.DataFrame, model: str, classic_params: dict[str, float], cfg: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, Any], pd.DataFrame]:
    n = df["N_params_B"].to_numpy(float)
    d = df["D_tokens_B"].to_numpy(float)
    q = df["Q_score"].to_numpy(float)
    y = df["residual_to_classic"].to_numpy(float)
    starts = int(cfg.get("quality_multistarts", 24))
    rows = []
    best: dict[str, Any] | None = None
    for i, theta0 in enumerate(_initials(model, starts, rng)):
        def residual(theta: np.ndarray) -> np.ndarray:
            pred = _quality_delta(model, theta, n, d, q, classic_params)
            return pred - y

        if model == "GQ3":
            beta = float(classic_params["beta"])
            bounds = (np.array([np.log(1e-6), -beta / 0.9 + 1e-6]), np.array([np.log(20.0), 1.0]))
            theta0 = np.clip(theta0, bounds[0] + 1e-9, bounds[1] - 1e-9)
        else:
            bounds = (-20.0, np.log(20.0))
        res = least_squares(residual, theta0, bounds=bounds, max_nfev=3000, xtol=1e-10, ftol=1e-10, gtol=1e-8)
        pred = _quality_delta(model, res.x, n, d, q, classic_params)
        cost = float(np.mean((pred - y) ** 2))
        params = _param_dict(model, res.x)
        rows.append({"model": model, "start": i, "success": bool(res.success), "cost": cost, "nfev": int(res.nfev), **params})
        if best is None or cost < best["cost"]:
            best = {"model": model, "success": bool(res.success), "cost": cost, "theta": res.x, "params": params}
    assert best is not None
    return best, pd.DataFrame(rows)


def predict_quality_delta(model: str, params: dict[str, float], classic_params: dict[str, float], n: np.ndarray, d: np.ndarray, q: np.ndarray) -> np.ndarray:
    if model == "GQ1":
        theta = np.log([params["gamma"]])
    elif model == "GQ2":
        theta = np.log([params["c_Q"], params["nu_Q"]])
    elif model == "GQ3":
        theta = np.array([np.log(params["b_Q"]), params["beta1"]])
    elif model == "GQ4":
        theta = np.log([params["gamma_N"], params["gamma_D"]])
    else:
        raise ValueError(model)
    return _quality_delta(model, theta, n, d, q, classic_params)


def _monotonic_violations(model: str, params: dict[str, float], classic_params: dict[str, float], b7: pd.DataFrame) -> int:
    count = 0
    for (n, d), g in b7.groupby(["N_params_B", "D_tokens_B"]):
        gg = g.sort_values("Q_score")
        pred = predict_quality_delta(model, params, classic_params, gg["N_params_B"].to_numpy(float), gg["D_tokens_B"].to_numpy(float), gg["Q_score"].to_numpy(float))
        count += int((np.diff(pred) > 1e-10).sum())
    return count


def _cv_model(b7: pd.DataFrame, model: str, classic_model: str, classic_params: dict[str, float], cfg: dict[str, Any], rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    loss_iqr = float(np.subtract(*np.quantile(b7["val_loss"], [0.75, 0.25])))
    loss_iqr = max(loss_iqr, 1e-12)
    for split in b7_splits(b7):
        mask = split["mask"]
        train = b7.loc[~mask].copy()
        valid = b7.loc[mask].copy()
        fit, _ = fit_quality(train, model, classic_params, cfg, rng)
        delta_pred = predict_quality_delta(
            model,
            fit["params"],
            classic_params,
            valid["N_params_B"].to_numpy(float),
            valid["D_tokens_B"].to_numpy(float),
            valid["Q_score"].to_numpy(float),
        )
        base_pred = classic_predict(
            classic_model,
            classic_params,
            valid["N_params_B"].to_numpy(float),
            valid["D_tokens_B"].to_numpy(float),
        )
        pred_abs = base_pred + delta_pred
        y_abs = valid["val_loss"].to_numpy(float)
        group_spearman = []
        for (_, _), g in valid.assign(predicted=pred_abs).groupby(["N_params_B", "D_tokens_B"]):
            if len(g) >= 3:
                group_spearman.append(spearman(g["Q_score"].to_numpy(float), g["predicted"].to_numpy(float)))
        paired_applicable = not (split["split_type"] == "leave_Q" and np.isclose(float(split["held_out"]), 1.0))
        paired_rmse = float("nan")
        if paired_applicable:
            paired_rmse = rmse(valid["delta_loss"].to_numpy(float), delta_pred)
        rows.append(
            {
                "model": model,
                "split_type": split["split_type"],
                "held_out_level": split["held_out"],
                "q_position": split.get("q_position", ""),
                "RMSE": rmse(y_abs, pred_abs),
                "MAE": mae(y_abs, pred_abs),
                "nRMSE": rmse(y_abs, pred_abs) / loss_iqr,
                "paired_delta_RMSE": paired_rmse,
                "paired_delta_status": "applicable" if paired_applicable else "not_applicable_Q1_held_out",
                "absolute_prediction_source": "classic_B1_plus_fold_trained_quality",
                "validation_loss_used_as_baseline": False,
                "group_spearman_median": float(np.nanmedian(group_spearman)) if group_spearman else np.nan,
                "monotonic_violations": _monotonic_violations(model, fit["params"], classic_params, b7),
                "param_count": len(fit["params"]),
            }
        )
        p = valid[["N_params_B", "D_tokens_B", "Q_score", "val_loss", "delta_loss"]].copy()
        p["model"] = model
        p["split_type"] = split["split_type"]
        p["held_out_level"] = split["held_out"]
        p["classic_prediction"] = base_pred
        p["quality_delta_prediction"] = delta_pred
        p["predicted"] = pred_abs
        p["residual"] = pred_abs - y_abs
        preds.append(p)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def _select_quality(cv: pd.DataFrame, rng: np.random.Generator) -> tuple[str, pd.DataFrame, dict[str, float]]:
    order = ["GQ1", "GQ2", "GQ3", "GQ4"]
    family = cv.groupby(["model", "split_type"], as_index=False).agg(
        family_RMSE=("RMSE", "mean"), family_nRMSE=("nRMSE", "mean"), folds=("RMSE", "size")
    )
    scores = family.groupby("model")["family_RMSE"].mean().to_dict()
    draws: dict[str, list[float]] = {m: [] for m in order}
    for _ in range(2000):
        sampled_by_family: dict[str, np.ndarray] = {}
        for split_type in sorted(cv["split_type"].unique()):
            count = int((cv["split_type"] == split_type).sum() // len(order))
            sampled_by_family[split_type] = rng.integers(0, count, size=count)
        for model in order:
            vals = []
            for split_type in sorted(cv["split_type"].unique()):
                arr = cv.loc[(cv["model"] == model) & (cv["split_type"] == split_type), "RMSE"].to_numpy(float)
                vals.append(float(np.mean(arr[sampled_by_family[split_type]])))
            draws[model].append(float(np.mean(vals)))
    ses = {m: float(np.std(draws[m], ddof=1)) for m in order}
    best_model = min(order, key=lambda m: scores[m])
    threshold = scores[best_model] + ses[best_model]
    eligible = {m for m in order if scores[m] <= threshold}
    for model in order:
        if model in eligible:
            return model, family, {"best_model": best_model, "best_score": scores[best_model], "best_se": ses[best_model], "one_se_threshold": threshold, **{f"score_{m}": scores[m] for m in order}, **{f"se_{m}": ses[m] for m in order}}
    return best_model, family, {"best_model": best_model, "best_score": scores[best_model], "best_se": ses[best_model], "one_se_threshold": threshold}


def _bootstrap_quality(b7: pd.DataFrame, selected: str, classic_params: dict[str, float], cfg: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    reps = int(cfg.get("bootstrap_reps", 120))
    cells = list(b7.groupby(["N_params_B", "D_tokens_B"]).groups.keys())
    rows = []
    for draw in range(reps):
        sample = [cells[i] for i in rng.integers(0, len(cells), size=len(cells))]
        boot = pd.concat([b7[(b7["N_params_B"] == n) & (b7["D_tokens_B"] == d)] for n, d in sample], ignore_index=True)
        try:
            fit, _ = fit_quality(boot, selected, classic_params, cfg, rng)
            rows.append({"draw": draw, "model": selected, "success": fit["success"], "constraint_status": "ok", **fit["params"]})
        except Exception as exc:
            rows.append({"draw": draw, "model": selected, "success": False, "constraint_status": exc.__class__.__name__})
    return pd.DataFrame(rows)


def run_quality(tables: dict[str, pd.DataFrame], classic: dict[str, Any], cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg["seed"]) + 22)
    selected_classic: FitResult = classic["selected"]
    params = selected_classic.params
    b7 = _quality_table(tables["B7"].copy(), selected_classic.model, params)
    q1 = b7[np.isclose(b7["Q_score"], b7["Q_score"].max())].copy()
    q1_pred = classic_predict(selected_classic.model, params, q1["N_params_B"].to_numpy(float), q1["D_tokens_B"].to_numpy(float))
    q1_nrmse = rmse(q1["val_loss"].to_numpy(float), q1_pred) / max(float(np.subtract(*np.quantile(b7["val_loss"], [0.75, 0.25]))), 1e-12)
    fit_mode = "absolute_anchor_compatible" if q1_nrmse <= float(cfg["acceptance"]["quality_nrmse_max"]) else "relative_delta_due_anchor_warn"

    fits = []
    starts = []
    cvs = []
    preds = []
    for model in ["GQ1", "GQ2", "GQ3", "GQ4"]:
        fit, start = fit_quality(b7, model, params, cfg, rng)
        fits.append(fit)
        starts.append(start)
        cv, pred = _cv_model(b7, model, selected_classic.model, params, cfg, rng)
        cvs.append(cv)
        preds.append(pred)
    cv_all = pd.concat(cvs, ignore_index=True)
    pred_all = pd.concat(preds, ignore_index=True)
    start_all = pd.concat(starts, ignore_index=True)
    raw_selected, family_scores, selection_diag = _select_quality(cv_all, rng)
    selected = raw_selected
    if selected == "GQ4":
        if not (selection_diag.get("score_GQ4", np.inf) < selection_diag.get("score_GQ1", np.inf) - selection_diag.get("se_GQ1", 0.0)):
            selected = "GQ1"
    selected_fit = next(f for f in fits if f["model"] == selected)
    bootstrap = _bootstrap_quality(b7, selected, params, cfg, rng)
    for df, name in [
        (cv_all, "quality_model_comparison.csv"),
        (family_scores, "quality_split_family_scores.csv"),
        (pred_all, "quality_block_predictions.csv.gz"),
        (start_all, "quality_multistart_log.csv"),
        (bootstrap, "quality_model_bootstrap.csv.gz"),
    ]:
        df.to_csv(output_dir / name, index=False, compression="gzip" if name.endswith(".gz") else None)

    b8 = tables["B8"].copy()
    b8_rows = []
    for dtype, part in b8.groupby("data_type"):
        for q_name, q_values in [("original_Q", part["Q_score"].to_numpy(float)), ("one_minus_Q", 1.0 - part["Q_score"].to_numpy(float))]:
            stats = []
            for (_, _), g in part.assign(q_tmp=q_values).groupby(["N_params_B", "D_tokens_B"]):
                if len(g) >= 3:
                    stats.append(spearman(g["q_tmp"].to_numpy(float), g["val_loss"].to_numpy(float)))
            b8_rows.append(
                {
                    "data_type": dtype,
                    "q_definition": q_name,
                    "rows": int(len(part)),
                    "group_spearman_median": float(np.nanmedian(stats)) if stats else np.nan,
                    "floor_count_loss_0_5": int(np.isclose(part["val_loss"], 0.5).sum()),
                    "statement": "stress_only_not_used_for_selection",
                }
            )
    b8_sens = pd.DataFrame(b8_rows)
    b8_sens.to_csv(output_dir / "b8_anomaly_sensitivity.csv", index=False)
    summary = {
        "fit_mode": fit_mode,
        "q1_anchor_nrmse": q1_nrmse,
        "raw_selected_model": raw_selected,
        "selected_model": selected,
        "selection_rule": "blocked_cv_one_standard_error_with_complex_model_gate",
        "primary_target": "absolute_loss_without_validation_loss_baseline",
        "split_family_weights": {k: 0.25 for k in sorted(cv_all["split_type"].unique())},
        "selection_diagnostics": selection_diag,
        "parameters": selected_fit["params"],
        "classic_parameter_strategy": "fixed_B1_selected",
        "cv_summary": cv_all.groupby("model")[["RMSE", "MAE", "nRMSE", "group_spearman_median", "monotonic_violations"]].mean().reset_index().to_dict(orient="records"),
        "split_family_summary": family_scores.to_dict(orient="records"),
        "leakage_guard": {"validation_loss_used_as_baseline": False, "loss_q1_role": "auxiliary_paired_metric_only", "leave_Q1_paired_status": "not_applicable"},
        "bootstrap_success_rate": float(bootstrap["success"].mean()) if len(bootstrap) else 0.0,
    }
    json_dump(output_dir / "quality_model_parameters.json", summary)
    return {"b7": b7, "cv": cv_all, "family_scores": family_scores, "predictions": pred_all, "bootstrap": bootstrap, "b8": b8_sens, "summary": summary}
