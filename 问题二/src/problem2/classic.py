from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc

from .common import equal_cluster_weights, json_dump, mae, rmse, spearman, weighted_huber_residual


@dataclass(frozen=True)
class FitResult:
    model: str
    params: dict[str, float]
    cost: float
    success: bool
    message: str
    nfev: int


def predict(model: str, params: dict[str, float], n: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = np.asarray(n, float)
    d = np.asarray(d, float)
    if model == "M0":
        return params["E"] + params["A"] * n ** (-params["alpha"]) + params["B"] * d ** (-params["beta"])
    if model == "MC":
        c = 6.0 * n * d
        return params["E"] + params["K"] * c ** (-params["zeta"])
    if model == "MND":
        return (
            params["E"]
            + params["A"] * n ** (-params["alpha"])
            + params["B"] * d ** (-params["beta"])
            + params["G"] * (n * d) ** (-params["xi"])
        )
    raise ValueError(model)


def _unpack(model: str, theta: np.ndarray) -> dict[str, float]:
    vals = np.exp(theta)
    if model == "M0":
        keys = ["E", "A", "B", "alpha", "beta"]
    elif model == "MC":
        keys = ["E", "K", "zeta"]
    elif model == "MND":
        keys = ["E", "A", "B", "alpha", "beta", "G", "xi"]
    else:
        raise ValueError(model)
    return {k: float(v) for k, v in zip(keys, vals)}


def _initials(model: str, y: np.ndarray, starts: int, rng: np.random.Generator) -> list[np.ndarray]:
    y_min = max(float(np.min(y)) * 0.4, 1e-4)
    y_span = max(float(np.max(y) - np.min(y)), 0.1)
    seeds: list[list[float]] = []
    if model == "M0":
        seeds.append([y_min, y_span, y_span, 0.30, 0.25])
        bounds = [(0.2, 2.5), (0.1, 8.0), (0.1, 8.0), (0.05, 1.0), (0.05, 1.0)]
    elif model == "MC":
        seeds.append([y_min, y_span, 0.25])
        bounds = [(0.2, 2.5), (0.1, 8.0), (0.05, 1.0)]
    else:
        seeds.append([y_min, y_span, y_span, 0.30, 0.25, 0.2 * y_span, 0.20])
        bounds = [(0.2, 2.5), (0.1, 8.0), (0.1, 8.0), (0.05, 1.0), (0.05, 1.0), (0.01, 3.0), (0.05, 1.0)]
    if starts > 1:
        seed = int(rng.integers(0, 2**31 - 1))
        unit = qmc.LatinHypercube(d=len(bounds), seed=seed).random(starts - 1)
        lo = np.log(np.array([a for a, _ in bounds], float))
        hi = np.log(np.array([b for _, b in bounds], float))
        seeds.extend(np.exp(lo + unit * (hi - lo)).tolist())
    return [np.log(np.asarray(s, float)) for s in seeds]


def fit_model(
    df: pd.DataFrame,
    model: str,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    forced_starts: int | None = None,
) -> tuple[FitResult, pd.DataFrame]:
    n = df["N_params_B"].to_numpy(float)
    d = df["D_tokens_B"].to_numpy(float)
    y = df["val_loss"].to_numpy(float)
    cluster_col = "bootstrap_cluster_instance" if "bootstrap_cluster_instance" in df.columns else "N_params_B"
    weights = equal_cluster_weights(df[cluster_col])
    starts = int(forced_starts or cfg.get("classic_multistarts", 50))
    initials = _initials(model, y, starts, rng)
    pilot = least_squares(
        lambda theta: np.sqrt(weights) * (predict(model, _unpack(model, theta), n, d) - y),
        initials[0], max_nfev=1500, method="trf",
    )
    pilot_resid = predict(model, _unpack(model, pilot.x), n, d) - y
    mad = float(np.median(np.abs(pilot_resid - np.median(pilot_resid))))
    delta = max(float(cfg.get("huber_delta_floor", cfg.get("huber_delta", 0.001))), 1.345 * 1.4826 * mad)
    records = []
    best: FitResult | None = None
    for idx, theta0 in enumerate(initials):
        def residual(theta: np.ndarray) -> np.ndarray:
            params = _unpack(model, theta)
            return weighted_huber_residual(predict(model, params, n, d) - y, weights, delta)

        res = least_squares(
            residual,
            theta0,
            max_nfev=5000,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-8,
            method="trf",
        )
        fit = FitResult(model, _unpack(model, res.x), float(2.0 * res.cost), bool(res.success), str(res.message), int(res.nfev))
        rec = {"model": model, "start": idx, "success": fit.success, "cost": fit.cost, "nfev": fit.nfev, "message": fit.message, "huber_delta": delta, **fit.params}
        records.append(rec)
        if best is None or fit.cost < best.cost:
            best = fit
    assert best is not None
    return best, pd.DataFrame(records)


def _cv_for_model(b1: pd.DataFrame, model: str, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    for held in sorted(b1["N_params_B"].unique()):
        train = b1[b1["N_params_B"] != held]
        valid = b1[b1["N_params_B"] == held]
        fit, _ = fit_model(train, model, cfg, rng, forced_starts=max(6, int(cfg.get("classic_multistarts", 50)) // 3))
        pred = predict(model, fit.params, valid["N_params_B"].to_numpy(float), valid["D_tokens_B"].to_numpy(float))
        y = valid["val_loss"].to_numpy(float)
        loss_iqr = np.subtract(*np.quantile(b1["val_loss"], [0.75, 0.25]))
        rows.append(
            {
                "held_out_N": held,
                "model": model,
                "RMSE": rmse(y, pred),
                "MAE": mae(y, pred),
                "nRMSE": rmse(y, pred) / loss_iqr,
                "Spearman_D": spearman(valid["D_tokens_B"].to_numpy(float), -pred),
                "bias": float(np.mean(pred - y)),
            }
        )
        part = valid[["N_params_B", "D_tokens_B", "val_loss"]].copy()
        part["model"] = model
        part["held_out_N"] = held
        part["predicted"] = pred
        part["residual"] = pred - y
        preds.append(part)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def _select_by_one_se(cv: pd.DataFrame) -> str:
    order = ["M0", "MC", "MND"]
    grouped = cv.groupby("model")["RMSE"].agg(["mean", "std", "count"]).reset_index()
    grouped["se"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"])
    best = grouped.loc[grouped["mean"].idxmin()]
    threshold = float(best["mean"] + best["se"])
    eligible = set(grouped.loc[grouped["mean"] <= threshold, "model"])
    for model in order:
        if model in eligible:
            return model
    return str(best["model"])


def _bootstrap(b1: pd.DataFrame, model: str, cfg: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    reps = int(cfg.get("bootstrap_reps", 120))
    groups = np.array(sorted(b1["N_params_B"].unique()))
    rows = []
    for draw in range(reps):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        counts = pd.Series(sampled).value_counts().sort_index()
        if np.unique(sampled).size < 4:
            rows.append({"draw": draw, "model": model, "success": False,
                         "unique_clusters": int(np.unique(sampled).size),
                         "sampled_cluster_instances": int(len(sampled)),
                         "duplicate_multiplicity_preserved": True,
                         "sample_counts": ";".join(f"{k}:{v}" for k, v in counts.items()),
                         "reason": "too_few_unique_clusters"})
            continue
        pieces = []
        for instance, g in enumerate(sampled):
            part = b1[b1["N_params_B"] == g].copy()
            part["bootstrap_cluster_instance"] = instance
            pieces.append(part)
        boot = pd.concat(pieces, ignore_index=True)
        try:
            fit, _ = fit_model(boot, model, cfg, rng, forced_starts=6)
            rows.append({"draw": draw, "model": model, "success": fit.success, "unique_clusters": int(np.unique(sampled).size), "sampled_cluster_instances": int(len(sampled)), "duplicate_multiplicity_preserved": True, "sample_counts": ";".join(f"{k}:{v}" for k, v in counts.items()), "reason": "", **fit.params})
        except Exception as exc:
            rows.append({"draw": draw, "model": model, "success": False, "unique_clusters": int(np.unique(sampled).size), "sampled_cluster_instances": int(len(sampled)), "duplicate_multiplicity_preserved": True, "sample_counts": ";".join(f"{k}:{v}" for k, v in counts.items()), "reason": exc.__class__.__name__})
    return pd.DataFrame(rows)


def warmup_sensitivity(
    b1: pd.DataFrame,
    selected: FitResult,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    output_dir: Path,
    threshold_d_tokens_b: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trimmed = b1[b1["D_tokens_B"] >= threshold_d_tokens_b].copy()
    removed = b1[b1["D_tokens_B"] < threshold_d_tokens_b].copy()
    fit, _ = fit_model(trimmed, selected.model, cfg, rng, forced_starts=max(20, int(cfg.get("classic_multistarts", 50)) // 2))
    rows = []
    relative = {}
    for name, full_value in selected.params.items():
        trimmed_value = float(fit.params[name])
        rel = (trimmed_value - float(full_value)) / max(abs(float(full_value)), 1e-12)
        relative[name] = rel
        rows.append({
            "parameter": name,
            "full_value": float(full_value),
            "trimmed_value": trimmed_value,
            "relative_change": rel,
            "abs_relative_change": abs(rel),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "classic_warmup_sensitivity.csv", index=False)
    summary = {
        "threshold_D_tokens_B": threshold_d_tokens_b,
        "full_rows": int(len(b1)),
        "removed_rows": int(len(removed)),
        "trimmed_rows": int(len(trimmed)),
        "removed_rows_per_N": {str(k): int(v) for k, v in removed.groupby("N_params_B").size().items()},
        "model": selected.model,
        "trimmed_parameters": fit.params,
        "relative_change": relative,
        "max_abs_relative_change": float(frame["abs_relative_change"].max()),
        "alpha_beta_max_abs_relative_change": float(frame.loc[frame["parameter"].isin(["alpha", "beta"]), "abs_relative_change"].max()),
        "role": "sensitivity_only_not_used_for_model_selection",
    }
    return frame, summary


def external_validation(tables: dict[str, pd.DataFrame], selected: FitResult, output_dir: Path) -> pd.DataFrame:
    rows = []
    for source in ["B2", "B4", "B5", "B10"]:
        df = tables[source].copy()
        if "is_converged" in df.columns:
            eval_df = df[df["is_converged"] == 1].copy()
        else:
            eval_df = df.copy()
        eval_df = eval_df[(eval_df["N_params_B"] > 0) & (eval_df["D_tokens_B"] > 0) & (eval_df["val_loss"] > 0)]
        if eval_df.empty:
            continue
        pred = predict(selected.model, selected.params, eval_df["N_params_B"].to_numpy(float), eval_df["D_tokens_B"].to_numpy(float))
        centered_true = eval_df["val_loss"].to_numpy(float)
        centered_pred = pred.copy()
        if source in {"B4", "B5"}:
            group_col = "source" if source == "B5" else "family"
            centered_true = eval_df["val_loss"].to_numpy(float) - eval_df.groupby(group_col)["val_loss"].transform("mean").to_numpy(float)
            centered_pred = pred - pd.Series(pred).groupby(eval_df[group_col].to_numpy()).transform("mean").to_numpy(float)
        rows.append(
            {
                "source": source,
                "n": int(len(eval_df)),
                "evidence_type": {"B2": "semi_synthetic", "B4": "external_family", "B5": "published", "B10": "estimated"}[source],
                "RMSE": rmse(eval_df["val_loss"].to_numpy(float), pred),
                "MAE": mae(eval_df["val_loss"].to_numpy(float), pred),
                "centered_MAE": mae(centered_true, centered_pred),
                "Spearman": spearman(eval_df["val_loss"].to_numpy(float), pred),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "classic_external_validation.csv", index=False)
    return out


def run_classic(tables: dict[str, pd.DataFrame], cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg["seed"]) + 11)
    b1 = tables["B1"].copy()
    fits = []
    starts = []
    cvs = []
    cv_preds = []
    for model in ["M0", "MC", "MND"]:
        fit, start_df = fit_model(b1, model, cfg, rng)
        fits.append(fit)
        starts.append(start_df)
        cv, pred = _cv_for_model(b1, model, cfg, rng)
        cvs.append(cv)
        cv_preds.append(pred)
    cv_all = pd.concat(cvs, ignore_index=True)
    pred_all = pd.concat(cv_preds, ignore_index=True)
    selected_model = _select_by_one_se(cv_all)
    selected = next(f for f in fits if f.model == selected_model)
    start_all = pd.concat(starts, ignore_index=True)
    bootstrap = _bootstrap(b1, selected_model, cfg, rng)
    warmup, warmup_summary = warmup_sensitivity(b1, selected, cfg, rng, output_dir)
    external = external_validation(tables, selected, output_dir)
    cv_all.to_csv(output_dir / "classic_scaling_cv.csv", index=False)
    pred_all.to_csv(output_dir / "classic_scaling_cv_predictions.csv.gz", index=False, compression="gzip")
    start_all.to_csv(output_dir / "classic_multistart_log.csv", index=False)
    bootstrap.to_csv(output_dir / "classic_scaling_bootstrap.csv.gz", index=False, compression="gzip")
    summary = {
        "selected_model": selected_model,
        "selection_rule": "one_standard_error",
        "parameters": selected.params,
        "cost": selected.cost,
        "success": selected.success,
        "message": selected.message,
        "cv_summary": cv_all.groupby("model")[["RMSE", "MAE", "nRMSE", "Spearman_D"]].mean().reset_index().to_dict(orient="records"),
        "multistart_success_rate": float(start_all.groupby("model")["success"].mean().get(selected_model, np.nan)),
        "bootstrap_success_rate": float(bootstrap["success"].mean()) if len(bootstrap) else 0.0,
        "bootstrap_reps": int(len(bootstrap)),
        "bootstrap_cluster_unit": "N_params_B_with_instance_multiplicity",
        "initial_design": "latin_hypercube_plus_canonical",
        "huber_scale": "training_residual_MAD",
        "warmup_sensitivity": warmup_summary,
    }
    json_dump(output_dir / "classic_scaling_parameters.json", summary)
    return {"selected": selected, "fits": fits, "cv": cv_all, "cv_predictions": pred_all, "multistart": start_all, "bootstrap": bootstrap, "warmup": warmup, "external": external, "summary": summary}
