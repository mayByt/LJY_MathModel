from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from .io import model_family

TARGETS = {
    "Average": "LB_Average", "IFEval": "LB_IFEval", "BBH": "LB_BBH",
    "MATH": "LB_MATH", "GPQA": "LB_GPQA", "MUSR": "LB_MUSR",
    "MMLU_PRO": "LB_MMLU_PRO",
}


def to_y(score, eps=0.5):
    return logit((np.asarray(score, float) + eps) / (100.0 + 2 * eps))


def to_score(y, eps=0.5):
    return 100.0 * expit(np.asarray(y, float))


@dataclass
class MonoModel:
    kind: str
    xmin: float
    xmax: float
    const: float | None = None
    intercept: float | None = None
    slope: float | None = None
    iso: object | None = None
    spline: object | None = None
    left_slope: float = 0.0
    right_slope: float = 0.0

    def predict(self, x):
        x = np.asarray(x, float)
        xc = np.clip(x, self.xmin, self.xmax)
        if self.kind == "constant":
            y = np.full_like(x, self.const)
        elif self.kind == "linear":
            y = self.intercept + self.slope * x
        elif self.kind == "isotonic":
            y = self.iso.predict(xc)
        else:
            y = self.spline(xc)
        if self.kind in {"isotonic", "spline"}:
            left = x < self.xmin
            right = x > self.xmax
            if self.kind == "isotonic":
                y_left = float(self.iso.predict([self.xmin])[0])
                y_right = float(self.iso.predict([self.xmax])[0])
            else:
                y_left = float(self.spline(self.xmin))
                y_right = float(self.spline(self.xmax))
            y[left] = y_left + self.left_slope * (x[left] - self.xmin)
            y[right] = y_right + self.right_slope * (x[right] - self.xmax)
        return y


def _weighted_linear(x, y, w):
    X = np.column_stack([np.ones(len(x)), x])
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    slope = min(float(coef[1]), 0.0)
    intercept = float(np.average(y - slope * x, weights=w))
    return intercept, slope


def fit_mono(kind, loss, score, weight, eps=0.5):
    x = np.log(np.asarray(loss, float))
    y = to_y(score, eps)
    w = np.asarray(weight, float)
    order = np.argsort(x)
    x, y, w = x[order], y[order], w[order]
    xmin, xmax = float(x.min()), float(x.max())
    if kind == "constant":
        return MonoModel(kind, xmin, xmax, const=float(np.average(y, weights=w)))
    intercept, slope = _weighted_linear(x, y, w)
    if kind == "linear":
        return MonoModel(kind, xmin, xmax, intercept=intercept, slope=slope)
    if kind == "isotonic":
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, y, sample_weight=w)
        return MonoModel(kind, xmin, xmax, iso=iso, left_slope=slope, right_slope=slope)
    bins = min(5, max(3, len(x) // 12))
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    xb, yb = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
        if mask.any():
            xb.append(float(np.average(x[mask], weights=w[mask])))
            yb.append(float(np.average(y[mask], weights=w[mask])))
    xb = np.asarray(xb)
    yb = np.minimum.accumulate(np.asarray(yb))
    if len(xb) < 2:
        return MonoModel("linear", xmin, xmax, intercept=intercept, slope=slope)
    spline = PchipInterpolator(xb, yb, extrapolate=False)
    ls = min(float(spline.derivative()(xb[0])), 0.0)
    rs = min(float(spline.derivative()(xb[-1])), 0.0)
    cap = max(abs(slope) * 3, 0.05)
    ls, rs = max(ls, -cap), max(rs, -cap)
    return MonoModel("spline", float(xb[0]), float(xb[-1]), spline=spline, left_slope=ls, right_slope=rs)


def comparability_weight(frame, medium_weight):
    high = frame["Loss_Comparability"].astype(str).str.lower().str.startswith("high")
    return np.where(high, 1.0, medium_weight)


def cross_validate(frame, target_col, medium_weight=0.25, eps=0.5):
    d = frame[["Model", "Val_Loss", target_col, "Loss_Comparability"]].dropna().copy()
    d["group"] = d["Model"].map(model_family)
    w = comparability_weight(d, medium_weight)
    kinds = ["constant", "linear", "spline", "isotonic"]
    rows = []
    groups = sorted(d["group"].unique())
    for kind in kinds:
        fold_mae = []
        for group in groups:
            te = d["group"].eq(group).to_numpy()
            tr = ~te
            if tr.sum() < 4 or te.sum() == 0 or w[tr].sum() <= 0 or w[te].sum() <= 0:
                continue
            model = fit_mono(kind, d.loc[tr, "Val_Loss"], d.loc[tr, target_col], w[tr], eps)
            pred = to_score(model.predict(np.log(d.loc[te, "Val_Loss"].to_numpy())), eps)
            err = np.abs(pred - d.loc[te, target_col].to_numpy())
            fold_mae.append(float(np.average(err, weights=w[te])))
        arr = np.asarray(fold_mae)
        rows.append({
            "target": target_col, "model": kind, "folds": len(arr),
            "cv_mae": float(np.mean(arr)) if len(arr) else np.nan,
            "cv_mae_se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else np.nan,
        })
    cv = pd.DataFrame(rows)
    cv["cv_scheme"] = "leave_source_family_out"
    good = cv.dropna(subset=["cv_mae"]).copy()
    if good.empty:
        fallback_rows = []
        positive = np.flatnonzero(w > 0)
        for kind in kinds:
            errors = []
            for hold in positive:
                tr = np.ones(len(d), dtype=bool)
                tr[hold] = False
                tr &= w > 0
                if tr.sum() < 4:
                    continue
                model = fit_mono(kind, d.loc[tr, "Val_Loss"], d.loc[tr, target_col], w[tr], eps)
                pred = to_score(model.predict(np.log([d.iloc[hold]["Val_Loss"]])), eps)[0]
                errors.append(abs(pred - d.iloc[hold][target_col]))
            arr = np.asarray(errors)
            fallback_rows.append({
                "target": target_col, "model": kind, "folds": len(arr),
                "cv_mae": float(np.mean(arr)) if len(arr) else np.nan,
                "cv_mae_se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else np.nan,
                "cv_scheme": "leave_one_out_single_family_sensitivity",
            })
        cv = pd.DataFrame(fallback_rows)
        good = cv.dropna(subset=["cv_mae"]).copy()
    best = good.loc[good["cv_mae"].idxmin()]
    cutoff = best["cv_mae"] + (best["cv_mae_se"] if np.isfinite(best["cv_mae_se"]) else 0)
    complexity = {"constant": 0, "linear": 1, "spline": 2, "isotonic": 3}
    eligible = good[good["cv_mae"] <= cutoff].copy()
    selected = min(eligible["model"], key=lambda k: complexity[k])
    const_mae = float(good.loc[good["model"].eq("constant"), "cv_mae"].iloc[0])
    selected_mae = float(good.loc[good["model"].eq(selected), "cv_mae"].iloc[0])
    weak = selected == "constant" or selected_mae >= const_mae
    cv["selected"] = cv["model"].eq(selected)
    cv["bridge_weak"] = weak
    return cv, selected, weak


def support_status(loss):
    if 2.0933 <= loss <= 2.5978:
        return "high_supported"
    if 1.65 <= loss <= 2.84:
        return "expanded_supported"
    return "out_of_bridge_range"


def fit_all(frame, medium_weight=0.25, eps=0.5):
    cv_parts = []
    models = {}
    meta = {}
    w = comparability_weight(frame, medium_weight)
    for target, col in TARGETS.items():
        cv, selected, weak = cross_validate(frame, col, medium_weight, eps)
        cv["target_name"] = target
        cv_parts.append(cv)
        model = fit_mono(selected, frame["Val_Loss"], frame[col], w, eps)
        models[target] = model
        grid = np.linspace(np.log(frame["Val_Loss"].min()), np.log(frame["Val_Loss"].max()), 200)
        monotone_violations = int((np.diff(model.predict(grid)) > 1e-10).sum())
        meta[target] = {
            "score_column": col, "selected_model": selected, "bridge_weak": bool(weak),
            "monotonicity_violations": monotone_violations,
            "loss_min": float(frame["Val_Loss"].min()), "loss_max": float(frame["Val_Loss"].max()),
        }
    return models, pd.concat(cv_parts, ignore_index=True), meta


def weight_sensitivity(frame, weights=(0.0, 0.1, 0.25, 0.5), eps=0.5):
    rows = []
    for mw in weights:
        models, cv, meta = fit_all(frame, mw, eps)
        for target in TARGETS:
            chosen = meta[target]["selected_model"]
            rec = cv[(cv["target_name"] == target) & (cv["model"] == chosen)].iloc[0]
            rows.append({
                "medium_weight": mw, "target": target, "selected_model": chosen,
                "cv_mae": rec["cv_mae"], "bridge_weak": meta[target]["bridge_weak"],
                "monotonicity_violations": meta[target]["monotonicity_violations"],
            })
    return pd.DataFrame(rows)


def _scenario_key(row):
    return (float(row["budget_FLOPs"]), int(row["context_length"]), str(row["quality_cost_type"]))


def map_problem3(c6, p3_root: Path, models, selected_meta, rng, draws=1000, medium_weight=0.25, eps=0.5):
    bridge = json.loads((p3_root / "problem3_bridge.json").read_text(encoding="utf-8"))
    if not bridge.get("PROBLEM3_READY"):
        raise RuntimeError("PROBLEM3_READY is false")
    entries = pd.DataFrame(bridge["entries"])
    boot = pd.read_csv(p3_root / "bootstrap_optimal_allocations.csv.gz")
    boot = boot[boot["support_mode"].eq("operational_extended")].copy()
    pools = {_scenario_key(g.iloc[0]): g["predicted_loss"].to_numpy() for _, g in boot.groupby(["budget_FLOPs", "context_length", "quality_cost_type"])}
    c6 = c6.copy()
    c6["group"] = c6["Model"].map(model_family)
    groups = sorted(c6["group"].unique())
    row_draws = []
    summaries = []
    boot_models = []
    for b in range(draws):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([c6[c6["group"].eq(g)] for g in sampled_groups], ignore_index=True)
        w = comparability_weight(sample, medium_weight)
        boot_models.append({
            target: fit_mono(selected_meta[target]["selected_model"], sample["Val_Loss"], sample[col], w, eps)
            for target, col in TARGETS.items()
        })
    for idx, entry in entries.iterrows():
        key = _scenario_key(entry)
        losses = pools.get(key)
        if losses is None or len(losses) == 0:
            lo, hi = entry["predicted_loss_interval"]
            losses = rng.triangular(lo, entry["predicted_loss"], hi, size=draws)
            upstream_mode = "interval_approximation"
        else:
            losses = rng.choice(losses, size=draws, replace=True)
            upstream_mode = "joint_bootstrap"
        target_draws = {t: [] for t in TARGETS}
        for b in range(draws):
            for target, col in TARGETS.items():
                m = boot_models[b][target]
                value = float(to_score(m.predict(np.log([losses[b]])), eps)[0])
                target_draws[target].append(value)
        base = {
            "scenario_index": idx, "budget_FLOPs": entry["budget_FLOPs"],
            "context_length": entry["context_length"], "quality_cost_type": entry["quality_cost_type"],
            "predicted_loss": entry["predicted_loss"], "support_status": support_status(float(entry["predicted_loss"])),
            "problem3_evidence": entry.get("support_status", entry.get("evidence_tier", "unknown")),
            "upstream_uncertainty_mode": upstream_mode,
        }
        for target, vals in target_draws.items():
            a = np.asarray(vals)
            summaries.append({
                **base, "target": target, "median": float(np.median(a)),
                "p10": float(np.quantile(a, .10)), "p90": float(np.quantile(a, .90)),
                "p025": float(np.quantile(a, .025)), "p975": float(np.quantile(a, .975)),
                "final_evidence_tier": "limited" if base["support_status"] == "out_of_bridge_range" or selected_meta[target]["bridge_weak"] else "supported",
            })
            for b, value in enumerate(a):
                row_draws.append({"scenario_index": idx, "target": target, "draw_id": b, "score": value, "loss": losses[b]})
    return pd.DataFrame(summaries), pd.DataFrame(row_draws)
