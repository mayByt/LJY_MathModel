from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import theilslopes


def pinball(residual, tau):
    return np.where(residual >= 0, tau * residual, (tau - 1) * residual)


@dataclass
class DynamicFit:
    tau: float
    lam: float
    beta0: float
    beta_x: float
    states: np.ndarray
    months: pd.PeriodIndex
    x_mean: float
    x_std: float
    objective: float
    success: bool

    def state_at(self, month):
        m = pd.Period(month, freq="M")
        if m in self.months:
            return float(self.states[self.months.get_loc(m)])
        positions = np.arange(len(self.months), dtype=float)
        target = (m.year - self.months[0].year) * 12 + m.month - self.months[0].month
        if target < 0:
            slope = np.polyfit(positions, self.states, 1)[0] if len(positions) > 1 else 0.0
            return float(self.states[0] + slope * target)
        recent = min(4, len(positions))
        local = np.polyfit(positions[-recent:], self.states[-recent:], 1)[0] if recent > 1 else 0.0
        global_slope = np.polyfit(positions, self.states, 1)[0] if len(positions) > 1 else 0.0
        slope = 0.5 * local + 0.5 * global_slope
        return float(self.states[-1] + slope * (target - positions[-1]))

    def predict(self, x, months):
        x = np.asarray(x, float)
        state = np.asarray([self.state_at(m) for m in months])
        return self.beta0 + self.beta_x * ((x - self.x_mean) / self.x_std) + state


def fit_dynamic(frame, tau=0.90, lam=1.0, weights=None, include_scale=True):
    d = frame.sort_values("submission_date").copy()
    months = pd.period_range(d["submission_date"].min().to_period("M"), d["submission_date"].max().to_period("M"), freq="M")
    midx = d["submission_date"].dt.to_period("M").map({m: i for i, m in enumerate(months)}).to_numpy()
    x = d["log10_compute"].to_numpy(float)
    y = d["ability_logit"].to_numpy(float)
    x_mean = float(np.mean(x))
    x_std = float(np.std(x)) or 1.0
    xs = (x - x_mean) / x_std
    w = np.ones(len(d)) if weights is None else np.asarray(weights, float)
    n_month = len(months)
    q0 = np.quantile(y, tau)
    init = np.r_[q0, 0.1 if include_scale else 0.0, np.zeros(n_month)]

    def objective(theta):
        b0, bx = theta[:2]
        a = theta[2:]
        pred = b0 + bx * xs + a[midx]
        residual = y - pred
        smooth_pinball = (tau - 0.5) * residual + 0.5 * np.sqrt(residual * residual + 1e-6)
        loss = np.sum(w * smooth_pinball) / np.sum(w)
        d2 = np.diff(a, 2)
        d1 = np.diff(a)
        penalty = lam * (np.sum(d2 * d2) + 0.05 * np.sum(d1 * d1)) + 100.0 * np.mean(a) ** 2
        return loss + penalty

    bounds = [(None, None), (0.0, 0.0) if not include_scale else (0.0, None)] + [(None, None)] * n_month
    result = minimize(objective, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "ftol": 1e-11})
    return DynamicFit(
        tau=tau, lam=lam, beta0=float(result.x[0]), beta_x=float(result.x[1]),
        states=np.asarray(result.x[2:]), months=months, x_mean=x_mean, x_std=x_std,
        objective=float(result.fun), success=bool(result.success),
    )


def fit_scale_only(frame, tau=0.90, weights=None):
    d = frame.copy()
    x = d["log10_compute"].to_numpy(float)
    y = d["ability_logit"].to_numpy(float)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, float)
    xm, xs = float(x.mean()), float(x.std()) or 1.0
    z = (x - xm) / xs
    def objective(theta):
        return float(np.sum(w * pinball(y - theta[0] - theta[1] * z, tau)) / np.sum(w))
    r = minimize(objective, [np.quantile(y, tau), .1], method="L-BFGS-B",
                 bounds=[(None, None), (0, None)])
    return r, xm, xs


def rolling_cv(frame, lambdas, tau=0.90):
    d = frame.sort_values("submission_date").copy()
    months = sorted(d["submission_date"].dt.to_period("M").unique())
    cut_indices = list(range(max(2, len(months) - 5), len(months) - 1))
    rows = []
    for ci in cut_indices:
        cutoff = months[ci]
        train = d[d["submission_date"].dt.to_period("M") <= cutoff]
        test = d[d["submission_date"].dt.to_period("M") > cutoff]
        test = test[test["submission_date"].dt.to_period("M") <= cutoff + 3]
        if len(train) < 12 or len(test) == 0 or train["submission_date"].dt.to_period("M").nunique() < 3:
            continue
        for lam in lambdas:
            model = fit_dynamic(train, tau, lam)
            pred = model.predict(test["log10_compute"], test["submission_date"].dt.to_period("M"))
            loss = float(np.mean(pinball(test["ability_logit"].to_numpy() - pred, tau)))
            rows.append({"fold_cutoff": str(cutoff), "model": "M-C", "lambda": lam, "n_train": len(train), "n_test": len(test), "pinball": loss})
        mt = fit_dynamic(train, tau, max(lambdas), include_scale=False)
        pred = mt.predict(test["log10_compute"], test["submission_date"].dt.to_period("M"))
        rows.append({"fold_cutoff": str(cutoff), "model": "M-time", "lambda": max(lambdas), "n_train": len(train), "n_test": len(test),
                     "pinball": float(np.mean(pinball(test["ability_logit"].to_numpy() - pred, tau)))})
        ms, xm, xs = fit_scale_only(train, tau)
        pred = ms.x[0] + ms.x[1] * ((test["log10_compute"].to_numpy() - xm) / xs)
        rows.append({"fold_cutoff": str(cutoff), "model": "M-scale", "lambda": np.nan, "n_train": len(train), "n_test": len(test),
                     "pinball": float(np.mean(pinball(test["ability_logit"].to_numpy() - pred, tau)))})
    cv = pd.DataFrame(rows)
    m = cv[cv["model"].eq("M-C")].groupby("lambda")["pinball"].agg(["mean", "std", "count"]).reset_index()
    if len(m) == 0:
        return cv, float(lambdas[len(lambdas) // 2])
    best = m.loc[m["mean"].idxmin()]
    cutoff = best["mean"] + (best["std"] / np.sqrt(best["count"]) if best["count"] > 1 else 0)
    chosen = float(m[m["mean"] <= cutoff].sort_values("lambda", ascending=False).iloc[0]["lambda"])
    return cv, chosen


def inv_score(y, eps=0.5):
    return float(100 * expit(y))


def decompose(model: DynamicFit, frame):
    monthly = frame.assign(month=frame["submission_date"].dt.to_period("M")).groupby("month")
    scale_frontier = monthly["log10_compute"].quantile(.9)
    usable = scale_frontier.index.intersection(model.months)
    t0, t1 = usable[0], usable[-1]
    x0, x1 = float(scale_frontier[t0]), float(scale_frontier[t1])
    f00 = float(model.predict([x0], [t0])[0])
    f10 = float(model.predict([x1], [t0])[0])
    f01 = float(model.predict([x0], [t1])[0])
    f11 = float(model.predict([x1], [t1])[0])
    scale = .5 * ((f10 - f00) + (f11 - f01))
    tech = .5 * ((f01 - f00) + (f11 - f10))
    total = f11 - f00
    s00, s10, s01, s11 = map(inv_score, [f00, f10, f01, f11])
    scale_s = .5 * ((s10 - s00) + (s11 - s01))
    tech_s = .5 * ((s01 - s00) + (s11 - s10))
    total_s = s11 - s00
    return {
        "t0": str(t0), "t1": str(t1), "x0_log10_compute": x0, "x1_log10_compute": x1,
        "f00_logit": f00, "f10_logit": f10, "f01_logit": f01, "f11_logit": f11,
        "scale_logit": scale, "tech_logit": tech, "total_logit": total,
        "closure_logit": abs(scale + tech - total),
        "scale_score": scale_s, "tech_score": tech_s, "total_score": total_s,
        "closure_score": abs(scale_s + tech_s - total_s),
    }


def bootstrap_frontier(frame, tau, lam, draws, rng):
    d = frame.reset_index(drop=True).copy()
    family = d["family"].astype(str)
    blocks = d["submission_date"].dt.to_period("2M").astype(str)
    output = []
    for draw in range(draws):
        fw = {g: rng.exponential() for g in family.unique()}
        bw = {g: rng.exponential() for g in blocks.unique()}
        w = np.asarray([fw[f] * bw[b] for f, b in zip(family, blocks)])
        try:
            model = fit_dynamic(d, tau, lam, w)
            dec = decompose(model, d)
            pos = np.arange(len(model.states), dtype=float)
            recent = min(4, len(pos))
            local = np.polyfit(pos[-recent:], model.states[-recent:], 1)[0] if recent > 1 else 0.0
            global_slope = np.polyfit(pos, model.states, 1)[0] if len(pos) > 1 else 0.0
            output.append({
                "draw_id": draw, "fit_success": model.success, "beta0": model.beta0,
                "beta_x": model.beta_x, "x_mean": model.x_mean, "x_std": model.x_std,
                "state_last": model.states[-1], "state_slope": 0.5 * local + 0.5 * global_slope,
                "last_month": str(model.months[-1]), **dec
            })
        except Exception:
            output.append({"draw_id": draw, "fit_success": False})
    return pd.DataFrame(output)


def compute_growth(c4, cutoff, lookback_months=24):
    d = c4.copy()
    d["date"] = pd.to_datetime(d["Publication date"], errors="coerce")
    d["compute"] = pd.to_numeric(d["Training compute (FLOP)"], errors="coerce")
    language = d["Domain"].fillna("").str.contains("language", case=False)
    opened = d["Open model weights?"].fillna("").str.lower().eq("yes")
    d = d[language & opened & d["compute"].gt(0) & d["date"].le(cutoff)].copy()
    start = cutoff - pd.DateOffset(months=lookback_months)
    recent = d[d["date"].ge(start)].copy()
    recent["month"] = recent["date"].dt.to_period("M")
    monthly = recent.groupby("month")["compute"].quantile(.9).rename("compute_q90").reset_index()
    monthly["month_index"] = monthly["month"].apply(lambda m: (m.year - monthly["month"].iloc[0].year) * 12 + m.month - monthly["month"].iloc[0].month)
    monthly["log10_compute_q90"] = np.log10(monthly["compute_q90"])
    if len(monthly) >= 3:
        slope, intercept, lo, hi = theilslopes(monthly["log10_compute_q90"], monthly["month_index"])
    else:
        slope = intercept = lo = hi = 0.0
    info = {
        "n_models": len(recent), "n_months": len(monthly),
        "monthly_log10_slope": float(slope), "annual_log10_slope": float(slope * 12),
        "annual_multiplicative_growth": float(10 ** (slope * 12)),
        "annual_log10_slope_low": float(lo * 12), "annual_log10_slope_high": float(hi * 12),
    }
    return monthly, info


def forecast(model, endpoint, x_endpoint, annual_log10_growth, scenarios, horizons=(12, 24)):
    rows = []
    for factor in scenarios:
        for horizon in horizons:
            month = endpoint.to_period("M") + horizon
            x = x_endpoint + annual_log10_growth * factor * horizon / 12
            y = float(model.predict([x], [month])[0])
            rows.append({
                "scenario_factor": factor, "horizon_months": horizon,
                "forecast_month": str(month), "forecast_log10_compute": x,
                "forecast_logit": y, "forecast_score": inv_score(y),
                "extrapolation_flag": "long_horizon_extrapolation" if horizon >= 24 else "extrapolation",
            })
    return pd.DataFrame(rows)
