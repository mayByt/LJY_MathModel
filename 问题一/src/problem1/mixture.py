from __future__ import annotations

import hashlib
import itertools
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskElasticNetCV, Ridge
from sklearn.model_selection import StratifiedKFold

from .common import json_dump, rank_metrics


MIXTURE_FILES = {
    "train_1m": ("train_mixture_1m.csv", "train_pile_loss_1m.csv"),
    "test_1m": ("test_mixture_1m.csv", "test_pile_loss_1m.csv"),
    "test_60m": ("test_mixture_60m.csv", "test_pile_loss_60m.csv"),
    "test_1B": ("test_mixture_1B.csv", "test_pile_loss_1B.csv"),
    "est_10b": ("est_mixture_10b.csv", "est_pile_loss_10b.csv"),
    "est_70b": ("est_mixture_70b.csv", "est_pile_loss_70b.csv"),
}


def _short_domain(column: str) -> str:
    return (
        column.replace("train_the_pile_", "")
        .replace("metric/the_pile_", "")
        .replace("_val_loss", "")
    )


def _hash_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()


def close_composition(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, 0.0)
    row_sum = p.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("存在配比和不为正的行")
    return p / row_sum


def load_mixture_tables(input_root: Path, output_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    table_root = input_root / "regmix_tables"
    datasets: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {"splits": {}}
    mixture_domains = None
    loss_domains = None
    for split, (mix_name, loss_name) in MIXTURE_FILES.items():
        mix = pd.read_csv(table_root / mix_name)
        loss = pd.read_csv(table_root / loss_name)
        if mix["index"].duplicated().any() or loss["index"].duplicated().any():
            raise ValueError(f"{split} 存在重复 index")
        if set(mix["index"]) != set(loss["index"]):
            raise ValueError(f"{split} 配比表与 Loss 表 index 不一致")
        merged = mix.merge(loss, on="index", how="inner", validate="one_to_one").sort_values("index")
        p_cols = [c for c in mix.columns if c != "index"]
        y_cols = [c for c in loss.columns if c != "index"]
        current_m = [_short_domain(c) for c in p_cols]
        current_y = [_short_domain(c) for c in y_cols]
        if mixture_domains is None:
            mixture_domains = current_m
            loss_domains = current_y
        elif current_m != mixture_domains or current_y != loss_domains:
            raise ValueError(f"{split} 域列顺序与训练表不一致")
        raw_p = merged[p_cols].to_numpy(float)
        p = close_composition(raw_p)
        y = merged[y_cols].to_numpy(float)
        datasets[split] = {"index": merged["index"].to_numpy(), "p": p, "y": y}
        audit["splits"][split] = {
            "rows": len(merged),
            "p_columns": len(p_cols),
            "loss_columns": len(y_cols),
            "raw_sum_max_deviation": float(np.max(np.abs(raw_p.sum(axis=1) - 1.0))),
            "closed_sum_max_deviation": float(np.max(np.abs(p.sum(axis=1) - 1.0))),
            "zero_components": int(np.sum(raw_p == 0)),
            "missing": int(np.isnan(np.column_stack([raw_p, y])).sum()),
        }
    audit["mixture_domains"] = mixture_domains
    audit["loss_domains"] = loss_domains
    same_1m_60m = np.allclose(datasets["test_1m"]["p"], datasets["test_60m"]["p"], atol=0, rtol=0)
    audit["test_1m_60m_identical_mixtures"] = bool(same_1m_60m)
    json_dump(output_dir / "data_audit_mixture.json", audit)
    return datasets, audit


def scheffe_features(p: np.ndarray, quadratic: bool = True) -> tuple[np.ndarray, list[tuple[str, int, int | None]]]:
    p = np.asarray(p, dtype=float)
    n, d = p.shape
    features = [p]
    terms: list[tuple[str, int, int | None]] = [("main", i, None) for i in range(d)]
    if quadratic:
        pairs = list(itertools.combinations(range(d), 2))
        inter = np.column_stack([p[:, i] * p[:, j] for i, j in pairs])
        features.append(inter)
        terms.extend(("interaction", i, j) for i, j in pairs)
    return np.column_stack(features), terms


def _folds(p: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = KMeans(n_clusters=max(n_splits, 5), random_state=seed, n_init=20).fit_predict(p)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(p, labels))


@dataclass
class ScaledRidgeModel:
    alpha: float
    scale: np.ndarray
    estimator: Ridge

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict(x / self.scale)

    @property
    def coef_raw(self) -> np.ndarray:
        return self.estimator.coef_ / self.scale[None, :]


def _rms_scale(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.sqrt(np.mean(np.square(x), axis=0)), 1e-8)


def fit_ridge_cv(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    alphas: np.ndarray,
) -> tuple[ScaledRidgeModel, np.ndarray, pd.DataFrame]:
    y_iqr = np.maximum(np.quantile(y, 0.75, axis=0) - np.quantile(y, 0.25, axis=0), 1e-8)
    alpha_rows = []
    predictions: dict[float, np.ndarray] = {}
    for alpha in alphas:
        oof = np.empty_like(y)
        for train_idx, val_idx in folds:
            scale = _rms_scale(x[train_idx])
            model = Ridge(alpha=float(alpha), fit_intercept=False)
            model.fit(x[train_idx] / scale, y[train_idx])
            oof[val_idx] = model.predict(x[val_idx] / scale)
        normalized_rmse = np.sqrt(np.mean(np.square((y - oof) / y_iqr[None, :])))
        median_spearman = float(np.nanmedian([spearmanr(y[:, k], oof[:, k]).statistic for k in range(y.shape[1])]))
        alpha_rows.append(
            {"alpha": float(alpha), "normalized_rmse": normalized_rmse, "median_spearman": median_spearman}
        )
        predictions[float(alpha)] = oof
    alpha_df = pd.DataFrame(alpha_rows)
    best_alpha = float(alpha_df.sort_values(["normalized_rmse", "alpha"]).iloc[0]["alpha"])
    scale = _rms_scale(x)
    estimator = Ridge(alpha=best_alpha, fit_intercept=False).fit(x / scale, y)
    return ScaledRidgeModel(best_alpha, scale, estimator), predictions[best_alpha], alpha_df


def _candidate_summary(y: np.ndarray, pred: np.ndarray, model: str) -> dict[str, Any]:
    spears = [spearmanr(y[:, k], pred[:, k]).statistic for k in range(y.shape[1])]
    kendalls = [kendalltau(y[:, k], pred[:, k]).statistic for k in range(y.shape[1])]
    iqr = np.maximum(np.quantile(y, 0.75, axis=0) - np.quantile(y, 0.25, axis=0), 1e-8)
    nrmse = float(np.sqrt(np.mean(np.square((y - pred) / iqr[None, :]))))
    return {
        "model": model,
        "normalized_rmse": nrmse,
        "median_spearman": float(np.nanmedian(spears)),
        "median_kendall": float(np.nanmedian(kendalls)),
    }


def _fit_elastic_oof(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cfg: dict[str, Any],
) -> tuple[Any, np.ndarray, dict[str, float]]:
    alpha_grid = np.geomspace(1e-6, 1e-1, int(cfg["elastic_alpha_count"]))
    scale = _rms_scale(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector = MultiTaskElasticNetCV(
            l1_ratio=list(map(float, cfg["elastic_l1_ratios"])),
            alphas=alpha_grid,
            cv=folds,
            fit_intercept=False,
            max_iter=5000,
            tol=1e-4,
            n_jobs=-1,
            selection="random",
            random_state=int(cfg["seed"]),
        ).fit(x / scale, y)
        best = (float(selector.alpha_), float(selector.l1_ratio_))
        oof = np.empty_like(y)
        for fold_id, (train_idx, val_idx) in enumerate(folds):
            fold_scale = _rms_scale(x[train_idx])
            est = MultiTaskElasticNet(
                alpha=best[0],
                l1_ratio=best[1],
                fit_intercept=False,
                max_iter=5000,
                tol=1e-4,
                selection="random",
                random_state=int(cfg["seed"]) + fold_id,
            ).fit(x[train_idx] / fold_scale, y[train_idx])
            oof[val_idx] = est.predict(x[val_idx] / fold_scale)
        estimator = MultiTaskElasticNet(
            alpha=best[0],
            l1_ratio=best[1],
            fit_intercept=False,
            max_iter=10000,
            tol=1e-5,
            selection="random",
            random_state=int(cfg["seed"]),
        ).fit(x / scale, y)
    return (estimator, scale), oof, {"alpha": best[0], "l1_ratio": best[1]}


def _tree_oof(
    p: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    kind: str,
    seed: int,
    n_estimators: int,
) -> tuple[Any, np.ndarray]:
    kwargs = dict(n_estimators=n_estimators, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=seed)
    cls = ExtraTreesRegressor if kind == "extra_trees" else RandomForestRegressor
    oof = np.empty_like(y)
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        model = cls(**{**kwargs, "random_state": seed + fold_id})
        model.fit(p[train_idx], y[train_idx])
        oof[val_idx] = model.predict(p[val_idx])
    final = cls(**kwargs).fit(p, y)
    return final, oof


def _crossfit_affine(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    out = np.empty_like(y)
    for train_idx, val_idx in folds:
        for k in range(y.shape[1]):
            design = np.column_stack([np.ones(len(train_idx)), pred[train_idx, k]])
            coef, *_ = np.linalg.lstsq(design, y[train_idx, k], rcond=None)
            out[val_idx, k] = coef[0] + coef[1] * pred[val_idx, k]
    return out


def _split_metrics(
    split: str,
    y: np.ndarray,
    pred: np.ndarray,
    loss_domains: list[str],
    top_fraction: float,
    calibrated_pred: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows = []
    absolute = calibrated_pred if calibrated_pred is not None else pred
    top_n = max(1, int(math.ceil(len(y) * top_fraction)))
    for k, domain in enumerate(loss_domains):
        metrics = rank_metrics(y[:, k], absolute[:, k])
        # 秩指标始终用未校准预测。
        metrics["spearman"] = float(spearmanr(y[:, k], pred[:, k]).statistic)
        metrics["kendall"] = float(kendalltau(y[:, k], pred[:, k]).statistic)
        true_top = set(np.argsort(y[:, k])[:top_n])
        pred_top = set(np.argsort(pred[:, k])[:top_n])
        metrics["top_overlap"] = len(true_top & pred_top) / top_n
        best_idx = int(np.argmin(pred[:, k]))
        regret = float(y[best_idx, k] - np.min(y[:, k]))
        iqr = max(float(np.quantile(y[:, k], 0.75) - np.quantile(y[:, k], 0.25)), 1e-8)
        metrics["regret"] = regret
        metrics["normalized_regret"] = regret / iqr
        metrics.update(split=split, loss_domain=domain, n=len(y), affine_calibrated=calibrated_pred is not None)
        rows.append(metrics)
    return rows


def _nearest_distance(p: np.ndarray, train_p: np.ndarray, scale: np.ndarray) -> float:
    return float(np.min(np.linalg.norm((train_p - p[None, :]) / scale[None, :], axis=1)))


def _support_radius(train_p: np.ndarray, scale: np.ndarray, quantile: float) -> float:
    dist = np.linalg.norm((train_p[:, None, :] - train_p[None, :, :]) / scale[None, None, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(np.quantile(np.min(dist, axis=1), quantile))


def quadratic_objective(coef: np.ndarray, p: np.ndarray) -> float:
    """计算二阶 Scheffé 综合目标；系数顺序与 ``scheffe_features`` 一致。"""
    p = np.asarray(p, dtype=float)
    d = len(p)
    pairs = itertools.combinations(range(d), 2)
    value = float(np.dot(coef[:d], p))
    value += sum(coef[d + t] * p[i] * p[j] for t, (i, j) in enumerate(pairs))
    return float(value)


def diagnose_optimization(
    coef: np.ndarray,
    p_star: np.ndarray,
    history: pd.DataFrame,
    train_p: np.ndarray,
    upper: np.ndarray,
    support_radius: float,
    support_scale: np.ndarray,
    random_count: int,
    seed: int,
) -> dict[str, Any]:
    """执行多起点一致性和随机可行点反证检查。"""
    feasible_history = history.loc[history["feasible"]].sort_values("objective")
    best_values = feasible_history["objective"].head(5).to_numpy(float)
    if len(best_values) >= 2:
        multistart_dispersion = float(
            (best_values.max() - best_values.min()) / max(abs(best_values.min()), 1e-12)
        )
    else:
        multistart_dispersion = math.inf

    rng = np.random.default_rng(seed)
    random_best = math.inf
    accepted = 0
    attempted = 0
    # 凸组合天然满足单纯形；再检查分量上界与经验支持域约束。
    while accepted < random_count and attempted < max(10000, random_count * 50):
        batch = min(512, random_count - accepted)
        ids = rng.integers(0, len(train_p), size=(batch, 3))
        weights = rng.dirichlet(np.ones(3), size=batch)
        candidates = np.einsum("bi,bij->bj", weights, train_p[ids])
        attempted += batch
        candidates = candidates[np.all(candidates <= upper[None, :] + 1e-12, axis=1)]
        for candidate in candidates:
            if _nearest_distance(candidate, train_p, support_scale) <= support_radius + 1e-12:
                random_best = min(random_best, quadratic_objective(coef, candidate))
                accepted += 1
                if accepted >= random_count:
                    break

    p_star_objective = quadratic_objective(coef, p_star)
    support_distance = _nearest_distance(p_star, train_p, support_scale)
    boundary_tolerance = max(1e-8, support_radius * 1e-8)
    boundary_active = bool(abs(support_radius - support_distance) <= boundary_tolerance)
    random_check_pass = bool(accepted > 0 and p_star_objective <= random_best + 1e-10)
    multistart_pass = bool(len(best_values) >= 5 and multistart_dispersion <= 1e-6)
    return {
        "best_feasible_starts_used": int(len(best_values)),
        "best_feasible_objectives": best_values.tolist(),
        "multistart_relative_dispersion": multistart_dispersion,
        "multistart_threshold": 1e-6,
        "multistart_pass": multistart_pass,
        "random_requested": int(random_count),
        "random_accepted": int(accepted),
        "random_attempted": int(attempted),
        "random_best_objective": None if not np.isfinite(random_best) else float(random_best),
        "p_star_objective": p_star_objective,
        "random_check_pass": random_check_pass,
        "support_boundary_active": boundary_active,
        "support_boundary_gap": float(support_radius - support_distance),
        "near_optimal_only": bool((not multistart_pass) or boundary_active),
    }


def rebuild_optimization_diagnostics(
    input_root: Path,
    cfg: dict[str, Any],
    output_dir: Path,
    audit: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """从正式结果真源重建优化诊断，供 ``finalize`` 使用而无需重新拟合。"""
    table_root = input_root / "regmix_tables"
    mix = pd.read_csv(table_root / MIXTURE_FILES["train_1m"][0])
    loss = pd.read_csv(table_root / MIXTURE_FILES["train_1m"][1])
    merged = mix.merge(loss, on="index", how="inner", validate="one_to_one").sort_values("index")
    p_cols = [c for c in mix.columns if c != "index"]
    y_cols = [c for c in loss.columns if c != "index"]
    p_train = close_composition(merged[p_cols].to_numpy(float))
    y_train = merged[y_cols].to_numpy(float)
    coef_df = pd.read_csv(output_dir / "mixture_coefficients.csv.gz")
    coef = np.vstack(
        [
            coef_df.loc[coef_df["loss_domain"] == domain, "coefficient"].to_numpy(float)
            for domain in audit["loss_domains"]
        ]
    )
    y_iqr = np.maximum(np.quantile(y_train, 0.75, axis=0) - np.quantile(y_train, 0.25, axis=0), 1e-8)
    combined_coef = np.mean(coef / y_iqr[:, None], axis=0)
    upper = np.quantile(p_train, float(cfg["support_quantile"]), axis=0)
    support_scale = np.maximum(np.std(p_train, axis=0), 1e-6)
    p_star_table = pd.read_csv(output_dir / "optimal_mixture.csv").set_index("domain")
    p_star = p_star_table.loc[audit["mixture_domains"], "p_star"].to_numpy(float)
    history = pd.read_csv(output_dir / "optimization_history.csv")
    diagnostics = diagnose_optimization(
        combined_coef,
        p_star,
        history,
        p_train,
        upper,
        float(summary["support_radius"]),
        support_scale,
        int(cfg["optimization_random_check"]),
        int(cfg["seed"]) + 307,
    )
    json_dump(output_dir / "optimization_diagnostics.json", diagnostics)
    summary["optimization_diagnostics"] = diagnostics
    return diagnostics


def optimize_mixture(
    coef: np.ndarray,
    train_p: np.ndarray,
    upper: np.ndarray,
    support_radius: float,
    support_scale: np.ndarray,
    starts: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    d = train_p.shape[1]
    pairs = list(itertools.combinations(range(d), 2))

    def objective(p: np.ndarray) -> float:
        return quadratic_objective(coef, p)

    def gradient(p: np.ndarray) -> np.ndarray:
        g = coef[:d].copy()
        for t, (i, j) in enumerate(pairs):
            b = coef[d + t]
            g[i] += b * p[j]
            g[j] += b * p[i]
        return g

    def support_constraint(p: np.ndarray) -> float:
        return support_radius - _nearest_distance(p, train_p, support_scale)

    rng = np.random.default_rng(seed)
    valid_rows = train_p[np.all(train_p <= upper[None, :] + 1e-12, axis=1)]
    if len(valid_rows) == 0:
        valid_rows = train_p.copy()
    candidates = [valid_rows[i] for i in rng.choice(len(valid_rows), size=min(starts, len(valid_rows)), replace=False)]
    attempts = 0
    while len(candidates) < starts and attempts < starts * 50:
        ids = rng.choice(len(train_p), size=3, replace=True)
        w = rng.dirichlet(np.ones(3))
        p = w @ train_p[ids]
        if np.all(p <= upper + 1e-12) and support_constraint(p) >= -1e-12:
            candidates.append(p)
        attempts += 1

    rows = []
    best_p = None
    best_value = math.inf
    for start_id, p0 in enumerate(candidates):
        res = minimize(
            objective,
            p0,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, float(u)) for u in upper],
            constraints=[
                {"type": "eq", "fun": lambda p: np.sum(p) - 1.0, "jac": lambda p: np.ones_like(p)},
                {"type": "ineq", "fun": support_constraint},
            ],
            options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
        )
        p = np.maximum(res.x, 0)
        p = p / p.sum()
        support_margin = support_constraint(p)
        feasible = (
            abs(p.sum() - 1.0) <= 1e-10
            and np.min(p) >= -1e-12
            and np.max(p - upper) <= 1e-10
            and support_margin >= -1e-8
        )
        value = objective(p)
        rows.append(
            {
                "start_id": start_id,
                "success": bool(res.success),
                "feasible": bool(feasible),
                "objective": value,
                "support_margin": support_margin,
                "iterations": int(res.nit),
                "message": str(res.message),
                **{f"p_{i}": float(value) for i, value in enumerate(p)},
            }
        )
        if feasible and value < best_value:
            best_value = value
            best_p = p.copy()
    if best_p is None:
        raise RuntimeError("多起点优化未找到可行解")
    return best_p, pd.DataFrame(rows)


def run_mixture_pipeline(
    input_root: Path,
    cfg: dict[str, Any],
    output_dir: Path,
    domain_mapping: pd.DataFrame,
) -> dict[str, Any]:
    datasets, audit = load_mixture_tables(input_root, output_dir)
    mix_domains = audit["mixture_domains"]
    loss_domains = audit["loss_domains"]
    train = datasets["train_1m"]
    p_train, y_train = train["p"], train["y"]
    folds = _folds(p_train, int(cfg["cv_folds"]), int(cfg["seed"]))
    alpha_grid = np.geomspace(
        10 ** float(cfg["ridge_alpha_min_exp"]),
        10 ** float(cfg["ridge_alpha_max_exp"]),
        int(cfg["ridge_alpha_count"]),
    )

    x_linear, linear_terms = scheffe_features(p_train, quadratic=False)
    x_quad, quad_terms = scheffe_features(p_train, quadratic=True)
    if x_quad.shape[1] != 153:
        raise AssertionError(f"二阶 Scheffé 特征应为 153，实际 {x_quad.shape[1]}")
    linear_model, linear_oof, linear_path = fit_ridge_cv(x_linear, y_train, folds, alpha_grid)
    quad_model, quad_oof, quad_path = fit_ridge_cv(x_quad, y_train, folds, alpha_grid)
    linear_path.assign(model="linear_ridge").to_csv(output_dir / "ridge_path_linear.csv", index=False)
    quad_path.assign(model="quadratic_ridge").to_csv(output_dir / "ridge_path_quadratic.csv", index=False)

    elastic_model, elastic_oof, elastic_params = _fit_elastic_oof(x_quad, y_train, folds, cfg)
    extra_model, extra_oof = _tree_oof(
        p_train, y_train, folds, "extra_trees", int(cfg["seed"]), int(cfg["tree_estimators"])
    )
    rf_model, rf_oof = _tree_oof(
        p_train, y_train, folds, "random_forest", int(cfg["seed"]), int(cfg["tree_estimators"])
    )

    candidates = [
        _candidate_summary(y_train, linear_oof, "linear_ridge"),
        _candidate_summary(y_train, quad_oof, "quadratic_ridge"),
        _candidate_summary(y_train, elastic_oof, "elastic_net"),
        _candidate_summary(y_train, extra_oof, "extra_trees"),
        _candidate_summary(y_train, rf_oof, "random_forest"),
    ]
    candidate_df = pd.DataFrame(candidates).sort_values("normalized_rmse")
    candidate_df.to_csv(output_dir / "mixture_internal_cv.csv", index=False)
    best_blackbox = candidate_df[candidate_df["model"].isin(["extra_trees", "random_forest"])].iloc[0]
    ridge_row = candidate_df[candidate_df["model"] == "quadratic_ridge"].iloc[0]
    relative_gap = (ridge_row["normalized_rmse"] - best_blackbox["normalized_rmse"]) / best_blackbox[
        "normalized_rmse"
    ]
    model_role = "quadratic_ridge_main" if relative_gap <= 0.05 else "dual_track"

    def predict_quad(p: np.ndarray) -> np.ndarray:
        return quad_model.predict(scheffe_features(p, quadratic=True)[0])

    metric_rows = []
    prediction_rows = []
    for split in ["test_1m", "test_60m", "test_1B", "est_10b", "est_70b"]:
        p = datasets[split]["p"]
        y = datasets[split]["y"]
        pred = predict_quad(p)
        calibrated = None
        if split in {"test_60m", "test_1B"}:
            ext_folds = _folds(p, min(int(cfg["cv_folds"]), max(2, len(p) // 20)), int(cfg["seed"]) + 91)
            calibrated = _crossfit_affine(y, pred, ext_folds)
        metric_rows.extend(
            _split_metrics(split, y, pred, loss_domains, float(cfg["top_fraction"]), calibrated)
        )
        for row_idx, idx in enumerate(datasets[split]["index"]):
            for k, domain in enumerate(loss_domains):
                prediction_rows.append(
                    {
                        "split": split,
                        "index": idx,
                        "loss_domain": domain,
                        "observed": y[row_idx, k],
                        "predicted": pred[row_idx, k],
                        "predicted_calibrated": calibrated[row_idx, k] if calibrated is not None else np.nan,
                    }
                )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "mixture_metrics.csv", index=False)
    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(output_dir / "mixture_predictions.csv.gz", index=False, compression="gzip")

    coef = quad_model.coef_raw
    coef_rows = []
    for k, loss_domain in enumerate(loss_domains):
        for t, (kind, i, j) in enumerate(quad_terms):
            coef_rows.append(
                {
                    "loss_domain": loss_domain,
                    "term_type": kind,
                    "domain_i": mix_domains[i],
                    "domain_j": mix_domains[j] if j is not None else "",
                    "coefficient": coef[k, t],
                }
            )

    # 固定 alpha 的训练行 bootstrap，估计系数和符号稳定性。
    rng = np.random.default_rng(int(cfg["seed"]) + 123)
    reps = int(cfg["interaction_bootstrap_reps"])
    coef_draws = np.empty((reps, len(loss_domains), x_quad.shape[1]), dtype=np.float32)
    p_ref_draws = np.empty((reps, len(mix_domains)), dtype=np.float32)
    bootstrap_sample_hashes = np.empty(reps, dtype="U64")
    for b in range(reps):
        ids = rng.integers(0, len(p_train), len(p_train))
        xb, yb = x_quad[ids], y_train[ids]
        scale = _rms_scale(xb)
        est = Ridge(alpha=quad_model.alpha, fit_intercept=False).fit(xb / scale, yb)
        coef_draws[b] = (est.coef_ / scale[None, :]).astype(np.float32)
        p_ref_draws[b] = np.mean(p_train[ids], axis=0).astype(np.float32)
        bootstrap_sample_hashes[b] = _hash_ids(ids)
    coef_df = pd.DataFrame(coef_rows)
    flat_draws = coef_draws.transpose(1, 2, 0).reshape(-1, reps)
    coef_df["ci_low"] = np.quantile(flat_draws, 0.025, axis=1)
    coef_df["ci_high"] = np.quantile(flat_draws, 0.975, axis=1)
    positive_prob = np.mean(flat_draws > 0, axis=1)
    coef_df["sign_probability"] = np.maximum(positive_prob, 1.0 - positive_prob)
    coef_df.to_csv(output_dir / "mixture_coefficients.csv.gz", index=False, compression="gzip")
    coef_df[coef_df["term_type"] == "interaction"].to_csv(output_dir / "interaction_stability.csv", index=False)

    y_med = np.median(y_train, axis=0)
    y_iqr = np.maximum(np.quantile(y_train, 0.75, axis=0) - np.quantile(y_train, 0.25, axis=0), 1e-8)
    combined_coef = np.mean(coef / y_iqr[:, None], axis=0)
    upper = np.quantile(p_train, float(cfg["support_quantile"]), axis=0)
    support_scale = np.maximum(np.std(p_train, axis=0), 1e-6)
    radius = _support_radius(p_train, support_scale, float(cfg["support_quantile"]))
    p_star, history = optimize_mixture(
        combined_coef,
        p_train,
        upper,
        radius,
        support_scale,
        int(cfg["optimization_starts"]),
        int(cfg["seed"]) + 211,
    )
    history.to_csv(output_dir / "optimization_history.csv", index=False)
    optimization_diagnostics = diagnose_optimization(
        combined_coef,
        p_star,
        history,
        p_train,
        upper,
        radius,
        support_scale,
        int(cfg["optimization_random_check"]),
        int(cfg["seed"]) + 307,
    )
    json_dump(output_dir / "optimization_diagnostics.json", optimization_diagnostics)

    # 方向边际效应：在训练平均配比处，将 j 的小份额转到 i。
    ref_p = np.mean(p_train, axis=0)
    pairs = list(itertools.combinations(range(len(mix_domains)), 2))
    gradients = np.empty((len(loss_domains), len(mix_domains)))
    for k in range(len(loss_domains)):
        g = coef[k, : len(mix_domains)].copy()
        for t, (i, j) in enumerate(pairs):
            b = coef[k, len(mix_domains) + t]
            g[i] += b * ref_p[j]
            g[j] += b * ref_p[i]
        gradients[k] = g
    centered_effect = gradients - gradients.mean(axis=1, keepdims=True)
    mean_effect = centered_effect.mean(axis=0)
    effect_df = pd.DataFrame(
        {
            "domain": mix_domains,
            "mean_directional_effect": mean_effect,
            "benefit_score": -mean_effect,
        }
    )
    effect_df.to_csv(output_dir / "mixture_directional_effects.csv", index=False)

    q_map = domain_mapping.set_index("domain")["q"].to_dict()
    q_values = np.array([q_map.get(d, np.nan) for d in mix_domains])
    valid = np.isfinite(q_values)
    quality_effect_association = {
        "spearman_q_vs_benefit": float(spearmanr(q_values[valid], -mean_effect[valid]).statistic),
        "kendall_q_vs_benefit": float(kendalltau(q_values[valid], -mean_effect[valid]).statistic),
        "n_domains": int(valid.sum()),
    }
    json_dump(output_dir / "quality_effect_association.json", quality_effect_association)

    # 优化 bootstrap；以原最优解和若干训练点为起点，控制计算量。
    opt_reps = int(cfg["optimization_bootstrap_reps"])
    p_draws = np.empty((opt_reps, len(mix_domains)), dtype=float)
    success = np.zeros(opt_reps, dtype=bool)
    for b in range(opt_reps):
        combined_b = np.mean(coef_draws[b].astype(float) / y_iqr[:, None], axis=0)
        try:
            pb, _ = optimize_mixture(
                combined_b,
                p_train,
                upper,
                radius,
                support_scale,
                max(5, int(cfg["optimization_starts"]) // 4),
                int(cfg["seed"]) + 1000 + b,
            )
            p_draws[b] = pb
            success[b] = True
        except RuntimeError:
            p_draws[b] = np.nan
    valid_draws = p_draws[success]
    optimal = pd.DataFrame(
        {
            "domain": mix_domains,
            "p_star": p_star,
            "ci_low": np.nanquantile(valid_draws, 0.025, axis=0) if len(valid_draws) else np.nan,
            "ci_high": np.nanquantile(valid_draws, 0.975, axis=0) if len(valid_draws) else np.nan,
            "zero_probability": np.mean(valid_draws < 1e-6, axis=0) if len(valid_draws) else np.nan,
            "upper_bound": upper,
        }
    )
    optimal.to_csv(output_dir / "optimal_mixture.csv", index=False)
    np.savez_compressed(output_dir / "optimal_mixture_bootstrap.npz", p=p_draws, success=success)

    # 问题二只消费本冻结接口，不再直接读取附件 A。
    bridge_payload: dict[str, Any] = {
        "interface_version": np.array("p1_to_p2_v2"),
        "domains": np.asarray(mix_domains),
        "loss_domains": np.asarray(loss_domains),
        "train_index": np.asarray(train["index"]),
        "p_train": p_train.astype(np.float64),
        "y_train": y_train.astype(np.float64),
        "coefficient_point": coef.astype(np.float64),
        "coefficient_draws": coef_draws,
        "p_ref_point": ref_p.astype(np.float64),
        "p_ref_draws": p_ref_draws,
        "bootstrap_sample_index_hash": bootstrap_sample_hashes,
        "loss_iqr": y_iqr.astype(np.float64),
        "ridge_alpha": np.array(float(quad_model.alpha)),
        "upper": upper.astype(np.float64),
        "support_scale": support_scale.astype(np.float64),
        "support_radius": np.array(float(radius)),
    }
    for bridge_split in ["test_1m", "test_60m", "test_1B"]:
        bridge_payload[f"{bridge_split}_index"] = np.asarray(datasets[bridge_split]["index"])
        bridge_payload[f"{bridge_split}_p"] = datasets[bridge_split]["p"].astype(np.float64)
        bridge_payload[f"{bridge_split}_y"] = datasets[bridge_split]["y"].astype(np.float64)
    np.savez_compressed(output_dir / "problem2_bridge_input.npz", **bridge_payload)

    result_summary = {
        "linear_alpha": linear_model.alpha,
        "quadratic_alpha": quad_model.alpha,
        "elastic_params": elastic_params,
        "model_role": model_role,
        "ridge_vs_best_blackbox_relative_nrmse_gap": float(relative_gap),
        "support_radius": radius,
        "optimization_success_rate": float(success.mean()),
        "p_star_sum": float(p_star.sum()),
        "p_star_min": float(p_star.min()),
        "p_star_max_upper_violation": float(np.max(p_star - upper)),
        "p_star_support_distance": _nearest_distance(p_star, p_train, support_scale),
        "optimization_diagnostics": optimization_diagnostics,
        "quality_effect_association": quality_effect_association,
    }
    json_dump(output_dir / "mixture_summary.json", result_summary)
    return {
        "datasets": datasets,
        "audit": audit,
        "candidate_metrics": candidate_df,
        "metrics": metrics_df,
        "predictions": predictions_df,
        "coefficients": coef_df,
        "effects": effect_df,
        "optimal": optimal,
        "summary": result_summary,
        "quad_model": quad_model,
    }
