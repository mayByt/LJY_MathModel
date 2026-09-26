from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs(project_root: Path, cfg: dict[str, Any]) -> tuple[Path, Path]:
    output = (project_root / cfg["output_root"]).resolve()
    figures = (project_root / cfg["figure_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return output, figures


def load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent
    cfg["input_root"] = str((project_root / cfg["input_root"]).resolve())
    return cfg, project_root


def apply_fast_mode(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg.update(
        bootstrap_reps=30,
        mapping_bootstrap_reps=20,
        interaction_bootstrap_reps=30,
        optimization_bootstrap_reps=20,
        critic_sample_per_domain=2000,
        optimization_starts=10,
        optimization_random_check=2000,
        ridge_alpha_count=15,
        elastic_alpha_count=8,
        tree_estimators=100,
    )
    return cfg


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def finite_or_nan(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def safe_softmax(values: Any) -> np.ndarray | None:
    try:
        a = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        return None
    shifted = a - np.max(a)
    exp = np.exp(shifted)
    denom = exp.sum()
    if not np.isfinite(denom) or denom <= 0:
        return None
    return exp / denom


def ordinal_expectation(values: Any, levels: int = 6) -> float:
    p = safe_softmax(values)
    if p is None or p.size != levels:
        return math.nan
    return float(np.dot(np.arange(levels, dtype=float), p) / (levels - 1))


def binary_positive_probability(values: Any) -> float:
    p = safe_softmax(values)
    if p is None or p.size != 2:
        return math.nan
    return float(p[1])


def weighted_median_matrix(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """逐行加权中位数；NaN 对应权重自动置零。"""
    v = np.asarray(values, dtype=float)
    w = np.broadcast_to(np.asarray(weights, dtype=float), v.shape).copy()
    w[~np.isfinite(v)] = 0.0
    vv = np.where(np.isfinite(v), v, np.inf)
    order = np.argsort(vv, axis=1)
    sorted_v = np.take_along_axis(vv, order, axis=1)
    sorted_w = np.take_along_axis(w, order, axis=1)
    csum = np.cumsum(sorted_w, axis=1)
    cutoff = 0.5 * sorted_w.sum(axis=1)
    idx = (csum >= cutoff[:, None]).argmax(axis=1)
    med = sorted_v[np.arange(v.shape[0]), idx]
    med[~np.isfinite(med)] = np.nan
    return med


def huber_location(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    c: float = 1.345,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> tuple[float, bool, int]:
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(x)
    if not mask.any():
        return math.nan, False, 0
    x = x[mask]
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)[mask]
    w = np.maximum(w, 0)
    if w.sum() <= 0:
        w = np.ones_like(x)
    if scales is None:
        med = np.median(x)
        mad = 1.4826 * np.median(np.abs(x - med))
        s = np.full_like(x, max(mad, 1e-6))
    else:
        s = np.maximum(np.asarray(scales, dtype=float)[mask], 1e-6)
    q = float(np.median(x))
    for iteration in range(1, max_iter + 1):
        r = (x - q) / s
        psi_weight = np.ones_like(r)
        far = np.abs(r) > c
        psi_weight[far] = c / np.abs(r[far])
        # d rho((x-q)/s) / dq 的 IRLS 等价权重包含 1/s^2。
        ww = w * psi_weight / np.square(s)
        q_new = float(np.dot(ww, x) / ww.sum())
        if abs(q_new - q) < tol:
            return float(np.clip(q_new, 0.0, 1.0)), True, iteration
        q = q_new
    # Huber 目标为凸函数；IRLS 慢收敛时直接求一阶条件的唯一根。
    try:
        from scipy.optimize import brentq

        def score(value: float) -> float:
            r = (x - value) / s
            psi = np.clip(r, -c, c)
            return float(np.sum(w * psi / s))

        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo <= tol:
            return float(np.clip((lo + hi) / 2.0, 0.0, 1.0)), True, max_iter
        root = brentq(score, lo, hi, xtol=tol, rtol=max(tol, 1e-12), maxiter=100)
        return float(np.clip(root, 0.0, 1.0)), True, max_iter
    except (ValueError, RuntimeError):
        return float(np.clip(q, 0.0, 1.0)), False, max_iter


def huber_mean_1d(values: Iterable[float], c: float = 1.345, tol: float = 1e-8) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan
    med = np.median(x)
    scale = max(1.4826 * np.median(np.abs(x - med)), 1e-6)
    q = float(med)
    for _ in range(100):
        r = (x - q) / scale
        w = np.ones_like(r)
        mask = np.abs(r) > c
        w[mask] = c / np.abs(r[mask])
        new = float(np.dot(w, x) / w.sum())
        if abs(new - q) < tol:
            return new
        q = new
    return q


def rank_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from scipy.stats import kendalltau, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "spearman": float(spearmanr(yt, yp).statistic),
        "kendall": float(kendalltau(yt, yp).statistic),
    }


def setup_matplotlib() -> str:
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")
    # Noto CJK 的部分 TTC 子集会被 Poppler 错误解析；优先选用经验证可同时
    # 覆盖拉丁字符与中文的文泉驿字体，保证论文插图跨阅读器显示一致。
    font_candidates = [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_path = next((path for path in font_candidates if path.exists()), None)
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font_name = "DejaVu Sans"
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    return font_name
