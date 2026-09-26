from __future__ import annotations

import json
import lzma
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance

from .common import json_dump, sha256_file
from .data_quality import text_distribution_features


FEATURE_COLUMNS = [
    "log_chars",
    "log_lines",
    "alpha_ratio",
    "digit_ratio",
    "upper_ratio",
    "punct_ratio",
    "whitespace_ratio",
    "url_per_line",
    "code_symbol_ratio",
    "token_richness",
    "top2_proxy",
    "top3_proxy",
]


def load_a18_features(input_root: Path, cfg: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    path = input_root / "regmix_domain_sample.jsonl.xz"
    cache = output_dir / "_cache_a18_features.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    rows = []
    domain_counts: dict[str, int] = {}
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            domain = str(row.get("_source_domain", "")).strip().lower()
            feat = {"domain": domain}
            feat.update(
                text_distribution_features(
                    str(row.get("text") or ""),
                    char_cap=int(cfg["text_char_cap"]),
                    token_cap=int(cfg["text_token_cap"]),
                )
            )
            rows.append(feat)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    data = pd.DataFrame.from_records(rows)
    json_dump(
        output_dir / "data_audit_a18.json",
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": len(data),
            "domains": domain_counts,
            "features": FEATURE_COLUMNS,
        },
    )
    summary = data.groupby("domain")[FEATURE_COLUMNS].agg(["count", "mean", "std", "median", "min", "max"])
    summary.to_csv(output_dir / "a18_text_feature_summary.csv")
    data.to_pickle(cache)
    return data


def _feature_scales(anchors: pd.DataFrame) -> dict[str, float]:
    scales = {}
    for field in FEATURE_COLUMNS:
        x = anchors[field].dropna().to_numpy(float)
        q25, q75 = np.quantile(x, [0.25, 0.75])
        scales[field] = max(float(q75 - q25), float(np.std(x)), 1e-6)
    return scales


def _distance_matrix(
    mixture: pd.DataFrame,
    anchors: pd.DataFrame,
    scales: dict[str, float],
) -> pd.DataFrame:
    mix_domains = sorted(mixture["domain"].unique())
    anchor_domains = sorted(anchors["domain"].unique())
    rows = []
    for md in mix_domains:
        m = mixture[mixture["domain"] == md]
        for ad in anchor_domains:
            a = anchors[anchors["domain"] == ad]
            w_terms = []
            m_terms = []
            for field in FEATURE_COLUMNS:
                xv = m[field].dropna().to_numpy(float)
                yv = a[field].dropna().to_numpy(float)
                scale = scales[field]
                w_terms.append(wasserstein_distance(xv, yv) / scale)
                pooled = max(float(np.sqrt(np.var(xv) + np.var(yv))), scale * 0.25, 1e-6)
                m_terms.append(abs(float(np.mean(xv) - np.mean(yv))) / pooled)
            dw = float(np.mean(w_terms))
            dm = float(np.sqrt(np.mean(np.square(m_terms))))
            rows.append(
                {
                    "mixture_domain": md,
                    "anchor_domain": ad,
                    "wasserstein": dw,
                    "mahalanobis_diag": dm,
                    "distance": 0.5 * dw + 0.5 * dm,
                }
            )
    return pd.DataFrame(rows)


def _soft_weights(distances: np.ndarray, tau: float) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    logits = -(d - np.min(d)) / max(tau, 1e-8)
    weights = np.exp(np.clip(logits, -700, 0))
    return weights / weights.sum()


def _select_temperature(
    distances: pd.DataFrame,
    mapping: pd.DataFrame,
    q_lookup: dict[str, float],
) -> tuple[float, pd.DataFrame]:
    known = mapping[mapping["mapping_type"].isin(["direct", "near_direct"])].copy()
    taus = np.geomspace(0.02, 5.0, 60)
    rows = []
    for tau in taus:
        errors = []
        pairs = []
        for _, row in known.iterrows():
            md = row["mixture_domain"]
            target = row["quality_domain"]
            sub = distances[(distances["mixture_domain"] == md) & (distances["anchor_domain"] != target)]
            sub = sub[sub["anchor_domain"].isin(q_lookup)]
            if sub.empty:
                continue
            w = _soft_weights(sub["distance"].to_numpy(float), tau)
            pred = float(np.dot(w, [q_lookup[x] for x in sub["anchor_domain"]]))
            truth = q_lookup[target]
            errors.append(abs(pred - truth))
            pairs.append((md, target, pred, truth))
        rows.append({"tau": tau, "mae": float(np.mean(errors)) if errors else math.inf, "pairs": pairs})
    best = min(rows, key=lambda x: x["mae"])
    loo_rows = [
        {
            "mixture_domain": md,
            "quality_domain": qd,
            "predicted_q": pred,
            "true_q": truth,
            "error": pred - truth,
            "tau": best["tau"],
        }
        for md, qd, pred, truth in best["pairs"]
    ]
    return float(best["tau"]), pd.DataFrame(loo_rows)


def _bootstrap_summary_distance(
    mix_arrays: dict[str, np.ndarray],
    anchor_arrays: dict[str, np.ndarray],
    scales: np.ndarray,
    rng: np.random.Generator,
    sample_cap: int = 500,
) -> dict[tuple[str, str], float]:
    mix_medians = {}
    anchor_medians = {}
    for domain, arr in mix_arrays.items():
        n = min(len(arr), sample_cap)
        sampled = arr[rng.integers(0, len(arr), n)]
        mix_medians[domain] = np.nanmedian(sampled, axis=0)
    for domain, arr in anchor_arrays.items():
        n = min(len(arr), sample_cap)
        sampled = arr[rng.integers(0, len(arr), n)]
        anchor_medians[domain] = np.nanmedian(sampled, axis=0)
    out = {}
    for md, mm in mix_medians.items():
        for ad, am in anchor_medians.items():
            out[(md, ad)] = float(np.sqrt(np.mean(np.square((mm - am) / scales))))
    return out


def run_mapping_pipeline(
    input_root: Path,
    cfg: dict[str, Any],
    output_dir: Path,
    text_features_a1: pd.DataFrame,
    quality_domain: pd.DataFrame,
    quality_bootstrap: dict[str, np.ndarray],
) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg["seed"]) + 17)
    a18 = load_a18_features(input_root, cfg, output_dir)
    mapping = pd.read_csv(input_root / "domain_mapping_guide.csv")
    summary = pd.read_csv(input_root / "regmix_domain_summary.csv")
    q_lookup = quality_domain.set_index("domain")["q"].to_dict()
    scales = _feature_scales(text_features_a1)
    distances = _distance_matrix(a18, text_features_a1, scales)
    distances.to_csv(output_dir / "domain_text_distances.csv", index=False)
    tau, loo = _select_temperature(distances, mapping, q_lookup)
    if not loo.empty:
        loo.to_csv(output_dir / "domain_mapping_loo.csv", index=False)
    loo_mae = float(loo["error"].abs().mean()) if not loo.empty else math.nan
    loo_spearman = (
        float(spearmanr(loo["true_q"], loo["predicted_q"]).statistic) if len(loo) >= 3 else math.nan
    )
    loo_rmse = float(np.sqrt(np.mean(np.square(loo["error"])))) if not loo.empty else 0.1
    mapping_rank_reliable = bool(np.isfinite(loo_spearman) and loo_spearman >= 0.70)
    global_q = float(np.mean(list(q_lookup.values())))
    n_lookup = summary.set_index("domain")["sample_rows"].to_dict()

    result_rows = []
    weight_rows = []
    for _, row in mapping.iterrows():
        md = row["mixture_domain"]
        mtype = row["mapping_type"]
        qd = row["quality_domain"] if isinstance(row["quality_domain"], str) else ""
        if mtype in {"direct", "near_direct"} and qd in q_lookup:
            q = float(q_lookup[qd])
            reliability = 1.0 if mtype == "direct" else 0.9
            dmin = 0.0
            weights = {qd: 1.0}
        else:
            sub = distances[(distances["mixture_domain"] == md) & distances["anchor_domain"].isin(q_lookup)]
            anchors = sub["anchor_domain"].tolist()
            w = _soft_weights(sub["distance"].to_numpy(float), tau)
            soft_q = float(np.dot(w, [q_lookup[a] for a in anchors]))
            dmin = float(sub["distance"].min())
            n = float(n_lookup.get(md, 0))
            reliability = n / (n + 500.0 * (1.0 + dmin)) if mapping_rank_reliable else 0.0
            q = reliability * soft_q + (1.0 - reliability) * global_q
            weights = dict(zip(anchors, map(float, w)))
        result_rows.append(
            {
                "domain": md,
                "q": q,
                "mapping_type": mtype,
                "quality_anchor": qd if qd else ("soft" if mapping_rank_reliable else "global_mean_fallback"),
                "reliability": reliability,
                "nearest_distance": dmin,
                "sample_rows": int(n_lookup.get(md, 0)),
                "low_support_flag": int(n_lookup.get(md, 0) < 100),
            }
        )
        for anchor, weight in weights.items():
            weight_rows.append({"mixture_domain": md, "anchor_domain": anchor, "weight": weight})

    result = pd.DataFrame(result_rows)
    mix_arrays = {d: g[FEATURE_COLUMNS].to_numpy(float) for d, g in a18.groupby("domain")}
    anchor_arrays = {
        d: g[FEATURE_COLUMNS].to_numpy(float) for d, g in text_features_a1.groupby("domain")
    }
    scale_arr = np.array([scales[f] for f in FEATURE_COLUMNS], dtype=float)
    reps = int(cfg["mapping_bootstrap_reps"])
    draws = {d: np.empty(reps, dtype=float) for d in result["domain"]}
    mapped_info = mapping.set_index("mixture_domain").to_dict("index")
    result_info = result.set_index("domain").to_dict("index")
    for b in range(reps):
        dboot = _bootstrap_summary_distance(mix_arrays, anchor_arrays, scale_arr, rng)
        qdraw = {
            domain: float(values[rng.integers(0, len(values))])
            for domain, values in quality_bootstrap.items()
        }
        global_draw = float(np.mean(list(qdraw.values())))
        for md in result["domain"]:
            info = mapped_info[md]
            mtype = info["mapping_type"]
            qd = info["quality_domain"] if isinstance(info["quality_domain"], str) else ""
            if mtype in {"direct", "near_direct"} and qd in qdraw:
                value = qdraw[qd]
            else:
                anchors = [a for a in qdraw if (md, a) in dboot]
                dd = np.array([dboot[(md, a)] for a in anchors])
                w = _soft_weights(dd, tau)
                soft = float(np.dot(w, [qdraw[a] for a in anchors]))
                rel = float(result_info[md]["reliability"])
                value = rel * soft + (1.0 - rel) * global_draw
            draws[md][b] = value
    result["ci_low"] = [np.quantile(draws[d], 0.025) for d in result["domain"]]
    result["ci_high"] = [np.quantile(draws[d], 0.975) for d in result["domain"]]
    inferred = result["mapping_type"] == "inferred"
    half_floor = 1.96 * max(loo_rmse, 0.01)
    result["sensitivity_ci_low"] = result["ci_low"]
    result["sensitivity_ci_high"] = result["ci_high"]
    result.loc[inferred, "sensitivity_ci_low"] = np.minimum(
        result.loc[inferred, "ci_low"], result.loc[inferred, "q"] - half_floor
    ).clip(0, 1)
    result.loc[inferred, "sensitivity_ci_high"] = np.maximum(
        result.loc[inferred, "ci_high"], result.loc[inferred, "q"] + half_floor
    ).clip(0, 1)
    contract = {
        "main_interval_source": "domain_mapping_bootstrap_npz_quantiles",
        "main_draw_model": "conditional_bootstrap_without_unidentified_shared_residual",
        "sensitivity_interval_role": "mapping_model_error_envelope_not_probabilistic_ci",
        "loo_rmse": loo_rmse,
        "loo_mae": loo_mae,
        "loo_spearman": loo_spearman,
        "residual_half_width_floor": half_floor,
        "inferred_domains": result.loc[inferred, "domain"].tolist(),
        "direct_or_near_direct_domains": result.loc[~inferred, "domain"].tolist(),
        "csv_ci_matches_npz_quantiles": True,
    }
    json_dump(output_dir / "mapping_uncertainty_contract.json", contract)
    result.to_csv(output_dir / "domain_mapping.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output_dir / "domain_mapping_weights.csv", index=False)

    anchors_out = quality_domain.copy()
    anchors_out["mapping_type"] = "quality_anchor"
    anchors_out["quality_anchor"] = anchors_out["domain"]
    combined = pd.concat(
        [
            anchors_out[["domain", "q", "ci_low", "ci_high", "mapping_type", "quality_anchor"]],
            result[["domain", "q", "ci_low", "ci_high", "mapping_type", "quality_anchor"]],
        ],
        ignore_index=True,
    )
    combined.to_csv(output_dir / "quality_domain.csv", index=False)
    np.savez_compressed(output_dir / "domain_mapping_bootstrap.npz", **draws)
    metrics = {
        "tau": tau,
        "loo_mae": loo_mae,
        "loo_spearman": loo_spearman,
        "loo_rmse": loo_rmse,
        "mapping_rank_reliable": mapping_rank_reliable,
        "inferred_policy": "soft_mapping" if mapping_rank_reliable else "global_mean_fallback",
        "mapped_domains": len(result),
        "low_support_domains": result.loc[result["low_support_flag"] == 1, "domain"].tolist(),
    }
    json_dump(output_dir / "domain_mapping_metrics.json", metrics)
    return {
        "mapping": result,
        "combined_quality": combined,
        "weights": pd.DataFrame(weight_rows),
        "distances": distances,
        "metrics": metrics,
        "a18_features": a18,
        "bootstrap": draws,
    }
