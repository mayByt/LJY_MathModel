from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor

from .common import json_dump, sha256_file


def scheffe_features(p: np.ndarray, domains: list[str]) -> tuple[np.ndarray, list[tuple[str, str | None, str]]]:
    p = np.asarray(p, float)
    feats, terms = [], []
    for i, d in enumerate(domains):
        feats.append(p[..., i]); terms.append((d, None, "main"))
    for i, di in enumerate(domains):
        for j in range(i + 1, len(domains)):
            feats.append(p[..., i] * p[..., j]); terms.append((di, domains[j], "interaction"))
    return np.stack(feats, axis=-1), terms


def _j(p: np.ndarray, domains: list[str], coef: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    x, _ = scheffe_features(np.atleast_2d(p), domains)
    return np.mean((x @ coef.T) / iqr[None, :], axis=1)


def _hash_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()


def _fit_transfer(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    xx = np.asarray(x).reshape(-1, 1)
    model = HuberRegressor(fit_intercept=True, epsilon=1.35, alpha=0.0, max_iter=300).fit(xx, y)
    pred = model.predict(xx)
    return {"slope": float(model.coef_[0]), "intercept": float(model.intercept_),
            "spearman_R_loss": float(spearmanr(xx[:, 0], y).statistic),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}


def _robust_slopes(x: np.ndarray, y: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Vectorized Huber IRLS for one predictor and multiple validation domains."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    slopes = np.sum((x - x.mean())[:, None] * (y - y.mean(axis=0)), axis=0) / max(float(np.sum((x - x.mean()) ** 2)), 1e-12)
    intercepts = y.mean(axis=0) - slopes * x.mean()
    for _ in range(iterations):
        resid = y - intercepts[None, :] - x[:, None] * slopes[None, :]
        center = np.median(resid, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(resid - center[None, :]), axis=0), 1e-9)
        u = np.abs(resid) / (1.345 * scale[None, :])
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
        sw = w.sum(axis=0); sx = (w * x[:, None]).sum(axis=0); sy = (w * y).sum(axis=0)
        sxx = (w * x[:, None] ** 2).sum(axis=0); sxy = (w * x[:, None] * y).sum(axis=0)
        denom = np.maximum(sw * sxx - sx * sx, 1e-12)
        new_slopes = (sw * sxy - sx * sy) / denom
        new_intercepts = (sy - new_slopes * sx) / np.maximum(sw, 1e-12)
        if np.max(np.abs(new_slopes - slopes)) < 1e-10:
            slopes, intercepts = new_slopes, new_intercepts
            break
        slopes, intercepts = new_slopes, new_intercepts
    return slopes


def run_bridge(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    p1 = Path(cfg["problem1_root"])
    manifest = json.loads((p1 / "problem1_manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads((p1 / "acceptance_report.json").read_text(encoding="utf-8"))
    mapping = pd.read_csv(p1 / "domain_mapping.csv")
    mix_summary = json.loads((p1 / "mixture_summary.json").read_text(encoding="utf-8"))
    optimal = pd.read_csv(p1 / "optimal_mixture.csv")
    consumed = [
        "acceptance_report.json",
        "domain_mapping.csv",
        "domain_mapping_bootstrap.npz",
        "mapping_uncertainty_contract.json",
        "mixture_summary.json",
        "optimal_mixture.csv",
        "optimal_mixture_bootstrap.npz",
        "problem2_bridge_input.npz",
    ]
    input_hashes = {name: sha256_file(p1 / name) for name in consumed}
    expected = manifest.get("outputs", {})
    mismatches = {k: {"expected": expected.get(k), "observed": v} for k, v in input_hashes.items() if expected.get(k) != v}
    if mismatches:
        raise RuntimeError(f"problem1 consumed-file hash mismatch: {sorted(mismatches)}")

    bridge_path = p1 / "problem2_bridge_input.npz"
    with np.load(bridge_path, allow_pickle=False) as z:
        frozen = {key: np.asarray(z[key]).copy() for key in z.files}
    version = str(frozen["interface_version"].item())
    if version != "p1_to_p2_v2":
        raise RuntimeError(f"unsupported problem1 bridge interface: {version}")

    domains = [str(x) for x in frozen["domains"]]
    loss_domains = [str(x) for x in frozen["loss_domains"]]
    p_train = frozen["p_train"].astype(float)
    y_train = frozen["y_train"].astype(float)
    coef_point = frozen["coefficient_point"].astype(float)
    coef_draws_all = frozen["coefficient_draws"].astype(float)
    p_ref = frozen["p_ref_point"].astype(float)
    p_ref_draws_all = frozen["p_ref_draws"].astype(float)
    iqr = frozen["loss_iqr"].astype(float)
    upper = frozen["upper"].astype(float)
    support_scale = frozen["support_scale"].astype(float)
    alpha = float(frozen["ridge_alpha"])
    reps = int(cfg.get("bootstrap_reps", 500))
    rng = np.random.default_rng(int(cfg["seed"]) + 404)
    if coef_draws_all.shape[0] < reps or p_ref_draws_all.shape[0] < reps:
        raise RuntimeError("problem1 bridge bootstrap draws fewer than requested")
    coef_draws = coef_draws_all[:reps]
    p_ref_draws = p_ref_draws_all[:reps]
    sample_hashes = frozen["bootstrap_sample_index_hash"][:reps].astype("U64")
    valid = np.isfinite(coef_draws).all(axis=(1, 2)) & np.isfinite(p_ref_draws).all(axis=1)
    reasons = np.where(valid, "", "nonfinite_problem1_bridge_draw").astype("U80")
    if len(domains) != 17 or coef_point.shape != (len(loss_domains), 153):
        raise RuntimeError("problem1 bridge dimensions do not match 17-domain quadratic Scheffe model")
    if not np.allclose(p_train.sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError("problem1 bridge training mixtures violate simplex closure")

    q = mapping.set_index("domain").reindex(domains)["q"].to_numpy(float)
    p_ref = p_ref / p_ref.sum()
    p_star = optimal.set_index("domain").reindex(domains)["p_star"].to_numpy(float)
    p_star /= p_star.sum()
    q_ref, q_star = float(p_ref @ q), float(p_star @ q)
    _, terms = scheffe_features(np.ones((1, len(domains))) / len(domains), domains)
    j_train = _j(p_train, domains, coef_point, iqr)
    j_ref = float(_j(p_ref, domains, coef_point, iqr)[0])
    r_scale = max(float(np.subtract(*np.quantile(j_train, [.75, .25]))), 1e-12)

    map_npz = np.load(p1 / "domain_mapping_bootstrap.npz")
    mapping_draws = np.column_stack([map_npz[d] for d in domains]).astype(float)
    near_npz = np.load(p1 / "optimal_mixture_bootstrap.npz")
    near_all = np.asarray(near_npz["p"], float)
    near_p = near_all[np.asarray(near_npz["success"], bool) & np.isfinite(near_all).all(axis=1)]
    q_ref_draws = np.empty(reps)
    j_ref_draws = np.empty(reps)
    r_scale_draws = np.empty(reps)
    for b in range(reps):
        jb = _j(p_train, domains, coef_draws[b], iqr)
        q_ref_draws[b] = float(p_ref_draws[b] @ mapping_draws[b % len(mapping_draws)])
        j_ref_draws[b] = float(_j(p_ref_draws[b], domains, coef_draws[b], iqr)[0])
        r_scale_draws[b] = max(float(np.subtract(*np.quantile(jb, [.75, .25]))), 1e-12)

    def R_of(p: np.ndarray, draw_id: int | None = None) -> np.ndarray:
        if draw_id is None:
            return (_j(p, domains, coef_point, iqr) - j_ref) / r_scale
        b = int(draw_id) % reps
        return (_j(p, domains, coef_draws[b], iqr) - j_ref_draws[b]) / r_scale_draws[b]

    np.savez_compressed(output_dir / "problem1_mixture_response_bootstrap.npz",
        coefficients=coef_draws, p_ref=p_ref_draws, q_ref=q_ref_draws, j_ref=j_ref_draws,
        r_scale=r_scale_draws, sample_index_hash=sample_hashes, valid=valid, reason=reasons,
        coefficient_point=coef_point.astype(np.float64), loss_iqr=iqr.astype(np.float64),
        p_ref_point=p_ref.astype(np.float64), j_ref_point=np.array(j_ref), r_scale_point=np.array(r_scale),
        domains=np.array(domains), loss_domains=np.array(loss_domains),
        terms=np.array([f"{kind}:{di}:{dj or ''}" for di, dj, kind in terms]))

    specs = [("1M", "test_1m"), ("60M", "test_60m"), ("1B", "test_1B")]
    transfer_data, rows = {}, []
    transfer_indices: dict[str, np.ndarray] = {}
    for scale_name, frozen_key in specs:
        pmat = frozen[f"{frozen_key}_p"].astype(float)
        yraw = frozen[f"{frozen_key}_y"].astype(float)
        indices = frozen[f"{frozen_key}_index"].copy()
        if pmat.shape[0] != yraw.shape[0] or yraw.shape[1] != len(loss_domains):
            raise RuntimeError(f"invalid frozen transfer arrays for {scale_name}")
        ystd = yraw / iqr[None, :]
        rval = R_of(pmat)
        transfer_data[scale_name] = (pmat, ystd)
        transfer_indices[scale_name] = indices
        for k, domain in enumerate(loss_domains):
            rec = _fit_transfer(rval, ystd[:, k])
            rec.update({"scale": scale_name, "loss_domain": domain, "n": len(rval), "role": "domain"})
            rows.append(rec)
    if not np.array_equal(transfer_indices["1M"], transfer_indices["60M"]):
        raise RuntimeError("frozen 1M/60M transfer indices are not paired")
    transfer_domains = pd.DataFrame(rows); summaries = []
    for scale_name, part in transfer_domains.groupby("scale", sort=False):
        pp, yy = transfer_data[scale_name]
        rval = R_of(pp)
        composite_loss = yy.mean(axis=1)
        composite_fit = _fit_transfer(rval, composite_loss)
        summaries.append({"scale": scale_name, "loss_domain": "__composite__", "n": int(part["n"].iloc[0]),
            "role": "composite", "slope": composite_fit["slope"],
            "intercept": composite_fit["intercept"],
            "spearman_R_loss": composite_fit["spearman_R_loss"], "rmse": composite_fit["rmse"],
            "domain_median_slope": float(part["slope"].median()),
            "domain_median_spearman_R_loss": float(part["spearman_R_loss"].median())})
    transfer = pd.concat([transfer_domains, pd.DataFrame(summaries)], ignore_index=True)

    boot_rows = []
    for b in range(reps):
        shared = rng.integers(0, 256, 256)
        rec = {"draw": b, "scheffe_draw_id": b, "valid": bool(valid[b]), "reason": reasons[b],
               "paired_index_hash": _hash_ids(shared)}
        for scale_name in ["1M", "60M"]:
            pp, yy = transfer_data[scale_name]
            rec[f"median_slope_{scale_name}"] = float(np.median(_robust_slopes(R_of(pp[shared], b), yy[shared])))
        pp, yy = transfer_data["1B"]; ids = rng.integers(0, len(pp), len(pp))
        rec["median_slope_1B"] = float(np.median(_robust_slopes(R_of(pp[ids], b), yy[ids])))
        s1, s60 = rec["median_slope_1M"], rec["median_slope_60M"]
        rec["delta_p"] = float(np.log(s60 / s1) / np.log(60)) if s1 > 0 and s60 > 0 else 0.0
        boot_rows.append(rec)
    transfer_boot = pd.DataFrame(boot_rows)
    transfer_boot.to_csv(output_dir / "mixture_transfer_bootstrap.csv.gz", index=False, compression="gzip")
    sign_prob = {s: float((transfer_boot[f"median_slope_{s}"] > 0).mean()) for s in ["1M", "60M", "1B"]}
    slopes = {r["scale"]: r["slope"] for r in summaries}
    composite_spearman = {r["scale"]: r["spearman_R_loss"] for r in summaries}
    gate = slopes["1M"] > 0 and slopes["60M"] > 0 and min(composite_spearman["1M"], composite_spearman["60M"]) >= .75
    delta_p = float(np.log(slopes["60M"] / slopes["1M"]) / np.log(60)) if gate else 0.0
    transfer["sign_probability_positive"] = transfer["scale"].map(sign_prob)
    transfer["delta_p_profile_role"] = "paired_estimate" if gate else "fixed_zero_due_direction_gate"
    transfer.to_csv(output_dir / "mixture_transfer_parameters.csv", index=False)
    warn_ids = sorted(r["id"] for r in acceptance["rules"] if r.get("status") == "WARN")
    r_star_draws = np.array([float(R_of(p_star, i)[0]) for i in range(reps) if valid[i]])
    interaction_rows = []
    for t, (di, dj, kind) in enumerate(terms):
        if kind != "interaction" or dj is None:
            continue
        point = float(np.mean(coef_point[:, t] / iqr))
        draws = np.mean(coef_draws[:, :, t] / iqr[None, :], axis=1)
        lo, hi = float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
        if hi < 0:
            relation = "complementary"
        elif lo > 0:
            relation = "substitutive"
        else:
            relation = "uncertain"
        interaction_rows.append({
            "domain_i": di,
            "domain_j": dj,
            "standardized_interaction": point,
            "ci_low": lo,
            "ci_high": hi,
            "probability_negative": float(np.mean(draws < 0)),
            "relation": relation,
            "interpretation": "negative_joint_loss_synergy" if relation == "complementary" else (
                "positive_joint_loss_competition" if relation == "substitutive" else "bootstrap_interval_crosses_zero"
            ),
        })
    interaction = pd.DataFrame(interaction_rows).sort_values(
        "standardized_interaction", key=lambda x: np.abs(x), ascending=False
    )
    interaction.to_csv(output_dir / "domain_substitution_complementarity.csv", index=False)
    interaction_counts = {str(k): int(v) for k, v in interaction.groupby("relation").size().items()}

    bridge = {
        "problem1_overall": acceptance.get("overall"), "problem1_summary": acceptance.get("summary"),
        "propagated_warn_ids": warn_ids,
        "warn_actions": {"MAP02": "retain mapping uncertainty", "O01": "p_star sensitivity only", "Q02": "exclude sample extremes"},
        "input_hashes": input_hashes, "input_hash_mismatches": mismatches, "input_hash_gate": "PASS",
        "problem1_bridge_interface": version,
        "raw_attachment_A_read_by_problem2": False,
        "domains": domains, "loss_domains": loss_domains, "scheffe_feature_count": len(terms),
        "quadratic_alpha": alpha, "model_role": mix_summary.get("model_role"),
        "p_ref": dict(zip(domains, map(float, p_ref))), "p_star": dict(zip(domains, map(float, p_star))),
        "Q_ref": q_ref, "Q_star": q_star,
        "Q_base_regression_targets": {"Q_ref_expected": .5389559588, "Q_star_expected": .5453680376,
            "Q_ref_abs_error": abs(q_ref - .5389559588), "Q_star_abs_error": abs(q_star - .5453680376)},
        "R_ref": float(R_of(p_ref)[0]), "R_star": float(R_of(p_star)[0]), "R_iqr_on_A4": r_scale,
        "R_star_bootstrap": {"median": float(np.median(r_star_draws)),
            "ci_low": float(np.quantile(r_star_draws, .025)), "ci_high": float(np.quantile(r_star_draws, .975))},
        "delta_p": delta_p, "transfer_sign_probability_positive": sign_prob,
        "transfer_composite_spearman_R_loss": composite_spearman,
        "transfer_gate": "composite_spearman_1M_60M_ge_0.75",
        "kappa0_or_lambda0_role": "scenario_not_identified_across_A_and_B_units",
        "mixture_strength_profiles": cfg.get("mixture_strength_profiles", [0.0, .25, .5, 1.0]),
        "scheffe_bootstrap_valid_rate": float(valid.mean()),
        "mixture_response_artifact": "problem1_mixture_response_bootstrap.npz",
        "mixture_response_artifact_contains_point_model": True,
        "scheffe_bootstrap_max_std": float(np.nanmax(np.nanstd(coef_draws.astype(float), axis=0))),
        "interaction_relation_counts": interaction_counts,
        "near_optimal_valid_draws": len(near_p),
        "mixture_support": {"upper": dict(zip(domains, map(float, upper))),
            "scale": dict(zip(domains, map(float, support_scale))), "radius": mix_summary.get("support_radius")},
        "manifest_project": manifest.get("project")}
    json_dump(output_dir / "problem1_bridge.json", bridge)
    return {"bridge": bridge, "transfer": transfer, "transfer_bootstrap": transfer_boot,
        "domains": domains, "loss_domains": loss_domains, "q": q, "mapping_draws": mapping_draws,
        "p_ref": p_ref, "p_star": p_star, "p_train": p_train, "near_optimal_p": near_p,
        "coef_draws": coef_draws, "p_ref_draws": p_ref_draws, "q_ref_draws": q_ref_draws,
        "j_ref_draws": j_ref_draws, "r_scale_draws": r_scale_draws, "valid_draws": valid, "R_of": R_of,
        "interaction": interaction}
