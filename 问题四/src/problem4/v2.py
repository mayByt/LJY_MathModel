from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import spearmanr, theilslopes

from .bridge import TARGETS, comparability_weight, fit_mono, support_status
from .io import SCORE_COLS, model_core, model_family, sha256, type_group


def score_to_link(score, eps=0.5):
    return logit((np.asarray(score, float) + eps) / (100.0 + 2.0 * eps))


def link_to_score(value, eps=0.5):
    return (100.0 + 2.0 * eps) * expit(np.asarray(value, float)) - eps


def weighted_quantile(values, quantile, weights=None):
    values = np.asarray(values, float)
    if weights is None:
        return float(np.quantile(values, quantile))
    weights = np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[ok], weights[ok]
    if not len(values):
        return np.nan
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return float(np.interp(quantile, cdf, values, left=values[0], right=values[-1]))


def family_label(name):
    known = model_family(name)
    if known != "other":
        return known
    token = str(name).lower().split("/")[-1].split()[0].split("-")[0]
    return token or "other"


def check_p3_interface(p3_root: Path):
    acceptance = json.loads((p3_root / "problem3_acceptance_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((p3_root / "problem3_manifest.json").read_text(encoding="utf-8"))
    bridge = json.loads((p3_root / "problem3_bridge.json").read_text(encoding="utf-8"))
    names = ["problem3_bridge.json", "problem3_acceptance_report.json",
             "bootstrap_optimal_allocations.csv.gz", "bootstrap_intervals.csv"]
    hashes, mismatches = {}, []
    for name in names:
        actual = sha256(p3_root / name)
        hashes[name] = actual
        recorded = [v for k, v in manifest.get("outputs", {}).items() if k.endswith("/" + name)]
        if recorded and recorded[0] != actual:
            mismatches.append(name)
    ready = (acceptance.get("summary", {}).get("FAIL") == 0 and
             bridge.get("PROBLEM3_READY") is True and len(bridge.get("entries", [])) == 45 and
             not mismatches)
    return ready, hashes, mismatches, bridge


def snapshot_tree(paths):
    rows = []
    for root in paths:
        root = Path(root)
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)})
    return pd.DataFrame(rows)


def enrich_matched(matched, ledger, c4):
    out = matched.copy()
    indices = ledger["c4_row"].tolist()
    fields = {
        "c4_compute_method_raw": "Training compute estimation method",
        "c4_confidence": "Confidence",
        "c4_compute_lower": "Training compute lower bound",
        "c4_compute_upper": "Training compute upper bound",
        "c4_reference": "Reference",
        "c4_link": "Link",
    }
    for dst, src in fields.items():
        vals = []
        for idx in indices:
            vals.append(c4.iloc[int(idx)][src] if pd.notna(idx) else np.nan)
        out[dst] = vals
    method = out["c4_compute_method_raw"].fillna("").str.lower()
    out["compute_evidence_v2"] = np.select(
        [method.str.contains("benchmark|comparison", regex=True),
         method.str.contains("reported"), method.str.contains("operation"),
         method.str.contains("hardware")],
        ["benchmark_or_comparison", "reported", "operation_counting", "hardware_based"],
        default="third_party_or_unknown",
    )
    return out


def build_panel_v2(matched, cutoff, eps, license_allowlist):
    p = matched.copy()
    p["submission_date"] = pd.to_datetime(p["Submission Date"], errors="coerce")
    p["type_group"] = p["Type"].map(type_group)
    p["family"] = p["Model"].map(family_label)
    epoch_yes = p["Epoch_AI_Open_Weights"].fillna("").astype(str).str.lower().eq("yes")
    c4_yes = p["c4_open_weights"].fillna("").astype(str).str.lower().eq("yes")
    p["open_weight_verified"] = epoch_yes | c4_yes
    access_ok = p["c4_accessibility"].fillna("").astype(str).str.contains(r"Open weights \(unrestricted\)", case=False, regex=True)
    license_ok = p["Hub License"].fillna("").astype(str).str.lower().isin(set(license_allowlist))
    p["strict_unrestricted"] = p["open_weight_verified"] & access_ok & license_ok
    p["complete_score"] = p[SCORE_COLS + ["Average ⬆️"]].notna().all(axis=1)
    p["future_leakage_excluded"] = p["c4_publication_date"].notna() & (p["c4_publication_date"] > cutoff)
    p["ability_logit"] = score_to_link(p["Average ⬆️"], eps)
    compute = pd.to_numeric(p["training_compute_flop"], errors="coerce")
    params = pd.to_numeric(p["#Params (B)"], errors="coerce") * 1e9
    data = pd.to_numeric(p["training_data_size"], errors="coerce")
    p["log10_compute"] = np.log10(compute.where(compute > 0))
    p["log10_params"] = np.log10(params.where(params > 0))
    p["log10_data"] = np.log10(data.where(data > 0))
    evidence_ok = p["compute_evidence_v2"].isin(["reported", "operation_counting", "hardware_based"])
    p["frontier_eligible"] = (
        p["open_weight_verified"] & p["type_group"].eq("pretrained") & p["complete_score"] &
        p["submission_date"].notna() & p["log10_compute"].notna() &
        p["match_status"].str.startswith("accepted") & evidence_ok & ~p["future_leakage_excluded"]
    )
    return p


def c3_historical_analysis(c3, eps, draws, rng):
    d = c3[c3["Source"].astype(str).str.contains("Historical", case=False, na=False)].copy()
    d = d.dropna(subset=["Average", "Params_B", "Year"])
    d = d[(d["Params_B"] > 0) & d["Average"].between(0, 100)]
    d["family"] = d["Model"].map(family_label)
    d["log10_params"] = np.log10(d["Params_B"] * 1e9)
    d["year_centered"] = d["Year"] - 2019
    d["ability_logit"] = score_to_link(d["Average"], eps)

    def fit(frame):
        X = np.column_stack([np.ones(len(frame)), frame["log10_params"], frame["year_centered"]])
        coef = np.linalg.lstsq(X, frame["ability_logit"], rcond=None)[0]
        return coef

    coef = fit(d)
    rows = [{"analysis": "central", "excluded_family": "", "intercept": coef[0],
             "log10_params_coef": coef[1], "year_coef": coef[2], "rows": len(d)}]
    for family in sorted(d["family"].unique()):
        sub = d[~d["family"].eq(family)]
        if len(sub) >= 8:
            c = fit(sub)
            rows.append({"analysis": "leave_one_family_out", "excluded_family": family,
                         "intercept": c[0], "log10_params_coef": c[1], "year_coef": c[2], "rows": len(sub)})
    families = sorted(d["family"].unique())
    boot = []
    for b in range(draws):
        sampled = rng.choice(families, len(families), replace=True)
        parts = [d[d["family"].eq(f)].assign(_cluster=j) for j, f in enumerate(sampled)]
        sample = pd.concat(parts, ignore_index=True)
        try:
            c = fit(sample)
            boot.append({"analysis": "bootstrap", "draw_id": b, "intercept": c[0],
                         "log10_params_coef": c[1], "year_coef": c[2], "rows": len(sample)})
        except np.linalg.LinAlgError:
            boot.append({"analysis": "bootstrap", "draw_id": b, "year_coef": np.nan, "rows": len(sample)})
    boot = pd.DataFrame(boot)
    annual = d.groupby("Year")["Average"].quantile(.9)
    ts = theilslopes(annual.to_numpy(), annual.index.to_numpy()) if len(annual) >= 3 else (np.nan,) * 4
    valid = boot["year_coef"].dropna()
    lofo = pd.DataFrame(rows)[lambda x: x["analysis"].eq("leave_one_family_out")]
    summary = {
        "rows": int(len(d)), "families": int(d["family"].nunique()),
        "year_coef": float(coef[2]), "year_coef_p10": float(valid.quantile(.1)),
        "year_coef_p90": float(valid.quantile(.9)), "year_coef_p025": float(valid.quantile(.025)),
        "year_coef_p975": float(valid.quantile(.975)), "positive_probability": float((valid > 0).mean()),
        "lofo_sign_stable": bool((np.sign(lofo["year_coef"]) == np.sign(coef[2])).all()) if len(lofo) else False,
        "annual_q90_theil_sen_slope": float(ts[0]),
    }
    summary["direction"] = ("supports_nonnegative_residual" if summary["positive_probability"] >= .8 and summary["lofo_sign_stable"]
                            else "direction_uncertain")
    return d, pd.concat([pd.DataFrame(rows), boot], ignore_index=True), summary


def aggregate_c8_v2(root: Path, leaderboard):
    official = {}
    for _, row in leaderboard.iterrows():
        official.setdefault(model_core(row["Model"]), []).append((float(row["BBH"]), float(row["MATH Lvl 5"]), row["Model"]))
    aggregates, subtasks, corrupt = [], [], []
    parseable = fallback_dirs = no_parse = 0
    math_keys = [
        "leaderboard_math_algebra_hard", "leaderboard_math_counting_and_prob_hard",
        "leaderboard_math_geometry_hard", "leaderboard_math_intermediate_algebra_hard",
        "leaderboard_math_num_theory_hard", "leaderboard_math_prealgebra_hard",
        "leaderboard_math_precalculus_hard",
    ]
    for directory in sorted((root / "detailed_results").iterdir()):
        if not directory.is_dir():
            continue
        selected = payload = None
        failed = []
        for path in sorted(directory.glob("*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8")); selected = path; break
            except Exception as exc:
                failed.append(path)
                corrupt.append({"model_directory": directory.name, "file": str(path),
                                "error_type": type(exc).__name__, "error": str(exc)[:300]})
        if selected is None:
            no_parse += 1; continue
        parseable += 1; fallback_dirs += int(bool(failed))
        results, counts, configs = payload.get("results", {}), payload.get("n-samples", {}), payload.get("configs", {})
        bbh_keys = payload.get("group_subtasks", {}).get("leaderboard_bbh", [])
        if not bbh_keys:
            bbh_keys = [k for k in results if k.startswith("leaderboard_bbh_") and k != "leaderboard_bbh"]
        bvals = []
        for key in bbh_keys:
            metric = results.get(key, {}).get("acc_norm,none")
            choices = configs.get(key, {}).get("doc_to_choice")
            lower = 1.0 / len(choices) if isinstance(choices, list) and len(choices) else np.nan
            count = counts.get(key, {})
            norm = max(0.0, (float(metric) - lower) / (1.0 - lower)) if metric is not None and np.isfinite(lower) else np.nan
            subtasks.append({"model_directory": directory.name, "json_file": str(selected), "benchmark": "BBH",
                             "task": key, "metric": "acc_norm,none", "raw_score": metric,
                             "random_lower_bound": lower, "normalized_score": norm,
                             "effective_n": count.get("effective"), "original_n": count.get("original"),
                             "git_hash": payload.get("git_hash"), "upper_git_hash": payload.get("upper_git_hash")})
            if np.isfinite(norm): bvals.append(norm)
        mvals = []
        for key in math_keys:
            metric = results.get(key, {}).get("exact_match,none")
            count = counts.get(key, {})
            n = count.get("effective", count.get("original"))
            subtasks.append({"model_directory": directory.name, "json_file": str(selected), "benchmark": "MATH",
                             "task": key, "metric": "exact_match,none", "raw_score": metric,
                             "random_lower_bound": 0.0, "normalized_score": metric,
                             "effective_n": n, "original_n": count.get("original"),
                             "git_hash": payload.get("git_hash"), "upper_git_hash": payload.get("upper_git_hash")})
            if metric is not None and n is not None: mvals.append((float(metric), float(n)))
        bbh = np.mean(bvals) * 100 if len(bvals) == len(bbh_keys) and bvals else np.nan
        math_score = sum(v*n for v, n in mvals) / sum(n for _, n in mvals) * 100 if len(mvals) == len(math_keys) else np.nan
        raw_name = payload.get("model_name") or payload.get("model_name_sanitized") or directory.name
        choices = official.get(model_core(raw_name), [])
        off_bbh, off_math = (choices[0][0], choices[0][1]) if len(choices) == 1 else (np.nan, np.nan)
        aggregates.append({"model_directory": directory.name, "json_file": str(selected), "fallback_used": bool(failed),
                           "model_name_json": raw_name, "normalized_core": model_core(raw_name),
                           "bbh_normalized_equal_pct": bbh, "bbh_subtasks": len(bvals),
                           "math_weighted_pct": math_score, "math_categories": len(mvals),
                           "official_bbh": off_bbh, "official_math": off_math,
                           "bbh_abs_diff": abs(bbh-off_bbh) if np.isfinite(bbh) and np.isfinite(off_bbh) else np.nan,
                           "math_abs_diff": abs(math_score-off_math) if np.isfinite(math_score) and np.isfinite(off_math) else np.nan})
    agg, sub = pd.DataFrame(aggregates), pd.DataFrame(subtasks)
    summary = {
        "directories": parseable + no_parse, "parseable_directories": parseable,
        "no_parseable_directories": no_parse, "fallback_directories": fallback_dirs,
        "corrupt_files": len(corrupt), "bbh_complete": int((agg["bbh_subtasks"] == 24).sum()),
        "math_complete": int((agg["math_categories"] == 7).sum()),
        "bbh_unique_comparable": int(agg["bbh_abs_diff"].notna().sum()),
        "math_unique_comparable": int(agg["math_abs_diff"].notna().sum()),
        "official_exact_bbh": int((agg["bbh_abs_diff"] <= 1e-6).sum()),
        "official_exact_math": int((agg["math_abs_diff"] <= 1e-6).sum()),
        "bbh_max_abs_diff": float(agg["bbh_abs_diff"].max()),
        "math_max_abs_diff": float(agg["math_abs_diff"].max()),
    }
    return agg, sub, pd.DataFrame(corrupt), summary


def bridge_cross_validate(frame, target_col, medium_weight, eps):
    d = frame[["Model", "Val_Loss", target_col, "Loss_Comparability"]].dropna().copy()
    d["group"] = d["Model"].map(family_label)
    d["row_id"] = d.index
    w = comparability_weight(d, medium_weight)
    kinds = ["constant", "linear", "spline", "isotonic"]
    pred_rows, summary = [], []
    for kind in kinds:
        fold_mae = []
        for group in sorted(d["group"].unique()):
            te = d["group"].eq(group).to_numpy(); tr = ~te
            if tr.sum() < 4 or te.sum() == 0 or w[tr].sum() <= 0 or w[te].sum() <= 0:
                continue
            model = fit_mono(kind, d.loc[tr, "Val_Loss"], d.loc[tr, target_col], w[tr], eps)
            train_pred = link_to_score(model.predict(np.log(d.loc[tr, "Val_Loss"])), eps)
            residual = d.loc[tr, target_col].to_numpy() - train_pred
            q10, q90 = weighted_quantile(residual, .1, w[tr]), weighted_quantile(residual, .9, w[tr])
            q025, q975 = weighted_quantile(residual, .025, w[tr]), weighted_quantile(residual, .975, w[tr])
            pred = link_to_score(model.predict(np.log(d.loc[te, "Val_Loss"])), eps)
            actual = d.loc[te, target_col].to_numpy(); wt = w[te]
            fold_mae.append(float(np.average(np.abs(pred-actual), weights=wt)))
            for pos, (_, row) in enumerate(d.loc[te].iterrows()):
                pred_rows.append({"target": target_col, "model": kind, "held_out_family": group,
                                  "row_id": int(row["row_id"]), "actual": actual[pos], "prediction": pred[pos],
                                  "weight": wt[pos], "p10": pred[pos]+q10, "p90": pred[pos]+q90,
                                  "p025": pred[pos]+q025, "p975": pred[pos]+q975})
        pr = pd.DataFrame([r for r in pred_rows if r["target"] == target_col and r["model"] == kind])
        if len(pr):
            err = pr["prediction"] - pr["actual"]
            mae = np.average(np.abs(err), weights=pr["weight"])
            rmse = math.sqrt(np.average(err**2, weights=pr["weight"]))
            rho = spearmanr(pr["actual"], pr["prediction"]).statistic if pr["actual"].nunique() > 1 else np.nan
            cov80 = np.average((pr["actual"] >= pr["p10"]) & (pr["actual"] <= pr["p90"]), weights=pr["weight"])
            cov95 = np.average((pr["actual"] >= pr["p025"]) & (pr["actual"] <= pr["p975"]), weights=pr["weight"])
        else:
            mae = rmse = rho = cov80 = cov95 = np.nan
        arr = np.asarray(fold_mae)
        summary.append({"target": target_col, "model": kind, "folds": len(arr), "cv_mae": mae,
                        "cv_rmse": rmse, "spearman": rho, "coverage80": cov80, "coverage95": cov95,
                        "mean_width80": float((pr["p90"]-pr["p10"]).mean()) if len(pr) else np.nan,
                        "mean_width95": float((pr["p975"]-pr["p025"]).mean()) if len(pr) else np.nan,
                        "cv_mae_se": float(np.std(arr, ddof=1)/np.sqrt(len(arr))) if len(arr)>1 else np.nan,
                        "cv_scheme": "leave_source_family_out"})
    summary = pd.DataFrame(summary)
    good = summary.dropna(subset=["cv_mae"])
    if good.empty:
        # With medium_weight=0 the high-comparability subset is the single
        # Pythia family.  Family holdout is then undefined, so keep this
        # sensitivity explicitly separate and use leave-one-row-out.
        pred_rows, fallback = [], []
        positive = np.flatnonzero(w > 0)
        for kind in kinds:
            fold_mae = []
            for hold in positive:
                tr = np.zeros(len(d), dtype=bool)
                tr[positive] = True
                tr[hold] = False
                if tr.sum() < 4:
                    continue
                model = fit_mono(kind, d.loc[tr, "Val_Loss"], d.loc[tr, target_col], w[tr], eps)
                train_pred = link_to_score(model.predict(np.log(d.loc[tr, "Val_Loss"])), eps)
                residual = d.loc[tr, target_col].to_numpy() - train_pred
                q10, q90 = weighted_quantile(residual,.1,w[tr]), weighted_quantile(residual,.9,w[tr])
                q025, q975 = weighted_quantile(residual,.025,w[tr]), weighted_quantile(residual,.975,w[tr])
                pred = float(link_to_score(model.predict(np.log([d.iloc[hold]["Val_Loss"]])), eps)[0])
                actual = float(d.iloc[hold][target_col]); fold_mae.append(abs(pred-actual))
                pred_rows.append({"target":target_col,"model":kind,"held_out_family":"single_family_loo",
                                  "row_id":int(d.iloc[hold]["row_id"]),"actual":actual,"prediction":pred,
                                  "weight":float(w[hold]),"p10":pred+q10,"p90":pred+q90,
                                  "p025":pred+q025,"p975":pred+q975})
            pr = pd.DataFrame([r for r in pred_rows if r["model"] == kind])
            if len(pr):
                err=pr["prediction"]-pr["actual"]
                mae=float(np.average(abs(err),weights=pr["weight"])); rmse=float(np.sqrt(np.average(err**2,weights=pr["weight"])))
                rho=spearmanr(pr["actual"],pr["prediction"]).statistic if pr["actual"].nunique()>1 else np.nan
                cov80=float(np.average((pr["actual"]>=pr["p10"])&(pr["actual"]<=pr["p90"]),weights=pr["weight"]))
                cov95=float(np.average((pr["actual"]>=pr["p025"])&(pr["actual"]<=pr["p975"]),weights=pr["weight"]))
            else:
                mae=rmse=rho=cov80=cov95=np.nan
            arr=np.asarray(fold_mae)
            fallback.append({"target":target_col,"model":kind,"folds":len(arr),"cv_mae":mae,"cv_rmse":rmse,
                             "spearman":rho,"coverage80":cov80,"coverage95":cov95,
                             "mean_width80":float((pr["p90"]-pr["p10"]).mean()) if len(pr) else np.nan,
                             "mean_width95":float((pr["p975"]-pr["p025"]).mean()) if len(pr) else np.nan,
                             "cv_mae_se":float(np.std(arr,ddof=1)/np.sqrt(len(arr))) if len(arr)>1 else np.nan,
                             "cv_scheme":"leave_one_out_single_family_sensitivity"})
        summary=pd.DataFrame(fallback); good=summary.dropna(subset=["cv_mae"])
    best = good.loc[good["cv_mae"].idxmin()]
    cutoff = best["cv_mae"] + (best["cv_mae_se"] if np.isfinite(best["cv_mae_se"]) else 0)
    complexity = {"constant": 0, "linear": 1, "spline": 2, "isotonic": 3}
    eligible = good[good["cv_mae"] <= cutoff]
    selected = min(eligible["model"], key=lambda x: complexity[x])
    const_mae = float(good.loc[good["model"].eq("constant"), "cv_mae"].iloc[0])
    sel_mae = float(good.loc[good["model"].eq(selected), "cv_mae"].iloc[0])
    weak = selected == "constant" or sel_mae >= const_mae
    summary["selected"] = summary["model"].eq(selected); summary["bridge_weak"] = weak
    return summary, pd.DataFrame(pred_rows), selected, weak


def fit_bridges(frame, medium_weight, eps):
    models, meta, summaries, predictions = {}, {}, [], []
    weights = comparability_weight(frame, medium_weight)
    for target, col in TARGETS.items():
        cv, pred, selected, weak = bridge_cross_validate(frame, col, medium_weight, eps)
        cv["target_name"] = target; pred["target_name"] = target
        model = fit_mono(selected, frame["Val_Loss"], frame[col], weights, eps)
        grid = np.linspace(np.log(frame["Val_Loss"].min()), np.log(frame["Val_Loss"].max()), 500)
        violations = int((np.diff(model.predict(grid)) > 1e-10).sum())
        models[target] = model
        meta[target] = {"score_column": col, "selected_model": selected, "bridge_weak": bool(weak),
                        "monotonicity_violations": violations, "loss_min": float(frame["Val_Loss"].min()),
                        "loss_max": float(frame["Val_Loss"].max())}
        summaries.append(cv); predictions.append(pred)
    return models, pd.concat(summaries, ignore_index=True), pd.concat(predictions, ignore_index=True), meta


def _scenario_key(row):
    return float(row["budget_FLOPs"]), int(row["context_length"]), str(row["quality_cost_type"])


def map_problem3_v2(c6, p3_root, models, meta, rng, draws, medium_weight, eps):
    entries = pd.DataFrame(json.loads((p3_root / "problem3_bridge.json").read_text(encoding="utf-8"))["entries"])
    boot = pd.read_csv(p3_root / "bootstrap_optimal_allocations.csv.gz")
    boot = boot[boot["support_mode"].eq("operational_extended")]
    pools = {_scenario_key(g.iloc[0]): g["predicted_loss"].to_numpy() for _, g in boot.groupby(["budget_FLOPs", "context_length", "quality_cost_type"])}
    c6 = c6.copy(); c6["group"] = c6["Model"].map(family_label)
    groups = sorted(c6["group"].unique()); boot_models = []
    for _ in range(draws):
        sample_groups = rng.choice(groups, len(groups), replace=True)
        sample = pd.concat([c6[c6["group"].eq(g)] for g in sample_groups], ignore_index=True)
        w = comparability_weight(sample, medium_weight)
        boot_models.append({t: fit_mono(meta[t]["selected_model"], sample["Val_Loss"], sample[col], w, eps)
                            for t, col in TARGETS.items()})
    raw, summary, budget = [], [], []
    for scenario_id, entry in entries.iterrows():
        key = _scenario_key(entry); pool = pools.get(key, np.array([]))
        center_loss = float(entry["predicted_loss"])
        if len(pool):
            losses = rng.choice(pool, draws, replace=True)
        else:
            lo, hi = entry["predicted_loss_interval"]
            losses = rng.triangular(lo, center_loss, hi, size=draws)
        for target in TARGETS:
            central_model = models[target]
            up = link_to_score(central_model.predict(np.log(losses)), eps)
            br = np.asarray([link_to_score(boot_models[b][target].predict(np.log([center_loss])), eps)[0] for b in range(draws)])
            joint = np.asarray([link_to_score(boot_models[b][target].predict(np.log([losses[b]])), eps)[0] for b in range(draws)])
            for mode, values in [("upstream_only", up), ("bridge_only", br), ("joint", joint)]:
                rec = {"scenario_index": scenario_id, "target": target, "uncertainty_mode": mode,
                       "median": float(np.median(values)), "p10": float(np.quantile(values,.1)),
                       "p90": float(np.quantile(values,.9)), "p025": float(np.quantile(values,.025)),
                       "p975": float(np.quantile(values,.975))}
                rec["width80"] = rec["p90"]-rec["p10"]; rec["width95"] = rec["p975"]-rec["p025"]
                summary.append(rec)
                budget.append({"estimand": "problem3_loss_to_benchmark", "scenario_id": scenario_id,
                               "target": target, "component": mode, **{k: rec[k] for k in ["p10","p90","p025","p975","width80","width95"]},
                               "draws": draws, "interpretation": "component widths are not additive"})
            for b, value in enumerate(joint):
                raw.append({"scenario_index": scenario_id, "target": target, "draw_id": b,
                            "score": value, "loss": losses[b], "uncertainty_mode": "joint"})
    sums = pd.DataFrame(summary)
    main = sums[sums["uncertainty_mode"].eq("joint")].copy()
    extra = entries.reset_index(names="scenario_index")[["scenario_index","budget_FLOPs","context_length","quality_cost_type","predicted_loss","support_status"]]
    main = main.merge(extra, on="scenario_index", how="left")
    main["bridge_support"] = main["predicted_loss"].map(support_status)
    main["final_evidence_tier"] = ["limited" if s=="out_of_bridge_range" or meta[t]["bridge_weak"] else "supported"
                                   for s,t in zip(main["bridge_support"],main["target"])]
    return main, sums, pd.DataFrame(raw), pd.DataFrame(budget)


@dataclass
class DynamicFitV2:
    tau: float
    lam: float
    rho: float
    beta0: float
    beta_x: float
    states: np.ndarray
    months: pd.PeriodIndex
    x_mean: float
    x_std: float
    objective: float
    success: bool

    def slopes(self):
        pos = np.arange(len(self.states), dtype=float)
        recent = min(4, len(pos))
        local = np.polyfit(pos[-recent:], self.states[-recent:], 1)[0] if recent > 1 else 0.0
        global_slope = np.polyfit(pos, self.states, 1)[0] if len(pos) > 1 else 0.0
        return float(local), float(global_slope), float(self.rho*local + (1-self.rho)*global_slope)

    def state_at(self, month):
        month = pd.Period(month, freq="M")
        if month in self.months:
            return float(self.states[self.months.get_loc(month)])
        target = (month.year-self.months[0].year)*12 + month.month-self.months[0].month
        _, _, slope = self.slopes()
        return float(self.states[-1] + slope*(target-(len(self.months)-1)))

    def predict(self, x, months):
        x = np.asarray(x, float)
        state = np.asarray([self.state_at(m) for m in months])
        return self.beta0 + self.beta_x*((x-self.x_mean)/self.x_std) + state


def pinball(residual, tau):
    return np.where(residual >= 0, tau*residual, (tau-1)*residual)


def fit_dynamic_v2(frame, tau=.9, lam=1.0, rho=.5, weights=None, include_scale=True):
    d = frame.sort_values("submission_date").copy()
    months = pd.period_range(d["submission_date"].min().to_period("M"), d["submission_date"].max().to_period("M"), freq="M")
    month_map = {m:i for i,m in enumerate(months)}
    midx = d["submission_date"].dt.to_period("M").map(month_map).to_numpy()
    x = d["log10_compute"].to_numpy(float); y = d["ability_logit"].to_numpy(float)
    xm, xs = float(np.mean(x)), float(np.std(x)) or 1.0; z = (x-xm)/xs
    w = np.ones(len(d)) if weights is None else np.asarray(weights,float)
    init = np.r_[np.quantile(y,tau), .1 if include_scale else 0., np.zeros(len(months))]
    def objective(theta):
        b0,bx=theta[:2]; state=theta[2:]; residual=y-(b0+bx*z+state[midx])
        smooth=(tau-.5)*residual+.5*np.sqrt(residual**2+1e-6)
        d1=np.diff(state); d2=np.diff(state,2)
        return np.sum(w*smooth)/np.sum(w)+lam*(np.sum(d2*d2)+.05*np.sum(d1*d1))+100*np.mean(state)**2
    bounds=[(None,None),(0,0) if not include_scale else (0,None)]+[(None,None)]*len(months)
    res=minimize(objective,init,method="L-BFGS-B",bounds=bounds,options={"maxiter":3000,"ftol":1e-11})
    return DynamicFitV2(tau,lam,rho,float(res.x[0]),float(res.x[1]),np.asarray(res.x[2:]),months,xm,xs,float(res.fun),bool(res.success))


def decompose_v2(model, frame, eps=.5):
    scale = frame.assign(month=frame["submission_date"].dt.to_period("M")).groupby("month")["log10_compute"].quantile(.9)
    usable = scale.index.intersection(model.months); t0,t1=usable[0],usable[-1]; x0,x1=float(scale[t0]),float(scale[t1])
    f00=float(model.predict([x0],[t0])[0]); f10=float(model.predict([x1],[t0])[0])
    f01=float(model.predict([x0],[t1])[0]); f11=float(model.predict([x1],[t1])[0])
    sl=.5*((f10-f00)+(f11-f01)); te=.5*((f01-f00)+(f11-f10)); total=f11-f00
    s00,s10,s01,s11=link_to_score([f00,f10,f01,f11],eps)
    ss=.5*((s10-s00)+(s11-s01)); ts=.5*((s01-s00)+(s11-s10)); total_s=s11-s00
    return {"t0":str(t0),"t1":str(t1),"x0_log10_compute":x0,"x1_log10_compute":x1,
            "f00_logit":f00,"f10_logit":f10,"f01_logit":f01,"f11_logit":f11,
            "scale_logit":sl,"tech_logit":te,"total_logit":total,"closure_logit":abs(sl+te-total),
            "scale_score":ss,"tech_score":ts,"total_score":total_s,"closure_score":abs(ss+ts-total_s)}


def rolling_cv_v2(frame, lambdas, rhos, tau=.9, horizons=(3,6)):
    d=frame.sort_values("submission_date").copy(); months=sorted(d["submission_date"].dt.to_period("M").unique()); rows=[]
    for horizon in horizons:
        for ci in range(2,len(months)-1):
            cutoff=months[ci]; test_end=cutoff+horizon
            train=d[d["submission_date"].dt.to_period("M")<=cutoff]
            test=d[(d["submission_date"].dt.to_period("M")>cutoff)&(d["submission_date"].dt.to_period("M")<=test_end)]
            if len(train)<12 or len(test)==0 or train["submission_date"].dt.to_period("M").nunique()<3: continue
            for lam in lambdas:
                for rho in rhos:
                    m=fit_dynamic_v2(train,tau,lam,rho); pred=m.predict(test["log10_compute"],test["submission_date"].dt.to_period("M"))
                    res=test["ability_logit"].to_numpy()-pred
                    rows.append({"fold_cutoff":str(cutoff),"horizon_months":horizon,"model":"M-C","lambda":lam,"rho":rho,
                                 "n_train":len(train),"n_test":len(test),"pinball":np.mean(pinball(res,tau)),"mae":np.mean(abs(res)),"rmse":math.sqrt(np.mean(res**2))})
            mt=fit_dynamic_v2(train,tau,max(lambdas),0,include_scale=False); pred=mt.predict(test["log10_compute"],test["submission_date"].dt.to_period("M")); res=test["ability_logit"].to_numpy()-pred
            rows.append({"fold_cutoff":str(cutoff),"horizon_months":horizon,"model":"M-time","lambda":max(lambdas),"rho":0,"n_train":len(train),"n_test":len(test),"pinball":np.mean(pinball(res,tau)),"mae":np.mean(abs(res)),"rmse":math.sqrt(np.mean(res**2))})
            z=train["log10_compute"].to_numpy(); y=train["ability_logit"].to_numpy(); xm,zs=z.mean(),z.std() or 1.; zz=(z-xm)/zs
            def obj(q): return np.mean(pinball(y-q[0]-q[1]*zz,tau))
            ms=minimize(obj,[np.quantile(y,tau),.1],method="L-BFGS-B",bounds=[(None,None),(0,None)])
            pred=ms.x[0]+ms.x[1]*(test["log10_compute"].to_numpy()-xm)/zs; res=test["ability_logit"].to_numpy()-pred
            rows.append({"fold_cutoff":str(cutoff),"horizon_months":horizon,"model":"M-scale","lambda":np.nan,"rho":np.nan,"n_train":len(train),"n_test":len(test),"pinball":np.mean(pinball(res,tau)),"mae":np.mean(abs(res)),"rmse":math.sqrt(np.mean(res**2))})
    cv=pd.DataFrame(rows); candidates=cv[cv["model"].eq("M-C")].groupby(["lambda","rho"])["pinball"].agg(["mean","std","count"]).reset_index()
    best=candidates.loc[candidates["mean"].idxmin()]; threshold=best["mean"]+(best["std"]/np.sqrt(best["count"]) if best["count"]>1 else 0)
    eligible=candidates[candidates["mean"]<=threshold].sort_values(["lambda","rho"],ascending=[False,True])
    chosen=eligible.iloc[0]
    return cv,float(chosen["lambda"]),float(chosen["rho"]),float(threshold)


def moving_block_weights(months, block_length, rng):
    unique=sorted(pd.Period(m,freq="M") for m in set(months)); n=len(unique); counts={m:0 for m in unique}
    starts=np.arange(max(1,n-block_length+1)); picked=[]
    while len(picked)<n:
        s=int(rng.choice(starts)); picked.extend(unique[s:min(s+block_length,n)])
    for m in picked[:n]: counts[m]+=1
    return counts


def bootstrap_frontier_joint(frame, taus, lam, rho, draws, block_length, rng, eps):
    d=frame.reset_index(drop=True).copy(); families=sorted(d["family"].astype(str).unique()); output=[]
    for b in range(draws):
        fw={f:rng.exponential() for f in families}; mw=moving_block_weights(d["submission_date"].dt.to_period("M"),block_length,rng)
        w=np.asarray([fw[str(f)]*mw[pd.Period(m,freq="M")] for f,m in zip(d["family"],d["submission_date"])]); models={}; ok=True
        try:
            for tau in taus: models[tau]=fit_dynamic_v2(d,tau,lam,rho,w)
            ok=all(m.success for m in models.values()); dec=decompose_v2(models[.9],d,eps)
        except Exception:
            ok=False; dec={}
        for tau in taus:
            m=models.get(tau)
            row={"draw_id":b,"tau":tau,"fit_success":bool(ok and m is not None)}
            if m is not None:
                local,global_s,slope=m.slopes(); row.update({"beta0":m.beta0,"beta_x":m.beta_x,"x_mean":m.x_mean,"x_std":m.x_std,
                    "state_last":m.states[-1],"state_slope_local":local,"state_slope_global":global_s,"state_slope":slope,"last_month":str(m.months[-1])})
                if tau==.9: row.update(dec)
            output.append(row)
    return pd.DataFrame(output)


def evidence_class(method):
    text=str(method).lower()
    if "benchmark" in text or "comparison" in text: return "benchmark_or_comparison"
    if "reported" in text: return "reported"
    if "operation" in text: return "operation_counting"
    if "hardware" in text: return "hardware_based"
    return "third_party_or_unknown"


def prepare_compute_panel(c4, cutoff, lookback, multipliers):
    d=c4.copy(); d["date"]=pd.to_datetime(d["Publication date"],errors="coerce"); d["compute"]=pd.to_numeric(d["Training compute (FLOP)"],errors="coerce")
    d["lower"]=pd.to_numeric(d["Training compute lower bound"],errors="coerce"); d["upper"]=pd.to_numeric(d["Training compute upper bound"],errors="coerce")
    d["evidence"]=d["Training compute estimation method"].map(evidence_class); d["family"]=d["Model"].map(family_label)
    d["confidence_clean"]=d["Confidence"].fillna("missing").astype(str)
    ok=(d["Domain"].fillna("").str.contains("language",case=False)&d["Open model weights?"].fillna("").str.lower().eq("yes")&d["compute"].gt(0)&d["date"].le(cutoff)&d["evidence"].isin(["reported","operation_counting","hardware_based"]))
    d=d[ok & d["date"].ge(cutoff-pd.DateOffset(months=lookback))].copy(); d["month"]=d["date"].dt.to_period("M")
    for idx,row in d.iterrows():
        lo,hi=row["lower"],row["upper"]
        if not (np.isfinite(lo) and lo>0): lo=row["compute"]*multipliers.get(row["confidence_clean"],multipliers["missing"])[0]
        if not (np.isfinite(hi) and hi>0): hi=row["compute"]*multipliers.get(row["confidence_clean"],multipliers["missing"])[1]
        d.at[idx,"lower_filled"]=min(lo,row["compute"]); d.at[idx,"upper_filled"]=max(hi,row["compute"])
        d.at[idx,"bound_imputed"]=not(np.isfinite(row["lower"]) and row["lower"]>0 and np.isfinite(row["upper"]) and row["upper"]>0)
    d["precision_weight"]=1/(np.log10(d["upper_filled"])-np.log10(d["lower_filled"])+.05)
    fam=d.sort_values("compute").groupby(["month","family"],as_index=False).tail(1)
    monthly=[]
    for month,g in fam.groupby("month"):
        monthly.append({"month":str(month),"model_families":g["family"].nunique(),"compute_q90":10**weighted_quantile(np.log10(g["compute"]),.9,g["precision_weight"]),
                        "log10_compute_q90":weighted_quantile(np.log10(g["compute"]),.9,g["precision_weight"])})
    monthly=pd.DataFrame(monthly).sort_values("month"); monthly["month_index"]=np.arange(len(monthly))
    return d,fam,monthly


def fit_growth(monthly, cutoff):
    m=monthly.copy(); periods=pd.PeriodIndex(m["month"],freq="M"); base=periods.min(); x=np.asarray([(p.year-base.year)*12+p.month-base.month for p in periods])
    y=m["log10_compute_q90"].to_numpy(); slope,intercept,lo,hi=theilslopes(y,x)
    end_period=cutoff.to_period("M"); end_x=(end_period.year-base.year)*12+end_period.month-base.month
    return {"monthly_log10_slope":float(slope),"annual_log10_slope":float(slope*12),"annual_log10_slope_low":float(lo*12),"annual_log10_slope_high":float(hi*12),
            "endpoint_log10_compute":float(intercept+slope*end_x),"n_months":len(m),"n_families_min":int(m["model_families"].min())}


def bootstrap_compute_growth(family_month, cutoff, lookback, draws, rng):
    families=sorted(family_month["family"].unique()); periods=sorted(family_month["month"].unique()); rows=[]
    for b in range(draws):
        sampled=rng.choice(families,len(families),replace=True); parts=[]
        for j,f in enumerate(sampled): parts.append(family_month[family_month["family"].eq(f)].assign(cluster_copy=j))
        s=pd.concat(parts,ignore_index=True); month_counts=moving_block_weights(periods,2,rng)
        s=s.loc[s.index.repeat([month_counts[p] for p in s["month"]])].copy()
        if s.empty: rows.append({"draw_id":b,"fit_success":False}); continue
        mode=np.log10(s["compute"]).to_numpy(); lo=np.log10(s["lower_filled"]).to_numpy(); hi=np.log10(s["upper_filled"]).to_numpy()
        sampled_compute=mode.copy(); uncertain=(hi-lo)>1e-12
        sampled_compute[uncertain]=rng.triangular(lo[uncertain],mode[uncertain],hi[uncertain])
        s["log_compute_draw"]=sampled_compute
        monthly=[]
        for month,g in s.groupby("month"):
            monthly.append({"month":str(month),"model_families":g["cluster_copy"].nunique(),"log10_compute_q90":weighted_quantile(g["log_compute_draw"],.9,g["precision_weight"])})
        monthly=pd.DataFrame(monthly)
        try: rows.append({"draw_id":b,"fit_success":True,**fit_growth(monthly,cutoff)})
        except Exception: rows.append({"draw_id":b,"fit_success":False})
    return pd.DataFrame(rows)


def forecast_joint(models, frontier_boot, compute_boot, cutoff, central_growth, factors, horizons, eps):
    central=[]
    for factor in factors:
        for horizon in horizons:
            month=cutoff.to_period("M")+horizon; raw=[]
            for tau,m in sorted(models.items()):
                x=central_growth["endpoint_log10_compute"]+central_growth["annual_log10_slope"]*factor*horizon/12
                y=float(m.predict([x],[month])[0]); raw.append((tau,y,x))
            ordered=np.sort([x[1] for x in raw])
            for (tau,y,x),yr in zip(raw,ordered):
                central.append({"tau":tau,"scenario_factor":factor,"horizon_months":horizon,"forecast_month":str(month),
                    "forecast_score_raw":float(link_to_score(y,eps)),"forecast_link_raw":y,"forecast_score":float(link_to_score(yr,eps)),"forecast_link":float(yr),
                    "quantile_rearranged":bool(abs(y-yr)>1e-12),"compute_endpoint_log10":central_growth["endpoint_log10_compute"],
                    "compute_growth_annual_log10":central_growth["annual_log10_slope"],"forecast_log10_compute":x,
                    "extrapolation_flag":"long_horizon_extrapolation" if horizon>=24 else "extrapolation"})
    central=pd.DataFrame(central); draws=[]
    valid_f=frontier_boot[frontier_boot["fit_success"]].copy(); valid_c=compute_boot[compute_boot["fit_success"]].copy()
    common=min(valid_f["draw_id"].nunique(),len(valid_c)); ids=sorted(valid_f["draw_id"].unique())[:common]; cb=valid_c.iloc[:common].set_index("draw_id",drop=False)
    for j,draw_id in enumerate(ids):
        fg=valid_f[valid_f["draw_id"].eq(draw_id)].sort_values("tau"); cr=valid_c.iloc[j]
        for factor in factors:
            for horizon in horizons:
                raw=[]; month=cutoff.to_period("M")+horizon
                for _,r in fg.iterrows():
                    x=cr["endpoint_log10_compute"]+cr["annual_log10_slope"]*factor*horizon/12
                    last=pd.Period(r["last_month"],freq="M"); gap=(month.year-last.year)*12+month.month-last.month
                    state=r["state_last"]+r["state_slope"]*gap
                    y=r["beta0"]+r["beta_x"]*((x-r["x_mean"])/r["x_std"])+state
                    raw.append((r["tau"],y,x))
                ordered=np.sort([x[1] for x in raw])
                for (tau,y,x),yr in zip(raw,ordered):
                    draws.append({"draw_id":int(draw_id),"tau":tau,"scenario_factor":factor,"horizon_months":horizon,
                        "forecast_link_raw":y,"forecast_score_raw":float(link_to_score(y,eps)),"forecast_link":float(yr),"forecast_score":float(link_to_score(yr,eps)),
                        "quantile_rearranged":bool(abs(y-yr)>1e-12),"compute_endpoint_log10":cr["endpoint_log10_compute"],
                        "compute_growth_annual_log10":cr["annual_log10_slope"],"forecast_log10_compute":x})
    draws=pd.DataFrame(draws); intervals=[]
    for key,g in draws.groupby(["tau","scenario_factor","horizon_months"]):
        vals=g["forecast_score"]
        intervals.append({"tau":key[0],"scenario_factor":key[1],"horizon_months":key[2],"p10":vals.quantile(.1),"p90":vals.quantile(.9),
                          "p025":vals.quantile(.025),"p975":vals.quantile(.975),"bootstrap_draw_count":len(vals),"interval_tau":key[0]})
    central=central.merge(pd.DataFrame(intervals),on=["tau","scenario_factor","horizon_months"],how="left")
    central["forecast_support"]="empirical_small_sample"; central["uncertainty_scope"]="frontier_state_compute_endpoint_growth_measurement"
    return central,draws


def nd_diagnostics(frontier):
    rows=[]
    for name,cols in [("M-N",["log10_params"]),("M-ND",["log10_params","log10_data"])]:
        d=frontier.dropna(subset=cols).copy(); rec={"model":name,"rows":len(d),"condition_number":np.nan,"fit_success":False}
        if len(d)>=8:
            X=sm.add_constant(d[cols]); rec["condition_number"]=float(np.linalg.cond(X.to_numpy()))
            try:
                fit=sm.QuantReg(d["ability_logit"],X).fit(q=.9,max_iter=3000); rec["fit_success"]=True
                for k,v in fit.params.items(): rec[f"coef_{k}"]=float(v)
            except Exception as exc: rec["error"]=str(exc)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_decomposition(boot):
    d=boot[(boot["tau"]==.9)&boot["fit_success"]].copy(); rows=[]
    for col in ["scale_logit","tech_logit","total_logit","scale_score","tech_score","total_score","beta_x"]:
        x=d[col].dropna(); rows.append({"quantity":col,"median":x.median(),"p10":x.quantile(.1),"p90":x.quantile(.9),
            "p025":x.quantile(.025),"p975":x.quantile(.975),"positive_probability":float((x>0).mean())})
    return pd.DataFrame(rows)
