from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCORE_COLS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, **kwargs)
    tmp.replace(path)


def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    raise TypeError(type(x).__name__)


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def model_core(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).lower().strip().split("/")[-1]
    s = re.sub(r"\b(hf|gguf|awq|gptq)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def owner_core(x) -> str:
    if pd.isna(x):
        return ""
    return norm_text(str(x).split("/")[0])


def model_family(x) -> str:
    s = str(x).lower()
    rules = [
        ("pythia", "pythia"), ("qwen", "qwen"), ("llama", "llama"),
        ("gemma", "gemma"), ("mistral", "mistral"), ("falcon", "falcon"),
        ("phi", "phi"), ("yi-", "yi"), ("yi_", "yi"), ("olmo", "olmo"),
        ("bloom", "bloom"), ("gpt", "gpt"), ("deepseek", "deepseek"),
    ]
    for needle, label in rules:
        if needle in s:
            return label
    return owner_core(x) or "other"


def type_group(x) -> str:
    s = str(x).lower()
    if "pretrained" in s:
        return "pretrained"
    if "merge" in s:
        return "merge"
    if any(k in s for k in ("chat", "rlhf", "dpo", "ift", "fine", "instruct")):
        return "posttrained"
    return "excluded_other"


def compute_evidence(row: pd.Series) -> str:
    note = " ".join(str(row.get(c, "")) for c in
                   ["Training compute notes", "Training compute estimation method"]).lower()
    if pd.isna(row.get("Training compute (FLOP)")):
        return "missing"
    if any(k in note for k in ("reported", "developer", "paper")):
        return "reported"
    if any(k in note for k in ("operation", "flop", "6nd")):
        return "operation_counting"
    if any(k in note for k in ("hardware", "chip", "gpu", "tpu")):
        return "hardware_based"
    return "third_party_or_benchmark"


def load_inputs(root: Path):
    names = {
        "C1": "leaderboard_cleaned.csv",
        "C2": "leaderboard_enhanced.csv",
        "C3": "leaderboard_extended_timeseries.csv",
        "C4": "epoch_all_ai_models.csv",
        "C5": "loss_benchmark_bridge.csv",
        "C6": "loss_benchmark_bridge_expanded.csv",
        "C7": "model_architecture_metadata.csv",
        "C9": "data/train-00000-of-00001.parquet",
    }
    frames = {}
    audit = {}
    for key, rel in names.items():
        path = root / rel
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        frames[key] = frame
        audit[key] = {
            "path": str(path), "sha256": sha256(path), "rows": len(frame),
            "columns": len(frame.columns), "duplicate_rows": int(frame.duplicated().sum()),
            "missing_cells": int(frame.isna().sum().sum()),
        }
    return frames, audit


def audit_contract(frames: dict, root: Path) -> dict:
    c1, c2, c3, c4, c5, c6, c9 = (frames[k] for k in ["C1", "C2", "C3", "C4", "C5", "C6", "C9"])
    six_mean = c1[SCORE_COLS].mean(axis=1)
    c5_set = set(map(tuple, c5.fillna("<NA>").astype(str).to_numpy()))
    c6_set = set(map(tuple, c6.fillna("<NA>").astype(str).to_numpy()))
    c8_dirs = list((root / "detailed_results").iterdir())
    c8_json = list((root / "detailed_results").glob("*/*.json"))
    dates = pd.to_datetime(c1["Submission Date"], errors="coerce")
    return {
        "c1_c2_score_equal": bool(np.allclose(c1[SCORE_COLS + ["Average ⬆️"]], c2[SCORE_COLS + ["Average ⬆️"]], equal_nan=True)),
        "c1_c9_rows_equal": len(c1) == len(c9),
        "average_max_abs_error": float(np.nanmax(np.abs(six_mean - c1["Average ⬆️"]))),
        "score_missing": int(c1[SCORE_COLS].isna().sum().sum()),
        "invalid_submission_dates": int(dates.isna().sum()),
        "submission_date_min": str(dates.min().date()),
        "submission_date_max": str(dates.max().date()),
        "license_missing": int(c1["Hub License"].isna().sum()),
        "duplicate_model_groups": int((c1.groupby("Model").size() > 1).sum()),
        "c3_sources": c3["Source"].value_counts(dropna=False).to_dict(),
        "c4_missing": {
            "parameters": int(c4["Parameters"].isna().sum()),
            "training_compute": int(c4["Training compute (FLOP)"].isna().sum()),
            "training_data": int(c4["Training dataset size (total)"].isna().sum()),
            "open_weights": int(c4["Open model weights?"].isna().sum()),
        },
        "c5_exact_subset_c6": c5_set <= c6_set,
        "c5_rows": len(c5), "c6_rows": len(c6),
        "c8_directories": len(c8_dirs), "c8_json_files": len(c8_json),
        "c7_role": "audited_only_not_used_in_model",
    }


def resolve_duplicates(c2: pd.DataFrame):
    d = c2.copy()
    d["_date"] = pd.to_datetime(d["Submission Date"], errors="coerce")
    d["_row"] = np.arange(len(d))
    d["type_group"] = d["Type"].map(type_group)
    d["model_version_id"] = (
        d["Model"].map(norm_text) + "|" + d["#Params (B)"].round(4).astype(str) +
        "|" + d["type_group"]
    )
    key = ["model_version_id", "Submission Date"]
    med_cols = SCORE_COLS + ["Average ⬆️"]
    rows, ledger = [], []
    for _, g in d.groupby(key, dropna=False, sort=False):
        keep = g.sort_values(["_date", "_row"], na_position="first").iloc[-1].copy()
        if len(g) > 1:
            keep[med_cols] = g[med_cols].median()
        rows.append(keep)
        for _, r in g.iterrows():
            ledger.append({
                "source_row": int(r["_row"]), "Model": r["Model"],
                "model_version_id": r["model_version_id"],
                "kept_source_row": int(keep["_row"]),
                "action": "kept_aggregated" if int(r["_row"]) == int(keep["_row"]) else "replaced_same_version_date",
                "reason": "same normalized model, parameter, type and submission date",
            })
    out = pd.DataFrame(rows).sort_values("_row").drop(columns=["_row"])
    return out.reset_index(drop=True), pd.DataFrame(ledger)


def _org_compatible(owner: str, org: str) -> bool:
    a, b = norm_text(owner), norm_text(org)
    aliases = [
        {"qwen", "alibaba", "alibabacloud"}, {"meta", "facebook"},
        {"huggingface", "huggingfacetb", "bigcode"}, {"01ai", "zerooneai"},
        {"eleutherai"}, {"google", "deepmind"}, {"microsoft"},
        {"mosaicml", "databricks"}, {"tii"}, {"stabilityai"},
        {"together", "togetherai"}, {"apple"},
    ]
    if not a or not b:
        return True
    if a in b or b in a:
        return True
    return any(a in group and any(x in b or b in x for x in group) for group in aliases)


def match_c2_c4(c2: pd.DataFrame, c4: pd.DataFrame, rel_tol: float = 0.05):
    e = c4.copy()
    e["_core"] = e["Model"].map(model_core)
    e["_org"] = e["Organization"].fillna("").astype(str)
    e["_param_b"] = pd.to_numeric(e["Parameters"], errors="coerce") / 1e9
    e["_compute"] = pd.to_numeric(e["Training compute (FLOP)"], errors="coerce")
    e["_data"] = pd.to_numeric(e["Training dataset size (total)"], errors="coerce")
    e["_pub"] = pd.to_datetime(e["Publication date"], errors="coerce")
    e["_evidence"] = e.apply(compute_evidence, axis=1)
    e["_idx4"] = np.arange(len(e))
    by_core = {k: g for k, g in e.groupby("_core") if k}
    ledger, picked = [], []
    for i, r in c2.iterrows():
        core = model_core(r["Model"])
        cand = by_core.get(core, e.iloc[0:0])
        reasons = []
        valid = []
        for _, q in cand.iterrows():
            p1, p2 = r["#Params (B)"], q["_param_b"]
            p_ok = pd.isna(p1) or pd.isna(p2) or abs(p1-p2)/max(abs(p1), abs(p2), 1e-12) <= rel_tol
            o_ok = _org_compatible(owner_core(r["Model"]), q["_org"])
            if p_ok and o_ok:
                valid.append(q)
            else:
                reasons.append(f"{int(q['_idx4'])}:parameter_or_org_conflict")
        status = "unmatched"
        chosen = None
        if len(valid) == 1:
            chosen = valid[0]; status = "accepted_exact"
        elif len(valid) > 1:
            with_compute = [q for q in valid if pd.notna(q["_compute"])]
            if len(with_compute) == 1:
                chosen = with_compute[0]; status = "accepted_exact_unique_compute"
            else:
                status = "ambiguous_excluded"
        ledger.append({
            "c2_row": i, "Model": r["Model"], "normalized_core": core,
            "candidate_count": len(cand), "valid_candidate_count": len(valid),
            "review_status": status,
            "c4_row": int(chosen["_idx4"]) if chosen is not None else np.nan,
            "evidence": "exact normalized name; parameter <= tolerance; organization compatible" if chosen is not None else ";".join(reasons[:5]),
        })
        picked.append(chosen)
    out = c2.copy()
    fields = {
        "training_compute_flop": "_compute", "training_data_size": "_data",
        "c4_publication_date": "_pub", "c4_open_weights": "Open model weights?",
        "c4_accessibility": "Model accessibility", "compute_evidence": "_evidence",
        "c4_domain": "Domain", "c4_task": "Task", "c4_organization": "Organization",
    }
    for dst, src in fields.items():
        out[dst] = [q[src] if q is not None else np.nan for q in picked]
    out["match_status"] = [x["review_status"] for x in ledger]
    return out, pd.DataFrame(ledger)


def build_panel(matched: pd.DataFrame, cutoff: pd.Timestamp):
    p = matched.copy()
    p["submission_date"] = pd.to_datetime(p["Submission Date"], errors="coerce")
    p["type_group"] = p["Type"].map(type_group)
    epoch_yes = p["Epoch_AI_Open_Weights"].fillna("").astype(str).str.lower().eq("yes")
    c4_yes = p["c4_open_weights"].fillna("").astype(str).str.lower().eq("yes")
    p["open_weight_verified"] = epoch_yes | c4_yes
    p["unrestricted"] = p["open_weight_verified"] & p["c4_accessibility"].fillna("").astype(str).str.contains("unrestricted", case=False)
    p["complete_score"] = p[SCORE_COLS + ["Average ⬆️"]].notna().all(axis=1)
    p["future_leakage_excluded"] = p["c4_publication_date"].notna() & (p["c4_publication_date"] > cutoff)
    p["family"] = p["Model"].map(model_family)
    p["ability_logit"] = np.log((p["Average ⬆️"] + 0.5) / (100.5 - p["Average ⬆️"]))
    compute = pd.to_numeric(p["training_compute_flop"], errors="coerce")
    params = pd.to_numeric(p["#Params (B)"], errors="coerce") * 1e9
    data = pd.to_numeric(p["training_data_size"], errors="coerce")
    p["log10_compute"] = np.log10(compute.where(compute > 0))
    p["log10_params"] = np.log10(params.where(params > 0))
    p["log10_data"] = np.log10(data.where(data > 0))
    p["frontier_eligible"] = (
        p["open_weight_verified"] & p["type_group"].eq("pretrained") &
        p["complete_score"] & p["submission_date"].notna() &
        p["log10_compute"].notna() & p["match_status"].str.startswith("accepted") &
        p["compute_evidence"].isin(["reported", "operation_counting", "hardware_based"]) &
        ~p["future_leakage_excluded"]
    )
    return p
