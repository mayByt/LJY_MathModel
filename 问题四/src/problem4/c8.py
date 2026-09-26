from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import model_core


MATH_KEYS = [
    "leaderboard_math_algebra_hard",
    "leaderboard_math_counting_and_prob_hard",
    "leaderboard_math_geometry_hard",
    "leaderboard_math_intermediate_algebra_hard",
    "leaderboard_math_num_theory_hard",
    "leaderboard_math_prealgebra_hard",
    "leaderboard_math_precalculus_hard",
]


def _weighted(d: dict, keys: list[str], metric: str):
    vals = []
    missing = []
    for key in keys:
        result = d.get("results", {}).get(key)
        counts = d.get("n-samples", {}).get(key, {})
        if not result or metric not in result:
            missing.append(key)
            continue
        n = counts.get("effective", counts.get("original"))
        if n is None or not np.isfinite(float(n)):
            missing.append(key)
            continue
        vals.append((key, float(result[metric]), float(n)))
    if not vals:
        return np.nan, 0, 0.0, missing
    den = sum(x[2] for x in vals)
    return sum(x[1] * x[2] for x in vals) / den, len(vals), den, missing


def aggregate_c8(root: Path, leaderboard: pd.DataFrame):
    official = {}
    for _, r in leaderboard.iterrows():
        official.setdefault(model_core(r["Model"]), []).append((float(r["BBH"]), float(r["MATH Lvl 5"]), r["Model"]))
    rows = []
    corrupt = []
    parseable_dirs = 0
    fallback_dirs = 0
    no_parse_dirs = 0
    for directory in sorted((root / "detailed_results").iterdir()):
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("*.json"), reverse=True)
        selected = None
        payload = None
        failed = []
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                selected = path
                break
            except Exception as exc:
                failed.append(path)
                corrupt.append({
                    "model_directory": directory.name, "file": str(path),
                    "error_type": type(exc).__name__, "error": str(exc)[:300],
                })
        if selected is None:
            no_parse_dirs += 1
            continue
        parseable_dirs += 1
        fallback = bool(failed)
        fallback_dirs += int(fallback)
        bbh_keys = [
            k for k in payload.get("results", {})
            if k.startswith("leaderboard_bbh_") and k != "leaderboard_bbh"
        ]
        bbh, bcnt, bn, bmiss = _weighted(payload, bbh_keys, "acc_norm,none")
        math, mcnt, mn, mmiss = _weighted(payload, MATH_KEYS, "exact_match,none")
        raw_name = payload.get("model_name") or payload.get("model_name_sanitized") or directory.name
        core = model_core(raw_name)
        choices = official.get(core, [])
        off_bbh = choices[-1][0] if len(choices) == 1 else np.nan
        off_math = choices[-1][1] if len(choices) == 1 else np.nan
        rows.append({
            "model_directory": directory.name,
            "json_file": str(selected), "fallback_used": fallback,
            "model_name_json": raw_name, "normalized_core": core,
            "bbh_weighted_pct": bbh * 100 if np.isfinite(bbh) else np.nan,
            "bbh_subtasks": bcnt, "bbh_effective_n": bn,
            "bbh_missing_subtasks": "|".join(bmiss),
            "math_weighted_pct": math * 100 if np.isfinite(math) else np.nan,
            "math_categories": mcnt, "math_effective_n": mn,
            "math_missing_categories": "|".join(mmiss),
            "official_bbh": off_bbh, "official_math": off_math,
            "bbh_abs_diff": abs(bbh * 100 - off_bbh) if np.isfinite(bbh) and np.isfinite(off_bbh) else np.nan,
            "math_abs_diff": abs(math * 100 - off_math) if np.isfinite(math) and np.isfinite(off_math) else np.nan,
        })
    frame = pd.DataFrame(rows)
    summary = {
        "directories": parseable_dirs + no_parse_dirs,
        "parseable_directories": parseable_dirs,
        "no_parseable_directories": no_parse_dirs,
        "fallback_directories": fallback_dirs,
        "corrupt_files": len(corrupt),
        "bbh_complete": int((frame["bbh_subtasks"] > 0).sum()) if len(frame) else 0,
        "math_complete": int((frame["math_categories"] == 7).sum()) if len(frame) else 0,
        "official_exact_bbh": int((frame["bbh_abs_diff"] <= 1e-6).sum()) if len(frame) else 0,
        "official_exact_math": int((frame["math_abs_diff"] <= 1e-6).sum()) if len(frame) else 0,
    }
    return frame, pd.DataFrame(corrupt), summary
