from __future__ import annotations

import collections
import hashlib
import json
import lzma
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, ks_2samp, wasserstein_distance

from .common import (
    binary_positive_probability,
    finite_or_nan,
    huber_location,
    huber_mean_1d,
    json_dump,
    ordinal_expectation,
    sha256_file,
    weighted_median_matrix,
)


PRRC_FIELDS = [
    "modernbert_cleanliness",
    "modernbert_readability",
    "modernbert_reasoning",
    "modernbert_professionalism",
]
SCALAR_FIELDS = [
    "dsir_books",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_doc_num_sentences",
    "rps_doc_word_count",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_doc_frac_unique_words",
    "rps_lines_numerical_chars_fraction",
    "dsir_math",
    "rps_doc_mean_word_length",
    "dsir_wiki",
    "rps_doc_frac_chars_top_3gram",
    "rps_doc_unigram_entropy",
]
RAW_COLUMNS = (
    ["fineweb_edu"]
    + [f"qurater_{i}" for i in range(4)]
    + ["ad_en", "fluency_en"]
    + PRRC_FIELDS
    + SCALAR_FIELDS
)
QUALITY_FIELDS = [
    "fineweb_edu",
    "qurater",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "fluency_en",
    "modernbert_readability",
    "modernbert_cleanliness",
    "rps_lines_ending_with_terminal_punctution_mark",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "ad_en",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_numerical_chars_fraction",
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_mean_word_length",
]
BLOCKS = {
    "education": ["fineweb_edu", "qurater", "dsir_books", "dsir_wiki", "dsir_math"],
    "expression": [
        "fluency_en",
        "modernbert_readability",
        "modernbert_cleanliness",
        "rps_lines_ending_with_terminal_punctution_mark",
    ],
    "reasoning": [
        "modernbert_reasoning",
        "modernbert_professionalism",
        "rps_doc_unigram_entropy",
        "rps_doc_frac_unique_words",
    ],
    "noise": [
        "ad_en",
        "rps_doc_frac_no_alph_words",
        "rps_doc_frac_chars_top_2gram",
        "rps_doc_frac_chars_top_3gram",
        "rps_lines_uppercase_letter_fraction",
        "rps_lines_numerical_chars_fraction",
    ],
    "structure": ["rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_mean_word_length"],
}
MODEL_FIELDS = {
    "fineweb_edu",
    "qurater",
    "fluency_en",
    "modernbert_readability",
    "modernbert_cleanliness",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "ad_en",
}
DSIR_FIELDS = {"dsir_books", "dsir_wiki", "dsir_math"}
POSITIVE_FIELDS = {
    "fineweb_edu",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "rps_lines_ending_with_terminal_punctution_mark",
}
SATURATING_FIELDS = {"rps_doc_unigram_entropy", "rps_doc_frac_unique_words"}
NEGATIVE_FIELDS = {
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
}
MID_FIELDS = {"rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_mean_word_length"}


@dataclass
class QualityLoadResult:
    raw: pd.DataFrame
    text_features_a1: pd.DataFrame
    text_proxies_a1: pd.DataFrame
    audit: dict[str, Any]


def _first_finite(value: Any) -> float:
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        return math.nan
    return finite_or_nan(value[0])


def extract_raw_scores(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    out["fineweb_edu"] = _first_finite(row.get("fineweb_edu"))
    qurater = row.get("qurater")
    for i in range(4):
        out[f"qurater_{i}"] = (
            finite_or_nan(qurater[i]) if isinstance(qurater, (list, tuple)) and len(qurater) == 4 else math.nan
        )
    out["ad_en"] = binary_positive_probability(row.get("ad_en"))
    out["fluency_en"] = binary_positive_probability(row.get("fluency_en"))
    for field in PRRC_FIELDS:
        out[field] = ordinal_expectation(row.get(field), levels=6)
    for field in SCALAR_FIELDS:
        out[field] = finite_or_nan(row.get(field))
    return out


def _count_nonfinite_original(row: dict[str, Any]) -> tuple[int, list[str]]:
    count = 0
    fields: list[str] = []
    for field in ["fineweb_edu", "qurater", "ad_en", "fluency_en", *PRRC_FIELDS, *SCALAR_FIELDS]:
        value = row.get(field)
        seq = value if isinstance(value, (list, tuple)) else [value]
        local = 0
        for item in seq:
            try:
                local += int(not math.isfinite(float(item)))
            except (TypeError, ValueError):
                local += 1
        if local:
            count += local
            fields.append(field)
    return count, fields


_URL_RE = re.compile(r"https?://|www\.", re.I)
_HTML_RE = re.compile(r"<(?:script|style|iframe|div|span|html|body|head)\b", re.I)
_TOKEN_RE = re.compile(r"\S+")
_PUNCT_SET = set(string.punctuation)
_CODE_SET = set("{}[]();<>:=_`\\|&^%$#@~")


def text_distribution_features(text: str, char_cap: int = 20000, token_cap: int = 1000) -> dict[str, float]:
    text = text or ""
    n_full = len(text)
    if n_full > char_cap:
        half = char_cap // 2
        sample = text[:half] + text[-half:]
    else:
        sample = text
    n = max(len(sample), 1)
    lines = sample.splitlines() or [sample]
    alpha = sum(ch.isalpha() for ch in sample) / n
    digit = sum(ch.isdigit() for ch in sample) / n
    upper = sum(ch.isupper() for ch in sample) / n
    whitespace = sum(ch.isspace() for ch in sample) / n
    punct = sum(ch in _PUNCT_SET for ch in sample) / n
    code = sum(ch in _CODE_SET for ch in sample) / n
    url = len(_URL_RE.findall(sample)) / max(len(lines), 1)
    tokens = _TOKEN_RE.findall(sample)[:token_cap]
    richness = len(set(tokens)) / max(len(tokens), 1)
    if len(tokens) >= 3:
        pairs = collections.Counter(zip(tokens, tokens[1:]))
        triples = collections.Counter(zip(tokens, tokens[1:], tokens[2:]))
        rep2 = max(pairs.values()) / max(len(tokens) - 1, 1)
        rep3 = max(triples.values()) / max(len(tokens) - 2, 1)
    else:
        rep2 = rep3 = 0.0
    return {
        "log_chars": math.log1p(n_full),
        "log_lines": math.log1p(text.count("\n") + 1),
        "alpha_ratio": alpha,
        "digit_ratio": digit,
        "upper_ratio": upper,
        "punct_ratio": punct,
        "whitespace_ratio": whitespace,
        "url_per_line": url,
        "code_symbol_ratio": code,
        "token_richness": richness,
        "top2_proxy": rep2,
        "top3_proxy": rep3,
    }


def text_validation_proxies(text: str) -> tuple[dict[str, float], list[int]]:
    text = text or ""
    n = max(len(text), 1)
    lines = text.splitlines() or [text]
    stripped = [line.strip() for line in lines if line.strip()]
    line_count = max(len(stripped), 1)
    controls = sum((ord(ch) < 32 and ch not in "\n\r\t") or ch == "�" for ch in text) / n
    html = len(_HTML_RE.findall(text)) / line_count
    url_only = sum(bool(re.fullmatch(r"\s*(?:https?://|www\.)\S+\s*", line, re.I)) for line in stripped) / line_count
    duplicate = 1.0 - len(set(stripped)) / line_count
    tokens = _TOKEN_RE.findall(text[:50000])
    long_token = sum(len(t) > 100 for t in tokens) / max(len(tokens), 1)
    candidates = []
    selected = (stripped[:3] + stripped[-3:])[:6]
    for line in selected:
        normalized = re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.lower())).strip()
        if 20 <= len(normalized) <= 200:
            digest = hashlib.blake2b(normalized.encode("utf-8", "ignore"), digest_size=8).digest()
            candidates.append(int.from_bytes(digest, "little"))
    return {
        "control_replacement_rate": controls,
        "html_script_density": html,
        "url_only_line_rate": url_only,
        "duplicate_line_rate": duplicate,
        "long_token_rate": long_token,
    }, candidates


def _quality_files(root: Path) -> dict[str, Path]:
    return {
        "A1": root / "slimpajama_quality_signal_sample.jsonl.xz",
        "A2": root / "slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz",
        "A3": root / "slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz",
    }


def load_quality_data(input_root: Path, cfg: dict[str, Any], output_dir: Path) -> QualityLoadResult:
    files = _quality_files(input_root)
    records: list[dict[str, Any]] = []
    a1_features: list[dict[str, Any]] = []
    a1_proxies: list[dict[str, Any]] = []
    template_counter: collections.Counter[int] = collections.Counter()
    template_lists: list[list[int]] = []
    overlap_reference: dict[tuple[str, str], np.ndarray] = {}
    row_counts: dict[str, int] = {}
    field_counts: dict[str, int] = {}
    nonfinite: dict[str, dict[str, Any]] = {}
    domain_counts: collections.Counter[str] = collections.Counter()

    for label, path in files.items():
        rows = 0
        nf_elements = 0
        nf_records = 0
        nf_fields: collections.Counter[str] = collections.Counter()
        domain_override = {"A2": "arxiv", "A3": "github"}.get(label)
        with lzma.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if rows == 0:
                    field_counts[label] = len(row)
                rows += 1
                domain = str(domain_override or row.get("_source_domain", "")).strip().lower()
                rid = str(row.get("id"))
                score = extract_raw_scores(row)
                n_bad, bad_fields = _count_nonfinite_original(row)
                if n_bad:
                    nf_records += 1
                    nf_elements += n_bad
                    nf_fields.update(bad_fields)
                in_a1 = label == "A1"
                if label == "A1" and domain in {"arxiv", "github"}:
                    overlap_reference[(domain, rid)] = np.array([score[c] for c in RAW_COLUMNS], dtype=float)
                if label != "A1" or domain not in {"arxiv", "github"}:
                    rec = {"id": rid, "domain": domain, "source": label, "in_a1": in_a1}
                    rec.update(score)
                    records.append(rec)
                    domain_counts[domain] += 1
                if label == "A1":
                    text = str(row.get("content") or "")
                    feat = {"id": rid, "domain": domain}
                    feat.update(text_distribution_features(text, cfg["text_char_cap"], cfg["text_token_cap"]))
                    a1_features.append(feat)
                    proxy, hashes = text_validation_proxies(text)
                    template_counter.update(hashes)
                    template_lists.append(hashes)
                    prow = {"id": rid, "domain": domain}
                    prow.update(proxy)
                    a1_proxies.append(prow)
        row_counts[label] = rows
        nonfinite[label] = {
            "records": nf_records,
            "elements": nf_elements,
            "fields": dict(nf_fields),
        }

    # 标记扩展集中属于 A1 的 ID，并核验重复记录质量字段。
    mismatch_count = 0
    overlap_found = collections.Counter()
    for rec in records:
        key = (rec["domain"], rec["id"])
        if key in overlap_reference:
            rec["in_a1"] = True
            overlap_found[rec["domain"]] += 1
            current = np.array([rec[c] for c in RAW_COLUMNS], dtype=float)
            if not np.allclose(current, overlap_reference[key], equal_nan=True, atol=0.0, rtol=0.0):
                mismatch_count += 1

    for row, hashes in zip(a1_proxies, template_lists):
        row["boilerplate_signature_rate"] = (
            sum(template_counter[h] >= 20 for h in hashes) / len(hashes) if hashes else 0.0
        )

    raw = pd.DataFrame.from_records(records)
    text_features = pd.DataFrame.from_records(a1_features)
    text_proxies = pd.DataFrame.from_records(a1_proxies)
    audit = {
        "files": {
            label: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for label, path in files.items()
        },
        "rows": row_counts,
        "field_counts": field_counts,
        "nonfinite": nonfinite,
        "a1_domain_counts": dict(text_features["domain"].value_counts()),
        "deduplicated_reference_rows": int(len(raw)),
        "deduplicated_domain_counts": dict(domain_counts),
        "overlap_found": dict(overlap_found),
        "overlap_quality_mismatch_count": mismatch_count,
        "raw_columns": RAW_COLUMNS,
    }
    json_dump(output_dir / "data_audit_quality.json", audit)
    return QualityLoadResult(raw=raw, text_features_a1=text_features, text_proxies_a1=text_proxies, audit=audit)


def _balanced_cdf_transform(
    values: np.ndarray,
    domains: np.ndarray,
    low_q: float,
    high_q: float,
) -> tuple[np.ndarray, dict[str, float]]:
    finite = np.isfinite(values)
    arrays: list[np.ndarray] = []
    for domain in sorted(set(domains)):
        arr = np.sort(values[(domains == domain) & finite])
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return np.full_like(values, np.nan, dtype=float), {"low": math.nan, "high": math.nan}

    lo_global = min(a[0] for a in arrays)
    hi_global = max(a[-1] for a in arrays)

    def cdf_scalar(x: float) -> float:
        return float(np.mean([np.searchsorted(a, x, side="right") / a.size for a in arrays]))

    def quantile(q: float) -> float:
        lo, hi = lo_global, hi_global
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if cdf_scalar(mid) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    low = quantile(low_q)
    high = quantile(high_q)
    clipped = np.clip(values, low, high)
    result = np.zeros(values.shape, dtype=float)
    result[:] = np.nan
    valid = np.isfinite(clipped)
    if valid.any():
        cdf = np.zeros(valid.sum(), dtype=float)
        xv = clipped[valid]
        for arr in arrays:
            cdf += np.searchsorted(arr, xv, side="right") / arr.size
        result[valid] = cdf / len(arrays)
    return result, {"low": float(low), "high": float(high)}


def _balanced_anchor(raw: pd.DataFrame, field: str, trusted: np.ndarray) -> tuple[float, float]:
    medians = []
    mads = []
    x = np.log1p(np.maximum(raw[field].to_numpy(float), 0.0))
    for domain in sorted(raw["domain"].unique()):
        vals = x[(raw["domain"].to_numpy() == domain) & trusted & np.isfinite(x)]
        if vals.size:
            med = float(np.median(vals))
            mad = float(1.4826 * np.median(np.abs(vals - med)))
            medians.append(med)
            if mad > 0:
                mads.append(mad)
    m = float(np.median(medians)) if medians else float(np.nanmedian(x))
    s = float(np.median(mads)) if mads else max(float(np.nanstd(x)), 1e-3)
    finite_x = x[np.isfinite(x)]
    iqr = float(np.quantile(finite_x, 0.75) - np.quantile(finite_x, 0.25)) if finite_x.size else 1.0
    return m, max(s, 0.05 * iqr, 1e-6)


def transform_quality(raw: pd.DataFrame, cfg: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    domains = raw["domain"].to_numpy(str)
    cdf_values: dict[str, np.ndarray] = {}
    specs: list[dict[str, Any]] = []
    for field in RAW_COLUMNS:
        z, bounds = _balanced_cdf_transform(
            raw[field].to_numpy(float), domains, cfg["winsor_low"], cfg["winsor_high"]
        )
        cdf_values[field] = z
        specs.append({"raw_field": field, "winsor_low": bounds["low"], "winsor_high": bounds["high"]})

    utilities = pd.DataFrame(index=raw.index)
    utilities["fineweb_edu"] = cdf_values["fineweb_edu"]
    utilities["qurater"] = np.nanmean(
        np.column_stack([cdf_values[f"qurater_{i}"] for i in range(4)]), axis=1
    )
    for field in ["dsir_books", "dsir_wiki", "dsir_math"]:
        utilities[field] = cdf_values[field]
    # 模型概率/有序期望本身已在 [0,1]，不再次做秩变换，以保留概率间距。
    for field in ["ad_en", "fluency_en", *PRRC_FIELDS]:
        utilities[field] = raw[field].to_numpy(float)
    for field in POSITIVE_FIELDS - {"fineweb_edu", "dsir_books", "dsir_wiki", "dsir_math"}:
        utilities[field] = cdf_values[field]
    for field in NEGATIVE_FIELDS:
        utilities[field] = 1.0 - cdf_values[field]
    for field in SATURATING_FIELDS:
        utilities[field] = np.minimum(cdf_values[field] / 0.95, 1.0)

    trusted = (
        (raw["ad_en"].to_numpy(float) > 0.8)
        & (raw["fluency_en"].to_numpy(float) > 0.8)
        & (raw["modernbert_cleanliness"].to_numpy(float) >= 0.8)
    )
    anchors: dict[str, tuple[float, float]] = {}
    for field in [*MID_FIELDS, "rps_lines_numerical_chars_fraction"]:
        m, s = _balanced_anchor(raw, field, trusted)
        anchors[field] = (m, s)
        x = np.log1p(np.maximum(raw[field].to_numpy(float), 0.0))
        mid = np.exp(-((x - m) ** 2) / (2.0 * s**2))
        if field == "rps_lines_numerical_chars_fraction":
            neg = 1.0 - cdf_values[field]
            utilities[field] = np.where(domains == "github", mid, neg)
        else:
            utilities[field] = mid

    utilities = utilities[QUALITY_FIELDS].clip(0.0, 1.0)
    transformed = pd.concat(
        [raw[["id", "domain", "source", "in_a1"]].reset_index(drop=True), utilities.reset_index(drop=True)],
        axis=1,
    )

    spec_df = pd.DataFrame(specs)
    spec_df["quality_field"] = spec_df["raw_field"].map(
        lambda x: "qurater" if x.startswith("qurater_") else x
    )
    spec_df["block"] = spec_df["quality_field"].map(
        lambda x: next((b for b, fs in BLOCKS.items() if x in fs), "")
    )
    spec_df["source"] = spec_df["quality_field"].map(
        lambda x: "DSIR" if x in DSIR_FIELDS else ("Model" if x in MODEL_FIELDS else "Rule")
    )
    spec_df["transform"] = spec_df["quality_field"].map(
        lambda x: "positive_cdf"
        if x in POSITIVE_FIELDS
        else (
            "negative_cdf"
            if x in NEGATIVE_FIELDS
            else ("saturating" if x in SATURATING_FIELDS else ("mid" if x in MID_FIELDS else "probability_or_expectation"))
        )
    )
    spec_df["anchor_m"] = spec_df["raw_field"].map(lambda x: anchors.get(x, (math.nan, math.nan))[0])
    spec_df["anchor_s"] = spec_df["raw_field"].map(lambda x: anchors.get(x, (math.nan, math.nan))[1])
    spec_df.to_csv(output_dir / "quality_transform_spec.csv", index=False)
    return transformed, spec_df


def _critic_weights(
    quality: pd.DataFrame, cfg: dict[str, Any], rng: np.random.Generator
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    sampled_parts = []
    for _, group in quality.groupby("domain", sort=True):
        n = min(len(group), int(cfg["critic_sample_per_domain"]))
        sampled_parts.append(group.iloc[rng.choice(len(group), size=n, replace=False)])
    sample = pd.concat(sampled_parts, ignore_index=True)
    results: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    reps = int(cfg["bootstrap_reps"])

    for block, fields in BLOCKS.items():
        corr = sample[fields].corr(method="spearman").fillna(0.0).to_numpy()
        sd = sample[fields].std(ddof=1).fillna(0.0).to_numpy()
        info = sd * np.sum(1.0 - np.abs(corr), axis=1)
        stability = []
        for field in fields:
            estimates = np.empty(reps, dtype=float)
            domain_arrays = [g[field].dropna().to_numpy(float) for _, g in sample.groupby("domain")]
            for b in range(reps):
                meds = []
                for arr in domain_arrays:
                    if arr.size:
                        meds.append(float(np.median(arr[rng.integers(0, arr.size, arr.size)])))
                estimates[b] = np.median(meds) if meds else np.nan
            mean = float(np.nanmean(estimates))
            cv = float(np.nanstd(estimates, ddof=1) / max(abs(mean), 1e-6))
            stability.append(1.0 / (1.0 + cv))
        score = info * np.asarray(stability)
        if not np.isfinite(score).any() or np.nansum(score) <= 0:
            weight = np.full(len(fields), 1.0 / len(fields))
        else:
            score = np.nan_to_num(score, nan=0.0)
            weight = score / score.sum()
        results[block] = dict(zip(fields, map(float, weight)))
        for f, sdev, inf, stab, w in zip(fields, sd, info, stability, weight):
            rows.append(
                {
                    "block": block,
                    "field": f,
                    "source": "DSIR" if f in DSIR_FIELDS else ("Model" if f in MODEL_FIELDS else "Rule"),
                    "sd": sdev,
                    "critic_information": inf,
                    "stability": stab,
                    "conditional_weight": w,
                    "total_weight": 0.2 * w,
                }
            )
    return results, pd.DataFrame(rows)


def score_quality(
    transformed: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
    text_proxies_a1: pd.DataFrame,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg["seed"]))
    weights, weight_df = _critic_weights(transformed, cfg, rng)
    weight_df.to_csv(output_dir / "quality_weights.csv", index=False)

    scored = transformed.copy()
    block_cols = []
    for block, fields in BLOCKS.items():
        w = np.array([weights[block][f] for f in fields], dtype=float)
        vals = scored[fields].to_numpy(float)
        valid = np.isfinite(vals)
        denom = valid @ w
        numer = np.nansum(vals * w[None, :], axis=1)
        col = f"block_{block}"
        scored[col] = np.divide(numer, denom, out=np.full(len(scored), np.nan), where=denom > 0)
        block_cols.append(col)

    bmat = scored[block_cols].to_numpy(float)
    scored["K"] = np.nanmax(bmat, axis=1) - np.nanmin(bmat, axis=1)
    thresholds = scored.groupby("domain")["K"].quantile(float(cfg["conflict_quantile"])).to_dict()
    k_threshold = scored["domain"].map(thresholds).to_numpy(float)
    high = np.nanmax(bmat, axis=1) >= float(cfg["conflict_high"])
    low = np.nanmin(bmat, axis=1) <= float(cfg["conflict_low"])
    conflict = (scored["K"].to_numpy(float) > k_threshold) & high & low
    scored["conflict"] = conflict

    argmax = np.nanargmax(np.where(np.isfinite(bmat), bmat, -np.inf), axis=1)
    argmin = np.nanargmin(np.where(np.isfinite(bmat), bmat, np.inf), axis=1)
    block_names = np.array(list(BLOCKS))
    conflict_labels = np.char.add(np.char.add(block_names[argmax], ">"), block_names[argmin])
    scored["conflict_type"] = np.where(conflict, conflict_labels, "none")

    total_weights = []
    for field in QUALITY_FIELDS:
        block = next(b for b, fs in BLOCKS.items() if field in fs)
        total_weights.append(0.2 * weights[block][field])
    total_weights_arr = np.asarray(total_weights)
    qvals = scored[QUALITY_FIELDS].to_numpy(float)
    med = weighted_median_matrix(qvals, total_weights_arr)
    valid = np.isfinite(qvals)
    denom = valid @ total_weights_arr
    qci_num = np.nansum(np.abs(qvals - med[:, None]) * total_weights_arr[None, :], axis=1)
    scored["QCI"] = np.divide(qci_num, denom, out=np.full(len(scored), np.nan), where=denom > 0)
    scored["strong_conflict"] = (np.nanmax(qvals, axis=1) > 0.8) & (np.nanmin(qvals, axis=1) < 0.2)

    scales: dict[tuple[str, str], float] = {}
    for domain, group in scored.groupby("domain"):
        for col in block_cols:
            x = group[col].dropna().to_numpy(float)
            m = np.median(x) if x.size else 0.5
            scales[(domain, col)] = max(1.4826 * np.median(np.abs(x - m)), 1e-3) if x.size else 0.1

    q = np.nanmean(bmat, axis=1)
    converged = np.ones(len(scored), dtype=bool)
    iterations = np.zeros(len(scored), dtype=int)
    for idx in np.flatnonzero(conflict):
        domain = scored.iloc[idx]["domain"]
        s = np.array([scales[(domain, col)] for col in block_cols])
        loc, conv, it = huber_location(
            bmat[idx],
            weights=np.full(len(block_cols), 0.2),
            scales=s,
            c=float(cfg["huber_c"]),
            tol=float(cfg["huber_tol"]),
            max_iter=int(cfg["huber_max_iter"]),
        )
        q[idx] = loc
        converged[idx] = conv
        iterations[idx] = it
    scored["Q"] = np.clip(q, 0.0, 1.0)
    scored["huber_converged"] = converged
    scored["huber_iterations"] = iterations
    scored["missing_block_count"] = np.sum(~np.isfinite(bmat), axis=1)

    # 预注册的聚合敏感性：不重新调参，只改变权重/稳健聚合规则。
    variant_values = {
        "field_equal": np.nanmean(qvals, axis=1),
        "block_equal_fields": np.nanmean(
            np.column_stack([np.nanmean(scored[fields].to_numpy(float), axis=1) for fields in BLOCKS.values()]),
            axis=1,
        ),
        "weighted_no_huber": np.nanmean(bmat, axis=1),
        "block_median": np.nanmedian(bmat, axis=1),
    }
    sensitivity_rows = []
    main_domain = scored.groupby("domain")["Q"].apply(huber_mean_1d)
    for name, values in variant_values.items():
        scored[f"Q_variant_{name}"] = np.clip(values, 0.0, 1.0)
        variant_domain = scored.assign(_variant=values).groupby("domain")["_variant"].apply(huber_mean_1d)
        common_domains = main_domain.index.intersection(variant_domain.index)
        tau = float(kendalltau(main_domain.loc[common_domains], variant_domain.loc[common_domains]).statistic)
        overlaps = []
        for _, group in scored.assign(_variant=values).groupby("domain"):
            n_dec = max(1, int(np.ceil(0.1 * len(group))))
            main_order = group["Q"].sort_values().index.to_numpy()
            alt_order = group["_variant"].sort_values().index.to_numpy()
            low_overlap = len(set(main_order[:n_dec]) & set(alt_order[:n_dec])) / n_dec
            high_overlap = len(set(main_order[-n_dec:]) & set(alt_order[-n_dec:])) / n_dec
            overlaps.append((low_overlap + high_overlap) / 2.0)
        sensitivity_rows.append(
            {"variant": name, "domain_kendall": tau, "median_extreme_overlap": float(np.median(overlaps))}
        )
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(output_dir / "quality_sensitivity.csv", index=False)

    sample_path = output_dir / "quality_sample.csv.gz"
    scored.to_csv(sample_path, index=False, compression="gzip")

    domain_rows = []
    bootstrap_draws: dict[str, np.ndarray] = {}
    reps = int(cfg["bootstrap_reps"])
    for domain, group in scored.groupby("domain", sort=True):
        values = group["Q"].dropna().to_numpy(float)
        draws = np.empty(reps, dtype=float)
        for b in range(reps):
            draws[b] = huber_mean_1d(values[rng.integers(0, len(values), len(values))], c=cfg["huber_c"])
        bootstrap_draws[domain] = draws
        lo, hi = np.quantile(draws, [0.025, 0.975])
        trim_n = int(0.1 * len(values))
        sorted_v = np.sort(values)
        trimmed = sorted_v[trim_n : len(values) - trim_n] if trim_n else sorted_v
        domain_rows.append(
            {
                "domain": domain,
                "n": len(values),
                "q": huber_mean_1d(values, c=cfg["huber_c"]),
                "ci_low": lo,
                "ci_high": hi,
                "median": np.median(values),
                "trimmed_mean_10pct": np.mean(trimmed),
                "conflict_rate": group["conflict"].mean(),
                "strong_conflict_rate": group["strong_conflict"].mean(),
                "missing_block_rate": (group["missing_block_count"] > 0).mean(),
            }
        )
    domain_df = pd.DataFrame(domain_rows)
    domain_df.to_csv(output_dir / "quality_domain_7.csv", index=False)
    np.savez_compressed(output_dir / "quality_domain_bootstrap.npz", **bootstrap_draws)

    comparisons = []
    for domain in ["arxiv", "github"]:
        g = scored[scored["domain"] == domain]
        a = g[g["in_a1"]]["Q"].dropna().to_numpy(float)
        r = g[~g["in_a1"]]["Q"].dropna().to_numpy(float)
        if len(a) and len(r):
            diff_draw = np.empty(reps)
            for b in range(reps):
                aa = a[rng.integers(0, len(a), len(a))]
                rr = r[rng.integers(0, len(r), len(r))]
                diff_draw[b] = huber_mean_1d(aa) - huber_mean_1d(rr)
            comparisons.append(
                {
                    "domain": domain,
                    "n_a1": len(a),
                    "n_remainder": len(r),
                    "huber_mean_diff": huber_mean_1d(a) - huber_mean_1d(r),
                    "median_diff": np.median(a) - np.median(r),
                    "wasserstein": wasserstein_distance(a, r),
                    "ks_statistic": ks_2samp(a, r).statistic,
                    "diff_ci_low": np.quantile(diff_draw, 0.025),
                    "diff_ci_high": np.quantile(diff_draw, 0.975),
                }
            )
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(output_dir / "sample_remainder_comparison.csv", index=False)

    proxy = text_proxies_a1.merge(scored[["id", "domain", "Q"]], on=["id", "domain"], how="inner")
    proxy_rows = []
    proxy_fields = [c for c in text_proxies_a1.columns if c not in {"id", "domain"}]
    for domain, group in proxy.groupby("domain"):
        low_q, high_q = group["Q"].quantile([0.1, 0.9])
        low_group = group[group["Q"] <= low_q]
        high_group = group[group["Q"] >= high_q]
        for field in proxy_fields:
            proxy_rows.append(
                {
                    "domain": domain,
                    "proxy": field,
                    "high_minus_low": high_group[field].mean() - low_group[field].mean(),
                    "n_high": len(high_group),
                    "n_low": len(low_group),
                }
            )
    pd.DataFrame(proxy_rows).to_csv(output_dir / "text_proxy_validation.csv", index=False)

    summary = {
        "sample_rows": len(scored),
        "quality_range": [float(scored["Q"].min()), float(scored["Q"].max())],
        "conflict_rate": float(scored["conflict"].mean()),
        "huber_convergence_rate": float(scored.loc[scored["conflict"], "huber_converged"].mean())
        if scored["conflict"].any()
        else 1.0,
        "missing_block_rate": float((scored["missing_block_count"] > 0).mean()),
        "sensitivity_min_domain_kendall": float(sensitivity_df["domain_kendall"].min()),
        "sensitivity_min_median_extreme_overlap": float(sensitivity_df["median_extreme_overlap"].min()),
    }
    json_dump(output_dir / "quality_summary.json", summary)
    return {
        "scored": scored,
        "domain": domain_df,
        "weights": weight_df,
        "bootstrap": bootstrap_draws,
        "comparison": comparison_df,
        "summary": summary,
    }


def run_quality_pipeline(input_root: Path, cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    load_cache = output_dir / "_cache_quality_load.pkl"
    transform_cache = output_dir / "_cache_quality_transformed.pkl"
    if load_cache.exists():
        cached = pd.read_pickle(load_cache)
        loaded = QualityLoadResult(**cached)
    else:
        loaded = load_quality_data(input_root, cfg, output_dir)
        pd.to_pickle(
            {
                "raw": loaded.raw,
                "text_features_a1": loaded.text_features_a1,
                "text_proxies_a1": loaded.text_proxies_a1,
                "audit": loaded.audit,
            },
            load_cache,
        )
    if transform_cache.exists():
        cached_transform = pd.read_pickle(transform_cache)
        transformed = cached_transform["transformed"]
        transform_spec = cached_transform["transform_spec"]
    else:
        transformed, transform_spec = transform_quality(loaded.raw, cfg, output_dir)
        pd.to_pickle(
            {"transformed": transformed, "transform_spec": transform_spec},
            transform_cache,
        )
    score_cache = output_dir / f"_cache_quality_score_v3_b{int(cfg['bootstrap_reps'])}_s{int(cfg['seed'])}.pkl"
    if score_cache.exists():
        scored = pd.read_pickle(score_cache)
    else:
        scored = score_quality(transformed, cfg, output_dir, loaded.text_proxies_a1)
        pd.to_pickle(scored, score_cache)
    return {
        **scored,
        "audit": loaded.audit,
        "text_features_a1": loaded.text_features_a1,
        "transform_spec": transform_spec,
    }
