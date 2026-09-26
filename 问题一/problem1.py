#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from problem1.acceptance import build_acceptance_report
from problem1.common import apply_fast_mode, ensure_dirs, json_dump, load_config, seed_everything, sha256_file, stable_hash
from problem1.data_quality import run_quality_pipeline
from problem1.mapping import run_mapping_pipeline
from problem1.mixture import rebuild_optimization_diagnostics, run_mixture_pipeline
from problem1.plots import generate_figures
from problem1.reporting import write_results_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="华为杯 F 题问题一：质量评分、领域映射与配比优化")
    parser.add_argument(
        "command",
        choices=["quality", "mapping", "mixture", "finalize", "validate", "all"],
        nargs="?",
        default="all",
    )
    parser.add_argument("--config", default="config/problem1.json")
    parser.add_argument("--fast", action="store_true", help="使用小 bootstrap 次数做开发冒烟测试")
    return parser.parse_args()


def load_existing_mixture(
    input_root: Path,
    cfg: dict,
    output_dir: Path,
    domain_mapping: pd.DataFrame,
) -> dict:
    """数值拟合已完成、仅作图或报告失败时，从结果真源恢复，不重新调参。"""
    required = [
        "data_audit_mixture.json",
        "mixture_internal_cv.csv",
        "mixture_metrics.csv",
        "mixture_predictions.csv.gz",
        "mixture_coefficients.csv.gz",
        "mixture_directional_effects.csv",
        "optimal_mixture.csv",
        "mixture_summary.json",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"无法 finalize，缺少结果文件: {missing}")
    audit = json.loads((output_dir / "data_audit_mixture.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "mixture_summary.json").read_text(encoding="utf-8"))
    effects = pd.read_csv(output_dir / "mixture_directional_effects.csv")
    q_lookup = domain_mapping.set_index("domain")["q"].to_dict()
    q = np.array([q_lookup.get(d, np.nan) for d in effects["domain"]], dtype=float)
    benefit = effects["benefit_score"].to_numpy(float)
    valid = np.isfinite(q) & np.isfinite(benefit)
    from scipy.stats import kendalltau, spearmanr

    association = {
        "spearman_q_vs_benefit": float(spearmanr(q[valid], benefit[valid]).statistic),
        "kendall_q_vs_benefit": float(kendalltau(q[valid], benefit[valid]).statistic),
        "n_domains": int(valid.sum()),
    }
    summary["quality_effect_association"] = association
    rebuild_optimization_diagnostics(input_root, cfg, output_dir, audit, summary)
    json_dump(output_dir / "quality_effect_association.json", association)
    json_dump(output_dir / "mixture_summary.json", summary)
    return {
        "audit": audit,
        "candidate_metrics": pd.read_csv(output_dir / "mixture_internal_cv.csv"),
        "metrics": pd.read_csv(output_dir / "mixture_metrics.csv"),
        "predictions": pd.read_csv(output_dir / "mixture_predictions.csv.gz"),
        "coefficients": pd.read_csv(output_dir / "mixture_coefficients.csv.gz"),
        "effects": effects,
        "optimal": pd.read_csv(output_dir / "optimal_mixture.csv"),
        "summary": summary,
    }


def main() -> int:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg, project_root = load_config(config_path)
    if args.fast:
        cfg = apply_fast_mode(cfg)
    seed_everything(int(cfg["seed"]))
    output_dir, figure_dir = ensure_dirs(project_root, cfg)
    report_dir = (project_root / cfg.get("report_root", "reports")).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run_log = {"command": args.command, "fast": args.fast, "started": started, "stages": []}

    input_root = Path(cfg["input_root"])
    print("[1/4] 读取质量数据并计算样本/领域质量", flush=True)
    quality = run_quality_pipeline(input_root, cfg, output_dir)
    run_log["stages"].append("quality")
    if args.command == "quality":
        json_dump(output_dir / "run_log.json", {**run_log, "elapsed_seconds": time.time() - started})
        return 0

    print("[2/4] 流式读取 A18 并完成 17 域质量映射", flush=True)
    mapping = run_mapping_pipeline(
        input_root,
        cfg,
        output_dir,
        quality["text_features_a1"],
        quality["domain"],
        quality["bootstrap"],
    )
    run_log["stages"].append("mapping")
    if args.command == "mapping":
        json_dump(output_dir / "run_log.json", {**run_log, "elapsed_seconds": time.time() - started})
        return 0

    if args.command == "finalize":
        print("[3/4] 读取已完成的配比结果，不重新调参", flush=True)
        mixture = load_existing_mixture(input_root, cfg, output_dir, mapping["mapping"])
    else:
        print("[3/4] 拟合 Scheffé 配比模型、跨规模验证并优化", flush=True)
        mixture = run_mixture_pipeline(input_root, cfg, output_dir, mapping["mapping"])
    run_log["stages"].append("mixture")
    if args.command == "mixture":
        json_dump(output_dir / "run_log.json", {**run_log, "elapsed_seconds": time.time() - started})
        return 0

    print("[4/4] 生成图表、验收报告和结果报告", flush=True)
    acceptance = build_acceptance_report(cfg, quality, mapping, mixture, output_dir)
    if bool(cfg.get("skip_plots", False)):
        figures = []
    else:
        figures = generate_figures(quality, mapping, mixture, figure_dir)
    report_path = report_dir / "RESULTS_REPORT.md"
    write_results_report(
        report_path,
        cfg,
        quality,
        mapping,
        mixture,
        acceptance,
        figures,
    )
    run_log["stages"].extend(["acceptance", "figures", "report"])
    final_elapsed = time.time() - started
    json_dump(output_dir / "run_log.json", {**run_log, "status": "COMPLETE", "elapsed_seconds": final_elapsed})
    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "problem1_manifest.json"
    }
    manifest = {
        "project": "华为杯F题问题一",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config": cfg,
        "config_hash": stable_hash(cfg),
        "fast_mode": args.fast,
        "elapsed_seconds": final_elapsed,
        "outputs": output_hashes,
        "figures": figures,
        "report_path": str(report_path),
        "report_hash": sha256_file(report_path),
        "acceptance": acceptance["summary"],
    }
    manifest["hash_verification"] = "PASS" if all(sha256_file(output_dir / name) == digest for name, digest in output_hashes.items()) else "FAIL"
    json_dump(output_dir / "problem1_manifest.json", manifest)
    print(json.dumps({"acceptance": acceptance["overall"], "elapsed_seconds": manifest["elapsed_seconds"], "hash_verification": manifest["hash_verification"]}, ensure_ascii=False))
    return 1 if acceptance["overall"] == "FAIL" or manifest["hash_verification"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
