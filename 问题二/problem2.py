#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from problem2.acceptance import build_acceptance_report
from problem2.classic import run_classic
from problem2.common import apply_fast_mode, device_record, ensure_dirs, freeze_repair_baseline, json_dump, load_config, manifest_payload, seed_everything, sha256_file
from problem2.data import audit_data
from problem2.generalized import run_generalized
from problem2.problem1_bridge import run_bridge
from problem2.quality import run_quality
from problem2.reporting import write_results_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="华为杯 F 题问题二：广义标度律建模与验收")
    parser.add_argument("command", nargs="?", default="all", choices=["audit", "classic", "quality", "bridge", "generalized", "validate", "report", "all"])
    parser.add_argument("--config", default="config/problem2.json")
    parser.add_argument("--fast", action="store_true", help="使用较少初值和 bootstrap 做开发冒烟测试")
    return parser.parse_args()


def _output_hashes(output_dir: Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "problem2_manifest.json"}


def main() -> int:
    args = parse_args()
    cfg_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg, project_root = load_config(cfg_path)
    if args.fast:
        cfg = apply_fast_mode(cfg)
    seed_everything(int(cfg["seed"]))
    output_dir, report_dir = ensure_dirs(project_root, cfg)
    freeze_repair_baseline(output_dir)
    started = time.time()
    run_log = {"command": args.command, "fast": args.fast, "device": device_record(cfg), "stages": [], "started": started}

    print("[1/7] 数据审计与异常隔离", flush=True)
    data = audit_data(cfg, output_dir)
    run_log["stages"].append("audit")
    if data["audit"]["hard_gate_status"] != "PASS":
        json_dump(output_dir / "run_log.json", {**run_log, "status": "STOPPED_DATA_HARD_GATE", "hard_failures": data["audit"]["hard_failures"], "elapsed_seconds": time.time() - started})
        return 1
    if args.command == "audit":
        json_dump(output_dir / "run_log.json", {**run_log, "elapsed_seconds": time.time() - started})
        return 0

    print("[2/7] 拟合经典标度律与留一规模验证", flush=True)
    classic = run_classic(data["tables"], cfg, output_dir)
    run_log["stages"].append("classic")
    if args.command == "classic":
        json_dump(output_dir / "run_log.json", {**run_log, "elapsed_seconds": time.time() - started})
        return 0

    print("[3/7] 拟合 B7 四类质量模型与成块验证", flush=True)
    quality = run_quality(data["tables"], classic, cfg, output_dir)
    run_log["stages"].append("quality")
    if args.command == "quality":
        json_dump(output_dir / "run_log.json", {**run_log, "elapsed_seconds": time.time() - started})
        return 0

    print("[4/7] 桥接问题一质量、配比响应和 WARN", flush=True)
    bridge = run_bridge(cfg, output_dir)
    run_log["stages"].append("bridge")
    if args.command == "bridge":
        json_dump(output_dir / "run_log.json", {**run_log, "elapsed_seconds": time.time() - started})
        return 0

    print("[5/7] 生成广义标度律接口、弹性和等价量", flush=True)
    generalized = run_generalized(classic, quality, bridge, data, cfg, output_dir)
    run_log["stages"].append("generalized")
    if args.command == "generalized":
        json_dump(output_dir / "run_log.json", {**run_log, "elapsed_seconds": time.time() - started})
        return 0

    print("[6/7] 执行验收规则", flush=True)
    acceptance = build_acceptance_report(data, classic, quality, bridge, generalized, output_dir)
    run_log["stages"].append("validate")
    if args.command == "validate":
        json_dump(output_dir / "run_log.json", {**run_log, "elapsed_seconds": time.time() - started})
        return 1 if acceptance["overall"] == "FAIL" else 0

    print("[7/7] 写入结果报告和 manifest", flush=True)
    final_elapsed = time.time() - started
    json_dump(output_dir / "run_log.json", {**run_log, "status": "COMPLETE", "interface_state": acceptance["interface_state"], "elapsed_seconds": final_elapsed})
    write_results_report(report_dir / "RESULTS_REPORT.md", cfg, data, classic, quality, bridge, generalized, acceptance)
    outputs = _output_hashes(output_dir)
    manifest = manifest_payload(cfg, outputs, acceptance, time.time() - started)
    manifest["skip_plots"] = bool(cfg.get("skip_plots", True))
    manifest["code_fingerprints"] = {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in sorted((PROJECT_ROOT / "src" / "problem2").glob("*.py"))}
    manifest["code_fingerprints"]["problem2.py"] = sha256_file(PROJECT_ROOT / "problem2.py")
    manifest["input_fingerprints"] = {k: v["sha256"] for k, v in data["audit"]["files"].items()}
    manifest["problem1_consumed_fingerprints"] = bridge["bridge"]["input_hashes"]
    manifest["report_hash"] = sha256_file(report_dir / "RESULTS_REPORT.md")
    manifest["hash_verification"] = "PASS" if all(sha256_file(output_dir / name) == digest for name, digest in outputs.items()) else "FAIL"
    json_dump(output_dir / "problem2_manifest.json", manifest)
    print(json.dumps({"acceptance": acceptance["overall"], "summary": acceptance["summary"], "elapsed_seconds": time.time() - started}, ensure_ascii=False))
    return 1 if acceptance["summary"].get("FAIL", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
