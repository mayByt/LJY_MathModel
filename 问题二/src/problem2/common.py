from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent
    cfg["input_root"] = str((project_root / cfg["input_root"]).resolve())
    cfg["problem1_root"] = str((project_root / cfg["problem1_root"]).resolve())
    return cfg, project_root


def apply_fast_mode(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out["bootstrap_reps"] = int(cfg.get("fast_bootstrap_reps", 30))
    out["classic_multistarts"] = int(cfg.get("fast_multistarts", 12))
    out["quality_multistarts"] = int(cfg.get("fast_quality_multistarts", 8))
    return out


def ensure_dirs(project_root: Path, cfg: dict[str, Any]) -> tuple[Path, Path]:
    out = (project_root / cfg["output_root"]).resolve()
    rep = (project_root / cfg["report_root"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    return out, rep


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def json_default(obj: Any) -> Any:
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
    raise TypeError(type(obj).__name__)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    r = np.asarray(y, float) - np.asarray(pred, float)
    return float(np.sqrt(np.mean(r * r)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(pred, float))))


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    from scipy.stats import spearmanr

    stat = spearmanr(y, pred, nan_policy="omit").statistic
    return float(stat) if np.isfinite(stat) else float("nan")


def weighted_huber_residual(resid: np.ndarray, weights: np.ndarray, delta: float) -> np.ndarray:
    r = np.asarray(resid, float)
    w = np.sqrt(np.asarray(weights, float))
    scale = max(delta, 1e-12)
    abs_r = np.abs(r)
    tail = np.sqrt(np.maximum(2.0 * scale * abs_r - scale * scale, 0.0))
    transformed = np.where(abs_r <= scale, r, np.sign(r) * tail)
    return w * transformed


def equal_cluster_weights(group: pd.Series) -> np.ndarray:
    counts = group.map(group.value_counts()).to_numpy(float)
    w = 1.0 / counts
    return w / w.sum()


def freeze_repair_baseline(output_dir: Path) -> Path:
    """Freeze the pre-repair result once, before a repaired run overwrites outputs."""
    target = output_dir / "repair_baseline.json"
    if target.exists():
        return target
    names = [
        "classic_scaling_parameters.json",
        "quality_model_parameters.json",
        "problem1_bridge.json",
        "generalized_scaling_parameters.json",
        "problem2_acceptance_report.json",
        "problem2_manifest.json",
    ]
    payload: dict[str, Any] = {
        "state": "PRE_REPAIR_BASELINE",
        "created_at_unix": time.time(),
        "files": {},
        "summaries": {},
    }
    for name in names:
        path = output_dir / name
        if not path.exists():
            continue
        payload["files"][name] = sha256_file(path)
        if path.suffix == ".json":
            try:
                payload["summaries"][name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload["summaries"][name] = {"read_error": True}
    json_dump(target, payload)
    return target


def device_record(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch  # type: ignore

        cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - depends on local optional dependency.
        return {
            "preferred_gpu": bool(cfg.get("gpu_preferred_if_faster", True)),
            "device_used": "cpu",
            "gpu_available": False,
            "reason": f"torch unavailable or unusable: {exc.__class__.__name__}",
        }
    return {
        "preferred_gpu": bool(cfg.get("gpu_preferred_if_faster", True)),
        "device_used": "cpu",
        "gpu_available": cuda,
        "reason": "scipy nonlinear least-squares path is CPU; GPU not faster/applicable for this workload",
    }


def manifest_payload(cfg: dict[str, Any], outputs: dict[str, str], acceptance: dict[str, Any], elapsed: float) -> dict[str, Any]:
    return {
        "project": "华为杯F题问题二",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config": cfg,
        "config_hash": stable_hash(cfg),
        "outputs": outputs,
        "acceptance": acceptance.get("summary", {}),
        "overall": acceptance.get("overall", "UNKNOWN"),
        "elapsed_seconds": elapsed,
        "created_at_unix": time.time(),
    }
