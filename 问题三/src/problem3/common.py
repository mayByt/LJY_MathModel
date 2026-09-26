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
    for key in ["input_root", "problem2_root", "problem2_code_root"]:
        cfg[key] = str((project_root / cfg[key]).resolve())
    return cfg, project_root


def ensure_dirs(project_root: Path, cfg: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = (project_root / cfg["output_root"]).resolve()
    report_dir = (project_root / cfg["report_root"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, report_dir


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(type(obj).__name__)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def device_record(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch  # type: ignore

        available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover
        return {
            "preferred_gpu": bool(cfg.get("gpu_preferred_if_faster", True)),
            "gpu_available": False,
            "device_used": "cpu",
            "reason": f"torch unavailable or unusable: {exc.__class__.__name__}",
        }
    return {
        "preferred_gpu": bool(cfg.get("gpu_preferred_if_faster", True)),
        "gpu_available": available,
        "device_used": "cpu",
        "reason": "two-dimensional scipy constrained optimization is CPU-bound; GPU path is not faster/applicable",
    }


def manifest_base(cfg: dict[str, Any], elapsed: float) -> dict[str, Any]:
    return {
        "project": "华为杯F题问题三",
        "version": "problem3_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config": cfg,
        "config_hash": stable_hash(cfg),
        "elapsed_seconds": elapsed,
        "created_at_unix": time.time(),
        "skip_plots": bool(cfg.get("skip_plots", True)),
    }
