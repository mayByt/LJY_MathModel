from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScalingParameters:
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    c_Q: float
    nu_Q: float
    Q0: float


@dataclass(frozen=True)
class SupportBounds:
    name: str
    n_min: float
    n_max: float
    d_min: float
    d_max: float


def params_from_payload(generalized: dict[str, Any], q0: float | None = None) -> ScalingParameters:
    classic = generalized["classic_parameters"]
    quality = generalized["quality_parameters"]
    return ScalingParameters(
        E=float(classic["E"]),
        A=float(classic["A"]),
        B=float(classic["B"]),
        alpha=float(classic["alpha"]),
        beta=float(classic["beta"]),
        c_Q=float(quality["c_Q"]),
        nu_Q=float(quality["nu_Q"]),
        Q0=float(generalized["Q_ref"] if q0 is None else q0),
    )


def params_from_bootstrap(row: Any) -> ScalingParameters:
    return ScalingParameters(
        E=float(row["classic_E"]),
        A=float(row["classic_A"]),
        B=float(row["classic_B"]),
        alpha=float(row["classic_alpha"]),
        beta=float(row["classic_beta"]),
        c_Q=float(row["quality_c_Q"]),
        nu_Q=float(row["quality_nu_Q"]),
        Q0=float(row["Q_ref_draw"]),
    )


def support_from_config(cfg: dict[str, Any], name: str) -> SupportBounds:
    raw = cfg["support_modes"][name]
    return SupportBounds(
        name=name,
        n_min=float(raw["N_min_B"]),
        n_max=float(raw["N_max_B"]),
        d_min=float(raw["D_min_B"]),
        d_max=float(raw["D_max_B"]),
    )
