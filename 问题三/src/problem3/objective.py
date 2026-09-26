from __future__ import annotations

import numpy as np

from .contracts import ScalingParameters


def quality_upper_bound(q0: float, slope: float, eps: float, cap: float = 1.0) -> float:
    saturation = q0 + (1.0 - eps - q0) / slope
    return float(min(cap, 1.0, saturation))


def effective_quality(q_a: np.ndarray | float, q0: float, slope: float) -> np.ndarray:
    return q0 + slope * (np.asarray(q_a, dtype=float) - q0)


def predict_loss(n_b: np.ndarray | float, d_b: np.ndarray | float, q_a: np.ndarray | float,
                 params: ScalingParameters, slope: float = 1.0) -> np.ndarray:
    n_b = np.asarray(n_b, dtype=float)
    d_b = np.asarray(d_b, dtype=float)
    q_eff = effective_quality(q_a, params.Q0, slope)
    if np.any(n_b <= 0) or np.any(d_b <= 0) or np.any(q_eff < 0) or np.any(q_eff >= 1.0):
        return np.full(np.broadcast(n_b, d_b, q_eff).shape, np.inf)
    return (
        params.E
        + params.A * np.power(n_b, -params.alpha)
        + params.B * np.power(d_b, -params.beta)
        + params.c_Q * np.power(1.0 - q_eff, params.nu_Q)
    )


def loss_derivatives_absolute(n_b: float, d_b: float, q_a: float, params: ScalingParameters,
                              slope: float) -> dict[str, float]:
    q_eff = float(effective_quality(q_a, params.Q0, slope))
    d_n = -params.alpha * params.A * n_b ** (-params.alpha - 1.0) / 1e9
    d_d = -params.beta * params.B * d_b ** (-params.beta - 1.0) / 1e9
    d_q = -params.c_Q * params.nu_Q * (1.0 - q_eff) ** (params.nu_Q - 1.0) * slope
    return {"dL_dN_abs": float(d_n), "dL_dD_abs": float(d_d), "dL_dQ": float(d_q)}
