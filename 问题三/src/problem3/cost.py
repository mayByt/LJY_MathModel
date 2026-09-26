from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostSpec:
    name: str
    amplitude: float
    shape: float


def quality_cost_value(q: np.ndarray | float, spec: CostSpec) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if spec.name == "exponential":
        return spec.amplitude * np.exp(spec.shape * q)
    if spec.name == "power":
        return spec.amplitude * np.power(q, spec.shape)
    if spec.name == "logarithmic":
        return spec.amplitude * np.log1p(spec.shape * q)
    raise ValueError(f"unknown quality cost type: {spec.name}")


def quality_cost_derivative(q: np.ndarray | float, spec: CostSpec) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if spec.name == "exponential":
        return spec.amplitude * spec.shape * np.exp(spec.shape * q)
    if spec.name == "power":
        return spec.amplitude * spec.shape * np.power(q, spec.shape - 1.0)
    if spec.name == "logarithmic":
        return spec.amplitude * spec.shape / (1.0 + spec.shape * q)
    raise ValueError(f"unknown quality cost type: {spec.name}")


def quality_cost_increment(q: np.ndarray | float, q0: float, spec: CostSpec) -> np.ndarray:
    delta = quality_cost_value(q, spec) - quality_cost_value(q0, spec)
    return np.maximum(delta, 0.0)


def total_costs(n_b: float, d_b: float, q: float, q0: float, context: int, eta: float,
                spec: CostSpec) -> dict[str, float]:
    n_abs, d_abs = n_b * 1e9, d_b * 1e9
    train = 6.0 * n_abs * d_abs
    quality = d_abs * float(quality_cost_increment(q, q0, spec))
    attention = eta * n_abs * d_abs * float(context)
    total = train + quality + attention
    return {"train": train, "quality": quality, "attention": attention, "total": total}


def cost_spec_from_config(cfg: dict, name: str, amplitude_scale: float = 1.0,
                          shape_scale: float = 1.0) -> CostSpec:
    raw = cfg["cost_parameters"][name]
    return CostSpec(name=name, amplitude=float(raw["amplitude"]) * amplitude_scale,
                    shape=float(raw["shape"]) * shape_scale)
