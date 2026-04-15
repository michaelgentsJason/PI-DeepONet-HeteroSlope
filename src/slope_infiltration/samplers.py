from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import BoundarySample, SlopeGeometry
from .heterogeneity import PiecewiseKsField


@dataclass(frozen=True)
class SampleBatch:
    residual: np.ndarray
    initial: np.ndarray
    rainfall_boundary: BoundarySample
    no_flow_boundary: BoundarySample


def _resample_until_clear(
    geometry: SlopeGeometry,
    field: PiecewiseKsField,
    rng: np.random.Generator,
    n_points: int,
    *,
    t_max: float | None,
    tolerance: float,
) -> np.ndarray:
    accepted: list[np.ndarray] = []
    total = 0

    while total < n_points:
        needed = max(n_points - total, 16)
        if t_max is None:
            points = geometry.sample_initial_points(rng, needed)
        else:
            points = geometry.sample_residual_points(rng, needed, t_max=t_max)

        distances = field.interface_distances(points[:, 1], points[:, 2])
        clear = points[distances >= tolerance]
        if clear.size:
            accepted.append(clear)
            total += clear.shape[0]

        if not field.zones:
            break

    return np.concatenate(accepted, axis=0)[:n_points]


def sample_training_batch(
    geometry: SlopeGeometry,
    field: PiecewiseKsField,
    *,
    rng: np.random.Generator | None = None,
    t_max: float = 1.0,
    n_residual: int = 2048,
    n_initial: int = 256,
    n_rainfall: int = 256,
    n_no_flow: int = 256,
    interface_tolerance: float = 1e-3,
) -> SampleBatch:
    generator = rng or np.random.default_rng()
    residual = _resample_until_clear(
        geometry,
        field,
        generator,
        n_residual,
        t_max=t_max,
        tolerance=interface_tolerance,
    )
    initial = _resample_until_clear(
        geometry,
        field,
        generator,
        n_initial,
        t_max=None,
        tolerance=interface_tolerance,
    )
    rainfall_boundary = geometry.sample_rainfall_boundary_points(
        generator,
        n_rainfall,
        t_max=t_max,
    )
    no_flow_boundary = geometry.sample_no_flow_boundary_points(
        generator,
        n_no_flow,
        t_max=t_max,
    )

    return SampleBatch(
        residual=residual,
        initial=initial,
        rainfall_boundary=rainfall_boundary,
        no_flow_boundary=no_flow_boundary,
    )
