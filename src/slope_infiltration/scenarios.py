from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import SlopeGeometry
from .heterogeneity import PiecewiseKsField


@dataclass(frozen=True)
class RainfallProfile:
    breakpoints: tuple[float, ...]
    fluxes: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.breakpoints) < 2:
            raise ValueError("at least two breakpoints are required")
        if len(self.fluxes) != len(self.breakpoints) - 1:
            raise ValueError("fluxes must have one fewer element than breakpoints")
        if any(b0 >= b1 for b0, b1 in zip(self.breakpoints[:-1], self.breakpoints[1:])):
            raise ValueError("breakpoints must be strictly increasing")

    @property
    def t_max(self) -> float:
        return self.breakpoints[-1]

    def flux(self, times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        values = np.zeros_like(t, dtype=float)

        for idx, flux_value in enumerate(self.fluxes):
            left = self.breakpoints[idx]
            right = self.breakpoints[idx + 1]
            is_last = idx == len(self.fluxes) - 1
            mask = (t >= left) & ((t <= right) if is_last else (t < right))
            values = np.where(mask, flux_value, values)

        return values

    def sensor_values(self, sensor_times: np.ndarray) -> np.ndarray:
        return self.flux(np.asarray(sensor_times, dtype=float))

    @classmethod
    def random_piecewise(
        cls,
        rng: np.random.Generator,
        *,
        t_max: float,
        n_segments: int,
        min_flux: float,
        max_flux: float,
    ) -> "RainfallProfile":
        if n_segments < 1:
            raise ValueError("n_segments must be at least 1")

        internal = np.sort(rng.uniform(0.0, t_max, size=n_segments - 1))
        breakpoints = np.concatenate([[0.0], internal, [t_max]])
        fluxes = rng.uniform(min_flux, max_flux, size=n_segments)
        return cls(
            breakpoints=tuple(float(v) for v in breakpoints),
            fluxes=tuple(float(v) for v in fluxes),
        )


@dataclass(frozen=True)
class HeterogeneousSlopeScenario:
    geometry: SlopeGeometry
    field: PiecewiseKsField
    rainfall: RainfallProfile
    initial_head: float

    def branch_features(self, sensor_times: np.ndarray) -> np.ndarray:
        rainfall_features = self.rainfall.sensor_values(sensor_times)
        geometry_features = np.asarray(
            [
                self.geometry.width,
                self.geometry.height,
                self.geometry.crest_width,
                self.geometry.toe_height,
                self.initial_head,
            ],
            dtype=float,
        )
        return np.concatenate(
            [
                np.asarray(rainfall_features, dtype=float).ravel(),
                self.field.feature_vector(),
                geometry_features,
            ]
        )
