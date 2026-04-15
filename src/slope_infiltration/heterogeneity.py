from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np


@dataclass(frozen=True)
class SoilZone:
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    saturated_conductivity: float

    def __post_init__(self) -> None:
        if not (self.x_min < self.x_max):
            raise ValueError("x_min must be smaller than x_max")
        if not (self.z_min < self.z_max):
            raise ValueError("z_min must be smaller than z_max")
        if self.saturated_conductivity <= 0.0:
            raise ValueError("saturated_conductivity must be positive")

    def contains(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (
            (x >= self.x_min)
            & (x <= self.x_max)
            & (z >= self.z_min)
            & (z <= self.z_max)
        )


@dataclass(frozen=True)
class PiecewiseKsField:
    background_ks: float
    zones: tuple[SoilZone, ...] = ()

    def __post_init__(self) -> None:
        if self.background_ks <= 0.0:
            raise ValueError("background_ks must be positive")

    def saturated_conductivity(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        z_array = np.asarray(z, dtype=float)
        ks = np.full_like(x_array, self.background_ks, dtype=float)

        for zone in self.zones:
            ks = np.where(zone.contains(x_array, z_array), zone.saturated_conductivity, ks)

        return ks

    def log_saturated_conductivity(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.log(self.saturated_conductivity(x, z))

    def feature_vector(self) -> np.ndarray:
        features = [log(self.background_ks), float(len(self.zones))]
        for zone in self.zones:
            features.extend(
                [
                    zone.x_min,
                    zone.x_max,
                    zone.z_min,
                    zone.z_max,
                    log(zone.saturated_conductivity),
                ]
            )
        return np.asarray(features, dtype=float)

    def interface_distances(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        z_array = np.asarray(z, dtype=float)
        distances = np.full_like(x_array, np.inf, dtype=float)

        for zone in self.zones:
            inside_z = (z_array >= zone.z_min) & (z_array <= zone.z_max)
            inside_x = (x_array >= zone.x_min) & (x_array <= zone.x_max)

            distances = np.minimum(
                distances,
                np.where(inside_z, np.abs(x_array - zone.x_min), np.inf),
            )
            distances = np.minimum(
                distances,
                np.where(inside_z, np.abs(x_array - zone.x_max), np.inf),
            )
            distances = np.minimum(
                distances,
                np.where(inside_x, np.abs(z_array - zone.z_min), np.inf),
            )
            distances = np.minimum(
                distances,
                np.where(inside_x, np.abs(z_array - zone.z_max), np.inf),
            )

        return distances

    @classmethod
    def two_layer(
        cls,
        *,
        upper_ks: float,
        lower_ks: float,
        split_z: float = 0.5,
    ) -> "PiecewiseKsField":
        return cls(
            background_ks=lower_ks,
            zones=(
                SoilZone(
                    x_min=0.0,
                    x_max=1.0,
                    z_min=split_z,
                    z_max=1.0,
                    saturated_conductivity=upper_ks,
                ),
            ),
        )
