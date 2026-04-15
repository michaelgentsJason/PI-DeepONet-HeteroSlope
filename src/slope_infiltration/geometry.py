from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np


@dataclass(frozen=True)
class BoundarySample:
    coordinates: np.ndarray
    normals: np.ndarray


@dataclass(frozen=True)
class SlopeGeometry:
    width: float = 1.0
    height: float = 1.0
    crest_width: float = 0.35
    toe_height: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.crest_width < self.width):
            raise ValueError("crest_width must be between 0 and width")
        if not (0.0 <= self.toe_height < self.height):
            raise ValueError("toe_height must be in [0, height)")

    @property
    def crest_length(self) -> float:
        return self.crest_width

    @property
    def slope_length(self) -> float:
        return hypot(self.width - self.crest_width, self.height - self.toe_height)

    @property
    def rainfall_boundary_length(self) -> float:
        return self.crest_length + self.slope_length

    def surface_elevation(self, x: np.ndarray | float) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        slope_fraction = np.clip(
            (x_array - self.crest_width) / (self.width - self.crest_width),
            0.0,
            1.0,
        )
        slope_z = self.height + slope_fraction * (self.toe_height - self.height)
        return np.where(x_array <= self.crest_width, self.height, slope_z)

    def contains(self, xz: np.ndarray) -> np.ndarray:
        points = np.asarray(xz, dtype=float)
        x = points[..., 0]
        z = points[..., 1]
        top = self.surface_elevation(x)
        return (
            (x >= 0.0)
            & (x <= self.width)
            & (z >= 0.0)
            & (z <= top)
        )

    def sample_interior_points(
        self,
        rng: np.random.Generator,
        n_points: int,
        *,
        max_tries_multiplier: int = 20,
    ) -> np.ndarray:
        accepted: list[np.ndarray] = []
        total = 0
        max_tries = max(n_points * max_tries_multiplier, 100)

        while total < n_points and max_tries > 0:
            candidates = np.column_stack(
                [
                    rng.uniform(0.0, self.width, size=n_points),
                    rng.uniform(0.0, self.height, size=n_points),
                ]
            )
            inside = self.contains(candidates)
            batch = candidates[inside]
            if batch.size:
                accepted.append(batch)
                total += batch.shape[0]
            max_tries -= 1

        if total < n_points:
            raise RuntimeError("failed to sample enough interior points for slope geometry")

        return np.concatenate(accepted, axis=0)[:n_points]

    def sample_residual_points(
        self,
        rng: np.random.Generator,
        n_points: int,
        *,
        t_max: float,
    ) -> np.ndarray:
        xz = self.sample_interior_points(rng, n_points)
        t = rng.uniform(0.0, t_max, size=(n_points, 1))
        return np.hstack([t, xz])

    def sample_initial_points(
        self,
        rng: np.random.Generator,
        n_points: int,
    ) -> np.ndarray:
        xz = self.sample_interior_points(rng, n_points)
        t = np.zeros((n_points, 1), dtype=float)
        return np.hstack([t, xz])

    def sample_rainfall_boundary_points(
        self,
        rng: np.random.Generator,
        n_points: int,
        *,
        t_max: float,
    ) -> BoundarySample:
        arc = rng.uniform(0.0, self.rainfall_boundary_length, size=n_points)
        t = rng.uniform(0.0, t_max, size=(n_points, 1))
        coords = np.zeros((n_points, 3), dtype=float)
        normals = np.zeros((n_points, 2), dtype=float)

        on_crest = arc <= self.crest_length
        crest_x = arc[on_crest]
        coords[on_crest, 0] = t[on_crest, 0]
        coords[on_crest, 1] = crest_x
        coords[on_crest, 2] = self.height
        normals[on_crest] = np.array([0.0, 1.0], dtype=float)

        on_slope = ~on_crest
        if np.any(on_slope):
            local_arc = arc[on_slope] - self.crest_length
            ratio = local_arc / self.slope_length
            dx = self.width - self.crest_width
            dz = self.toe_height - self.height
            slope_x = self.crest_width + ratio * dx
            slope_z = self.height + ratio * dz
            normal = np.array([-dz, dx], dtype=float)
            normal /= np.linalg.norm(normal)

            coords[on_slope, 0] = t[on_slope, 0]
            coords[on_slope, 1] = slope_x
            coords[on_slope, 2] = slope_z
            normals[on_slope] = normal

        return BoundarySample(coordinates=coords, normals=normals)

    def sample_no_flow_boundary_points(
        self,
        rng: np.random.Generator,
        n_points: int,
        *,
        t_max: float,
        include_right_face: bool = True,
    ) -> BoundarySample:
        segment_lengths = [
            ("left", self.height),
            ("bottom", self.width),
        ]
        if include_right_face and self.toe_height > 0.0:
            segment_lengths.append(("right", self.toe_height))

        names = [name for name, _ in segment_lengths]
        probs = np.array([length for _, length in segment_lengths], dtype=float)
        probs /= probs.sum()
        picks = rng.choice(names, size=n_points, p=probs)

        t = rng.uniform(0.0, t_max, size=(n_points, 1))
        coords = np.zeros((n_points, 3), dtype=float)
        normals = np.zeros((n_points, 2), dtype=float)

        left = picks == "left"
        coords[left, 0] = t[left, 0]
        coords[left, 1] = 0.0
        coords[left, 2] = rng.uniform(0.0, self.height, size=np.count_nonzero(left))
        normals[left] = np.array([-1.0, 0.0], dtype=float)

        bottom = picks == "bottom"
        coords[bottom, 0] = t[bottom, 0]
        coords[bottom, 1] = rng.uniform(0.0, self.width, size=np.count_nonzero(bottom))
        coords[bottom, 2] = 0.0
        normals[bottom] = np.array([0.0, -1.0], dtype=float)

        right = picks == "right"
        if np.any(right):
            coords[right, 0] = t[right, 0]
            coords[right, 1] = self.width
            coords[right, 2] = rng.uniform(0.0, self.toe_height, size=np.count_nonzero(right))
            normals[right] = np.array([1.0, 0.0], dtype=float)

        return BoundarySample(coordinates=coords, normals=normals)
