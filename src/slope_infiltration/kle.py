from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class ExponentialKLE1D:
    domain_length: float
    mean_log_ks: float
    variance_log_ks: float
    correlation_length: float
    n_modes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "omegas", self._solve_omegas())
        object.__setattr__(self, "eigenvalues", self._eigenvalues(self.omegas))
        object.__setattr__(self, "normalizers", self._normalizers(self.omegas))

    def _characteristic(self, omega: float) -> float:
        eta = self.correlation_length
        length = self.domain_length
        return (eta**2 * omega**2 - 1.0) * np.sin(omega * length) - 2.0 * eta * omega * np.cos(omega * length)

    def _solve_omegas(self) -> np.ndarray:
        roots: list[float] = []
        upper = max(2.0 * self.n_modes * np.pi / self.domain_length, 1.0)

        while len(roots) < self.n_modes:
            grid = np.linspace(1e-8, upper, 20000)
            values = self._characteristic(grid)
            sign_changes = np.where(np.sign(values[:-1]) * np.sign(values[1:]) < 0.0)[0]

            roots = []
            for idx in sign_changes:
                left = grid[idx]
                right = grid[idx + 1]
                root = brentq(self._characteristic, left, right)
                if not roots or abs(root - roots[-1]) > 1e-8:
                    roots.append(root)
                if len(roots) >= self.n_modes:
                    break

            upper *= 1.5

        return np.asarray(roots[: self.n_modes], dtype=float)

    def _eigenvalues(self, omegas: np.ndarray) -> np.ndarray:
        eta = self.correlation_length
        sigma2 = self.variance_log_ks
        return 2.0 * eta * sigma2 / (eta**2 * omegas**2 + 1.0)

    def _normalizers(self, omegas: np.ndarray) -> np.ndarray:
        eta = self.correlation_length
        length = self.domain_length
        return np.sqrt(((eta**2 * omegas**2 + 1.0) * length) / 2.0 + eta)

    def basis(self, depth: np.ndarray) -> np.ndarray:
        z = np.asarray(depth, dtype=float)[..., None]
        omega = self.omegas[None, :]
        values = eta_omega = self.correlation_length * omega
        return (eta_omega * np.cos(omega * z) + np.sin(omega * z)) / self.normalizers[None, :]

    def basis_derivative(self, depth: np.ndarray) -> np.ndarray:
        z = np.asarray(depth, dtype=float)[..., None]
        omega = self.omegas[None, :]
        numerator = -self.correlation_length * omega**2 * np.sin(omega * z) + omega * np.cos(omega * z)
        return numerator / self.normalizers[None, :]

    def log_ks(self, depth: np.ndarray, xi: np.ndarray) -> np.ndarray:
        xi_array = np.asarray(xi, dtype=float)
        weights = np.sqrt(self.eigenvalues) * xi_array
        return self.mean_log_ks + self.basis(depth) @ weights

    def dlog_ks_dd(self, depth: np.ndarray, xi: np.ndarray) -> np.ndarray:
        xi_array = np.asarray(xi, dtype=float)
        weights = np.sqrt(self.eigenvalues) * xi_array
        return self.basis_derivative(depth) @ weights
