from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class VanGenuchtenParameters:
    alpha: float = 3.6
    n: float = 1.56
    theta_r: float = 0.078
    theta_s: float = 0.43

    @property
    def m(self) -> float:
        return 1.0 - 1.0 / self.n


def theta_from_head(head: jnp.ndarray, params: VanGenuchtenParameters) -> jnp.ndarray:
    term = (1.0 + jnp.abs(params.alpha * head) ** params.n) ** (-params.m)
    theta = params.theta_r + (params.theta_s - params.theta_r) * term
    return jnp.where(head > 0.0, params.theta_s, theta)


def effective_saturation(
    head: jnp.ndarray,
    params: VanGenuchtenParameters,
) -> jnp.ndarray:
    theta = theta_from_head(head, params)
    return (theta - params.theta_r) / (params.theta_s - params.theta_r)


def relative_conductivity_from_head(
    head: jnp.ndarray,
    params: VanGenuchtenParameters,
) -> jnp.ndarray:
    saturation = effective_saturation(head, params)
    inner = 1.0 - (1.0 - saturation ** (1.0 / params.m)) ** params.m
    kr = jnp.sqrt(saturation) * inner**2
    return jnp.where(head > 0.0, 1.0, kr)


def hydraulic_conductivity(
    head: jnp.ndarray,
    ks_value: jnp.ndarray,
    params: VanGenuchtenParameters,
) -> jnp.ndarray:
    return ks_value * relative_conductivity_from_head(head, params)


def darcy_flux_normal(
    head: jnp.ndarray,
    grad_x: jnp.ndarray,
    grad_z: jnp.ndarray,
    ks_value: jnp.ndarray,
    normal_x: jnp.ndarray,
    normal_z: jnp.ndarray,
    params: VanGenuchtenParameters,
) -> jnp.ndarray:
    conductivity = hydraulic_conductivity(head, ks_value, params)
    flux_x = -conductivity * grad_x
    flux_z = -conductivity * (grad_z + 1.0)
    return flux_x * normal_x + flux_z * normal_z
