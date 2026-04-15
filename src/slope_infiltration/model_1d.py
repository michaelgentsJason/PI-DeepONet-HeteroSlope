from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random, vmap
from jax.example_libraries import optimizers

from .jax_nets import architecture, modified_mlp
from .physics import VanGenuchtenParameters, hydraulic_conductivity, relative_conductivity_from_head, theta_from_head


@dataclass(frozen=True)
class Domain1D:
    t_min: float
    t_max: float
    z_top: float
    z_bottom: float


@dataclass(frozen=True)
class LossWeights1D:
    initial: float = 5.0
    top_flux: float = 20.0
    bottom: float = 5.0
    residual: float = 1.0


class PI_DeepONet1D:
    def __init__(
        self,
        rainfall_input_dim: int,
        ks_input_dim: int,
        hidden_dim: int,
        hydrology: VanGenuchtenParameters,
        domain: Domain1D,
        loss_weights: LossWeights1D,
        ks_log_fn,
        dks_log_dz_fn,
        learning_rate: float = 1e-3,
    ) -> None:
        self.hydrology = hydrology
        self.domain = domain
        self.loss_weights = loss_weights
        self.ks_log_fn = ks_log_fn
        self.dks_log_dz_fn = dks_log_dz_fn
        self.head_scale = 50.0

        rain_layers = architecture(rainfall_input_dim, depth=3, width=64, output_size=hidden_dim)
        ks_layers = architecture(ks_input_dim, depth=3, width=64, output_size=hidden_dim)
        trunk_layers = architecture(2, depth=3, width=64, output_size=hidden_dim)

        self.rain_init, self.rain_apply = modified_mlp(rain_layers, activation=jnp.tanh)
        self.ks_init, self.ks_apply = modified_mlp(ks_layers, activation=jnp.tanh)
        self.trunk_init, self.trunk_apply = modified_mlp(trunk_layers, activation=jnp.tanh)

        params = (
            self.rain_init(random.PRNGKey(11)),
            self.ks_init(random.PRNGKey(22)),
            self.trunk_init(random.PRNGKey(33)),
            jnp.array(0.0),
        )

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_state = self.opt_init(params)

    def normalize_t(self, t: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * (t - self.domain.t_min) / (self.domain.t_max - self.domain.t_min) - 1.0

    def normalize_z(self, z: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * (z - self.domain.z_bottom) / (self.domain.z_top - self.domain.z_bottom) - 1.0

    def operator_net(self, params, rainfall_u, ks_u, t, z):
        rain_params, ks_params, trunk_params, bias = params
        branch_rain = self.rain_apply(rain_params, rainfall_u)
        branch_ks = self.ks_apply(ks_params, ks_u)
        trunk_in = jnp.stack([self.normalize_t(t), self.normalize_z(z)])
        trunk = self.trunk_apply(trunk_params, trunk_in)
        latent = branch_rain * branch_ks * trunk
        return -jnp.exp(jnp.sum(latent) + bias)

    def head_gradients(self, params, rainfall_u, ks_u, t, z):
        h_t = grad(self.operator_net, argnums=3)(params, rainfall_u, ks_u, t, z)
        h_z = grad(self.operator_net, argnums=4)(params, rainfall_u, ks_u, t, z)
        h_zz = grad(grad(self.operator_net, argnums=4), argnums=4)(params, rainfall_u, ks_u, t, z)
        return h_t, h_z, h_zz

    def hydraulic_terms(self, h, z, ks_u):
        log_ks = self.ks_log_fn(z, ks_u)
        ks_value = jnp.exp(log_ks)
        dlogks_dz = self.dks_log_dz_fn(z, ks_u)
        rel_k = relative_conductivity_from_head(h, self.hydrology)
        conductivity = hydraulic_conductivity(h, ks_value, self.hydrology)
        dconductivity_dh = grad(lambda hh: hydraulic_conductivity(hh, ks_value, self.hydrology))(h)
        dconductivity_dz = conductivity * dlogks_dz
        return conductivity, dconductivity_dh, dconductivity_dz, rel_k

    def residual_point(self, params, rainfall_u, ks_u, t, z):
        h = self.operator_net(params, rainfall_u, ks_u, t, z)
        capacity = grad(lambda hh: theta_from_head(hh, self.hydrology))(h)
        h_t, h_z, h_zz = self.head_gradients(params, rainfall_u, ks_u, t, z)
        conductivity, dconductivity_dh, dconductivity_dz, _ = self.hydraulic_terms(h, z, ks_u)
        return capacity * h_t - (
            dconductivity_dz * (h_z + 1.0)
            + dconductivity_dh * h_z * (h_z + 1.0)
            + conductivity * h_zz
        )

    def top_flux_point(self, params, rainfall_u, ks_u, t, z):
        h = self.operator_net(params, rainfall_u, ks_u, t, z)
        _, h_z, _ = self.head_gradients(params, rainfall_u, ks_u, t, z)
        conductivity, _, _, _ = self.hydraulic_terms(h, z, ks_u)
        return -conductivity * (h_z + 1.0)

    def bottom_grad_point(self, params, rainfall_u, ks_u, t, z):
        _, h_z, _ = self.head_gradients(params, rainfall_u, ks_u, t, z)
        return h_z

    def loss_initial(self, params, rainfall_u, ks_u, points, initial_head):
        pred = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, rainfall_u, ks_u, points[:, 0], points[:, 1])
        return jnp.mean(((pred - initial_head) / self.head_scale) ** 2)

    def loss_top_flux(self, params, rainfall_u, ks_u, points, target_flux):
        pred = vmap(self.top_flux_point, (None, 0, 0, 0, 0))(params, rainfall_u, ks_u, points[:, 0], points[:, 1])
        return jnp.mean((pred - target_flux) ** 2)

    def loss_bottom(self, params, rainfall_u, ks_u, points):
        grad_bottom = vmap(self.bottom_grad_point, (None, 0, 0, 0, 0))(params, rainfall_u, ks_u, points[:, 0], points[:, 1])
        return jnp.mean(grad_bottom**2)

    def loss_residual(self, params, rainfall_u, ks_u, points):
        res = vmap(self.residual_point, (None, 0, 0, 0, 0))(params, rainfall_u, ks_u, points[:, 0], points[:, 1])
        return jnp.mean(res**2)

    def total_loss(self, params, batch):
        loss_ic = self.loss_initial(params, batch["rain_ic"], batch["ks_ic"], batch["ic_points"], batch["initial_head"])
        loss_top = self.loss_top_flux(params, batch["rain_top"], batch["ks_top"], batch["top_points"], batch["top_flux"])
        loss_bottom = self.loss_bottom(params, batch["rain_bottom"], batch["ks_bottom"], batch["bottom_points"])
        loss_res = self.loss_residual(params, batch["rain_res"], batch["ks_res"], batch["res_points"])
        total = (
            self.loss_weights.initial * loss_ic
            + self.loss_weights.top_flux * loss_top
            + self.loss_weights.bottom * loss_bottom
            + self.loss_weights.residual * loss_res
        )
        return total, (loss_ic, loss_top, loss_bottom, loss_res)

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        grads = grad(lambda p: self.total_loss(p, batch)[0])(params)
        return self.opt_update(i, grads, opt_state)

    @partial(jit, static_argnums=(0,))
    def predict_head(self, params, rainfall_u, ks_u, points):
        return vmap(self.operator_net, (None, 0, 0, 0, 0))(params, rainfall_u, ks_u, points[:, 0], points[:, 1])
