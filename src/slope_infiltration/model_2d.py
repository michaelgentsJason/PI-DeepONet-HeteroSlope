from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
from jax import grad, jit, random, vmap
from jax.example_libraries import optimizers

from .jax_nets import architecture, modified_mlp
from .physics import VanGenuchtenParameters, darcy_flux_normal, hydraulic_conductivity, theta_from_head


@dataclass(frozen=True)
class Domain2D:
    t_min: float
    t_max: float
    x_min: float
    x_max: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class LossWeights2D:
    initial: float = 5.0
    rainfall: float = 20.0
    no_flow: float = 5.0
    residual: float = 1.0


class PI_DeepONet2D:
    def __init__(
        self,
        rainfall_input_dim: int,
        hetero_input_dim: int,
        hidden_dim: int,
        hydrology: VanGenuchtenParameters,
        domain: Domain2D,
        loss_weights: LossWeights2D,
        ks_value_fn,
        learning_rate: float = 1e-3,
    ) -> None:
        self.hydrology = hydrology
        self.domain = domain
        self.loss_weights = loss_weights
        self.ks_value_fn = ks_value_fn
        self.head_scale = 50.0

        rain_layers = architecture(rainfall_input_dim, depth=3, width=64, output_size=hidden_dim)
        hetero_layers = architecture(hetero_input_dim, depth=3, width=64, output_size=hidden_dim)
        trunk_layers = architecture(3, depth=3, width=64, output_size=hidden_dim)

        self.rain_init, self.rain_apply = modified_mlp(rain_layers, activation=jnp.tanh)
        self.hetero_init, self.hetero_apply = modified_mlp(hetero_layers, activation=jnp.tanh)
        self.trunk_init, self.trunk_apply = modified_mlp(trunk_layers, activation=jnp.tanh)

        params = (
            self.rain_init(random.PRNGKey(111)),
            self.hetero_init(random.PRNGKey(222)),
            self.trunk_init(random.PRNGKey(333)),
            jnp.array(0.0),
        )

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_state = self.opt_init(params)

    def normalize_t(self, t: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * (t - self.domain.t_min) / (self.domain.t_max - self.domain.t_min) - 1.0

    def normalize_x(self, x: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * (x - self.domain.x_min) / (self.domain.x_max - self.domain.x_min) - 1.0

    def normalize_z(self, z: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * (z - self.domain.z_min) / (self.domain.z_max - self.domain.z_min) - 1.0

    def operator_net(self, params, rainfall_u, hetero_u, t, x, z):
        rain_params, hetero_params, trunk_params, bias = params
        branch_rain = self.rain_apply(rain_params, rainfall_u)
        branch_hetero = self.hetero_apply(hetero_params, hetero_u)
        trunk_in = jnp.stack([self.normalize_t(t), self.normalize_x(x), self.normalize_z(z)])
        trunk = self.trunk_apply(trunk_params, trunk_in)
        latent = branch_rain * branch_hetero * trunk
        return -jnp.exp(jnp.sum(latent) + bias)

    def gradients(self, params, rainfall_u, hetero_u, t, x, z):
        h_t = grad(self.operator_net, argnums=3)(params, rainfall_u, hetero_u, t, x, z)
        h_x = grad(self.operator_net, argnums=4)(params, rainfall_u, hetero_u, t, x, z)
        h_z = grad(self.operator_net, argnums=5)(params, rainfall_u, hetero_u, t, x, z)
        h_xx = grad(grad(self.operator_net, argnums=4), argnums=4)(params, rainfall_u, hetero_u, t, x, z)
        h_zz = grad(grad(self.operator_net, argnums=5), argnums=5)(params, rainfall_u, hetero_u, t, x, z)
        return h_t, h_x, h_z, h_xx, h_zz

    def residual_point(self, params, rainfall_u, hetero_u, t, x, z):
        h = self.operator_net(params, rainfall_u, hetero_u, t, x, z)
        capacity = grad(lambda hh: theta_from_head(hh, self.hydrology))(h)
        h_t, h_x, h_z, h_xx, h_zz = self.gradients(params, rainfall_u, hetero_u, t, x, z)
        ks_value = self.ks_value_fn(x, z)
        conductivity = hydraulic_conductivity(h, ks_value, self.hydrology)
        dconductivity_dh = grad(lambda hh: hydraulic_conductivity(hh, ks_value, self.hydrology))(h)
        return capacity * h_t - dconductivity_dh * (h_x**2 + h_z**2) - conductivity * (h_xx + h_zz) - dconductivity_dh * h_z

    def rainfall_flux_point(self, params, rainfall_u, hetero_u, t, x, z, nx, nz):
        h = self.operator_net(params, rainfall_u, hetero_u, t, x, z)
        _, h_x, h_z, _, _ = self.gradients(params, rainfall_u, hetero_u, t, x, z)
        ks_value = self.ks_value_fn(x, z)
        return darcy_flux_normal(h, h_x, h_z, ks_value, nx, nz, self.hydrology)

    def no_flow_flux_point(self, params, rainfall_u, hetero_u, t, x, z, nx, nz):
        return self.rainfall_flux_point(params, rainfall_u, hetero_u, t, x, z, nx, nz)

    def loss_initial(self, params, rainfall_u, hetero_u, points, initial_head):
        pred = vmap(self.operator_net, (None, 0, 0, 0, 0, 0))(params, rainfall_u, hetero_u, points[:, 0], points[:, 1], points[:, 2])
        return jnp.mean(((pred - initial_head) / self.head_scale) ** 2)

    def loss_rainfall(self, params, rainfall_u, hetero_u, points, normals, target_flux):
        pred = vmap(self.rainfall_flux_point, (None, 0, 0, 0, 0, 0, 0, 0))(
            params,
            rainfall_u,
            hetero_u,
            points[:, 0],
            points[:, 1],
            points[:, 2],
            normals[:, 0],
            normals[:, 1],
        )
        return jnp.mean((pred - target_flux) ** 2)

    def loss_no_flow(self, params, rainfall_u, hetero_u, points, normals):
        pred = vmap(self.no_flow_flux_point, (None, 0, 0, 0, 0, 0, 0, 0))(
            params,
            rainfall_u,
            hetero_u,
            points[:, 0],
            points[:, 1],
            points[:, 2],
            normals[:, 0],
            normals[:, 1],
        )
        return jnp.mean(pred**2)

    def loss_residual(self, params, rainfall_u, hetero_u, points):
        pred = vmap(self.residual_point, (None, 0, 0, 0, 0, 0))(params, rainfall_u, hetero_u, points[:, 0], points[:, 1], points[:, 2])
        return jnp.mean(pred**2)

    def total_loss(self, params, batch):
        loss_ic = self.loss_initial(params, batch["rain_ic"], batch["hetero_ic"], batch["ic_points"], batch["initial_head"])
        loss_rain = self.loss_rainfall(
            params,
            batch["rain_bc"],
            batch["hetero_bc"],
            batch["rain_points"],
            batch["rain_normals"],
            batch["rain_flux"],
        )
        loss_noflow = self.loss_no_flow(
            params,
            batch["rain_nf"],
            batch["hetero_nf"],
            batch["nf_points"],
            batch["nf_normals"],
        )
        loss_res = self.loss_residual(params, batch["rain_res"], batch["hetero_res"], batch["res_points"])
        total = (
            self.loss_weights.initial * loss_ic
            + self.loss_weights.rainfall * loss_rain
            + self.loss_weights.no_flow * loss_noflow
            + self.loss_weights.residual * loss_res
        )
        return total, (loss_ic, loss_rain, loss_noflow, loss_res)

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        grads = grad(lambda p: self.total_loss(p, batch)[0])(params)
        return self.opt_update(i, grads, opt_state)

    @partial(jit, static_argnums=(0,))
    def predict_head(self, params, rainfall_u, hetero_u, points):
        return vmap(self.operator_net, (None, 0, 0, 0, 0, 0))(params, rainfall_u, hetero_u, points[:, 0], points[:, 1], points[:, 2])
