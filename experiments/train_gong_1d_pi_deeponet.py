from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from slope_infiltration.kle import ExponentialKLE1D
from slope_infiltration.model_1d import Domain1D, LossWeights1D, PI_DeepONet1D
from slope_infiltration.physics import VanGenuchtenParameters, theta_from_head


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_batch(config: dict, rainfall_u: np.ndarray, ks_u: np.ndarray, rng: np.random.Generator) -> dict:
    physics = config["physics"]
    training = config["training"]

    def repeat_branch(count: int, values: np.ndarray) -> jnp.ndarray:
        return jnp.asarray(np.tile(values[None, :], (count, 1)))

    n_res = training["n_residual"]
    n_ic = training["n_initial"]
    n_top = training["n_top"]
    n_bottom = training["n_bottom"]

    t_final = physics["t_final_days"]
    depth = physics["depth_cm"]

    res_t = rng.uniform(0.0, t_final, size=(n_res, 1))
    res_z = rng.uniform(-depth, 0.0, size=(n_res, 1))
    ic_points = np.hstack([np.zeros((n_ic, 1)), rng.uniform(-depth, 0.0, size=(n_ic, 1))])
    top_points = np.hstack([rng.uniform(0.0, t_final, size=(n_top, 1)), np.zeros((n_top, 1))])
    bottom_points = np.hstack([rng.uniform(0.0, t_final, size=(n_bottom, 1)), -depth * np.ones((n_bottom, 1))])

    return {
        "res_points": jnp.asarray(np.hstack([res_t, res_z])),
        "ic_points": jnp.asarray(ic_points),
        "top_points": jnp.asarray(top_points),
        "bottom_points": jnp.asarray(bottom_points),
        "rain_res": repeat_branch(n_res, rainfall_u),
        "ks_res": repeat_branch(n_res, ks_u),
        "rain_ic": repeat_branch(n_ic, rainfall_u),
        "ks_ic": repeat_branch(n_ic, ks_u),
        "rain_top": repeat_branch(n_top, rainfall_u),
        "ks_top": repeat_branch(n_top, ks_u),
        "rain_bottom": repeat_branch(n_bottom, rainfall_u),
        "ks_bottom": repeat_branch(n_bottom, ks_u),
        "initial_head": jnp.asarray(physics["initial_head_cm"]),
        "top_flux": jnp.asarray(physics["top_flux_cm_per_day"]),
    }


def save_results(
    output_dir: Path,
    model: PI_DeepONet1D,
    params,
    rainfall_u: np.ndarray,
    ks_u: np.ndarray,
    config: dict,
    kle: ExponentialKLE1D,
    losses: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_df = pd.DataFrame(losses)
    loss_df.to_csv(output_dir / "loss_history.csv", index=False)
    with open(output_dir / "model_params.pkl", "wb") as handle:
        pickle.dump(params, handle)

    depth = config["physics"]["depth_cm"]
    t_final = config["physics"]["t_final_days"]
    t_grid = np.linspace(0.0, t_final, 101)
    z_grid = np.linspace(-depth, 0.0, 100)

    tt, zz = np.meshgrid(t_grid, z_grid)
    points = np.column_stack([tt.ravel(), zz.ravel()])
    rain_eval = np.tile(rainfall_u[None, :], (points.shape[0], 1))
    ks_eval = np.tile(ks_u[None, :], (points.shape[0], 1))

    head = np.asarray(model.predict_head(params, jnp.asarray(rain_eval), jnp.asarray(ks_eval), jnp.asarray(points))).reshape(zz.shape)
    theta = np.asarray(theta_from_head(jnp.asarray(head), model.hydrology))

    depth_positive = -z_grid
    log_ks = kle.log_ks(depth_positive, ks_u)
    ks = np.exp(log_ks)
    ks_df = pd.DataFrame({"depth_cm": depth_positive, "log_ks": log_ks, "ks_cm_per_day": ks})
    ks_df.to_csv(output_dir / "ks_profile.csv", index=False)

    head_df = pd.DataFrame(head, index=z_grid, columns=t_grid)
    theta_df = pd.DataFrame(theta, index=z_grid, columns=t_grid)
    head_df.to_csv(output_dir / "head_field.csv")
    theta_df.to_csv(output_dir / "theta_field.csv")

    plt.figure(figsize=(7, 4))
    plt.plot(loss_df["iteration"], loss_df["loss_total"], label="total")
    plt.plot(loss_df["iteration"], loss_df["loss_res"], label="res")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(4, 5))
    plt.plot(ks, depth_positive)
    plt.gca().invert_yaxis()
    plt.xlabel(r"$K_s$ (cm/day)")
    plt.ylabel("Depth (cm)")
    plt.tight_layout()
    plt.savefig(output_dir / "ks_profile.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.imshow(
        theta,
        aspect="auto",
        origin="lower",
        extent=[t_grid.min(), t_grid.max(), z_grid.min(), z_grid.max()],
        cmap="viridis",
    )
    plt.colorbar(label="theta")
    plt.xlabel("Time (day)")
    plt.ylabel("Elevation z (cm)")
    plt.tight_layout()
    plt.savefig(output_dir / "theta_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.imshow(
        head,
        aspect="auto",
        origin="lower",
        extent=[t_grid.min(), t_grid.max(), z_grid.min(), z_grid.max()],
        cmap="plasma",
    )
    plt.colorbar(label="Head (cm)")
    plt.xlabel("Time (day)")
    plt.ylabel("Elevation z (cm)")
    plt.tight_layout()
    plt.savefig(output_dir / "head_heatmap.png", dpi=180)
    plt.close()

    metadata = {
        "rainfall_branch_dim": int(rainfall_u.shape[0]),
        "ks_branch_dim": int(ks_u.shape[0]),
        "heterogeneity_coefficients": ks_u.tolist(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    config_path = ROOT / "configs" / "gong_1d_heterogeneous_pi_deeponet.json"
    config = load_config(config_path)

    physics = config["physics"]
    training = config["training"]
    weights = config["loss_weights"]
    sensor_count = config["branch_inputs"]["rainfall_sensor_count"]

    kle = ExponentialKLE1D(
        domain_length=physics["depth_cm"],
        mean_log_ks=physics["mean_log_ks"],
        variance_log_ks=physics["variance_log_ks"],
        correlation_length=physics["correlation_length_cm"],
        n_modes=physics["n_kle_modes"],
    )

    rng = np.random.default_rng(training["seed"])
    xi_rng = np.random.default_rng(config["branch_inputs"]["heterogeneity_seed"])
    xi = xi_rng.normal(0.0, 1.0, size=physics["n_kle_modes"]).astype(np.float32)
    rainfall_sensors = np.linspace(0.0, physics["t_final_days"], sensor_count, dtype=np.float32)
    rainfall_values = np.full_like(rainfall_sensors, physics["top_flux_cm_per_day"], dtype=np.float32)

    hydrology = VanGenuchtenParameters(
        alpha=physics["alpha"],
        n=physics["n"],
        theta_r=physics["theta_r"],
        theta_s=physics["theta_s"],
    )

    domain = Domain1D(
        t_min=0.0,
        t_max=physics["t_final_days"],
        z_top=0.0,
        z_bottom=-physics["depth_cm"],
    )

    def ks_log_fn(z, coeffs):
        depth_positive = -z
        basis = jnp.asarray(kle.basis(np.asarray([0.0])), dtype=jnp.float32)
        weights = jnp.asarray(np.sqrt(kle.eigenvalues), dtype=jnp.float32) * coeffs
        omega = jnp.asarray(kle.omegas, dtype=jnp.float32)
        normalizers = jnp.asarray(kle.normalizers, dtype=jnp.float32)
        depth = depth_positive
        modes = (physics["correlation_length_cm"] * omega * jnp.cos(omega * depth) + jnp.sin(omega * depth)) / normalizers
        return physics["mean_log_ks"] + jnp.sum(weights * modes)

    def dks_log_dz_fn(z, coeffs):
        depth = -z
        weights = jnp.asarray(np.sqrt(kle.eigenvalues), dtype=jnp.float32) * coeffs
        omega = jnp.asarray(kle.omegas, dtype=jnp.float32)
        normalizers = jnp.asarray(kle.normalizers, dtype=jnp.float32)
        derivative_depth = (
            -physics["correlation_length_cm"] * omega**2 * jnp.sin(omega * depth)
            + omega * jnp.cos(omega * depth)
        ) / normalizers
        return -jnp.sum(weights * derivative_depth)

    model = PI_DeepONet1D(
        rainfall_input_dim=sensor_count,
        ks_input_dim=physics["n_kle_modes"],
        hidden_dim=training["hidden_dim"],
        hydrology=hydrology,
        domain=domain,
        loss_weights=LossWeights1D(**weights),
        ks_log_fn=ks_log_fn,
        dks_log_dz_fn=dks_log_dz_fn,
        learning_rate=training["learning_rate"],
    )

    loss_history: list[dict] = []
    start = perf_counter()

    for step in range(training["iterations"]):
        batch = make_batch(config, rainfall_values, xi, rng)
        model.opt_state = model.step(step, model.opt_state, batch)

        if step % training["log_every"] == 0 or step == training["iterations"] - 1:
            params = model.get_params(model.opt_state)
            loss_total, loss_parts = model.total_loss(params, batch)
            record = {
                "iteration": step,
                "loss_total": float(loss_total),
                "loss_ic": float(loss_parts[0]),
                "loss_top": float(loss_parts[1]),
                "loss_bottom": float(loss_parts[2]),
                "loss_res": float(loss_parts[3]),
            }
            loss_history.append(record)
            print(record)

    elapsed = perf_counter() - start
    params = model.get_params(model.opt_state)
    output_dir = ROOT / "outputs" / "gong_1d_pi_deeponet"
    save_results(output_dir, model, params, rainfall_values, xi, config, kle, loss_history)
    print(f"training_seconds={elapsed:.2f}")
    print(f"results_dir={output_dir}")


if __name__ == "__main__":
    main()
