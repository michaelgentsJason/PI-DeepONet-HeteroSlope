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

from slope_infiltration import (
    Domain2D,
    LossWeights2D,
    PI_DeepONet2D,
    PiecewiseKsField,
    RainfallProfile,
    SlopeGeometry,
    SoilZone,
    VanGenuchtenParameters,
    sample_training_batch,
    theta_from_head,
)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_case(config: dict):
    geometry = SlopeGeometry(**config["geometry"])
    zones = tuple(SoilZone(**zone) for zone in config["soil"].get("zones", []))
    field = PiecewiseKsField(background_ks=config["soil"]["background_ks"], zones=zones)
    rainfall = RainfallProfile(
        breakpoints=tuple(config["rainfall"]["breakpoints"]),
        fluxes=tuple(config["rainfall"]["fluxes"]),
    )
    return geometry, field, rainfall


def make_batch(config: dict, geometry: SlopeGeometry, field: PiecewiseKsField, rainfall_u: np.ndarray, hetero_u: np.ndarray, rng: np.random.Generator) -> dict:
    batch = sample_training_batch(
        geometry,
        field,
        rng=rng,
        t_max=config["physics"]["t_final_days"],
        n_residual=config["training"]["n_residual"],
        n_initial=config["training"]["n_initial"],
        n_rainfall=config["training"]["n_rainfall"],
        n_no_flow=config["training"]["n_no_flow"],
        interface_tolerance=config["training"]["interface_tolerance"],
    )

    def repeat_branch(count: int, values: np.ndarray) -> jnp.ndarray:
        return jnp.asarray(np.tile(values[None, :], (count, 1)))

    return {
        "res_points": jnp.asarray(batch.residual),
        "ic_points": jnp.asarray(batch.initial),
        "rain_points": jnp.asarray(batch.rainfall_boundary.coordinates),
        "rain_normals": jnp.asarray(batch.rainfall_boundary.normals),
        "nf_points": jnp.asarray(batch.no_flow_boundary.coordinates),
        "nf_normals": jnp.asarray(batch.no_flow_boundary.normals),
        "rain_res": repeat_branch(batch.residual.shape[0], rainfall_u),
        "hetero_res": repeat_branch(batch.residual.shape[0], hetero_u),
        "rain_ic": repeat_branch(batch.initial.shape[0], rainfall_u),
        "hetero_ic": repeat_branch(batch.initial.shape[0], hetero_u),
        "rain_bc": repeat_branch(batch.rainfall_boundary.coordinates.shape[0], rainfall_u),
        "hetero_bc": repeat_branch(batch.rainfall_boundary.coordinates.shape[0], hetero_u),
        "rain_nf": repeat_branch(batch.no_flow_boundary.coordinates.shape[0], rainfall_u),
        "hetero_nf": repeat_branch(batch.no_flow_boundary.coordinates.shape[0], hetero_u),
        "initial_head": jnp.asarray(config["physics"]["initial_head_cm"]),
        "rain_flux": jnp.asarray(config["rainfall"]["fluxes"][0]),
    }


def save_results(
    output_dir: Path,
    model: PI_DeepONet2D,
    params,
    geometry: SlopeGeometry,
    field: PiecewiseKsField,
    rainfall_u: np.ndarray,
    hetero_u: np.ndarray,
    config: dict,
    losses: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_df = pd.DataFrame(losses)
    loss_df.to_csv(output_dir / "loss_history.csv", index=False)
    with open(output_dir / "model_params.pkl", "wb") as handle:
        pickle.dump(params, handle)

    nx = 90
    nz = 90
    t_eval = config["physics"]["t_final_days"]
    x_grid = np.linspace(0.0, geometry.width, nx)
    z_grid = np.linspace(0.0, geometry.height, nz)
    xx, zz = np.meshgrid(x_grid, z_grid)
    mask = zz <= geometry.surface_elevation(xx)
    points = np.column_stack([np.full(xx.size, t_eval), xx.ravel(), zz.ravel()])
    rain_eval = np.tile(rainfall_u[None, :], (points.shape[0], 1))
    hetero_eval = np.tile(hetero_u[None, :], (points.shape[0], 1))

    head = np.asarray(
        model.predict_head(
            params,
            jnp.asarray(rain_eval),
            jnp.asarray(hetero_eval),
            jnp.asarray(points),
        )
    ).reshape(xx.shape)
    theta = np.asarray(theta_from_head(jnp.asarray(head), model.hydrology))
    ks = field.saturated_conductivity(xx, zz)

    head = np.where(mask, head, np.nan)
    theta = np.where(mask, theta, np.nan)
    ks = np.where(mask, ks, np.nan)

    grid_df = pd.DataFrame(
        {
            "x_cm": xx.ravel(),
            "z_cm": zz.ravel(),
            "inside_domain": mask.ravel().astype(int),
            "ks_cm_per_day": ks.ravel(),
            "head_cm": head.ravel(),
            "theta": theta.ravel(),
        }
    )
    grid_df.to_csv(output_dir / "final_snapshot_grid.csv", index=False)

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

    def draw_field(data: np.ndarray, label: str, filename: str, cmap: str) -> None:
        masked = np.ma.masked_invalid(data)
        plt.figure(figsize=(6.8, 4.8))
        plt.pcolormesh(xx, zz, masked, shading="auto", cmap=cmap)
        plt.colorbar(label=label)
        plt.plot(x_grid, geometry.surface_elevation(x_grid), color="black", linewidth=1.2)
        plt.xlabel("x (cm)")
        plt.ylabel("z (cm)")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    draw_field(ks, "Ks (cm/day)", "ks_snapshot.png", "terrain")
    draw_field(theta, "theta", "theta_snapshot.png", "viridis")
    draw_field(head, "head (cm)", "head_snapshot.png", "plasma")

    metadata = {
        "rainfall_branch_dim": int(rainfall_u.shape[0]),
        "hetero_branch_dim": int(hetero_u.shape[0]),
        "surface_length_cm": float(geometry.rainfall_boundary_length),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    config = load_config(ROOT / "configs" / "gong_2d_slope_heterogeneous_pi_deeponet.json")
    geometry, field, rainfall = build_case(config)

    hydrology = VanGenuchtenParameters(
        alpha=config["physics"]["alpha"],
        n=config["physics"]["n"],
        theta_r=config["physics"]["theta_r"],
        theta_s=config["physics"]["theta_s"],
    )
    domain = Domain2D(
        t_min=0.0,
        t_max=config["physics"]["t_final_days"],
        x_min=0.0,
        x_max=geometry.width,
        z_min=0.0,
        z_max=geometry.height,
    )

    rainfall_sensor_count = config["branch_inputs"]["rainfall_sensor_count"]
    sensor_times = np.linspace(0.0, config["physics"]["t_final_days"], rainfall_sensor_count, dtype=np.float32)
    rainfall_u = rainfall.sensor_values(sensor_times).astype(np.float32)
    hetero_u = np.concatenate(
        [
            field.feature_vector(),
            np.asarray([geometry.width, geometry.height, geometry.crest_width, geometry.toe_height], dtype=float),
        ]
    ).astype(np.float32)

    def ks_value_fn(x, z):
        ks = jnp.full_like(x, config["soil"]["background_ks"])
        for zone in config["soil"].get("zones", []):
            inside = (
                (x >= zone["x_min"])
                & (x <= zone["x_max"])
                & (z >= zone["z_min"])
                & (z <= zone["z_max"])
            )
            ks = jnp.where(inside, zone["saturated_conductivity"], ks)
        return ks

    model = PI_DeepONet2D(
        rainfall_input_dim=rainfall_u.shape[0],
        hetero_input_dim=hetero_u.shape[0],
        hidden_dim=config["training"]["hidden_dim"],
        hydrology=hydrology,
        domain=domain,
        loss_weights=LossWeights2D(**config["loss_weights"]),
        ks_value_fn=ks_value_fn,
        learning_rate=config["training"]["learning_rate"],
    )

    rng = np.random.default_rng(config["training"]["seed"])
    history: list[dict] = []
    start = perf_counter()

    for step in range(config["training"]["iterations"]):
        batch = make_batch(config, geometry, field, rainfall_u, hetero_u, rng)
        model.opt_state = model.step(step, model.opt_state, batch)

        if step % config["training"]["log_every"] == 0 or step == config["training"]["iterations"] - 1:
            params = model.get_params(model.opt_state)
            total, parts = model.total_loss(params, batch)
            record = {
                "iteration": step,
                "loss_total": float(total),
                "loss_ic": float(parts[0]),
                "loss_rain": float(parts[1]),
                "loss_noflow": float(parts[2]),
                "loss_res": float(parts[3]),
            }
            history.append(record)
            print(record)

    elapsed = perf_counter() - start
    params = model.get_params(model.opt_state)
    output_dir = ROOT / "outputs" / "gong_2d_pi_deeponet"
    save_results(output_dir, model, params, geometry, field, rainfall_u, hetero_u, config, history)
    print(f"training_seconds={elapsed:.2f}")
    print(f"results_dir={output_dir}")


if __name__ == "__main__":
    main()
