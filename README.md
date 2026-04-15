# PI-DeepONet-HeteroSlope

Physics-Informed DeepONet for Rainfall Infiltration Analysis in Heterogeneous Slopes

## Overview

This project extends the PI-DeepONet framework to analyze rainfall infiltration in **heterogeneous slopes** using pure deep learning without numerical simulation. Based on the work presented in Gong et al. (2025), this implementation focuses on non-uniform soil properties and their effects on unsaturated flow dynamics.

## Background

Traditional numerical methods for solving Richards' equation in heterogeneous soils require fine computational meshes and significant computing resources. This project leverages Physics-Informed Neural Networks (PINNs) with DeepONet architecture to learn the operator mapping between:

- **Input**: Rainfall scenarios + Heterogeneous hydraulic conductivity fields
- **Output**: Pressure head distribution over space and time

## Key Features

- **Physics-Informed DeepONet Architecture**: Encodes Darcy's law and Richards' equation into the loss function
- **Heterogeneous Soil Modeling**: Supports piecewise saturated hydraulic conductivity ($K_s$) fields with multiple soil zones
- **Van Genuchten Model**: Implements soil water characteristic curves for unsaturated flow
- **2D Slope Geometry**: Models inclined surfaces with complex boundary conditions
- **JAX-based Implementation**: GPU-accelerated training with automatic differentiation

## Project Structure

```
PI-DeepONet-HeteroSlope/
├── src/
│   └── slope_infiltration/
│       ├── __init__.py          # Package exports
│       ├── config_io.py         # Configuration loading
│       ├── geometry.py          # Slope geometry and boundaries
│       ├── heterogeneity.py     # Soil zone and Ks field definitions
│       ├── jax_nets.py          # Neural network architectures
│       ├── kle.py               # Karhunen-Loeve expansion for random fields
│       ├── model_1d.py          # 1D DeepONet implementation
│       ├── model_2d.py          # 2D DeepONet implementation
│       ├── physics.py           # VanGenuchten model and hydraulic functions
│       ├── samplers.py          # Training point sampling
│       └── scenarios.py         # Rainfall and slope scenarios
├── experiments/
│   ├── train_gong_1d_pi_deeponet.py   # 1D training script
│   └── train_gong_2d_pi_deeponet.py   # 2D training script
├── DeepONet_Codes/              # Original DeepONet reference implementation
└── case/                         # Reference papers and data
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install jax jaxlib numpy scipy matplotlib pandas opt_einsum

# Or use the existing environment from the codebase
```

## Quick Start

### 2D Heterogeneous Slope Training

```python
from pathlib import Path
from slope_infiltration import (
    Domain2D, LossWeights2D, PI_DeepONet2D,
    PiecewiseKsField, RainfallProfile, SlopeGeometry,
    SoilZone, VanGenuchtenParameters, sample_training_batch
)

# Define slope geometry
geometry = SlopeGeometry(
    x_min=0.0, x_max=10.0, z_min=0.0, z_max=10.0, slope_angle=0.349
)

# Define heterogeneous soil zones
zones = (
    SoilZone(x_min=0.0, x_max=5.0, z_min=0.0, z_max=10.0, saturated_conductivity=0.001),
    SoilZone(x_min=5.0, x_max=10.0, z_min=0.0, z_max=10.0, saturated_conductivity=0.01),
)
field = PiecewiseKsField(background_ks=0.005, zones=zones)

# Define domain and hydrology
domain = Domain2D(t_min=0.0, t_max=1.0, x_min=0.0, x_max=10.0, z_min=0.0, z_max=10.0)
hydrology = VanGenuchtenParameters(alpha=3.6, n=1.56, theta_r=0.078, theta_s=0.43)

# Initialize model
model = PI_DeepONet2D(
    rainfall_input_dim=1,
    hetero_input_dim=field.feature_vector().shape[0],
    hidden_dim=64,
    hydrology=hydrology,
    domain=domain,
    loss_weights=LossWeights2D(),
    ks_value_fn=field.saturated_conductivity,
)
```

### Run Training

```bash
cd experiments
python train_gong_2d_pi_deeponet.py --config path/to/config.json
```

## Physics Formulation

### Governing Equation (Richards' Equation)

The model solves the mixed form of Richards' equation:

$$\frac{\partial \theta}{\partial t} = \nabla \cdot [K(h) \nabla h] + K(h) \cdot \nabla z$$

where:
- $\theta$ is volumetric water content
- $h$ is pressure head
- $K$ is hydraulic conductivity
- $z$ is vertical coordinate

### Constitutive Relations (Van Genuchten-Mualem)

- Water content: $\theta(h) = \theta_r + (\theta_s - \theta_r)(1 + |\alpha h|^n)^{-m}$
- Hydraulic conductivity: $K(h) = K_s \cdot K_r(h)$
- Relative conductivity: $K_r(h) = \sqrt{S_e} \left[1 - (1 - S_e^{1/m})^m\right]^2$

where $m = 1 - 1/n$ and $S_e$ is effective saturation.

## Heterogeneity Modeling

The saturated hydraulic conductivity $K_s$ is modeled as a piecewise constant field:

$$K_s(x, z) = K_{s,0} + \sum_{i=1}^{N_z} \Delta K_{s,i} \cdot \mathbf{1}_{Z_i}(x, z)$$

where $Z_i$ are soil zones with distinct hydraulic properties.

## Reference

If you use this code, please cite:

```
Gong et al. (2025). Physics Informed Neural Network With Enhanced Parameter State Coupling for Inverse.
Water Resources Research.
```

## Acknowledgments

This project is based on the PI-DeepONet framework for unsaturated flow in slopes. We extend the original work to handle heterogeneous soil conditions using pure deep learning approaches.

## License

MIT License
