# Experiment Summary

## 1D Heterogeneous PI-DeepONet

- Experiment name: `gong_1d_pi_deeponet`
- Script: `experiments/train_gong_1d_pi_deeponet.py`
- Config: `configs/gong_1d_heterogeneous_pi_deeponet.json`
- Physics setup:
  - van Genuchten parameters follow Gong (2025) basic setup
  - `theta_r = 0.078`
  - `theta_s = 0.43`
  - `alpha = 0.036 cm^-1`
  - `n = 1.56`
  - `mu_lnKs = 3.0`
  - `var_lnKs = 3.0`
  - `corr_length = 50 cm`
  - `depth = 99 cm`
  - `initial_head = -50 cm`
  - `top_flux = -1 cm/day`
  - `t_final = 10 day`
- Heterogeneity representation:
  - 1D KLE
  - `Nkl = 9`
- Training result:
  - iterations: `1500`
  - training time: `42.95 s`
  - final total loss: `0.068227`
  - final `loss_ic`: `0.011831`
  - final `loss_top`: `0.000256`
  - final `loss_bottom`: `0.000082`
  - final `loss_res`: `0.003546`
- Output ranges:
  - `Ks`: `8.1081` to `59.7315 cm/day`
  - head: `-63.1149` to `-9.9997 cm`
  - `theta`: `0.2816` to `0.4074`
- Result assessment:
  - training is stable
  - losses decrease substantially
  - predicted field ranges are plausible for a forward pilot

## 2D Heterogeneous Slope PI-DeepONet

- Experiment name: `gong_2d_pi_deeponet`
- Script: `experiments/train_gong_2d_pi_deeponet.py`
- Config: `configs/gong_2d_slope_heterogeneous_pi_deeponet.json`
- Physics setup:
  - same hydraulic parameters as the 1D Gong-style setup
  - 2D slope geometry:
    - `width = 99 cm`
    - `height = 99 cm`
    - `crest_width = 35 cm`
  - `initial_head = -50 cm`
  - constant rainfall flux on slope surface: `-1 cm/day`
  - `t_final = 10 day`
- Heterogeneity representation:
  - piecewise constant `Ks(x,z)`
  - background `Ks = 20.0855 cm/day`
  - low-`Ks` zone: `8 cm/day`
  - high-`Ks` zone: `45 cm/day`
- Training result:
  - iterations: `1000`
  - training time: `39.36 s`
  - final total loss: `1.434561`
  - final `loss_ic`: `0.056082`
  - final `loss_rain`: `0.053105`
  - final `loss_noflow`: `0.018135`
  - final `loss_res`: `0.001374`
- Output ranges inside slope domain:
  - `Ks`: `8.0` to `45.0 cm/day`
  - head: `-1614.4432` to `-23.6027 cm`
  - `theta`: `0.1142` to `0.3644`
  - grid points inside domain: `5467`
- Result assessment:
  - the 2D model trains and produces saved outputs successfully
  - boundary and residual losses decrease clearly
  - however, the head range is still too wide in parts of the slope domain
  - this means the 2D run is a successful computational pilot, but not yet a calibrated final-quality result

## Saved Outputs

- 1D results: `outputs/gong_1d_pi_deeponet/`
- 2D results: `outputs/gong_2d_pi_deeponet/`

## Immediate Next Recommendations

- Use the 1D result as the current stable baseline.
- Treat the 2D slope result as a proof-of-execution run.
- Next tuning targets for 2D:
  - increase residual and boundary sampling density near the wetting front and material interfaces
  - introduce residual-based adaptive refinement
  - tune loss weights and network width
  - optionally warm-start from an easier homogeneous 2D case
