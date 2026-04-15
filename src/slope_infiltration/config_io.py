from __future__ import annotations

import json
from pathlib import Path

from .geometry import SlopeGeometry
from .heterogeneity import PiecewiseKsField, SoilZone
from .scenarios import HeterogeneousSlopeScenario, RainfallProfile


def load_case_config(path: str | Path) -> tuple[HeterogeneousSlopeScenario, dict]:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    geometry = SlopeGeometry(**raw["geometry"])
    zones = tuple(SoilZone(**zone) for zone in raw["soil"].get("zones", []))
    field = PiecewiseKsField(
        background_ks=raw["soil"]["background_ks"],
        zones=zones,
    )
    rainfall = RainfallProfile(
        breakpoints=tuple(raw["rainfall"]["breakpoints"]),
        fluxes=tuple(raw["rainfall"]["fluxes"]),
    )
    scenario = HeterogeneousSlopeScenario(
        geometry=geometry,
        field=field,
        rainfall=rainfall,
        initial_head=raw["initial_head"],
    )
    sampling = dict(raw.get("sampling", {}))
    return scenario, sampling
