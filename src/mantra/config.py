#!/usr/bin/env python
# src/mantra/config.py
"""
Small shared configuration objects for MANTRA.

Right now this is intentionally tiny and only defines:

  - QCConfig:   thresholds for basic QC on raw AnnData
  - TraitsConfig: primary + auxiliary traits for readout

Each script is still responsible for loading its own YAML and
constructing module-specific configs (CNMFConfig, EnergyTrainConfig, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# -----------------------
# QC
# -----------------------

@dataclass
class QCConfig:
    """
    QC thresholds applied to the raw GWPS AnnData before HVG selection.

    Mirrors the `qc:` block in configs/params.yml, e.g.:

      qc:
        min_cells: 500
        min_genes: 200
        max_pct_mt: 15
    """
    min_cells: int = 500
    min_genes: int = 200
    max_pct_mt: float = 15.0


def load_qc_config(params_path: str | Path) -> QCConfig:
    """
    Load QCConfig from a params YAML file.

    If no `qc` block is present, returns QCConfig() defaults.
    Unknown keys in the `qc` block are ignored.
    """
    params: Dict[str, Any] = yaml.safe_load(Path(params_path).read_text()) or {}
    qc_block: Dict[str, Any] = params.get("qc", {}) or {}

    cfg = QCConfig(
        min_cells=int(qc_block.get("min_cells", QCConfig.min_cells)),
        min_genes=int(qc_block.get("min_genes", QCConfig.min_genes)),
        max_pct_mt=float(qc_block.get("max_pct_mt", QCConfig.max_pct_mt)),
    )
    return cfg


# -----------------------
# Traits
# -----------------------

@dataclass
class TraitsConfig:
    """
    Trait selection for downstream readout (SMR/TWAS-derived).

    Mirrors the `traits:` block in configs/params.yml, e.g.:

      traits:
        primary: MCH
        extras: [RDW, IRF]
    """
    primary: str = "MCH"
    extras: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extras is None:
            self.extras = ["RDW", "IRF"]


def load_traits_config(params_path: str | Path) -> TraitsConfig:
    """
    Load TraitsConfig from a params YAML file.

    If no `traits` block is present, returns TraitsConfig() defaults.
    """
    params: Dict[str, Any] = yaml.safe_load(Path(params_path).read_text()) or {}
    traits_block: Dict[str, Any] = params.get("traits", {}) or {}

    primary: str = traits_block.get("primary", TraitsConfig.primary)  # type: ignore[arg-type]
    extras_raw: Optional[Any] = traits_block.get("extras", None)

    if extras_raw is None:
        extras = None
    elif isinstance(extras_raw, list):
        extras = [str(x) for x in extras_raw]
    else:
        # allow a single scalar like "RDW" → ["RDW"]
        extras = [str(extras_raw)]

    return TraitsConfig(primary=primary, extras=extras)
