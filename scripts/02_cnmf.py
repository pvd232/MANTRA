#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, TypedDict, Literal

import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore
import yaml
from numpy.typing import NDArray
from sklearn.decomposition import NMF

# ---- Typed config ------------------------------------------------------------

InitType = Literal["random", "nndsvd", "nndsvda", "nndsvdar", "custom"]


class CNMFParams(TypedDict):
    K: int
    max_iter: int
    init: InitType
    random_state: int


class Params(TypedDict):
    hvg_n_top_genes: int
    pct_mito_max: float
    min_genes_per_cell: int
    cnmf: CNMFParams


def _coerce_init(x: str) -> InitType:
    allowed: tuple[InitType, ...] = (
        "random",
        "nndsvd",
        "nndsvda",
        "nndsvdar",
        "custom",
    )
    if x not in allowed:
        raise ValueError(f"cnmf.init must be one of {allowed}, got {x!r}")
    return x


def load_params(path: Path) -> Params:
    raw: Dict[str, Any] = yaml.safe_load(path.read_text())
    cnmf_raw: Dict[str, Any] = raw["cnmf"]
    params: Params = {
        "hvg_n_top_genes": int(raw["hvg_n_top_genes"]),
        "pct_mito_max": float(raw["pct_mito_max"]),
        "min_genes_per_cell": int(raw["min_genes_per_cell"]),
        "cnmf": {
            "K": int(cnmf_raw["K"]),
            "max_iter": int(cnmf_raw["max_iter"]),
            "init": _coerce_init(str(cnmf_raw["init"])),
            "random_state": int(cnmf_raw["random_state"]),
        },
    }
    return params


# ---- CLI ---------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fit cNMF programs on unperturbed HVGs.")
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument("--in", dest="inp", required=True, help="out/interim")
    ap.add_argument("--out", required=True, help="out/interim")
    return ap


# ---- Utils -------------------------------------------------------------------


def as_dense(X: Any) -> NDArray[np.float64]:
    """Convert AnnData matrix to a dense numpy array (float64)."""
    if hasattr(X, "toarray"):
        return X.toarray().astype(np.float64, copy=False)  # type: ignore[no-any-return]
    return np.asarray(X, dtype=np.float64)


# ---- Main --------------------------------------------------------------------


def main() -> None:
    args = build_argparser().parse_args()
    P = load_params(Path(args.params))

    ad = sc.read_h5ad("data/interim/unperturbed_qc.h5ad")

    mask: NDArray[np.bool_] = ad.var["highly_variable"].to_numpy()  # type: ignore[no-any-return]
    X: NDArray[np.float64] = as_dense(ad[:, mask].X) # type: ignore

    # Pylance is now satisfied because "init" is InitType (a Literal union)
    nmf = NMF(
        n_components=P["cnmf"]["K"],
        init=P["cnmf"]["init"],
        random_state=P["cnmf"]["random_state"],
        max_iter=P["cnmf"]["max_iter"],
    )
    H: NDArray[np.float64] = nmf.fit_transform(np.maximum(X, 0.0))
    W: NDArray[np.float64] = nmf.components_.T  # genes×K

    ad.obsm["cnmf_H"] = H
    ad.varm["cnmf_W"] = W

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    gene_names = ad.var_names[mask]
    cols = [f"P{k}" for k in range(P["cnmf"]["K"])]

    pd.DataFrame(W, index=gene_names, columns=cols).to_csv(
        out_dir / "program_loadings_W.csv"
    )

    manifest: dict[str, Any] = {
        "time": time.time(),
        "git": os.popen("git rev-parse --short HEAD").read().strip(),
        "outputs": ["out/interim/program_loadings_W.csv"],
        "params": P["cnmf"],
    }
    (out_dir / "manifest_cnmf.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
