#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import scanpy as sc

from mantra.programs.cnmf import CNMFConfig, run_cnmf, save_cnmf_result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run consensus NMF on a QC'd AnnData object to obtain "
            "gene-program loadings W [G,K] and related artifacts."
        )
    )

    p.add_argument(
        "--ad",
        type=str,
        required=True,
        help="Input QC AnnData (e.g. data/interim/k562_gwps_unperturbed_qc.h5ad)",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for W arrays and manifest.",
    )
    p.add_argument(
        "--params",
        type=str,
        default=None,
        help="Optional YAML params file; if provided, uses the 'cnmf' section as defaults.",
    )

    # core CNMF config
    p.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of programs / components (rank K).",
    )
    p.add_argument(
        "--use-hvg",
        action="store_true",
        help="Restrict to ad.var['highly_variable']==True if present.",
    )
    p.add_argument(
        "--obsm-key",
        type=str,
        default=None,
        help="If set, use ad.obsm[KEY] instead of ad.X (e.g. 'X_pca').",
    )
    p.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Optional layer name to use instead of ad.X.",
    )
    p.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional cap on #cells used for NMF (global subsample).",
    )

    # consensus params
    p.add_argument(
        "--n-restarts",
        type=int,
        default=10,
        help="Number of independent NMF runs for consensus.",
    )
    p.add_argument(
        "--bootstrap-fraction",
        type=float,
        default=0.8,
        help="Fraction of cells to use per run (0< f ≤ 1).",
    )

    # NMF hyperparams
    p.add_argument(
        "--max-iter",
        type=int,
        default=400,
        help="Max iterations for NMF solver.",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance for NMF.",
    )
    p.add_argument(
        "--init",
        type=str,
        default="nndsvda",
        help="NMF initialization (e.g. 'nndsvda', 'random').",
    )
    p.add_argument(
        "--alpha-W",
        type=float,
        default=0.0,
        help="L1/L2 regularization for W (sklearn alpha_W).",
    )
    p.add_argument(
        "--alpha-H",
        type=float,
        default=0.0,
        help="L1/L2 regularization for H (sklearn alpha_H).",
    )
    p.add_argument(
        "--l1-ratio",
        type=float,
        default=0.0,
        help="Mix between L1 (1.0) and L2 (0.0) penalties.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for NMF + bootstraps.",
    )

    # KMeans hyperparams
    p.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=300,
        help="Max iterations for KMeans consensus clustering.",
    )
    p.add_argument(
        "--kmeans-tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance for KMeans.",
    )
    p.add_argument(
        "--kmeans-n-init",
        type=int,
        default=10,
        help="Number of KMeans initializations.",
    )

    p.add_argument(
        "--name",
        type=str,
        default="k562_cnmf",
        help="Short name used as prefix for output files.",
    )

    return p


def make_cfg(args: argparse.Namespace) -> CNMFConfig:
    base: Dict[str, Any] = {}
    if args.params is not None:
        base = yaml.safe_load(Path(args.params).read_text()).get("cnmf", {})

    def get_param(key: str, default):
        # CLI overrides YAML; else YAML; else given default
        v = getattr(args, key, None)
        if v is not None:
            return v
        if key in base:
            return base[key]
        return default

    cfg = CNMFConfig(
        n_components=get_param("k", args.k),
        use_hvg=get_param("use_hvg", args.use_hvg),
        obsm_key=get_param("obsm_key", args.obsm_key),
        layer=get_param("layer", args.layer),
        max_cells=get_param("max_cells", args.max_cells),
        n_restarts=get_param("n_restarts", args.n_restarts),
        bootstrap_fraction=get_param("bootstrap_fraction", args.bootstrap_fraction),
        max_iter=get_param("max_iter", args.max_iter),
        tol=get_param("tol", args.tol),
        init=get_param("init", args.init),
        random_state=get_param("seed", args.seed),
        alpha_W=get_param("alpha_W", args.alpha_W),
        alpha_H=get_param("alpha_H", args.alpha_H),
        l1_ratio=get_param("l1_ratio", args.l1_ratio),
        kmeans_max_iter=get_param("kmeans_max_iter", args.kmeans_max_iter),
        kmeans_tol=get_param("kmeans_tol", args.kmeans_tol),
        kmeans_n_init=get_param("kmeans_n_init", args.kmeans_n_init),
        name=get_param("name", args.name),
    )
    return cfg


def main() -> None:
    args = build_argparser().parse_args()

    ad_path = Path(args.ad)
    out_dir = Path(args.out)

    print(f"[CNMF] Loading AnnData from {ad_path} ...", flush=True)
    ad = sc.read_h5ad(ad_path)
    print(f"[CNMF] ad.n_obs={ad.n_obs}, ad.n_vars={ad.n_vars}", flush=True)

    cfg = make_cfg(args)
    print(f"[CNMF] Config: {cfg}", flush=True)

    result = run_cnmf(ad, cfg)
    save_cnmf_result(out_dir, ad, result, prefix=cfg.name)


if __name__ == "__main__":
    main()
