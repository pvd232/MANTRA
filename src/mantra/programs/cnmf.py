#!/usr/bin/env python
# src/mantra/programs/cnmf.py
"""
Consensus NMF (cNMF) routines for program discovery on AnnData.

Public API:
  - run_cnmf(ad: AnnData, cfg: CNMFConfig) -> CNMFResults
  - save_cnmf_result(out_dir: Path, ad: AnnData, result: CNMFResults, prefix: str)

These utilities:
  - select a non-negative gene expression matrix from AnnData
  - apply HVG restriction, min-cells filter, and optional top-gene truncation
  - run NMF n_restarts times with different seeds
  - cluster per-run programs in gene space via KMeans
  - compute stable consensus programs and basic stability metrics
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

from mantra.programs.config import CNMFConfig, CNMFResults


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _select_matrix_from_anndata(
    ad: sc.AnnData,
    cfg: CNMFConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare the matrix X for NMF:

        X_sel:    (n_cells_sel, G)
        cell_idx: indices of cells used   (here: all cells)
        gene_idx: indices of genes used   (after HVG / filters)

    Steps:
      - optional HVG restriction (use_hvg_only, hvg_key)
      - min_cells_per_gene filter
      - per-cell library-size normalization (if scale_cells)
    """
    ad_view = ad.copy()

    # --- 1) HVG restriction ---
    if cfg.use_hvg_only:
        if cfg.hvg_key in ad_view.var:
            hvg_mask = ad_view.var[cfg.hvg_key].to_numpy().astype(bool)
            n_hvg = int(hvg_mask.sum())
            if n_hvg == 0:
                print(
                    f"[CNMF] HVG key '{cfg.hvg_key}' present but no genes flagged; "
                    "using all genes.",
                    flush=True,
                )
            else:
                print(
                    f"[CNMF] Restricting to HVGs via '{cfg.hvg_key}': n_vars = {n_hvg}",
                    flush=True,
                )
                ad_view = ad_view[:, hvg_mask].copy()
        else:
            print(
                f"[CNMF] HVG key '{cfg.hvg_key}' not found in ad.var; using all genes.",
                flush=True,
            )

    # --- 2) min_cells_per_gene filter ---
    X_tmp = ad_view.X
    if sp.issparse(X_tmp):
        detected = np.asarray((X_tmp > 0).sum(axis=0)).ravel()
    else:
        detected = (X_tmp > 0).sum(axis=0)
    gene_keep = detected >= int(cfg.min_cells_per_gene)
    n_kept = int(gene_keep.sum())
    if n_kept == 0:
        raise RuntimeError(
            f"[CNMF] No genes pass min_cells_per_gene={cfg.min_cells_per_gene}"
        )
    if n_kept < ad_view.n_vars:
        print(
            f"[CNMF] Filtered genes by min_cells_per_gene={cfg.min_cells_per_gene}: "
            f"{ad_view.n_vars} → {n_kept}",
            flush=True,
        )
        ad_view = ad_view[:, gene_keep].copy()

    # --- 3) materialize X and ensure non-negativity ---
    X = ad_view.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    if (X < 0).any():
        raise ValueError(
            "[CNMF] Input matrix has negative entries. "
            "NMF assumes non-negative data. Check preprocessing."
        )

    # --- 5) per-cell scaling (library-size normalization) ---
    if cfg.scale_cells:
        libsize = X.sum(axis=1, keepdims=True)  # (N,1)
        libsize[libsize == 0.0] = 1.0
        X = X / libsize
        print("[CNMF] Applied per-cell library-size normalization.", flush=True)

    n_cells, G = X.shape
    cell_idx = np.arange(n_cells, dtype=int)
    gene_idx = np.arange(G, dtype=int)

    print(f"[CNMF] Final matrix for NMF: shape={X.shape}", flush=True)
    return X, cell_idx, gene_idx


def _fit_single_nmf(
    X: np.ndarray,
    cfg: CNMFConfig,
    run_seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Fit a single NMF:

        X ≈ W_cells @ H
        W_cells: (n_cells_run, K)
        H:       (K, G)

    Return:
        W_cells, H, frob_rmse, n_iter
    """
    n_cells_run, G = X.shape
    K = int(cfg.n_components)

    print(
        f"[CNMF]   NMF run with seed={run_seed}, n_cells={n_cells_run}, G={G}, K={K}",
        flush=True,
    )

    nmf = NMF(
        n_components=K,
        init="nndsvda",
        max_iter=int(cfg.max_iter),
        tol=float(cfg.tol),
        random_state=run_seed,
        alpha_W=float(cfg.alpha),
        alpha_H=float(cfg.alpha),
        l1_ratio=float(cfg.l1_ratio),
        solver="cd",
        verbose=0,
    )

    W_cells = nmf.fit_transform(X)      # (n_cells_run, K)
    H = nmf.components_                 # (K, G)

    recon = W_cells @ H
    frob_err = np.linalg.norm(X - recon, ord="fro") / np.sqrt(X.size)

    print(
        f"[CNMF]   NMF done. Frobenius RMSE per entry={frob_err:.4f}, "
        f"n_iter={nmf.n_iter_}",
        flush=True,
    )

    return W_cells, H, float(frob_err), int(nmf.n_iter_)


# ---------------------------------------------------------------------
# Public API: run_cnmf + save_cnmf_result
# ---------------------------------------------------------------------


def run_cnmf(
    ad: sc.AnnData,
    cfg: CNMFConfig,
) -> CNMFResults:
    """
    Consensus NMF:

      1) Select X from AnnData (HVG / filters / scaling).
      2) For r = 1..n_restarts:
           - run NMF with seed = random_state + r
           - collect gene-level program loadings W_genes^{(r)} = H^{(r)T}  [G, K]
      3) Stack all programs into matrix P: (R*K, G), row L2-normalized.
      4) KMeans on P into K clusters (K = n_components).
      5) Consensus W_full = cluster_centers^T  [G, K]
      6) Optionally filter to stable programs using run_coverage
         (fraction of runs each cluster appears in).

    Returns:
      CNMFResults with:
        - W_consensus:      [G, K_stable]
        - programs_all:     [R*K, G] (per-run programs in gene space)
        - cluster_assignments: [R*K]
        - run_coverage:     [K]
        - gene_names:       [G]
        - cell_idx:         indices of cells used
        - config:           CNMFConfig
    """
    X_base, cell_idx, gene_idx = _select_matrix_from_anndata(ad, cfg)
    n_cells_base, G = X_base.shape
    K = int(cfg.n_components)
    R = int(cfg.n_restarts)

    print(
        f"[CNMF] Consensus NMF: base matrix shape={X_base.shape}, "
        f"K={K}, n_restarts={R}",
        flush=True,
    )

    frob_rmse_runs = []
    n_iter_runs = []
    all_programs = []      # will hold (R*K, G)

    for r in range(R):
        run_seed = int(cfg.random_state) + r
        W_cells, H_run, rmse, n_iter = _fit_single_nmf(X_base, cfg, run_seed)

        frob_rmse_runs.append(rmse)
        n_iter_runs.append(n_iter)

        # gene-level program loadings: W_genes [G, K]
        W_genes = H_run.T  # (G, K)

        # L2-normalize each program so clustering is about *shape* not scale
        norms = np.linalg.norm(W_genes, axis=0, keepdims=True) + 1e-8
        W_genes_norm = W_genes / norms  # (G, K)

        # append each program as a separate row in program space
        for j in range(K):
            all_programs.append(W_genes_norm[:, j])

    programs_all = np.stack(all_programs, axis=0)  # (R*K, G)
    print(
        f"[CNMF] Collected {programs_all.shape[0]} program vectors "
        f"for consensus clustering.",
        flush=True,
    )

    # ----- KMeans clustering in program space -----
    kmeans = KMeans(
        n_clusters=K,
        random_state=int(cfg.random_state),
        n_init=int(cfg.consensus_kmeans_n_init),
        verbose=0,
    )
    labels = kmeans.fit_predict(programs_all)       # (R*K,)
    centers = kmeans.cluster_centers_              # (K, G) in normalized space

    # ----- Stability metrics per consensus program -----
    program_counts = np.bincount(labels, minlength=K)  # [K]

    run_coverage = np.zeros(K, dtype=np.float32)  # [K]
    for r in range(R):
        start = r * K
        end = (r + 1) * K
        labels_r = labels[start:end]  # cluster IDs for run r's K programs
        present = np.unique(labels_r)
        run_coverage[present] += 1.0
    run_coverage /= float(R)

    print(
        "[CNMF] Stability summary: "
        f"coverage min={run_coverage.min():.3f}, "
        f"median={np.median(run_coverage):.3f}, "
        f"max={run_coverage.max():.3f}",
        flush=True,
    )

    # ----- Build consensus W and apply coverage filtering -----
    W_full = centers.T  # (G, K)

    # enforce non-negativity (numerical noise)
    W_full = np.clip(W_full, a_min=0.0, a_max=None)

    if cfg.filter_by_coverage:
        stable_mask = run_coverage >= float(cfg.min_run_coverage)
        if not stable_mask.any():
            print(
                f"[CNMF] WARNING: no programs pass min_run_coverage={cfg.min_run_coverage}; "
                "keeping all programs.",
                flush=True,
            )
            stable_mask = np.ones(K, dtype=bool)
    else:
        stable_mask = np.ones(K, dtype=bool)

    W_consensus = W_full[:, stable_mask]  # (G, K_stable)

    print(
        f"[CNMF] Consensus W_full shape: {W_full.shape}, "
        f"W_consensus (stable) shape: {W_consensus.shape}. "
        f"Mean RMSE across runs: {np.mean(frob_rmse_runs):.4f}",
        flush=True,
    )

    gene_names = np.array(ad.var_names)[gene_idx]

    result = CNMFResults(
        W_consensus=W_consensus.astype(np.float32),
        programs_all=programs_all.astype(np.float32),
        cluster_assignments=labels.astype(np.int32),
        run_coverage=run_coverage.astype(np.float32),
        gene_names=gene_names,
        cell_idx=cell_idx.astype(np.int32),
        config=cfg,
    )

    return result


def save_cnmf_result(
    out_dir: Path,
    ad: sc.AnnData,
    result: CNMFResults,
    prefix: str = "k562_cnmf",
) -> None:
    """
    Save consensus NMF artifacts:

      - {prefix}_W_consensus.npy      : [G, K_stable]
      - {prefix}_programs_all.npy     : [R*K, G]
      - {prefix}_cluster_labels.npy   : [R*K]
      - {prefix}_genes.npy            : [G]
      - {prefix}_cells.npy            : [N_used] (indices into ad.obs_names)
      - {prefix}_run_coverage.npy     : [K]
      - {prefix}_manifest.yml         : lightweight YAML manifest
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    W_consensus = result.W_consensus             # [G, K_stable]
    programs_all = result.programs_all           # [R*K, G]
    labels = result.cluster_assignments          # [R*K]
    run_coverage = result.run_coverage           # [K]
    genes = result.gene_names                    # [G]
    cell_idx = result.cell_idx                   # [N_used]

    cells = np.array(ad.obs_names)[cell_idx]

    # main artifacts
    np.save(out_dir / f"{prefix}_W_consensus.npy", W_consensus)
    np.save(out_dir / f"{prefix}_programs_all.npy", programs_all)
    np.save(out_dir / f"{prefix}_cluster_labels.npy", labels)
    np.save(out_dir / f"{prefix}_genes.npy", genes)
    np.save(out_dir / f"{prefix}_cells.npy", cells)
    np.save(out_dir / f"{prefix}_run_coverage.npy", run_coverage)

    # lightweight manifest
    import yaml  # type: ignore

    manifest: Dict[str, Any] = {
        "shape": {
            "W_consensus": list(W_consensus.shape),
            "programs_all": list(programs_all.shape),
        },
        "genes_n": int(genes.size),
        "cells_n": int(cells.size),
        "genes_head": [str(g) for g in genes[:10]],
        "cells_head": [str(c) for c in cells[:10]],
        "run_coverage": run_coverage.tolist(),
        "config": asdict(result.config),
    }

    with (out_dir / f"{prefix}_manifest.yml").open("w") as f:
        yaml.safe_dump(manifest, f)

    print(f"[CNMF] Saved consensus W + manifest to {out_dir}", flush=True)
