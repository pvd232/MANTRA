#!/usr/bin/env python
# src/mantra/grn/make_adj.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scanpy as sc  # type: ignore
import torch


def make_grn_adj(
    ad_path: Path,
    energy_ckpt_path: Path,
    out_path: Path,
    *,
    corr_thresh: float = 0.2,
    topk: Optional[int] = 10,
) -> Dict[str, Any]:
    """
    Build a gene–gene adjacency matrix A [G, G] in the EGGFM HVG space.

    The adjacency is constructed from gene–gene Pearson correlations
    across cells in `ad`, aligned to the HVG gene ordering in the
    EGGFM checkpoint:

        1) Load EGGFM checkpoint to get HVG gene list / ordering.
        2) Align the AnnData var_names to that ordering.
        3) Compute gene–gene Pearson correlations across cells.
        4) Take |corr| as similarity, threshold by corr_thresh.
        5) Optionally keep only top-k neighbors per gene (row-wise).
        6) Row-normalize so each row sums to 1.0, with self-loops
           for isolated genes.

    Parameters
    ----------
    ad_path:
        Path to QC'd unperturbed AnnData (e.g. data/interim/k562_gwps_unperturbed_qc.h5ad)
        that lives in the same gene space as the EGGFM HVGs.
    energy_ckpt_path:
        Path to EGGFM .pt checkpoint with 'var_names' or 'feature_names'
        defining the HVG gene list and ordering.
    out_path:
        Path to .npy file to write adjacency matrix A [G, G].
    corr_thresh:
        Minimum absolute Pearson correlation to keep an edge.
    topk:
        If not None and >0, keep at most top-k non-zero neighbors per
        gene (per row) before row-normalization.

    Returns
    -------
    dict with basic stats:
        - G_eff: number of genes in adjacency (after mapping)
        - corr_thresh
        - topk
        - n_edges: number of non-zero entries in A
        - out_path: path to saved .npy
    """
    ad_path = Path(ad_path)
    energy_ckpt_path = Path(energy_ckpt_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[adj] loading AnnData: {ad_path}", flush=True)
    ad = sc.read_h5ad(str(ad_path))
    print(
        f"[adj] AnnData loaded: n_obs={ad.n_obs}, n_vars={ad.n_vars}",
        flush=True,
    )

    # ---------- 1) Load EGGFM checkpoint to get HVG genes ----------
    print(f"[adj] loading energy checkpoint: {energy_ckpt_path}", flush=True)
    ckpt = torch.load(energy_ckpt_path, map_location="cpu")
    if "var_names" in ckpt:
        hvg_genes = np.array(ckpt["var_names"])
    elif "feature_names" in ckpt:
        hvg_genes = np.array(ckpt["feature_names"])
    else:
        raise KeyError(
            "Checkpoint missing 'var_names'/'feature_names'; "
            "cannot infer HVG gene list for adjacency."
        )

    G_ckpt = hvg_genes.shape[0]
    print(f"[adj] n_HVG from checkpoint: G_ckpt={G_ckpt}", flush=True)

    # ---------- 2) Map HVG genes into AnnData var_names ----------
    var_full = np.array(ad.var_names)
    gene_to_idx: Dict[str, int] = {g: i for i, g in enumerate(var_full)}

    hvg_idx_full = []
    missing_genes = []
    for g in hvg_genes:
        idx = gene_to_idx.get(g)
        if idx is None:
            missing_genes.append(g)
        else:
            hvg_idx_full.append(idx)

    if missing_genes:
        print(
            f"[adj][warn] {len(missing_genes)} HVG genes from checkpoint not in AnnData.var_names. "
            f"Examples: {missing_genes[:10]}",
            flush=True,
        )

    hvg_idx_full_np = np.array(hvg_idx_full, dtype=int)
    G_eff = int(hvg_idx_full_np.shape[0])
    print(
        f"[adj] using G_eff={G_eff} genes after mapping into AnnData",
        flush=True,
    )
    if G_eff == 0:
        raise RuntimeError(
            "No HVG genes from checkpoint found in AnnData var_names!"
        )

    ad_view = ad[:, hvg_idx_full_np].copy()  # now var_names aligned to EGGFM genes

    # ---------- 3) Materialize expression matrix X [N_cells, G_eff] ----------
    X = ad_view.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    n_cells, G = X.shape
    print(f"[adj] X shape for correlation: {X.shape}", flush=True)

    # ---------- 4) Compute gene–gene Pearson correlation ----------
    # corr[g1, g2] = corr across cells (columns are genes)
    print("[adj] computing Pearson correlations...", flush=True)
    corr = np.corrcoef(X, rowvar=False)  # [G, G]
    corr = np.nan_to_num(corr, nan=0.0)

    # ---------- 5) Build adjacency by |corr| + threshold + top-k ----------
    A = np.abs(corr)            # similarity in [0,1]
    np.fill_diagonal(A, 0.0)    # remove self-edge from raw corr

    # threshold
    thresh = float(corr_thresh)
    if thresh > 0.0:
        A[A < thresh] = 0.0
    print(
        f"[adj] applied corr_thresh={thresh:.3f}; "
        f"non-zeros pre-topk={int((A > 0).sum())}",
        flush=True,
    )

    # optional top-k per row
    if topk is not None and int(topk) > 0:
        k = int(topk)
        print(f"[adj] applying top-k sparsification with k={k}", flush=True)
        for i in range(G):
            row = A[i]
            # indices where row > 0
            nz_mask = row > 0
            if nz_mask.sum() > k:
                # kth largest value among non-zero entries
                nz_vals = row[nz_mask]
                cutoff = np.partition(nz_vals, -k)[-k]
                # zero out anything below cutoff
                row[row < cutoff] = 0.0
                A[i] = row

    # ---------- 6) Row-normalize and ensure self-loops for isolated ----------
    row_sums = A.sum(axis=1, keepdims=True)
    isolated = (row_sums[:, 0] == 0.0)

    if isolated.any():
        n_iso = int(isolated.sum())
        print(
            f"[adj] {n_iso} genes were isolated; adding self-loops.",
            flush=True,
        )
        idx_iso = np.where(isolated)[0]
        A[idx_iso, idx_iso] = 1.0  # set diagonal entries for isolated genes
        row_sums = A.sum(axis=1, keepdims=True)

    A = A / row_sums


    n_edges = int((A > 0).sum())
    print(
        f"[adj] final adjacency shape={A.shape}, non-zeros={n_edges}",
        flush=True,
    )

    np.save(out_path, A.astype(np.float32))
    print(f"[adj] saved adjacency to {out_path}", flush=True)

    return {
        "G_eff": G_eff,
        "corr_thresh": thresh,
        "topk": int(topk) if topk is not None else None,
        "n_edges": n_edges,
        "out_path": out_path,
    }
