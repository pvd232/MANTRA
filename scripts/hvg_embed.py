#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, SpectralEmbedding

# Optional PHATE
try:
    import phate  # type: ignore
except ImportError:
    phate = None  # type: ignore


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute multiple dimensionality reductions on a QC’d AnnData with HVGs "
            "and store them as embeddings in .obsm."
        )
    )
    p.add_argument(
        "--ad",
        type=str,
        required=True,
        help="Input AnnData file (QC’d, with HVGs annotated).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output AnnData file (default: overwrite --ad).",
    )
    p.add_argument(
        "--n-components",
        type=int,
        default=20,
        help="Target embedding dimensionality for most methods (default: 20).",
    )
    p.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="k for kNN-based methods (diffmap, UMAP, PHATE, Isomap, spectral).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--hvg-trunc",
        type=int,
        default=None,
        help=(
            "Number of HVGs to keep in 'X_hvg_trunc'. "
            "Default: use n-components if not set."
        ),
    )
    p.add_argument(
        "--only-hvg-phate",
        action="store_true",
        help=(
            "Run ONLY HVG-truncated embedding and PHATE (if installed), "
            "skipping PCA/diffmap/UMAP/Isomap/spectral."
        ),
    )
    return p


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def to_dense(x):
    """Convert sparse -> dense; ensure numpy array."""
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def ensure_hvg_view(ad: sc.AnnData) -> sc.AnnData:
    """
    Return an AnnData subset to HVGs (if annotated), otherwise a full copy.

    All embeddings are computed on this HVG view, but stored in the
    original AnnData (ad.obsm[...] on the full object).
    """
    if "highly_variable" in ad.var:
        mask = ad.var["highly_variable"].to_numpy().astype(bool)
        n_hvg = int(mask.sum())
        if n_hvg > 0:
            print(f"[HVG] Using HVGs only: n_vars = {n_hvg}", flush=True)
            return ad[:, mask].copy()
        else:
            print(
                "[HVG] 'highly_variable' present but no genes flagged; "
                "using all genes."
            )
            return ad.copy()
    else:
        print("[HVG] No 'highly_variable' flag; using all genes.", flush=True)
        return ad.copy()


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------


def add_hvg_truncated(
    ad: sc.AnnData,
    ad_hvg: sc.AnnData,
    n_genes: int,
    key: str = "X_hvg_trunc",
) -> None:
    """
    'HVG truncated' embedding: keep top genes by normalized dispersion
    (or first n) and store raw expression for those genes in .obsm[key].

    Shape: (n_cells, n_genes)
    """
    print(f"[HVG-trunc] Computing HVG-truncated embedding (n_genes={n_genes})", flush=True)

    if "dispersions_norm" in ad_hvg.var:
        disp = ad_hvg.var["dispersions_norm"].to_numpy()
        order = np.argsort(disp)[::-1]  # descending
        keep_idx = order[: min(n_genes, ad_hvg.n_vars)]
    else:
        print(
            "  No 'dispersions_norm' in ad.var; taking first "
            f"{min(n_genes, ad_hvg.n_vars)} HVGs."
        )
        keep_idx = np.arange(min(n_genes, ad_hvg.n_vars))

    X_hvg = ad_hvg.X[:, keep_idx]
    X_hvg = to_dense(X_hvg).astype(np.float32)

    ad.obsm[key] = X_hvg
    ad.uns[f"{key}_genes"] = ad_hvg.var_names[keep_idx].tolist()
    print(f"  Stored '{key}' with shape {X_hvg.shape}.", flush=True)


def add_pca_scanpy(
    ad: sc.AnnData,
    ad_hvg: sc.AnnData,
    n_components: int,
    seed: int,
) -> None:
    """
    PCA via Scanpy (ARPACK-based on covariance). Stores ad.obsm['X_pca'].
    """
    print(f"[PCA] Computing Scanpy PCA with n_comps={n_components}", flush=True)
    sc.tl.pca(
        ad_hvg,
        n_comps=n_components,
        svd_solver="arpack",
        random_state=seed,
    )
    X_pca = ad_hvg.obsm["X_pca"][:, :n_components].astype(np.float32)
    ad.obsm["X_pca"] = X_pca
    print("  Stored 'X_pca' with shape", X_pca.shape, flush=True)


def add_neighbors_diffmap_umap(
    ad: sc.AnnData,
    ad_hvg: sc.AnnData,
    n_components: int,
    n_neighbors: int,
    seed: int,
) -> None:
    """
    Build kNN graph on PCA space (from ad_hvg), then:
      - Diffusion Map → ad.obsm['X_diffmap']
      - UMAP         → ad.obsm['X_umap']
    """
    print(
        f"[neighbors+DM+UMAP] neighbors: n_neighbors={n_neighbors}, "
        f"based on X_pca; n_comps={n_components}",
        flush=True,
    )

    sc.pp.neighbors(
        ad_hvg,
        n_neighbors=n_neighbors,
        use_rep="X_pca",
        random_state=seed,
    )

    print("  Computing Diffusion Map...", flush=True)
    sc.tl.diffmap(ad_hvg, n_comps=n_components)
    X_dm = ad_hvg.obsm["X_diffmap"][:, :n_components].astype(np.float32)
    ad.obsm["X_diffmap"] = X_dm
    print("  Stored 'X_diffmap' with shape", X_dm.shape, flush=True)

    print("  Computing UMAP...", flush=True)
    sc.tl.umap(
        ad_hvg,
        n_components=n_components,
        random_state=seed,
    )
    X_umap = ad_hvg.obsm["X_umap"].astype(np.float32)
    ad.obsm["X_umap"] = X_umap
    print("  Stored 'X_umap' with shape", X_umap.shape, flush=True)


def add_phate(
    ad: sc.AnnData,
    ad_hvg: sc.AnnData,
    n_components: int,
    n_neighbors: int,
    seed: int,
) -> None:
    """
    PHATE embedding stored as ad.obsm['X_phate'], if phate is installed.
    """
    if phate is None:
        print("[PHATE] phate not installed; skipping.", flush=True)
        return

    print(
        f"[PHATE] Computing PHATE (n_components={n_components}, knn={n_neighbors})",
        flush=True,
    )
    X = to_dense(ad_hvg.X)
    ph = phate.PHATE(
        n_components=n_components,
        knn=n_neighbors,
        n_jobs=-1,
        random_state=seed,
    )
    X_phate = ph.fit_transform(X).astype(np.float32)

    ad.obsm["X_phate"] = X_phate
    print("  Stored 'X_phate' with shape", X_phate.shape, flush=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = build_argparser().parse_args()

    np.random.seed(args.seed)

    in_path = Path(args.ad)
    out_path = Path(args.out) if args.out is not None else in_path

    print(f"[load] Reading AnnData from {in_path} ...", flush=True)
    ad = sc.read_h5ad(in_path)

    # Work on HVG subset for all manifold methods
    ad_hvg = ensure_hvg_view(ad)

    # 1) HVG truncated (top genes by dispersion)
    hvg_trunc_n = args.hvg_trunc or args.n_components
    add_hvg_truncated(ad, ad_hvg, n_genes=hvg_trunc_n)

    if args.only_hvg_phate:
        # Just HVG-trunc + PHATE, skip everything else
        add_phate(
            ad,
            ad_hvg,
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
            seed=args.seed,
        )
    else:
        # 2) PCA (Scanpy)
        add_pca_scanpy(ad, ad_hvg, n_components=args.n_components, seed=args.seed)

        # 3) Diffusion Map + UMAP (kNN graph on PCA)
        add_neighbors_diffmap_umap(
            ad,
            ad_hvg,
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
            seed=args.seed,
        )

        # 4) PHATE (optional if installed)
        add_phate(
            ad,
            ad_hvg,
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
            seed=args.seed,
        )

      

    print(f"[save] Writing updated AnnData with embeddings to {out_path} ...", flush=True)
    ad.write(out_path)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
