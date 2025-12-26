from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scanpy as sc
from anndata import AnnData
import logging

class ManifoldLearner:
    """
    Manifold Learning wrapper using Diffusion Maps to derive
    geometry-aware operators (Laplacian L_M, Metric G).
    """
    def __init__(self, n_neighbors: int = 15, n_pcs: int = 20, sigma: float = 0.0):
        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs
        self.sigma = sigma
        self.adata_ = None

    def fit(self, adata: AnnData) -> ManifoldLearner:
        """
        Compute neighbors and diffusion map on the provided AnnData.
        Stores the processed AnnData internally.
        """
        self.adata_ = adata.copy()
        
        # 1. PCA (if not already present/sufficient)
        if 'X_pca' not in self.adata_.obsm:
             sc.pp.pca(self.adata_, n_comps=self.n_pcs)

        # 2. Neighbors graph
        sc.pp.neighbors(self.adata_, n_neighbors=self.n_neighbors, n_pcs=self.n_pcs)

        # 3. Diffusion Maps
        # sc.tl.diffmap computes the transition matrix T and eigenvectors
        sc.tl.diffmap(self.adata_)
        
        return self

    def get_laplacian(self, mode: str = 'normalized') -> sp.csr_matrix:
        """
        Returns the geometry-aware Laplacian L_M.
        
        L_M is derived from the affinity matrix W (from neighbors).
        Standard graph Laplacian: L = I - D^-1/2 W D^-1/2 (normalized)
        """
        if self.adata_ is None:
            raise ValueError("ManifoldLearner is not fitted. Call fit() first.")
            
        # Scanpy stores connectivities in obsp['connectivities'] or neighbors['connectivities']
        
        W = self.adata_.obsp['connectivities']
        
        # Degree matrix
        degree = np.array(W.sum(axis=1)).flatten()
        
        if mode == 'normalized':
            # D^-1/2
            d_inv_sqrt = np.power(degree, -0.5)
            # Handle division by zero (though degree should be > 0 in connected graph)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            D_inv_sqrt = sp.diags(d_inv_sqrt)
            
            # I - D^-1/2 W D^-1/2
            I = sp.eye(W.shape[0])
            L = I - D_inv_sqrt @ W @ D_inv_sqrt
            
        else: # combinatorial
            D = sp.diags(degree)
            L = D - W
            
        return L

    def get_eigendecomposition(self):
        """Returns diffusion components (eigenvectors) and eigenvalues."""
        if self.adata_ is None or 'X_diffmap' not in self.adata_.obsm:
            raise ValueError("Model not fitted or diffmap failed.")
        
        return self.adata_.obsm['X_diffmap'], self.adata_.uns.get('diffmap_evals', None)
