import sys
import os
import pytest
import numpy as np
from anndata import AnnData

# Ensure we can import from models/
current_dir = os.path.dirname(os.path.abspath(__file__))
# tests/ -> v02_manifold/
exp_root = os.path.dirname(current_dir)
sys.path.append(exp_root)

# MOCK TORCH so we can import from models without it
import types
torch_mock = types.ModuleType("torch")
torch_mock.nn = types.ModuleType("torch.nn")
torch_mock.nn.Module = object
class MockTensor:
    pass
torch_mock.Tensor = MockTensor
sys.modules["torch"] = torch_mock
sys.modules["torch.nn"] = torch_mock.nn

from models.manifold import ManifoldLearner

@pytest.fixture
def synthetic_adata():
    # 100 cells, 50 genes
    X = np.random.rand(100, 50).astype(np.float32)
    return AnnData(X=X)

def test_manifold_fit(synthetic_adata):
    learner = ManifoldLearner(n_neighbors=5, n_pcs=10)
    learner.fit(synthetic_adata)
    
    assert learner.adata_ is not None
    assert 'X_diffmap' in learner.adata_.obsm
    assert 'connectivities' in learner.adata_.obsp

def test_get_laplacian(synthetic_adata):
    learner = ManifoldLearner(n_neighbors=5, n_pcs=10)
    learner.fit(synthetic_adata)
    
    L = learner.get_laplacian(mode='normalized')
    
    # Check shape [N, N]
    assert L.shape == (100, 100)
    
    # Check it's a sparse matrix
    import scipy.sparse as sp
    assert sp.issparse(L)

def test_unfitted_error():
    learner = ManifoldLearner()
    with pytest.raises(ValueError):
        learner.get_laplacian()
