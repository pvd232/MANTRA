# Experiment Journal: V5 Multi-Scale Memory

## Phase 3.0: Hierarchical Decoupling (Initial Run)
- **Status**: **FAILED** (Regression: -5.19%)
- **Hypothesis**: The NLP-inspired <15% density target would force the dual manifolds to pick up strictly informative features.
- **Outcome**: 
    - **Fine Density**: 4.5% (Good)
    - **Coarse Density**: 0.02% (Failed)
- **Pathology**: **Semantic Starvation**. The Coarse manifold (Pathway-level) is too stable to trigger Z-scored surprisal gates under current dataset conditions. It stayed empty, starving the GNN of hierarchical priors.

## Phase 3.1: Relaxed Gating (Salvage Pivot)
- **Status**: **IN PROGRESS**
- **Strategy**: 
    - **Coarse Manifold**: Switch to **High-Density (100% during warming)** to establish a global pathway baseline.
    - **Fine Manifold**: Maintain **Z-scored Regret (15%)** to capture gene-level outliers.
## Phase 3.2: Recursive Multi-Scale (The Matrix Pivot)
- **Status**: **PLANNED**
- **Critical Insight**: The 1.47% gain in V4 came from **Recursive Structural Conditioning**. V5.0 and V5.1 failed because they treated hierarchical priors as post-hoc residuals. A biological engine must use priors as **Message-Passing Biases**.
- **The Plan**: Inject the Dual-Manifold context directly into the GNN hidden layers.
- **SMR Bridge**: Validate not on MSE, but on Trait Prediction fidelity ($\Delta P \to \Delta y$).
