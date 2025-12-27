# Experiment V1: Nexus Integration & Initial Pathologies

## Hypothesis
Directly injecting historical biological signatures from a Centroid-Addressable Manifold (CAM) into the GNN's latent space would allow the model to correct its own structural errors based on past observations.

## Walkthrough: The V1 Protocol
1. **Embedding Injection**: Historical gene expression deltas ($\Delta E$) were cached in a CAM and retrieved during the GNN forward pass.
2. **Deterministic Hash**: Retrieval was based on a simple linear hash of the regulator index.
3. **Training Objective**: End-to-end MSE minimization of the reconstructed delta-expression.

## Findings: The Pathology Era
V1 failed to provide a meaningful performance uplift due to two critical pathologies:

### 1. The "Double Counting" Pathology
The model was receiving the historical signature *before* performing its own inference. This created a circular dependency where the model would simply pass through the historical value rather than computing a refined prediction, leading to gradient stagnation.

### 2. "Index Cheating" (Mode A Path)
Because retrieval was deterministic on the regulator index, the transformer blocks learned to "cheat" by memorizing the index-to-target mapping via the retrieval values, rather than performing actual biological reasoning.

## Conclusion & Pivot
V1 proved that direct injection into the hidden layers of a raw GNN is unstable without a residual framework.
- **Decision**: Pivot to **Mode B++** (Global Residual Correctors).
- **Target**: Predicted $\Delta a_{resid} = \Delta a_{obs} - \Delta a_{baseline}$.
