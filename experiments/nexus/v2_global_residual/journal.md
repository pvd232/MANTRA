# Experiment V2: Global Residual Correctors (Mode B++)

## Hypothesis
By framing Nexus as a "Global Residual Corrector" that operates outside the GNN message-passing loop, we can eliminate circular dependencies and force the model to focus on correcting baseline errors.

## Walkthrough: The V2 Protocol
1. **Baseline Freeze**: The GRNGNN baseline was frozen.
2. **Residual Target**: Training targets were computed as $R = \Delta P_{obs} - \Delta P_{baseline}$.
3. **Honest Indexing**: Prediction was restricted to the `STATE` token hidden state (Index 2) to ensure the model could not "see" the residual it was supposed to write.
4. **Persistent Memory**: Shifted to a persistent manifold that maintains state across batches, initialized via a **Sequential Priming** pass (BSZ=1).

## Findings: The +1.02% Breakthrough
V2 provided the first successful validation of the Nexus architecture:
- **Baseline MSE**: 0.7525
- **Nexus V2 MSE**: 0.7448
- **Relative Gain**: **+1.02%**

### Analysis
The model successfully "memorized" high-variance residuals (pathological regulators) that the GNN consistently mispredicted. However, the use of **Index-Based Bucketing** meant that the model remained a "Fact Checker"—it could only correct regulators it had seen during training.

## Conclusion & Pivot
V2 confirmed the stability of the Global Residual approach but highlighted the need for **Generalization**.
- **Decision**: Pivot to **Functional Slotting** (V3) to share memory across gene families.
