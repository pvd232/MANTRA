# Debugging Guide & Best Practices 🐞

**"If you don't audit the gradients, you are training on hope."**

This document outlines the standard fixes for common pathologies in [PROJECT_TYPE] models.

---

## 1. The Physiology of Training (Weights & Gradients)

### A. The Weight Update Ratio
We must ensure weights are actually moving, but not too fast.
$$ Ratio = \frac{\eta \cdot \|\nabla W\|}{\|W\|} $$
-   **Target**: **1e-3** (The "Goldilocks Zone").
-   **< 1e-5**: Model is frozen (Check Learning Rate, Detached Graphs).
-   **> 1e-1**: Model is unstable (Check Clipping, Batch Size).

### B. Gradient Norm Topology
-   **Norm = 0.0**: **Broken Graph**. Check `requires_grad`, `.detach()`, or index operations.
-   **Norm > 10.0**: **Exploding**. Use `clip_grad_norm_`.
-   **Norm = NaN**: **Overflow**. Check for division by zero or `log(0)`.

---

## 2. Shape Hygiene 📐

**Symptom**: `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)`.

### Best Practices
1.  **Assert Inputs**:
    ```python
    B, T, D = x.shape
    assert D == self.hidden_dim, f"Mismatch: {D} != {self.hidden_dim}"
    ```
2.  **Use Einops**: Avoid `x.view()`. Use `rearrange(x, 'b t (h d) -> ...')` to be explicit.

---

## 3. Convergence Pathology 📉

1.  **The "Hockey Stick"**: Loss drops instantly then flatlines.
    *   *Diagnosis*: LR too high. Local Fast Basin.
2.  **The "Slow Burn"**: Loss decreases linearly.
    *   *Diagnosis*: LR too low.
3.  **The "NaN Death"**: Loss becomes NaN at step N.
    *   *Diagnosis*: Optimization instability or Normalization Drift.

---

## 4. GPU/Hardware Tips 💻
-   **OOM (Out of Memory)**: Reduce Batch Size, use Gradient Accumulation.
-   **Slow Training**: Check CPU-GPU sync points (e.g. `print(tensor)` inside loop).
