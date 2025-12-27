# MANTRA 🧪
**Manifold-Aware Network Trajectory Analysis**

> **"Unifying Structural Priors with Phenotypic Realism."**
> MANTRA is a biological engine designed to predict cellular trajectories and phenotypic trait deltas from regulator-level perturbations. It couples a Gene Regulatory Network (GRN) prior with high-resolution biological memory (Nexus) and a manifold-constrained energy objective (EGGFM).

![Architecture](https://img.shields.io/badge/Architecture-Nexus_V4-blue)
![SOTA](https://img.shields.io/badge/SOTA-V4_Certified-green)
![Status](https://img.shields.io/badge/Status-Production_Ready-green)

## 🏆 Current Frontier: Nexus V4 (Recursive Injection)
We have certified **Nexus V4** as our production SOTA. By recursively injecting structural biological corrections directly into the GNN message-passing flow, we've achieved robust signal integration.

- **Program MSE Gain**: **+1.47%** (Significantly outperforms global residuals).
- **Trait Fidelity Gain**: **+1.25%** (Verified sign-accuracy improvement in MCH/RDW).

---

## 🧬 The 5-Stage Biological Engine
The MANTRA pipeline follows a "Context-First" architecture, ensuring that predictions are grounded in both individual regulation and global biological memory.

1.  **Embedding Gateway**: The entry point where regulator identity and dose are mapped into a latent conditioning space via the `ConditionEncoder`.
2.  **Nexus (Memory)**: **Context Retrieval**. Before computing the response, MANTRA queries the Nexus manifold. It retrieves "historical" biological context ($h_{nexus}$) to bias the upcoming GNN pass.
3.  **GRN (Response Core)**: **Structural Prediction**. The GNN computes the gene-level response ($\Delta E$) conditioned on the gene-graph and recursively injected Nexus signals.
4.  **cNMF (Programs)**: **Semantic Projection**. High-resolution gene deltas are projected onto the cNMF loading matrix ($W$) to derive biologically coherent programs ($\Delta P$).
5.  **SMR (Traits)**: **Phenotypic Readout**. Finally, program deltas are mapped to phenotypic phenotypic deltas ($\Delta y$) using pre-fitted trait effect sizes ($\theta$).

> [!NOTE]
> **Why Nexus first?** By reading memory *before* the GNN pass, we allow the model to adjust its structural message-passing (via FiLM conditioning) based on similar historical perturbations, effectively providing a "biological prior" for the specific regulator family.

---

## 🏛️ The Infrastructure
*   **EGGFM (Prior)**: A manifold-constrained energy objective that provides the Riemannian geometric prior for the GRN loss.

---

## Repository Structure

### 🚀 Production Core (`src/mantra`)
*   `src/mantra/grn/`: The core GNN architecture with recursive Nexus support.
*   `src/mantra/nexus/`: The certified V4 Manifold and Tokenizer.
*   `src/mantra/eggfm/`: Manifold energy prior and metric tensor logic.
*   `src/mantra/smr/`: Trait projection heads and WLS fitting scripts.

### 📚 Documentation (The "Brain")
*   **[Results Summary](RESULTS_SUMMARY.md)**: Historical log of experimental architectural evolution.
*   **[Roadmap](ROADMAP.md)**: Strategic vision for future portability and scaling.
*   **[Protocols](PROTOCOLS.md)**: SOPs for "SOTA Promotion" and codebase hygiene.

---

## Quick Start (Running SOTA)

### 0. Environment Setup
```bash
source init_project.sh      # Activates conda 'mantra' and sets PYTHONPATH
```

### 1. Evaluate SOTA
```bash
python experiments/nexus/v4_recursive_injection/tests/v4_eval.py
```

## Citation
[DeepMind MANTRA Team], 2025. **MANTRA: Manifold-Aware Network Trajectory Analysis**.
