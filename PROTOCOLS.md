# Research Protocols & Standard Operating Procedures 🛡️

**"Slow is Smooth, Smooth is Fast."**

---

## 1. The "Experiment-to-SOTA" Lifecycle 🔄

We use a two-tiered structure to balance research agility with production stability.

### Tier 1: The Sandbox (`experiments/`)
-   **Purpose**: Rapid prototyping and architectural scaling.
-   **Rule**: Every experiment is self-contained (COPY of models, local tests).
-   **Requirement**: Must have a `journal.md` and a verified `audit.py`.

### Tier 2: The Production Core (`src/mantra/`)
-   **Purpose**: Stable, verified engine for downstream pipelines (SMR, Clinical).
-   **Rule**: Only contains code "Certified" via the Promotion Protocol.

### The Promotion Protocol
To promote a model from `experiments/` to `src/mantra/`:
1.  **Audit**: Run `vXX_eval.py` on the hold-out validation set.
2.  **Metrics**: Performance must be significantly superior to current SOTA (or provide critical biological context).
3.  **Refactor**: Clean up experimental "spaghetti" (hardcoded paths, print statements).
4.  **Handoff**: Merge into `src/mantra/` and update `README.md` leaderboard.

---

## 2. Pipeline Integration Rules 🧬
The "4-Stage MANTRA Engine" follows a recursive dependency chain:
1.  **Pre-requisite**: EGGFM Energy (+Metric Tensor).
2.  **Input Conditioning**: cNMF Programs ($W$) + Nexus Memory ($M$).
3.  **Core Task**: GNN Prediction ($\Delta E$) with Recursive Nexus Injection.
4.  **Output Readout**: SMR Trait Mapping ($\Delta y = \Delta P \cdot \theta$).

**CRITICAL**: When modifying a component in the middle of the chain (e.g., Nexus), you MUST trigger an "Integrated Audit" that measures downstream impact on Trait Fidelity.

---

## 3. The "Mini-Train" Rule 🏃
Before launching a full run:
1.  Run the training script for **200 steps** (or 1 epoch on small data).
2.  **Verify Convergence**: Does loss go down?
3.  **Verify Artifacts**: Are checkpoints saving?
*Never launch a 2-day job without a 2-minute test.*

---

## 4. Code Hygiene & Git 🧼
-   **Explicit Imports**: No `from model import *`.
-   **Heavy File Ban**: No `.pt`, `.npy`, or large `.csv` in git. Use `data/` or `out/`.
-   **Assertion Guardrails**: `assert` tensor shapes in `forward()`.

---

## 5. Multi-Environment Workflow 🖥️
-   **Agent Env (L4/22GB)**: Coding, refactoring, and mini-trains.
-   **Heavy Env (A100/80GB)**: Scaled production training.
-   **Syncing**: Always bring `training_log.txt` back to the experiment folder for analysis.
