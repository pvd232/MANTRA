# Research Protocols & Standard Operating Procedures 🛡️

**"Slow is Smooth, Smooth is Fast."**

This document defines the engineering standards required to contribute to this project. Violations of these protocols lead to "Code Rot" and are unacceptable.

---

## 1. The "Experiment-First" Architecture 🧪

We do not use a monolithic `src/` folder. We use an **Experiment-First** structure.

### The Standard Layout
Every experiment MUST live in `experiments/v{VERSION}_{DESCRIPTION}/`.
Inside that folder, it MUST be self-contained:
-   `models/`: A COPY of the model file used. Do not import from previous versions.
-   `logs/`: Training logs.
-   `audits/`: JSON results and validation reports.
-   `tests/`: Unit tests and diagnostic scripts specific to this version.
-   `journal.md`: The narrative log of the experiment (The "Captain's Log").
-   `train.py`: The entry point script.

**Why?**
This allows us to delete or archive any old experiment without breaking the current SOTA. It decouples progress.

### B. The Global Root Strategy 🌍
While `experiments/` is for *change*, the Root is for *permanence*.
-   **`scripts/`**: **V-Agnostic Utilities**. Data preparation, global plotting, or repo maintenance tools. (If it imports a specific model, it belongs in `experiments/`).
-   **`docs/`**: **Deep Knowledge**. Mathematical proofs, architectural whitepapers, and guides that outlive any single version.
-   **`data/`**: **Immutable Source**. Raw datasets. Experiments treat this as Read-Only.
-   **`configs/`**: **Shared Baselines**. Global configuration constants (if needed).
-   **`final_report/`**: **The Publication**. When a milestone is reached, polished figures and summaries are copied here for external consumption (e.g. arXiv/GitHub pages).

---

## 2. The Agentic Workflow 🤖

### A. Matrix Mode (Deep Work)
When entering a complex task:
1.  **Plan**: Create `implementation_plan.md`.
2.  **Execute**: Write code, strictly following the plan.
3.  **Verify**: Run `audit.py` or `tests/`.
4.  **Document**: Update `journal.md` and `walkthrough.md`.

### B. The "Mini-Train" Rule
Before launching a full run:
1.  Run the training script for **200 steps** (or 1 epoch on small data).
2.  **Verify Convergence**: Does loss go down?
3.  **Verify Artifacts**: Are checkpoints saving? Are logs writing?
*Never launch a 2-day job without a 2-minute test.*

### C. The "Mini-Audit" Rule
Before claiming success or running a benchmark:
1.  **Constraint**: Must run in **< 5 minutes**.
2.  **Task**: A simplified, synthetic version of the Hard Task (e.g. "Toy Recall" vs "Book Recall").
3.  **Threshold**: Must hit **100% Accuracy** (or equivalent 0.0 Loss).
*If it fails the Toy Task, it will fail the Real Task. Fail fast.*

---

## 3. Code Hygiene 🧹

-   **Explicit Imports**: Avoid `from model import *`.
-   **Config Separation**: Hyperparameters should be at the top of the script or in a config object, not buried in loops.
-   **Assertion Guardrails**: Use `assert` statements aggressively to validate tensor shapes at the start of `forward()`.

## 4. Final Report Protocol 📝

When an experiment concludes:
1.  **Success**: Mark as SOTA in `LEADERBOARD.md`. Merge findings into `README.md`.
2.  **Failure**: Document *why* in `journal.md`. Move folder to `archive/` if it is clutter.
3.  **Handoff**: Write a summary in `journal.md` for the next agent.

---

## 5. Git Hygiene & Version Control 🐙

### A. The "Heavy File" Ban
**Never** commit:
-   Checkpoints (`.pt`, `.ckpt`)
-   Large Logs (`.log`, `.txt` > 1MB)
-   Datasets (`.bin`, `.jsonl`)
**Why?**: Bloats the repo size forever. Use `.gitignore` religiously.

### B. The "folder-as-branch" Strategy
In this research repo, we prefer **New Folders (`experiments/vXX`)** over **Git Branches** for experimental variations.
-   **Benefit**: You can diff `v1/models/model.py` vs `v2/models/model.py` directly in the editor.
-   **Benefit**: Multiple experiments can run simultaneously on the same machine without switching git branches (which would break running jobs).

### C. The "Clean Start" Rule
**Before** launching a training run:
1.  **Commit your code**.
2.  Log the **Git Hash** in your `journal.md` or training log.
*This ensures that if the run is a success 3 days later, you know EXACTLY what code produced it.*

---

## 6. Multi-Environment & Hardware Layout 🖥️☁️

We operate in a **Hybrid Compute** environment.

### A. The Agent Environment (GCP Compute Engine / L4)
*   **Role**: **Home Base**. Agents live here.
*   **Capabilities**: Coding, Logic, Mini-Trains (L4 GPU, **22GB vRAM**).
*   **Rule**: ALL code is written and versioned here.

### B. The external "Heavy Lift" (Google Colab / A100 Cluster)
*   **Role**: **Muscle**. User manually runs massive jobs here.
*   **Capabilities**: High-RAM Training (A100 GPU, **80GB vRAM**).
*   **Process**:
    1.  **Export**: User zips the code from Agent Env.
    2.  **Train**: User runs job on Colab.
    3.  **Import**: User uploads logs/checkpoints back to Agent Env via `gdown`.

### C. The "Airlock" Protocol
When handling external results:
1.  **Never** manually edit code on Colab and forget to update the Agent.
2.  **Always** bring the `log.txt` back to `experiments/vXX/logs/` so the Agent can analyze it.
