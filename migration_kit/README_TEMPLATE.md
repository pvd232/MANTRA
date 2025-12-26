# [PROJECT_NAME]: [ONE_LINE_PITCH]

> **"[INSPIRATIONAL_QUOTE_OR_MANTRA]"**
> [Brief technical definition of the project's core hypothesis]

![Architecture](https://img.shields.io/badge/Architecture-[ARCH_NAME]-blue)
![SOTA](https://img.shields.io/badge/SOTA-V[XX]_Verified-green)
![Status](https://img.shields.io/badge/Status-Active-green)

## 🏆 Current Frontier: [CURRENT_VERSION]
We are currently actively developing **[CURRENT_VERSION]**, which aims to solve [CURRENT_BLOCKER].

-   **Legacy SOTA (V[XX])**: Achieved [METRIC_A] but failed at [METRIC_B].
-   **Stable Core (V[XX])**: The robust baseline used for comparisons.
-   **Next Gen (V[XX])**: Experimental branch testing [NEW_HYPOTHESIS].

---

## Key Innovations

### 1. [INNOVATION_1_NAME]
[Description of the first major breakthrough].
*   **Result**: [Quantifiable impact, e.g., 2x speedup].

### 2. [INNOVATION_2_NAME]
[Description of the second major breakthrough].
*   **Result**: [Quantifiable impact, e.g., 99% accuracy].

---

## Repository Structure

### 🚀 Active Models
*   `experiments/[EXP_ID]/models/[MODEL_FILE].py`: **The Current Leader**.
*   `experiments/[EXP_ID]/models/`: Contains the specific architecture for that experiment.

### 📚 Documentation (The "Brain")
*   **[Project Overview](RESULTS_SUMMARY.md)**: **MUST READ**. A comprehensive summary of the project goals, history, and SOPs.
*   **[Protocols & Workflow](PROTOCOLS.md)**: Standard Operating Procedures (DoS and DON'Ts).
*   **[Leaderboard](experiments/LEADERBOARD.md)**: The active scoreboard of all experiments.
*   **[Metrics Methodology](experiments/METRICS_METHODOLOGY.md)**: Definitions of success.
*   **[Debugging Guide](experiments/DEBUGGING_GUIDE.md)**: "Medical Textbook" for common failures.

---

## Quick Start (Running SOTA)

```python
import sys
sys.path.append("experiments/[EXP_ID]/models")
from [MODEL_FILE] import [MODEL_CLASS]

# 1. Initialize
model = [MODEL_CLASS](
    # [CONFIG_PARAMS]
)

# 2. Run
# [EXAMPLE_USAGE_CODE]
```

---

## Agent Setup Instructions 🤖

**If you are an AI agent initializing this project for the first time:**

1. **Load Context** (use `view_file`):
   - `README.md` (this file)
   - `ROADMAP.md`
   - `PROTOCOLS.md`
   - `migration_kit/AGENT_CHEATSHEET.md`

2. **Understand Current State**:
   - Check `ROADMAP.md` for current phase
   - Read latest experiment journal: `experiments/v[LATEST]/journal.md`
   - Review `experiments/LEADERBOARD.md` for SOTA

3. **Before Starting Work**:
   - Confirm you understand the North Star metrics
   - Know which experiment version you're working on
   - Load the previous experiment's journal into context

**Your workflow reference is in** [`migration_kit/AGENT_CHEATSHEET.md`](migration_kit/AGENT_CHEATSHEET.md).

---

## Citation
[AUTHORS], [YEAR]. **[PROJECT_TITLE]**.
[REPO_URL]
