# Project Manifest 📜

**Objective**: This is the central registry for project-level assets. It ensures any agent entering the project can find the "Ground Truths" immediately.

---

## 🏗️ The Current Frontier
- **SOTA Model**: `experiments/v[XX]_[NAME]/models/`
- **Current Objective**: [Link to task.md]
- **Active Phase**: [Link to Phase in ROADMAP.md]

## 💾 Core Infrastructure
| Asset | Location | Description |
|---|---|---|
| **Raw Data (Archive)** | `G Drive/data/raw/` | The immutable source datasets |
| **Raw Data (Train)** | `HF/Kaggle Datasets` | Optimized for cluster download speeds |
| **Processed Data** | `data/processed/` | Version-specific processed data (Local SSD) |
| **Master Weights** | `experiments/v[XX]/checkpoints/` | The current champion's weights |
| **Leaderboard** | `experiments/LEADERBOARD.md` | The record of all valid audits |
| **Methods** | `experiments/METRICS_METHODOLOGY.md` | How we define success |

## 💻 Hardware Environment
- **Primary Trainer**: [e.g. A100 80GB]
- **Dev Environment**: [e.g. L4 22GB]
- **Min VRAM Required**: [e.g. 18GB]

## 🗺️ Essential Reading
1. [`README.md`](README.md): High-level overview.
2. [`ROADMAP.md`](ROADMAP.md): Strategic vision and history.
3. [`PROTOCOLS.md`](PROTOCOLS.md): How we operate (The Law).
4. [`AGENT_CHEATSHEET.md`](migration_kit/AGENT_CHEATSHEET.md): Current workflow reference.

---

## 🕵️ Data & Asset Registry
| Hash (SHA-256) | File/Folder | Date Added |
|---|---|---|
| `[HASH]` | `data/raw/dataset_v1.bin` | [DATE] |

---

## 📝 Update Protocol
**When an experiment becomes "SOTA"**:
1. Update `The Current Frontier` section above.
2. Update the `Master Weights` path.
3. If new data was introduced, add to the `Data Registry`.

*This manifest is updated by the transitioning agent at the end of a milestone.*
