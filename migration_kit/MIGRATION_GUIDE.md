# The Agentic Research Migration Guide 📦✨

**Target Audience**: AI agents conducting ML research experiments.

This folder (`migration_kit/`) contains the protocol definitions for agent-driven research workflows. If you are an agent starting a new experiment, this is your instruction manual.

> [!TIP]
> **Need Step-by-Step Instructions?**
> If you are setting up this repository or a new experiment for the first time, read **[`SETUP.md`](SETUP.md)** first. It contains the exact commands to execute.

## 1. Why This Structure Exists (The "Why")

You (the agent) face three primary challenges:
1.  **Context Window Limits**: You can't hold the entire codebase in memory. Deep nesting (`experiments/v1/baseline/models/transformer.py`) wastes your token budget.
2.  **Context Loss Between Sessions**: Each session, you start fresh. Without proper documentation, you'll repeat the previous agent's mistakes.
3.  **Code Rot**: Importing across experiment versions creates brittle dependencies. When `v67` imports from `v65`, and someone deletes `v65`, everything breaks.

**Our Solution**:
- **Flat, Descriptive Roots**: `experiments/v1_baseline` instead of `experiments/v1/baseline` → More semantic meaning in fewer tokens
- **Self-Contained Units**: Each experiment has its own `models/`, `logs/`, `journal.md` → You can `view_file` one experiment without loading 66 others
- **Copy, Don't Import**: Duplicate model files across versions → No cross-version dependencies, clean isolation

## 2. The Setup (The "How")

> [!IMPORTANT]
> **Executable Instructions**
> For the exact commands to initialize this repository or a new experiment, see **[`SETUP.md`](SETUP.md)**.
>
> **Summary of Setup Phase**:
> 1.  **Repo Migration**: Copying templates and establishing the "Descriptive Root" structure.
> 2.  **Experiment Initialization**: Creating the self-contained "Cell" (`models/`, `logs/`, `journal.md`).

---

## 3. Maintenance Tools 🛠️

Use these scripts (via `run_command`) to keep the repo clean:

- **`cleanup_structure.py`**: Automatically moves loose files (media, models, logs) into correct subfolders
- **`audit_structure_template.py`**: Generates a report of files violating the protocol

**When to run**: After completing an experiment, before calling `notify_user`.

---

## 4. The Golden Rules 🌟

**These are HARD requirements. Breaking them will cause failures in future sessions.**

1.  **Journal First**: Before calling `run_command` to launch training, call `write_to_file` to initialize `journal.md`.
2.  **Audit First**: Before fixing a bug, call `write_to_file` to create `tests/repro_[BUG].py`. Document the failure mode.
3.  **Mini-Train**: Before launching a 6-hour training run, call `run_command` with `--steps 200` to verify the pipeline works.
4.  **Descriptive Roots**: When calling `run_command` with `mkdir`, use `experiments/v5_attention_fix`, NOT `experiments/v5`.

> [!IMPORTANT]
> **For Agents**: Load [`AGENT_CHEATSHEET.md`](AGENT_CHEATSHEET.md) into your context at the start of every experiment session. It contains the complete workflow checklist.

**Remember**: Your session is stateless. The journal is your memory across sessions. Write EVERYTHING down.
