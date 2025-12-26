# Agent Setup & Migration Instructions 📥

**Target Audience**: Agents initiating the migration process or setting up a new experiment.

This document outlines the **exact steps** to adopt the "Experiment-First" workflow.

---

## ⚠️ Flexibility Disclaimer
**Adapt to Your Environment**:
Every project is different. While this kit assumes a flat structure, you **MUST** adapt these paths if your project uses a different layout (e.g., `src/` for library code, `lib/` for shared modules).

*   **Standard**: `scripts/cleanup_structure.py`
*   **Adaptation**: `src/scripts/cleanup_structure.py` (if that's where you keep scripts)

**Do not break existing imports.** If moving files, check for `sys.path` implications.

---

## 1. Migration Protocol (First Time Setup)

If this repository does not yet use the Migration Kit structure:

### Step 1: Install Infrastructure
Use `run_command` to copy the core templates to the root:
```bash
cp migration_kit/README_TEMPLATE.md README.md
cp migration_kit/PROTOCOLS_TEMPLATE.md PROTOCOLS.md
cp migration_kit/MANIFEST_TEMPLATE.md PROJECT_MANIFEST.md
cp migration_kit/gitignore_template .gitignore
```

### Step 2: Initialize Governance
Establish the project tracking files:
1.  **Roadmap**:
    ```bash
    cp migration_kit/ROADMAP_TEMPLATE.md ROADMAP.md
    ```
    *Action*: Edit `ROADMAP.md` to define phases and North Star metrics.

2.  **Leaderboard**:
    ```bash
    # Create empty leaderboard
    echo "| Version | Metric A | Metric B | Notes |" > experiments/LEADERBOARD.md
    echo "|---|---|---|---|" >> experiments/LEADERBOARD.md
    ```

3.  **Methodology**:
    ```bash
    cp migration_kit/METRICS_TEMPLATE.md experiments/METRICS_METHODOLOGY.md
    cp migration_kit/DEBUGGING_TEMPLATE.md experiments/DEBUGGING_GUIDE.md
    ```

### Step 3: Establish "Descriptive Roots"
Identify the current working directory or "latest" experiment.
1.  Create a new, properly named folder: `mkdir -p experiments/v1_baseline`
2.  Move current working files there: `mv model.py train.py experiments/v1_baseline/`
3.  **Verify** imports still work (run a smoke test).

---

## 2. Experiment Initialization (Routine)

**Every time you start a NEW experiment (`vXX`), follow this sequence:**

### Step 1: Create the "Cell"
Create your self-contained directory:
```bash
mkdir -p experiments/v68_lane_unification/models
mkdir -p experiments/v68_lane_unification/logs
mkdir -p experiments/v68_lane_unification/tests
```

### Step 2: Clone the DNA
**Copy, Do Not Import**. Copy the model files from the *previous* best version to your new folder.
```bash
cp experiments/v67_regret_gating/models/*.py experiments/v68_lane_unification/models/
```
*Why? This isolates your experiment. You can now modify `v68` code without breaking `v67`.*

### Step 3: Initialize the Journal (Crucial)
Copy the template to your new folder:
```bash
cp migration_kit/journal_template.md experiments/v68_lane_unification/journal.md
```
*Action*: Open `journal.md`. Fill in the **Hypothesis** immediately.

### Step 4: Create the Smoke Test
Copy the smoke test template and adapt it:
```bash
cp migration_kit/smoke_test_template.py experiments/v68_lane_unification/tests/smoke_test_v68.py
```
*Action*: Edit the file. Update imports to point to `../models`. Run it.
**Hard Rule**: If `smoke_test_v68.py` fails, DO NOT START TRAINING.

---

## 3. The "Clean Exit"

When you are finished with your migration or setup tasks:

1.  **Delete Stub Files**: If you created temporary scripts to move files around, delete them.
    ```bash
    rm cleanup_migration.py temp_move.sh
    ```
2.  **Commit**:
    ```bash
    git add .
    git commit -m "MIGRATION: Adopted Agentic Workflow v1.0"
    ```

**You are now ready to begin Science.**
