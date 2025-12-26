# Metrics & Methodology 📏

**"You get what you measure."**

This document defines the ground truth metrics for [PROJECT_NAME].

---

## 1. The North Star Metrics ⭐

| Metric | Unit | Description | Success Threshold |
|---|---|---|---|
| **[METRIC_A]** | [UNIT] | [Description, e.g., Validation Loss] | < [VALUE] |
| **[METRIC_B]** | [UNIT] | [Description, e.g., Recall @ K] | > [VALUE]% |
| **[METRIC_C]** | [UNIT] | [Description, e.g., Throughput] | > [VALUE] / sec |

---

## 2. The Audit Levels 🕵️

We do not trust "training loss". We run explicit audits.

### Level 1: The Smoke Test 💨
*   **Goal**: Ensure the code runs without crashing.
*   **Dataset**: Random/Synthetic tensor data.
*   **Steps**: 10 forward/backward passes.
*   **Pass**: Exit Code 0.

### Level 2: The Diagnostic Audit 🩺
*   **Goal**: Verify the model can learn [SIMPLE_TASK].
*   **Dataset**: [OVERFIT_DATASET].
*   **Pass**: 100% Accuracy / 0.0 Loss.

### Level 3: The SOTA Certification 🏅
*   **Goal**: Prove superiority over the baseline.
*   **Dataset**: Full [TEST_SET].
*   **Pass**: Outperform [PREVIOUS_SOTA] on [METRIC_A].

---

## 3. Terminology Dictionary 📖

*   **[TERM_1]**: [Definition].
*   **[TERM_2]**: [Definition].
*   **[TERM_3]**: [Definition].
