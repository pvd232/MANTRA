# Gate Density: A Comprehensive Overview

## 1. Definition & Importance
**Gate Density** is defined as the percentage of tokens in a sequence that are written to the Content-Addressable Manifold (CAM). It effectively measures the **"Write Rate"** of the model.

### Why it matters:
*   **Efficiency**: Higher density means more writes, which translates to more memory operations. In ealier versions (V67), "Read-Before-Write" strategies meant high density significantly slowed down throughput.
*   **Memory Capacity**: With a finite memory buffer, high density leads to **"Greedy Overwrite"**. If the model writes too aggressively (>20-30%), it overwrites its own long-term history too quickly, causing "catastrophic forgetting".
*   **Signal-to-Noise**: A density that is too high implies the model is storing noise. A density that is too low implies it is missing critical information.

---

## 2. Target Values
The project has established clear targets based on the "PG-19 Elbow" analysis.

*   **Hard Target**: **<15%** (Achieved in V68b)
*   **Ideal "Efficiency" Target**: **~8-12%**
*   **Current Reality (V68b)**: **<15%** (Certified SOTA)

| Budget | Approx ID Threshold | Status | Description |
| :--- | :--- | :--- | :--- |
| **>30%** | ID > 1,000 | **Forbidden** | Causes rapid cache eviction ("Greedy Overwrite"). |
| **~23%** | ID > 4,000 | **V67 Sweet Spot** | Required for >80% Common Recall in V67. |
| **<15%** | ID > 10,000 | **The Goal** | Achieved by **V68b**. SOTA efficiency and production deployment. |
| **~8%** | ID > 18,000 | **Stretch Goal** | "SOTA Efficiency" tier. |

---

## 3. Impact on Recall (The Trade-Off)
There is a fundamental tension between **Gate Density** and **Common Sense Recall**.

*   **Rare Recall (Entity Retrieval)**: Generally robust to low density. Rare words (names, unique nouns) have high self-information (surprisal) and are almost always selected by gating mechanisms.
*   **Common Recall ("Common Sense")**: Highly sensitive to density. Common words ("the", "not", "is") have low surprisal.
    *   **At <15% Density**: Earlier models (V66) completely ignored these, leading to **0% Common Recall**. The model became an "Idiot Savant"—remembering names but losing the thread of conversation.
    *   **At ~23% Density**: Models like V67 (Strategy B) recover **~83% Common Recall**.
    *   **The "Recall Cliff"**: Reducing V67 density from 23% to 15% caused Common Recall to plummet from ~83% to <60%.

**The Challenge**: The project's central problem has been finding a way to keep density <15% without falling off the "Recall Cliff" for common tokens.

---

## 4. Evolution of Gate Density

### V65: The Oracle Era (Static)
*   **Mechanism**: Hard-coded filter (`ID > 5000`).
*   **Density**: Fixed at **~18.7%**.
*   **Result**: 
    *   Rare Recall: High.
    *   Common Recall: **0.0%**.
    *   Verdict: Good baseline, but "blind" to context.

### V66: Learned Salience (The "Intelligence" Trap)
*   **Mechanism**: Learned "Surprisal" (Entropy).
*   **Density**: **13.5% - 18.5%** (Constrained by top-K budget).
*   **Result**: 
    *   Optimized for "Surprisal per bit". 
    *   Since rare words are more surprising, it preferentially stored them and starved common words.
    *   Common Recall remained **0.0%**.

### V67: Regret Gating (The "Common Sense" Hunt)
*   **Mechanism**: "Regret" (Current Surprisal > Average Surprisal).
*   **Density**: **~19% - 28%**.
*   **Evolution**:
    *   **V67.2 (Unified IFS)**: ~17% density. Failed (1% recall).
    *   **V67.3 (Stratified)**: **28% density**. Great recall (81%), but too heavy.
    *   **Strategy B (Conditional)**: **23.4% density**. The "Robust Winner" with 83% recall.
    *   **Strategy C (Predictive)**: **15.3% density** (Target hit!), but Rare Recall collapsed to 42%.

### V68b: Lane Unification (Production SOTA)
*   **Mechanism**: **Centroid-Rigid Addressing**. Unified Lexical/Semantic lanes with **Unsupervised Regret Gating**.
*   **Density**: **<15%**.
*   **Result**:
    *   **100% Dictionary Recall**.
    *   **99.9% Common Recall** @ 1M context.
    *   Throughput: **159k tok/s**.
    *   **Status**: Production-Ready.

### V70: Phantom Recurrence (Validation)
*   **Experiment**: Explored "Hole Report" failures in intermediate versions (V69).
*   **Finding**: Confirmed that discrete slots with "Phantom Regret" protections are superior to pure recurrence.
*   **Outcome**: Validated the architectural choices of V68b.

---

## Summary
Gate Density is the throttle of the NEXUS engine. 
*   **Too High**: The engine floods (memory thrashing, slow speed).
*   **Too Low**: The engine stalls (Recall Cliff, loss of meaning).
*   **Current State**: **V68b** has mastered this balance, achieving the **<15% target** while delivering 99.9% Common Recall and 159k tok/s throughput. It stands as the certified Production SOTA.
