# MANTRA: Technical Exposition of the Nexus Integration

The project uses a tripartite model architecture with downstream readouts:
1.  **The GNN ($f_\theta$)**: Performs message-passing on a gene-gene interaction graph to predict $\Delta E$.
2.  **The EGGFM (Energy Prior)**: Ensures the predicted $\Delta E$ remains on a biologically stable manifold.
3.  **The Nexus (The Oracle)**: A **Global Residual Corrector** that provides additive gains in Program Space.
4.  **The cNMF Projection**: Maps high-dimensional Gene Deltas ($\Delta E$) to a compressed **Program Space** ($\Delta P \in \mathbb{R}^K$).
5.  **The SMR Readout**: Maps Program Deltas ($\Delta P$) to **Trait Expression Deltas** ($\Delta T$) using SMR-derived effect sizes.

---

## 2. Mathematical Framework

### 2.1 The GNN Baseline
The GNN predicts a coarse latent activation vector $a_{GNN} \in \mathbb{R}^K$:
$$a_{GNN} = \text{GNN}(R, D | \mathcal{G})$$
where $\mathcal{G}$ is the prior gene interaction graph.

### 2.2 The Nexus Global Reservoir (V2)
Nexus functions as a global residual corrector. Unlike NLP transformers that use episodic memory, Nexus V2 utilizes a **State-Persistent Manifold** (CAM).

The Nexus correction $\Delta a_{Nexus}$ is retrieved via:
$$\Delta a_{Nexus} = \text{Retrieve}(\text{Embed}(R, D), \mathcal{M})$$
where $\mathcal{M}$ is the Global Manifold. The training objective for Nexus is the **Residual Loss**:
$$\mathcal{L}_{Nexus} = \| \Delta a_{obs} - (a_{GNN} + \Delta a_{Nexus}) \|^2$$

---

## 3. The "Biological Gap": Why Nexus?
GNNs are excellent at **Generalizing**. However, they are terrible at **Memorizing Exceptions**.
*   **The Problem**: If we force the GNN to learn these exceptions, it "overfits" and loses its ability to generalize.
*   **The Nexus Solution**: Nexus acts as a **Global Residual Cache**. It "offloads" the memorization task. If a regulator has a weird, non-generalizable signature, Nexus captures that fact in its memory slots and serves it as an additive correction.

---

## 4. Deep Dive: `nexus_v2` (The Global Reservoir)

### 4.1 From Episodic to Persistent (`Global Cache`)
In standard NLP, Transformers have "Episodic" memory. In `nexus_v2`, the memory is **Persistent**. The **Centroid-Addressable Manifold (CAM)** is a single, global buffer with shape `(1, Slots, Hidden)`. 

### 4.2 Honest Header Indexing (The Causality Barrier)
The model consumes records formatted as: `[REG] [DOSE] [STATE] | [PROGRAM CONTENT]`.
*   **The Fix**: We only ever predict from **Index 2** (the `STATE` token). 
*   **The Logic**: This forces the model to perform a "Retrieval" from its memory slots using only the metadata (`Reg/Dose`).

### 4.3 The Sequential Priming (The "Warming" Pass)
We use a **Sequential Warming Pass** (BSZ=1) to populate the CAM with high-fidelity fingerprints of every known perturbation.

---

## 5. Core Mechanics Deep-Dive

### 5.1 Unsupervised Gating (The "Regret" Mechanic)
Nexus uses a **Surprisal-driven Gating** system. It evaluates whether a perturbation is "memorable" by measuring the model's surprise (Entropy) during a forward pass:
- **Entropy Tracking**: For every regulator token, Nexus maintains a Running EMA of prediction entropy ($\mu$) and variance ($\sigma^2$).
- **Gating Logic**: A memory commitment is triggered if:
$$ \frac{g_{entropy} - \mu}{\sigma} \times (1 - \text{sim}_{max}) > \gamma_{regret} $$
- **Significance**: This ensures Nexus only commits states it cannot already explain or retrieve. It prevents the manifold from being flooded with redundant information.

### 5.2 Hybrid Retrieval: Lexical Hash & Semantic Search
Nexus breaks the "Search" problem into two orthogonal stages:

1.  **Stage 1: Lexical Bucketing (The Hash)**
    - **Mechanism**: The regulator ID is passed through a deterministic modulo operation: `bucket_idx = reg_id % n_buckets`.
    - **Purpose**: This acts as a **Lexical Hash** that guarantees "biological identicals" (same regulator) are always routed to the same memory neighborhood. It transforms a global search into a local one ($32$ slots instead of $16,000$).
2.  **Stage 2: Semantic Alignment (The Search)**
    - **Mechanism**: Within the bucket, Nexus performs a Dot-Product Attention between the **Unified Query** and the **Slot Keys**.
    - **Query Fusion**: The query is a weighted sum: 
      $$ Q_{unified} = \alpha \cdot H_{semantic} + (1-\alpha) \cdot C_{lexical} $$
      where $H_{semantic}$ is the transformer's latent state and $C_{lexical}$ is a learned centroid anchor.
    - **Purpose**: This allows the model to find the most "semantically relevant" memory slot for the current cell-state context, even among multiple memories for the same regulator.

### 5.3 Prioritized Read (Honest Match)
If a stored Token ID in the manifold exactly matches the incoming `reg_id`, Nexus performs a **Hard Read** (Lexical override). This allows for near-perfect reconstruction of known biological "Facts," while the Semantic Search handles the "Nuance."

---

## 6. Performance Analysis & Roadmap

### 6.1 Why +1.02% is just the beginning
While 1% sounds humble, in biological program space ($K=75$), this represents significant additive gains on **Outlier Regulators**. The GNN is already a very "strong" biological generator; Nexus is currently acting as a **Safety Net** for known exceptions.

### 6.2 Scaling for 10x Gains
To move from +1% to +10%, we are planning the following architectural pivots:
1.  **Functional Slotting**: Instead of bucket mapping by *index*, we will bucket by *functional class* (e.g., all Zinc Finger TFs share buckets). This will allow Nexus to generalize "memory" of one transcription factor to another closely related one.
2.  **Recursive Correction**: Instead of an additive $\Delta P$ at the end of the pipeline, we will feed the Nexus retrieval back into the GNN's message-passing layers.
3.  **Multi-head Memory**: Implementing multiple manifolds for different biological scales (Pathway-level vs. Gene-level).

---

## 5. Data Flow & Shapes (Technical Summary)

| Stage | Input Shape | Model | Output Shape | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `[B]` | - | `reg_idx`, `dose` | Perturbation condition. |
| **GNN** | `[B]` | `GRNGNN` | $\Delta E_{gnn}$ $[B, G]$ | $G \approx 20,000$ genes. |
| **cNMF** | $\Delta E_{gnn}$ | $W_{cnmf}$ | $\Delta P_{gnn}$ $[B, K]$ | $K = 75$ latent programs. |
| **Nexus Q**| `[B, 3]` | `Tokenizer` | `tokens` $[B, 3]$ | `[REG, DOSE, STATE]` headers. |
| **Nexus K**| `tokens` | `NexusAdapter` | $\Delta P_{corr}$ $[B, K]$ | Residual memory retrieval. |
| **Combine** | $\Delta P_{gnn}$ + $\Delta P_{corr}$ | - | $\Delta P_{final}$ $[B, K]$ | Stabilized program vector. |
| **SMR** | $\Delta P_{final}$ | `TraitHead` | $\Delta T$ $[B, T]$ | Final trait expression deltas. |

---
**Status**: Integrated as a constructive stabilizer within the cNMF-SMR cascade.
