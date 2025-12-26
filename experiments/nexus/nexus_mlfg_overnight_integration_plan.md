# NEXUS × MANTRA (MLFG) — Overnight Integration & Implementation Plan

This document is a **do-it-tonight** plan to drop your current Nexus implementation in as an **independent module** inside MANTRA, train it on **your existing MLFG artifacts**, and wire it into your **regulator → gene → program → trait** pipeline as an additive “rare-signal capture + state-aware retrieval” adapter.

It is written to match the Nexus design in the latest paper (CAM + PSS + surprisal-gated writes + residual fusion). fileciteturn2file0  
It also assumes you’re using your existing `HybridTransformerV68b` / `CentroidAddressableManifold` implementation as the core engine. fileciteturn2file1

---

## 0) Goal and success criterion (keep it tight)

### Goal
Add a **NEXUS memory layer** that improves MANTRA’s robustness on:
- rare regulators / rare programs
- subtle state-conditional effects
- noisy / sparse deltas

…**without** rewriting your core science model.

### “Overnight success” definition
By tomorrow morning you should have:
1. A **trained Nexus module** on MLFG-derived token sequences (or supervised residuals).
2. A **drop-in adapter** that produces a correction in **program space** (preferred) or gene space.
3. A **single ablation** showing at least one of:
   - improved **trait sign accuracy** / calibration
   - better performance specifically on **rare/long-tail** regulators or programs
   - reduced over-smoothing artifacts after manifold projection

---

## 1) Where Nexus sits in your current pipeline (exactly)

Your current MANTRA-style forward pass is conceptually:

1. **GRN-GNN**: `ΔE_gnn = GNN(reg_idx, dose, A)`
2. **Manifold constraint**: `ΔE_final = Project_manifold(ΔE_gnn)`  *(EGGFM / Laplacian / geometry)*
3. **Programs**: `Δa = Wᵀ ΔE_final`
4. **Trait readout**: `ΔTrait = θᵀ Δa`

### Recommended insertion point (A): **between (1) and (2)**
Nexus is best used as a **retrieval/correction adapter** before your manifold operator:

- `ΔE_pre = ΔE_gnn + ΔE_corr`
- `ΔE_final = Project_manifold(ΔE_pre)`

This is where rare perturbation signatures tend to get blurred by global smoothing.

### Preferred correction space: **program space**
Instead of correcting a full gene vector, predict **Δa_corr** and map back:

- `Δa_pre = Δa_gnn + Δa_corr`  
- `ΔE_corr = W Δa_corr`

This is faster, smaller, and stays aligned with your interpretability story.

---

## 2) Map Nexus paper concepts to MLFG objects (so your write gate is meaningful)

Nexus (latest paper) defines:
- Local window transformer `L=1024`
- CAM memory with **bucketed slots** `(B × S)`
- Deterministic lexical bank routing `b(x)=x mod B`
- **Surprisal Z-score** gating from language head entropy
- PSS eviction via priority score `P = λs s + λr r + λn n`
- Residual fusion of retrieved value into local state  
(“working memory + long-term memory”). fileciteturn2file0

### In MLFG, define “tokens” so the above has a direct meaning
A perturbation example becomes a short “record”:

**Header tokens**
- `<REG=r>`
- `<DOSE=d_bin>`
- `<STATE=s>` *(optional tonight; you can set all to 0 if needed)*

**Content tokens (recommended)**
- **Program delta tokens** from observed program change `Δa_obs` (or `deltaP_obs`):
  - pick top `P` programs by |Δa|
  - bin magnitudes into (say) 9 bins and keep sign
  - token format: `<PROG=k,SIGN=+/-,BIN=b>`

Optional later:
- gene tokens (top-K genes) for finer retrieval

This gives you a tractable vocabulary for lexical hashing (`token_id mod B`) and a surprisal signal from a language head.

---

## 3) Data you need (you already have almost all of it)

### Minimum fields per perturbation example
From your aggregated NPZ (you mentioned keys like `(reg_idx, deltaE, deltaP_obs, deltaY_obs, dose)`):
- `reg_idx`: `[N]`
- `dose`: `[N]`
- `deltaP_obs`: `[N, K_prog]`  *(or compute via `Wᵀ deltaE` if needed)*
- optional: `deltaY_obs`: `[N, T_traits]`
- optional: `deltaE_obs`: `[N, G]`

**Tonight’s simplification:** you can ignore `STATE` and set `STATE=0` everywhere.
If you *do* have a state embedding already, use a k-means bin ID.

---

## 4) Directory structure inside MANTRA (clean “independent module” drop-in)

Add a new module folder:

```
src/mantra/nexus/
  __init__.py
  hybrid_transformer_v68b.py        # your current implementation
  tokenizer.py                      # MLFG tokenization + vocab
  datasets.py                       # NPZ -> token stream / examples
  adapter.py                        # NexusAdapter used by MANTRA pipeline
  train.py                          # training entrypoint (or scripts/)
  configs.py                        # dataclasses / defaults
```

Keep it isolated so you can rip it out cleanly.

---

## 5) Two viable “overnight” training modes (choose one)

### Mode A (paper-faithful): **Token LM with surprisal-gated CAM writes**
- Train Nexus as a **language model** over your perturbation token stream.
- The CAM gate uses entropy/surprisal as in the paper. fileciteturn2file0
- After training, the memory contains rare program tokens and the reader can retrieve them.

**Pros:** matches your Nexus story.  
**Cons:** to get a *correction vector*, you still need a small decoding head.

### Mode B (recommended overnight): **Supervised residual correction (fast, clean)**
Train Nexus as a “retrieval-conditioned regressor/classifier”:
- Inputs: the record tokens + retrieved CAM value
- Output target: `Δa_resid = Δa_true − Δa_pred_baseline` (or directly `Δa_true`)
- Loss: MSE on Δa (or top-P programs), plus optional auxiliary LM loss

**Pros:** fastest path to a measurable MANTRA gain.  
**Cons:** slightly less “pure” w.r.t. the paper framing (still uses CAM + PSS + retrieval).

If your goal is MLFG deliverable value **tomorrow**, do **Mode B**.

---

## 6) Build the MLFG token vocabulary (tonight, minimal)

### Token types and counts
Let:
- `R` = # regulators used in the perturb dataset
- `D` = # dose bins (e.g., 4)
- `S` = # state bins (e.g., 1 tonight)
- `K` = # programs
- `BINS` = magnitude bins (e.g., 9)
- `SIGNS` = 2

Vocabulary size:
- `R + D + S + (K * BINS * SIGNS) + special_tokens`

Special tokens:
- `<BOS> <EOS> <ENDREC> <PAD>`

### Recommended record length
- Header: 3 tokens
- Content: `P=8–16` program tokens
- Tail: `<ENDREC>`
Total record length ≈ 12–20 tokens → very cheap.

---

## 7) Dataset builder (from NPZ) — exact output files

### Input
- `train_npz`: your existing MANTRA training NPZ with deltas
- `val_npz`: optional

### Output artifacts
1) `out/nexus/vocab.json`
2) `out/nexus/train.bin` (or `.pt`) token stream **or** per-example token tensors
3) `out/nexus/val.bin`
4) Optional metadata:
   - `out/nexus/example_index.parquet` mapping token offsets → (reg,dose,idx)

### Construction procedure
For each example `i`:
1. `reg = reg_idx[i]`
2. `dose_bin = bucket(dose[i])`
3. `Δa = deltaP_obs[i]`
4. select top `P` program indices by `abs(Δa)`
5. for each selected program:
   - sign = sign(Δa_k)
   - mag_bin = bucket(|Δa_k|)
   - emit token `<PROG=k,SIGN, BIN>`
6. emit `<ENDREC>`

Concatenate records in random order.

---

## 8) Training config that matches your Nexus implementation (sanity defaults)

Start with something close to the paper’s scale but reduced for bio tokens:

- `d_model = 256`
- `n_layers = 4`
- `L_window = 1024` (irrelevant if your sequences are tiny, but keep)
- CAM geometry:
  - **safe**: `n_buckets=512, slots_per_bucket=32`
  - **aggressive**: `n_buckets=2048, slots_per_bucket=8`
  - **extreme**: `n_buckets=4096, slots_per_bucket=4` (paper’s “width beats depth”) fileciteturn2file0
- `tau` (soft attention temp) ≈ 0.1
- `alpha_val` (lexical/semantic blend) ≈ 1.0 if you rely on pure semantic within-bank read (per paper discussion)

Training:
- bf16 if possible
- batch size tuned to GPU
- 1–3 epochs over your token stream is usually enough for the memory behavior to emerge (given tiny vocab)

---

## 9) Wire it into MANTRA: `NexusAdapter` API (what MANTRA calls)

### Adapter inputs (from MANTRA training step)
At minimum:
- `reg_idx: [B]`
- `dose: [B]`
- `deltaA_pred: [B, K]` *(from your current GRN + manifold baseline, or from GNN+W)*
- optional: `state_bin: [B]`

### Adapter outputs
- `deltaA_corr: [B, K]` *(or sparse top-P + magnitudes)*
- optional diagnostics:
  - write density
  - gate statistics
  - collision/bucket occupancy summaries

### Combine in MANTRA
- `deltaA_pre = deltaA_pred + deltaA_corr`
- `deltaTrait = thetaᵀ deltaA_pre`

Or if you correct genes:
- `deltaE_pre = deltaE_pred + deltaE_corr`

---

## 10) Minimal evaluation (tomorrow morning)

You only need **one** clean plot/table to justify additivity:

### Suggested eval slices
1) **Rare regulators**: define rare = bottom quantile of frequency in train
2) **Rare programs**: bottom quantile by activation frequency

### Metrics (choose 2–3)
- Trait:
  - sign accuracy / AUROC
  - calibration slope or correlation
- Program-level:
  - cosine similarity of Δa_pred vs Δa_obs
  - top-P overlap / Spearman on program ranks

### Ablation table (tiny)
- baseline MANTRA
- MANTRA + Nexus (no writes / memory off)
- MANTRA + Nexus (memory on)

If “memory off” ≈ baseline but “memory on” improves rare slice → you win.

---

## 11) Concrete “tonight checklist” (the whole run)

### A. Prep
- [ ] Confirm NPZ keys exist: `reg_idx`, `dose`, `deltaP_obs` (and ideally `deltaY_obs`)
- [ ] Decide `P` (top programs per record) and `BINS`
- [ ] Create output dir: `out/nexus/`

### B. Build data
- [ ] Build vocab + tokenized records
- [ ] Save `vocab.json`, `train.bin`, `val.bin`

### C. Train
- [ ] Train Nexus in Mode B (supervised residual) first
- [ ] Save checkpoint: `out/nexus/model.pt`
- [ ] Dump diagnostics: write density, bucket occupancy

### D. Integrate
- [ ] Add `NexusAdapter` to MANTRA training step (one line: `deltaA_corr = nexus(...)`)
- [ ] Re-run a small evaluation batch

### E. Reportable artifact
- [ ] Make an ablation table + a single rare-slice metric plot

---

## 12) Pitfalls and fast fixes (so you don’t lose the night)

### If you don’t have `deltaP_obs`
Compute it:
- `deltaP_obs = Wᵀ deltaE_obs` (use your existing cNMF `W`)

### If state bins aren’t ready
Set `STATE=0`. Add state routing later.

### If the model “writes too much”
Increase threshold / tighten gate:
- raise `regret_gamma` or increase surprisal Z threshold

### If the model never writes
Lower threshold, or increase priming writes early in stream (your V68b already has a priming mask concept).

---

## Appendix: Why this matches the Nexus paper (so you can justify it)
Your latest Nexus write/read loop is:
- deterministic bank routing `b(x)=x mod B`
- surprisal-gated writes (entropy-derived Z-score)
- discrete slot eviction by priority (PSS)
- differentiable within-bank reads + residual fusion into local state fileciteturn2file0

This MLFG integration preserves those mechanics, but swaps “natural language tokens” for **biological record tokens** representing perturbation context and program effects.

---

## Confidence
**0.80** that you can get a working integration + a meaningful *rare-slice* improvement by tomorrow **if** your aggregated NPZ contains stable `deltaP_obs` and your baseline predictions are non-trivial.
