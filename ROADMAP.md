# Project Roadmap 🗺️

**Last Updated**: 2025-12-22 (Initial Draft based on Interim Report)

This document provides the strategic vision for **MANTRA** (Manifold-Aware Network Trajectory Analysis).

---

## Vision & North Star 🌟

**High-Level Goal**: Predict red blood cell traits from CRISPRi perturbations using a unified objective that couples a GRN prior, manifold-aware filtering, and SMR/TWAS-based trait readouts.

**Success Criteria** (Project-Level):
- [ ] **Trait Concordance**: Validated R^2 and Pearson correlation against K562 baselines.
- [ ] **Dose Monotonicity**: Higher sgRNA-UMI quartiles should yield stronger trait deltas.
- [ ] **Manifold Realism**: Perturbations should respect the geometry of unperturbed cell states (kNN overlap, geodesic preservation).
- [ ] **Portability**: Validation on HCT116 (future).

**Current Status**: **Phase 1 (Foundation)**. Evaluating baseline performance on K562.

---

## Phase Breakdown

### Phase 1: Foundation & Baseline (Current)
**Objective**: Establish the MANTRA architecture on K562 cells and validate against non-manifold baselines.

**Key Components**:
- **GRN Prior**: Derived from GWPS/Perturb-seq.
- **Manifold Learning**: EGGFM-derived Riemannian metric tensor on unperturbed cells.
- **Program Discovery**: cNMF modules.
- **Trait Readout**: SMR/TWAS-informed program weights.

**Key Experiments**:
| Version | Goal | Status | Outcome |
|---|---|---|---|
| `v00_baselines` | Replicate Ota beta-regression & linear baselines | 📅 Planned | Benchmark for improvement. |
| `v01_initial_grn` | Initial GRN migration & smoke test | ✅ Complete | Modular structure established. |
| `v02_manifold` | Learn (M, G) on unperturbed K562 | 📅 Planned | Metric tensor & Laplacian L_M. |
| `v03_mantra_beta` | Full MANTRA pipeline (Eq 2) | 📅 Planned | Validate geom smoothing. |

**Learnings**:
- (To be populated)

**Next Phase Trigger**: MANTRA outperforms linear baselines on K562 validation set.

---

### Phase 2: Dose & Geometry Refinement
**Objective**: Incorporate dose-stratification (Quartiles Q1-Q4) and rigorous manifold realism checks.

**Key Experiments**:
| Version | Goal | Status | Outcome |
|---|---|---|---|
| `v04_dose_strat` | Implement monotonic margin loss | 📅 Planned | Better dose-response consistency. |
| `v05_ablation` | 2x2 Grid (GRN +/- , Manifold +/-) | 📅 Planned | Quantify gain from each component. |
| `v06_hvg_sensitivity` | Test robustness to HVG selection | 📅 Planned | Stability check. |

---

### Phase 3: Portability (HCT116)
**Objective**: Extend the framework to HCT116 cells to demonstrate portability of the manifold-aware approach.

**Planned Milestones**:
- [ ] Generate/Load HCT116 unperturbed manifold.
- [ ] Cross-cell line prediction validation.

---

## Parking Lot 🅿️
- **HCT116 Extension**: Preregistered for final report.
- **Alternative Smoothers**: Compare Tikhonov vs. Diffusion Heat Kernel.

---

## Update Protocol 📝
**After each experiment**:
1. Update "Current Status".
2. Mark relevant experiment as ✅ Complete or ❌ Failed.
3. Add key metrics (R^2, Recall) to the Outcome column.
