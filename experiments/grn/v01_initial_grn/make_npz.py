# src/mantra/grn/make_npz.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scanpy as sc  # type: ignore
from scipy import sparse
import torch


def make_grn_npz(
    ad_raw_path: Path,
    energy_ckpt_path: Path,
    out_dir: Path,
    *,
    reg_col: str = "gene",
    dose_col: str = "gem_group",
    control_value: str = "non-targeting",
    max_pct_mt: float = 0.2,
    min_umi: float = 2000.0,
    min_cells_per_group: int = 10,
    val_frac: float = 0.2,
    seed: int = 7,
    cnmf_W_path: Optional[Path] = None,
    traits_dim: int = 3,
) -> Dict[str, Any]:
    """
    Stream K562 GWPS .h5ad and build train/val NPZs in the EGGFM HVG space.

    Parameters
    ----------
    ad_raw_path:
        Path to big backed AnnData (e.g. data/raw/k562_gwps.h5ad).
    energy_ckpt_path:
        Path to EGGFM checkpoint with `var_names` / `feature_names`.
    out_dir:
        Directory where `train.npz` and `val.npz` will be written.
    reg_col:
        obs column with perturbed target gene / regulator.
    dose_col:
        obs column with dose / gem group (ignored here, but kept for API).
    control_value:
        Value in `reg_col` denoting non-targeting controls.
    max_pct_mt:
        Max allowed mitochondrial fraction for QC.
    min_umi:
        Min UMI_count per cell for QC.
    min_cells_per_group:
        Min # cells for a regulator to be kept.
    val_frac:
        Fraction of regulator-level samples for validation.
    seed:
        RNG seed for train/val split.
    cnmf_W_path:
        Optional path to W.npy [G,K]; if None, uses ΔP_obs = ΔE.
    traits_dim:
        Dimensionality of ΔY_obs stub (e.g. 3 for MCH, RDW, IRF).

    Returns
    -------
    dict with basic stats: N_train, N_val, G, n_regulators_used, etc.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Get HVG genes from EGGFM checkpoint ----------
    print(f"[ckpt] loading energy checkpoint: {energy_ckpt_path}", flush=True)
    ckpt = torch.load(energy_ckpt_path, map_location="cpu")
    if "var_names" in ckpt:
        hvg_genes = np.array(ckpt["var_names"])
    elif "feature_names" in ckpt:
        hvg_genes = np.array(ckpt["feature_names"])
    else:
        raise KeyError(
            "Checkpoint missing 'var_names'/'feature_names'; "
            "cannot infer HVG gene list."
        )
    G = hvg_genes.shape[0]
    print(f"[ckpt] n_HVG from checkpoint: G={G}", flush=True)

    # ---------- 2) Open raw AnnData in backed mode ----------
    print(f"[load] raw AnnData (backed): {ad_raw_path}", flush=True)
    ad_raw = sc.read_h5ad(str(ad_raw_path), backed="r")
    n_cells_raw = ad_raw.n_obs
    print(
        f"[info] raw AnnData: n_obs={ad_raw.n_obs}, n_vars={ad_raw.n_vars}",
        flush=True,
    )

    # Map HVG genes to raw var_names
    var_full = np.array(ad_raw.var_names)
    gene_to_idx: Dict[str, int] = {g: i for i, g in enumerate(var_full)}

    hvg_idx_full = []
    missing_genes = []
    for g in hvg_genes:
        idx = gene_to_idx.get(g)
        if idx is None:
            missing_genes.append(g)
        else:
            hvg_idx_full.append(idx)

    if missing_genes:
        print(
            f"[warn] {len(missing_genes)} HVG genes from checkpoint not in raw var_names. "
            f"Examples: {missing_genes[:10]}",
            flush=True,
        )
    hvg_idx_full_np = np.array(hvg_idx_full, dtype=int)
    G_eff = hvg_idx_full_np.shape[0]
    print(f"[info] using G={G_eff} genes after mapping into raw AnnData", flush=True)
    if G_eff == 0:
        raise RuntimeError("No HVG genes from checkpoint found in raw AnnData var_names!")

    # ---------- 3) Build cell-level QC mask from obs ----------
    obs = ad_raw.obs

    if "mitopercent" not in obs.columns:
        raise ValueError(
            "'mitopercent' not found in obs; "
            f"available columns: {list(obs.columns)}"
        )
    if "UMI_count" not in obs.columns:
        raise ValueError(
            "'UMI_count' not found in obs; "
            f"available columns: {list(obs.columns)}"
        )

    mitopercent = obs["mitopercent"].to_numpy()
    umi_count = obs["UMI_count"].to_numpy()

    mito_ok = mitopercent < float(max_pct_mt)
    umi_ok = umi_count > float(min_umi)

    qc_cells = mito_ok & umi_ok
    print(
        f"[qc] {qc_cells.sum()} / {n_cells_raw} cells pass "
        f"(mitopercent<{max_pct_mt}, UMI_count>{min_umi})",
        flush=True,
    )

    # ---------- 4) Control / perturbed, reg ----------
    if reg_col not in obs.columns:
        raise ValueError(
            f"reg-col '{reg_col}' not in obs; "
            f"available columns: {list(obs.columns)}"
        )
    if dose_col not in obs.columns:
        raise ValueError(
            f"dose-col '{dose_col}' not in obs; "
            f"available columns: {list(obs.columns)}"
        )

    reg_raw = obs[reg_col].to_numpy()
    reg = np.array(reg_raw)

    is_ctrl = (reg == control_value) & qc_cells
    is_pert = (reg != control_value) & qc_cells

    n_ctrl = int(is_ctrl.sum())
    n_pert = int(is_pert.sum())
    print(f"[split] control cells (reg={control_value!r}): {n_ctrl}", flush=True)
    print(f"[split] perturbed cells: {n_pert}", flush=True)
    if n_ctrl == 0:
        raise RuntimeError(
            f"No control cells found with {reg_col} == {control_value!r}"
        )

    # ---------- 5) Global control mean in HVG space (dose-free) ----------
    print("[ctrl] computing GLOBAL control mean in HVG space...", flush=True)

    ctrl_mask = is_ctrl
    n_ctrl_qc = int(ctrl_mask.sum())
    if n_ctrl_qc == 0:
        raise RuntimeError("No control cells after QC filtering!")

    ad_ctrl = ad_raw[ctrl_mask, :].to_memory()
    ad_ctrl_hvg = ad_ctrl[:, hvg_idx_full_np]

    X_ctrl = ad_ctrl_hvg.X
    if sparse.issparse(X_ctrl):
        X_ctrl = X_ctrl.toarray()
    X_ctrl = X_ctrl.astype(np.float32)

    global_ctrl_mean = X_ctrl.mean(axis=0, keepdims=True)  # [1, G_eff]
    print(
        f"[ctrl] global control mean over {n_ctrl_qc} cells; G={global_ctrl_mean.shape[1]}",
        flush=True,
    )

    # ---------- 6) Aggregate ΔE per regulator (ignore dose for K562) ----------
    print("[agg] aggregating ΔE per regulator (dose-free)...", flush=True)

    regs_pert = np.unique(reg[is_pert])
    print(f"[agg] {len(regs_pert)} unique perturbed regulators", flush=True)

    reg_to_idx = {r: i for i, r in enumerate(regs_pert)}

    deltaE_list = []
    reg_idx_list = []
    dose_list = []

    min_cells = int(min_cells_per_group)

    for r in regs_pert:
        mask_r = is_pert & (reg == r)
        n = int(mask_r.sum())
        if n < min_cells:
            continue

        ad_r = ad_raw[mask_r, :].to_memory()
        ad_r_hvg = ad_r[:, hvg_idx_full_np]

        X_r = ad_r_hvg.X
        if sparse.issparse(X_r):
            X_r = X_r.toarray()
        X_r = X_r.astype(np.float32)

        x_r_mean = X_r.mean(axis=0, keepdims=True)   # [1, G_eff]
        deltaE = x_r_mean - global_ctrl_mean         # [1, G_eff]

        deltaE_list.append(deltaE)
        reg_idx_list.append(reg_to_idx[r])
        dose_list.append(0.0)  # dummy dose; GRN has use_dose = False

        if len(deltaE_list) % 500 == 0:
            print(
                f"  [agg] processed {len(deltaE_list)} regulators so far...",
                flush=True,
            )

    if len(deltaE_list) == 0:
        raise RuntimeError("No regulators with enough cells after QC!")

    deltaE = np.vstack(deltaE_list).astype(np.float32)
    reg_idx_arr = np.array(reg_idx_list, dtype=np.int64)
    dose_arr = np.array(dose_list, dtype=np.float32)

    N, G_eff = deltaE.shape
    print(f"[agg] built {N} regulator-level samples; each ΔE has G={G_eff} genes", flush=True)

    # ---------- 7) ΔP_obs via W (optional) ----------
    if cnmf_W_path is not None:
        print(f"[prog] loading cNMF W: {cnmf_W_path}", flush=True)
        W = np.load(cnmf_W_path).astype(np.float32)  # [G_eff, K]
        if W.shape[0] != G_eff:
            raise ValueError(
                f"W has {W.shape[0]} genes but ΔE has {G_eff}; check HVG alignment."
            )
        deltaP = deltaE @ W  # [N, K]
    else:
        print("[prog] no W provided; using ΔP_obs = ΔE", flush=True)
        deltaP = deltaE.copy()

    # ---------- 8) Stub ΔY_obs ----------
    T = int(traits_dim)
    deltaY = np.zeros((N, T), dtype=np.float32)

    # ---------- 9) Train/val split ----------
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    N_val = int(val_frac * N)
    val_idx = perm[:N_val]
    train_idx = perm[N_val:]

    def _save_npz(path: Path, idx: np.ndarray) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            reg_idx=reg_idx_arr[idx],
            deltaE=deltaE[idx],
            deltaP_obs=deltaP[idx],
            deltaY_obs=deltaY[idx],
            dose=dose_arr[idx],
        )
        print(f"[save] {path} (N={len(idx)})", flush=True)

    train_path = out_dir / "train.npz"
    val_path = out_dir / "val.npz"

    _save_npz(train_path, train_idx)
    _save_npz(val_path, val_idx)
    print("[done]", flush=True)

    return {
        "N": int(N),
        "N_train": int(train_idx.size),
        "N_val": int(val_idx.size),
        "G_eff": int(G_eff),
        "n_regulators_used": int(len(deltaE_list)),
        "train_path": train_path,
        "val_path": val_path,
    }
