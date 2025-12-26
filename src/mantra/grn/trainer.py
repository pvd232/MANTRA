# src/mantra/grn/trainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from mantra.grn.models import GRNGNN, TraitHead
from mantra.grn.config import GRNTrainConfig, GRNLossConfig
from mantra.nexus.adapter import NexusAdapter


class GRNTrainer:
    """
    Trainer for the GRN GNN block with an arbitrary energy prior.

    Usage:
        trainer = GRNTrainer(
            grn_model=grn,
            trait_head=trait_head,
            A=A,
            x_ref=x_ref,
            W=W,
            energy_prior=hvg_prior or embed_prior,
            loss_cfg=loss_cfg,
            train_cfg=train_cfg,
            device="cuda",
        )
        trainer.fit(train_loader, val_loader)
    """
    def __init__(
        self,
        grn_model: GRNGNN,
        trait_head: Optional[TraitHead],
        A,
        x_ref,
        W,
        energy_prior: nn.Module,
        loss_cfg: GRNLossConfig,
        train_cfg: GRNTrainConfig,
        nexus_adapter: Optional[NexusAdapter] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.grn_model = grn_model.to(self.device)
        self.trait_head = trait_head.to(self.device) if trait_head is not None else None

        self.A = A.to(self.device)
        self.x_ref = x_ref.to(self.device)
        self.W = W.to(self.device)

        self.energy_prior = energy_prior.to(self.device)
        self.loss_cfg = loss_cfg
        self.train_cfg = train_cfg
        self.nexus_adapter = nexus_adapter.to(self.device) if nexus_adapter is not None else None

        params = list(self.grn_model.parameters())
        if self.trait_head is not None:
            params += list(self.trait_head.parameters())

        self.optimizer = optim.Adam(
            params,
            lr=float(train_cfg.lr),          # force-cast in case YAML gave a string
            weight_decay=train_cfg.weight_decay,
        )
        
        self.scheduler = None
        if getattr(train_cfg, "use_cosine_lr", False):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg.max_epochs,
                eta_min=float(train_cfg.cosine_eta_min),
            )

    # ------------------- public API -------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        cfg = self.train_cfg

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improve = 0

        for epoch in range(cfg.max_epochs):
            train_stats = self.train_epoch(train_loader)

            # log current LR
            curr_lr = self.optimizer.param_groups[0]["lr"]
            msg = (
                f"[GRN] Epoch {epoch+1}/{cfg.max_epochs} "
                f"train_loss={train_stats['loss']:.4f} "
                f"expr={train_stats['L_expr']:.4f} "
                f"geo={train_stats['L_geo']:.4f} "
                f"prog={train_stats['L_prog']:.4f} "
                f"trait={train_stats['L_trait']:.4f} "
                f"lr={curr_lr:.2e}"
            )

            if val_loader is not None:
                val_stats = self.eval_epoch(val_loader)
                val_loss = val_stats["loss"]
                val_expr = val_stats["L_expr"]

                msg += f" | val_loss={val_loss:.4f} val_expr={val_expr:.4f}"

                # early stopping on expression only (as you have)
                improved = val_expr + cfg.early_stop_min_delta < best_val_loss
                if improved:
                    best_val_loss = val_expr
                    best_state = self._snapshot_state()
                    self.best_model_state = best_state["grn"]
                    self.best_trait_state = best_state.get("trait_head")
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
            else:
                train_expr = train_stats["L_expr"]
                improved = train_expr + cfg.early_stop_min_delta < best_val_loss
                if improved:
                    best_val_loss = train_expr
                    best_state = self._snapshot_state()
                    self.best_model_state = best_state["grn"]
                    self.best_trait_state = best_state.get("trait_head")
                epochs_without_improve = 0

            print(msg, flush=True)

            # NEW: step scheduler once per epoch
            if self.scheduler is not None:
                self.scheduler.step()

            if (
                val_loader is not None
                and cfg.early_stop_patience > 0
                and epochs_without_improve >= cfg.early_stop_patience
            ):
                print(
                    f"[GRN] Early stopping at epoch {epoch+1} "
                    f"(best_val_loss={best_val_loss:.4f})",
                    flush=True,
                )
                break

        if best_state is not None:
            self._load_state(best_state)
            
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.grn_model.train()
        if self.trait_head is not None:
            self.trait_head.train()

        total = 0
        sum_loss = sum_expr = sum_geo = sum_prog = sum_trait = 0.0

        for batch in loader:
            stats = self._forward_batch(batch, train_mode=True)

            bsz = batch["reg_idx"].shape[0]
            total += bsz
            sum_loss  += stats["loss"].item()    * bsz
            sum_expr  += stats["L_expr"].item()  * bsz
            sum_geo   += stats["L_geo"].item()   * bsz
            sum_prog  += stats["L_prog"].item()  * bsz
            sum_trait += stats["L_trait"].item() * bsz

        return {
            "loss":   sum_loss  / total,
            "L_expr": sum_expr  / total,
            "L_geo":  sum_geo   / total,
            "L_prog": sum_prog  / total,
            "L_trait":sum_trait / total,
        }

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.grn_model.eval()
        if self.trait_head is not None:
            self.trait_head.eval()

        total = 0
        sum_loss = sum_expr = sum_geo = sum_prog = sum_trait = 0.0

        for batch in loader:
            stats = self._forward_batch(batch, train_mode=False)

            bsz = batch["reg_idx"].shape[0]
            total += bsz
            sum_loss  += stats["loss"].item()    * bsz
            sum_expr  += stats["L_expr"].item()  * bsz
            sum_geo   += stats["L_geo"].item()   * bsz
            sum_prog  += stats["L_prog"].item()  * bsz
            sum_trait += stats["L_trait"].item() * bsz

        return {
            "loss":   sum_loss  / total,
            "L_expr": sum_expr  / total,
            "L_geo":  sum_geo   / total,
            "L_prog": sum_prog  / total,
            "L_trait":sum_trait / total,
        }

    # ------------------- internals -------------------

    def _forward_batch(self, batch: Dict[str, torch.Tensor], train_mode: bool) -> Dict[str, torch.Tensor]:
        batch = {k: v.to(self.device) for k, v in batch.items()}

        out = compute_grn_losses(
            model=self.grn_model,
            A=self.A,
            batch=batch,
            x_ref=self.x_ref,
            energy_prior=self.energy_prior,
            W=self.W,
            loss_cfg=self.loss_cfg,
            trait_head=self.trait_head,
            nexus_adapter=self.nexus_adapter,
        )
        loss = out["loss"]

        if train_mode:
            self.optimizer.zero_grad()
            loss.backward()
            if self.train_cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.grn_model.parameters())
                    + ([] if self.trait_head is None else list(self.trait_head.parameters())),
                    self.train_cfg.grad_clip,
                )
            self.optimizer.step()

        return out

    def _snapshot_state(self):
        state = {
            "grn": self.grn_model.state_dict(),
        }
        if self.trait_head is not None:
            state["trait_head"] = self.trait_head.state_dict()
        return state

    def _load_state(self, state):
        self.grn_model.load_state_dict(state["grn"])
        if self.trait_head is not None and "trait_head" in state:
            self.trait_head.load_state_dict(state["trait_head"])


def compute_grn_losses(
    model: GRNGNN,
    A: torch.Tensor,                           # [G, G]
    batch: dict[str, torch.Tensor],
    x_ref: torch.Tensor,                       # [G]
    energy_prior: nn.Module,                      # EnergyScorerPrior
    W: torch.Tensor,                           # [G, K]
    loss_cfg: GRNLossConfig,
    trait_head: Optional[nn.Module] = None,
    nexus_adapter: Optional[NexusAdapter] = None,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device

    reg_idx = batch["reg_idx"].to(device)      # [B]
    deltaE_obs = batch["deltaE"].to(device)    # [B, G]

    dose = batch.get("dose", None)
    if dose is not None:
        dose = dose.to(device)

    # 1) ΔE prediction
    deltaE_pred = model(reg_idx=reg_idx, dose=dose, A=A)  # [B, G]

    # 2) Expression loss
    L_expr = ((deltaE_pred - deltaE_obs) ** 2).mean()

    # 3) Geometric prior (frozen EGGFM)
    if loss_cfg.lambda_geo != 0.0:
        x_hat = x_ref.unsqueeze(0) + deltaE_pred   # [B, G]

        # energy at current prediction
        energy = energy_prior(x_hat)               # [B]

        # energy at reference control point (cacheable)
        if energy_ref is None:
            with torch.no_grad():
                energy_ref = energy_prior(x_ref.unsqueeze(0)).mean()

        rel_energy = energy - energy_ref           # [B]
        rel_energy_pos = torch.relu(rel_energy)    # only penalize high-energy states

        L_geo = float(loss_cfg.lambda_geo) * rel_energy_pos.mean()
    else:
        L_geo = torch.zeros((), device=device)    

    # 4) Program-level supervision
    deltaP_pred = deltaE_pred @ W              # [B, K]

    # [Nexus Integration]
    if nexus_adapter is not None:
        deltaP_corr = nexus_adapter(reg_idx=reg_idx, dose=dose)
        deltaP_pred = deltaP_pred + deltaP_corr

    L_prog = torch.zeros((), device=device)
    if "deltaP_obs" in batch and loss_cfg.lambda_prog != 0.0:
        deltaP_obs = batch["deltaP_obs"].to(device)
        L_prog = loss_cfg.lambda_prog * ((deltaP_pred - deltaP_obs) ** 2).mean()
        
    # 5) Trait head (optional)
    L_trait = torch.zeros((), device=device)
    if trait_head is not None and "deltaY_obs" in batch and loss_cfg.lambda_trait != 0.0:
        deltaY_obs = batch["deltaY_obs"].to(device)
        deltaY_pred = trait_head(deltaP_pred)
        L_trait = loss_cfg.lambda_trait * ((deltaY_pred - deltaY_obs) ** 2).mean()

    L_total = L_expr + L_geo + L_prog + L_trait

    return {
        "loss": L_total,
        "L_expr": L_expr.detach(),
        "L_geo": L_geo.detach(),
        "L_prog": L_prog.detach(),
        "L_trait": L_trait.detach(),
        "deltaE_pred": deltaE_pred.detach(),
        "deltaP_pred": deltaP_pred.detach(),
    }
