# src/mantra/grn/trainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from mantra.grn.models import GRNGNN, TraitHead, GRNLossConfig, compute_grn_losses

@dataclass
class GRNTrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 50
    grad_clip: float = 0.0
    early_stop_patience: int = 0      # 0 = off
    early_stop_min_delta: float = 0.0

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
            energy_fn=hvg_prior or embed_prior,
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
        energy_fn: nn.Module,
        loss_cfg: GRNLossConfig,
        train_cfg: GRNTrainConfig,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.grn_model = grn_model.to(self.device)
        self.trait_head = trait_head.to(self.device) if trait_head is not None else None

        self.A = A.to(self.device)
        self.x_ref = x_ref.to(self.device)
        self.W = W.to(self.device)

        self.energy_fn = energy_fn.to(self.device)
        self.loss_cfg = loss_cfg
        self.train_cfg = train_cfg

        params = list(self.grn_model.parameters())
        if self.trait_head is not None:
            params += list(self.trait_head.parameters())

        self.optimizer = optim.Adam(
            params,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

    # ------------------- public API -------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Simple training loop with optional early stopping on val loss.
        """
        cfg = self.train_cfg

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improve = 0

        for epoch in range(cfg.max_epochs):
            train_stats = self.train_epoch(train_loader)
            msg = (
                f"[GRN] Epoch {epoch+1}/{cfg.max_epochs} "
                f"train_loss={train_stats['loss']:.4f} "
                f"expr={train_stats['L_expr']:.4f} "
                f"geo={train_stats['L_geo']:.4f} "
                f"prog={train_stats['L_prog']:.4f} "
                f"trait={train_stats['L_trait']:.4f}"
            )

            if val_loader is not None:
                val_stats = self.eval_epoch(val_loader)
                val_loss = val_stats["loss"]
                msg += f" | val_loss={val_loss:.4f}"
                improved = val_loss + cfg.early_stop_min_delta < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    best_state = self._snapshot_state()
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
            else:
                # no val set → just track best train loss
                improved = train_stats["loss"] + cfg.early_stop_min_delta < best_val_loss
                if improved:
                    best_val_loss = train_stats["loss"]
                    best_state = self._snapshot_state()
                epochs_without_improve = 0  # no early stopping

            print(msg, flush=True)

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
            energy_fn=self.energy_fn,
            W=self.W,
            loss_cfg=self.loss_cfg,
            trait_head=self.trait_head,
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
