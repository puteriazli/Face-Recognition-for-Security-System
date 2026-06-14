"""
trainer.py — Two-Phase Training Loop.

Phase 1 (CrossEntropy, backbone frozen):
  Model belajar discriminasi dasar dengan cepat.
  LR besar (0.001), hanya head yang di-update.
  Target akhir phase 1: ~70-80% val_accuracy.

Phase 2 (CrossEntropy + Label Smoothing, full fine-tune):
  Seluruh model di-refine dengan LR kecil.
  Backbone LR lebih kecil dari head LR (differential LR).
  CosineAnnealingLR untuk konvergensi halus.
  Target akhir phase 2: >95% val_accuracy.
"""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


# ── Container history ─────────────────────────────────────────────────────────

class History:
    """Menyimpan metrik per epoch untuk plotting nanti."""
    def __init__(self):
        self.train_loss: list = []
        self.val_loss:   list = []
        self.train_acc:  list = []
        self.val_acc:    list = []
        self.lr:         list = []
        self.phase:      list = []   # "P1" atau "P2" per epoch

    def best_val_acc(self) -> float:
        return max(self.val_acc) if self.val_acc else 0.0

    def best_epoch(self) -> int:
        return self.val_acc.index(max(self.val_acc)) + 1 if self.val_acc else 0

    def to_dict(self) -> dict:
        return vars(self)


# ── Trainer utama ─────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, model, train_loader, val_loader, cfg: dict, device: torch.device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.tr           = cfg["training"]
        self.amp          = self.tr["mixed_precision"]

        # CrossEntropy dengan label smoothing (lebih stabil dari plain CE)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg["loss"]["label_smoothing"]
        )

        self.scaler  = GradScaler(enabled=self.amp)   # untuk mixed precision
        self.history = History()
        self.writer  = SummaryWriter(cfg["paths"]["logs"])

        self._best_acc       = 0.0
        self._no_improve     = 0
        self._ckpt_best      = Path(cfg["paths"]["checkpoints"]) / "best_model.pt"
        self._ckpt_best.parent.mkdir(parents=True, exist_ok=True)

    # ── Setup optimizer per phase ─────────────────────────────────────────────

    def _opt_phase1(self):
        """Phase 1: hanya parameter head yang di-optimize."""
        self.model.freeze_backbone()
        params = [p for p in self.model.head.parameters() if p.requires_grad]
        self.opt = Adam(params, lr=self.tr["phase1_lr"], weight_decay=1e-4)
        self.sch = CosineAnnealingLR(self.opt, T_max=self.tr["phase1_epochs"], eta_min=1e-5)
        log.info(f"Phase 1 optimizer: Adam lr={self.tr['phase1_lr']} (head only)")

    def _opt_phase2(self):
        """
        Phase 2: differential LR — backbone pakai LR lebih kecil dari head.
        Penting agar pretrained weight backbone tidak rusak.
        """
        self.model.unfreeze_all()
        params = [
            {"params": self.model.backbone.parameters(), "lr": self.tr["phase2_bb_lr"]},
            {"params": self.model.head.parameters(),     "lr": self.tr["phase2_lr"]},
        ]
        self.opt = Adam(params, weight_decay=1e-4)
        self.sch = CosineAnnealingLR(
            self.opt, T_max=self.tr["phase2_epochs"], eta_min=1e-6
        )
        log.info(f"Phase 2 optimizer: Adam bb_lr={self.tr['phase2_bb_lr']} "
                 f"head_lr={self.tr['phase2_lr']} (full model)")

    # ── Satu epoch train ─────────────────────────────────────────────────────

    def _train_epoch(self) -> tuple:
        self.model.train()
        tot_loss = correct = total = 0

        for imgs, labels in self.train_loader:
            imgs   = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.opt.zero_grad(set_to_none=True)   # lebih efisien dari zero_grad()

            with autocast(enabled=self.amp):       # mixed precision forward pass
                logits = self.model(imgs)           # → (B, num_classes)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()     # backward dengan scaled gradient
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(              # clip gradient agar stabil
                self.model.parameters(), self.tr["grad_clip"])
            self.scaler.step(self.opt)             # update weight
            self.scaler.update()                   # update scaler untuk iterasi berikut

            with torch.no_grad():
                preds    = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                tot_loss += loss.item() * labels.size(0)

        return tot_loss / total, correct / total

    # ── Satu epoch validasi ───────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self) -> tuple:
        self.model.eval()
        tot_loss = correct = total = 0

        for imgs, labels in self.val_loader:
            imgs   = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.amp):
                logits = self.model(imgs)
                loss   = self.criterion(logits, labels)

            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            tot_loss += loss.item() * labels.size(0)

        return tot_loss / total, correct / total

    # ── Simpan checkpoint ─────────────────────────────────────────────────────

    def _save_best(self, epoch: int, val_acc: float):
        torch.save({
            "epoch"      : epoch,
            "model_state": self.model.state_dict(),
            "opt_state"  : self.opt.state_dict(),
            "val_acc"    : val_acc,
            "cfg"        : self.cfg,
        }, self._ckpt_best)
        log.info(f"  ✅ Best checkpoint disimpan (val_acc={val_acc:.4f})")

    # ── Loop utama ────────────────────────────────────────────────────────────

    def train(self) -> History:
        p1 = self.tr["phase1_epochs"]
        p2 = self.tr["phase2_epochs"]

        # ── Phase 1 ─────────────────────────────────────────────────────────
        self._opt_phase1()
        log.info("=" * 55)
        log.info(f"PHASE 1 — CrossEntropy, backbone frozen [{p1} epoch]")
        log.info("=" * 55)

        for ep in range(p1):
            t0 = time.time()
            tr_loss, tr_acc   = self._train_epoch()
            va_loss, va_acc   = self._val_epoch()
            self.sch.step()
            lr = self.opt.param_groups[-1]["lr"]

            self.history.train_loss.append(tr_loss)
            self.history.val_loss.append(va_loss)
            self.history.train_acc.append(tr_acc)
            self.history.val_acc.append(va_acc)
            self.history.lr.append(lr)
            self.history.phase.append("P1")

            self.writer.add_scalars("Loss",     {"train": tr_loss, "val": va_loss}, ep)
            self.writer.add_scalars("Accuracy", {"train": tr_acc,  "val": va_acc},  ep)
            self.writer.add_scalar("LR", lr, ep)

            log.info(f"[P1 {ep+1:02d}/{p1}] "
                     f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
                     f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} | "
                     f"lr={lr:.2e} | {time.time()-t0:.0f}s")

            if va_acc > self._best_acc:
                self._best_acc = va_acc
                self._save_best(ep, va_acc)

        log.info(f"Phase 1 selesai. Best val_acc: {self._best_acc:.4f}\n")

        # Reset early stopping counter untuk phase 2
        self._no_improve = 0

        # ── Phase 2 ─────────────────────────────────────────────────────────
        self._opt_phase2()
        log.info("=" * 55)
        log.info(f"PHASE 2 — Full fine-tuning, differential LR [{p2} epoch]")
        log.info("=" * 55)

        for ep2 in range(p2):
            ep_global = p1 + ep2   # nomor epoch global untuk TensorBoard
            t0 = time.time()
            tr_loss, tr_acc = self._train_epoch()
            va_loss, va_acc = self._val_epoch()
            self.sch.step()
            lr = self.opt.param_groups[-1]["lr"]

            self.history.train_loss.append(tr_loss)
            self.history.val_loss.append(va_loss)
            self.history.train_acc.append(tr_acc)
            self.history.val_acc.append(va_acc)
            self.history.lr.append(lr)
            self.history.phase.append("P2")

            self.writer.add_scalars("Loss",     {"train": tr_loss, "val": va_loss}, ep_global)
            self.writer.add_scalars("Accuracy", {"train": tr_acc,  "val": va_acc},  ep_global)
            self.writer.add_scalar("LR", lr, ep_global)

            log.info(f"[P2 {ep2+1:02d}/{p2}] "
                     f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
                     f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} | "
                     f"lr={lr:.2e} | {time.time()-t0:.0f}s")

            if va_acc > self._best_acc:
                self._best_acc = va_acc
                self._no_improve = 0
                self._save_best(ep_global, va_acc)
            else:
                self._no_improve += 1
                if self._no_improve >= self.tr["patience"]:
                    log.info(f"Early stopping di epoch {ep_global+1}.")
                    break

        self.writer.close()

        # Simpan history ke JSON
        hist_path = Path(self.cfg["paths"]["logs"]) / "history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history.to_dict(), f, indent=2)

        log.info(f"\n{'='*55}")
        log.info(f"Training selesai. Best val_acc: {self._best_acc:.4f} "
                 f"(epoch {self.history.best_epoch()})")
        log.info(f"{'='*55}")
        return self.history

    def load_best(self):
        """Load checkpoint terbaik ke model."""
        ckpt = torch.load(self._ckpt_best, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        log.info(f"Best model di-load dari epoch {ckpt['epoch']+1} "
                 f"(val_acc={ckpt['val_acc']:.4f})")
        return self.model
