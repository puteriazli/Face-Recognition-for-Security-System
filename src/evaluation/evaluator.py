"""
evaluator.py — Evaluasi lengkap: accuracy, F1, confusion matrix, t-SNE, TAR@FAR.
Semua grafik disimpan ke results/. Semua angka disimpan ke JSON.
"""

import json
import logging
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import (
    classification_report, confusion_matrix,
    top_k_accuracy_score, roc_auc_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

log = logging.getLogger(__name__)


class Evaluator:

    def __init__(self, model, test_loader, cfg: dict, device: torch.device):
        self.model  = model
        self.loader = test_loader
        self.cfg    = cfg
        self.device = device
        self.amp    = cfg["training"]["mixed_precision"]
        self.out    = Path(cfg["paths"]["results"])
        self.out.mkdir(parents=True, exist_ok=True)

    # ── Ekstrak semua prediksi dari test set ──────────────────────────────────

    @torch.no_grad()
    def _predict(self):
        """Jalankan model pada seluruh test set, kumpulkan logits + label."""
        self.model.eval()
        all_logits, all_labels = [], []

        for imgs, labels in self.loader:
            imgs = imgs.to(self.device, non_blocking=True)
            with autocast(enabled=self.amp):
                logits = self.model(imgs)   # (B, num_classes)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

        logits = torch.cat(all_logits).numpy()   # (N, num_classes)
        labels = torch.cat(all_labels).numpy()   # (N,)
        preds  = logits.argmax(axis=1)           # (N,)
        return logits, preds, labels

    @torch.no_grad()
    def _extract_embeddings(self):
        """Ekstrak embedding 512-d untuk t-SNE dan TAR@FAR."""
        self.model.eval()
        all_embs, all_labels = [], []

        for imgs, labels in self.loader:
            imgs = imgs.to(self.device, non_blocking=True)
            with autocast(enabled=self.amp):
                emb = self.model(imgs, return_embedding=True)   # (B, 512)
            all_embs.append(emb.cpu())
            all_labels.append(labels)

        return torch.cat(all_embs).numpy(), torch.cat(all_labels).numpy()

    # ── Evaluasi utama ────────────────────────────────────────────────────────

    def run(self, idx_to_class: dict, history=None) -> dict:
        """
        Jalankan semua evaluasi dan simpan hasilnya.
        Args:
            idx_to_class: {0: "Ronaldo", 1: "Messi", ...}
            history     : objek History dari training (opsional, untuk kurva)
        Returns:
            dict berisi semua metrik numerik
        """
        log.info("Memulai evaluasi lengkap pada test set...")
        n_cls       = len(idx_to_class)
        class_names = [idx_to_class[i] for i in range(n_cls)]

        logits, preds, labels = self._predict()
        probs  = torch.tensor(logits).softmax(dim=1).numpy()   # untuk ROC-AUC

        # ── Metrik klasifikasi ────────────────────────────────────────────────
        acc1 = float((preds == labels).mean())
        acc5 = float(top_k_accuracy_score(labels, logits, k=min(5, n_cls)))
        rpt  = classification_report(labels, preds, target_names=class_names,
                                     output_dict=True, zero_division=0)

        try:  # ROC-AUC membutuhkan binarized labels
            lb   = label_binarize(labels, classes=list(range(n_cls)))
            auc  = float(roc_auc_score(lb, probs, multi_class="ovr", average="macro"))
        except Exception:
            auc = -1.0

        # ── Metrik biometrik ──────────────────────────────────────────────────
        embs, emb_labels = self._extract_embeddings()
        tar_far, eer     = self._tar_far(embs, emb_labels)

        # ── Plot ──────────────────────────────────────────────────────────────
        self._plot_confusion_matrix(labels, preds, class_names)
        self._plot_per_class_f1(rpt, class_names)
        self._plot_tsne(embs, emb_labels, class_names)
        if history is not None:
            self._plot_curves(history)

        # ── Simpan & print ringkasan ──────────────────────────────────────────
        summary = {
            "top1_accuracy"    : round(acc1, 4),
            "top5_accuracy"    : round(acc5, 4),
            "macro_precision"  : round(rpt["macro avg"]["precision"], 4),
            "macro_recall"     : round(rpt["macro avg"]["recall"],    4),
            "macro_f1"         : round(rpt["macro avg"]["f1-score"],  4),
            "weighted_f1"      : round(rpt["weighted avg"]["f1-score"], 4),
            "roc_auc_macro"    : round(auc,  4),
            "eer"              : round(eer,  4),
            "tar_at_far_0.1"   : round(tar_far.get(0.1,  -1), 4),
            "tar_at_far_0.01"  : round(tar_far.get(0.01, -1), 4),
            "tar_at_far_0.001" : round(tar_far.get(0.001,-1), 4),
            "n_test_samples"   : int(len(labels)),
            "n_classes"        : n_cls,
        }

        with open(self.out / "evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*55)
        print("  HASIL EVALUASI — TEST SET")
        print("="*55)
        for k, v in summary.items():
            if isinstance(v, float):
                status = "🎯" if k == "top1_accuracy" and v >= 0.90 else "  "
                print(f"  {status} {k:25s}: {v:.4f}")
        print("="*55)

        print("\n--- Classification Report (macro) ---")
        print(classification_report(labels, preds, target_names=class_names,
                                    zero_division=0))
        return summary

    # ── TAR@FAR & EER ────────────────────────────────────────────────────────

    def _tar_far(self, embs, labels, n_pairs: int = 5000):
        """
        Hitung TAR (True Accept Rate) pada berbagai FAR (False Accept Rate).
        TAR@FAR: metrik standar industri untuk sistem biometrik.
        EER: titik di mana FAR == FRR (Equal Error Rate), semakin kecil semakin baik.
        """
        rng = np.random.default_rng(42)
        n   = len(labels)
        genuine, impostor = [], []

        for _ in range(n_pairs):
            i, j = rng.choice(n, 2, replace=False)
            sim  = float(np.dot(embs[i], embs[j]))   # cosine sim (sudah L2-norm)
            (genuine if labels[i] == labels[j] else impostor).append(sim)

        genuine  = np.array(genuine)
        impostor = np.array(impostor)

        tar_far = {}
        for far_t in [0.1, 0.01, 0.001]:
            thr = np.percentile(impostor, 100 * (1 - far_t))
            tar_far[far_t] = float((genuine >= thr).mean())
            log.info(f"  TAR@FAR={far_t:.3f}: {tar_far[far_t]:.4f}")

        # EER
        thrs = np.linspace(impostor.min(), impostor.max(), 500)
        frr  = np.array([(genuine < t).mean()  for t in thrs])
        far  = np.array([(impostor >= t).mean() for t in thrs])
        idx  = np.argmin(np.abs(frr - far))
        eer  = float((frr[idx] + far[idx]) / 2)
        log.info(f"  EER: {eer:.4f}")

        return tar_far, eer

    # ── Plot helpers ──────────────────────────────────────────────────────────

    def _plot_confusion_matrix(self, labels, preds, class_names):
        n  = len(class_names)
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(max(12, n//3), max(10, n//3)))
        sns.heatmap(cm, annot=(n<=25), fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names if n<=25 else [],
                    yticklabels=class_names if n<=25 else [])
        ax.set_title("Confusion Matrix — Test Set", fontsize=13)
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Label Asli")
        plt.tight_layout()
        fig.savefig(self.out / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        log.info("Confusion matrix disimpan")

    def _plot_per_class_f1(self, report, class_names):
        f1s = [report[c]["f1-score"] for c in class_names if c in report]
        idx = np.argsort(f1s)
        clrs = ["#4CAF50" if f >= 0.9 else "#FF9800" if f >= 0.7 else "#f44336"
                for f in np.array(f1s)[idx]]
        fig, ax = plt.subplots(figsize=(10, max(6, len(f1s)//4)))
        ax.barh(range(len(f1s)), np.array(f1s)[idx], color=clrs)
        ax.set_yticks(range(len(f1s)))
        ax.set_yticklabels([class_names[i] for i in idx], fontsize=7)
        ax.axvline(0.9, color="green", linestyle="--", label="90%")
        ax.set_title("F1-Score per Kelas"); ax.legend()
        plt.tight_layout()
        fig.savefig(self.out / "per_class_f1.png", dpi=150)
        plt.close(fig)

    def _plot_tsne(self, embs, labels, class_names):
        log.info("Menghitung t-SNE ...")
        n   = min(1000, len(embs))
        idx = np.random.choice(len(embs), n, replace=False)
        proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embs[idx])
        lbl  = labels[idx]
        cmap = plt.cm.get_cmap("tab20", len(class_names))
        fig, ax = plt.subplots(figsize=(12, 10))
        for ci in range(len(class_names)):
            m = lbl == ci
            if m.any():
                ax.scatter(proj[m,0], proj[m,1], c=[cmap(ci)],
                           label=class_names[ci], alpha=0.7, s=15)
        ax.set_title("t-SNE Embedding Space (Test Set)")
        if len(class_names) <= 20:
            ax.legend(fontsize=6, markerscale=2, loc="best")
        plt.tight_layout()
        fig.savefig(self.out / "tsne.png", dpi=150)
        plt.close(fig)
        log.info("t-SNE disimpan")

    def _plot_curves(self, history):
        epochs = range(1, len(history.train_loss) + 1)
        p1_end = history.phase.index("P2") if "P2" in history.phase else len(epochs)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, metric, title in zip(
            axes,
            [("train_loss","val_loss"), ("train_acc","val_acc"), ("lr",)],
            ["Loss", "Accuracy", "Learning Rate"]
        ):
            if metric[0] == "lr":
                ax.plot(epochs, history.lr, "g-o", markersize=3)
                ax.set_yscale("log")
            else:
                ax.plot(epochs, getattr(history, metric[0]), "b-o",
                        label="Train", markersize=3)
                ax.plot(epochs, getattr(history, metric[1]), "r-o",
                        label="Val",   markersize=3)
                if title == "Accuracy":
                    ax.axhline(0.9, color="green", linestyle="--",
                               label="Target 90%")
                ax.legend()

            # Garis pemisah phase 1 / phase 2
            if p1_end < len(epochs):
                ax.axvline(p1_end + 0.5, color="purple", linestyle=":",
                           alpha=0.7, label="P1→P2")

            ax.set_title(title); ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)

        plt.suptitle("Training Curves — InceptionResnetV1 + VGGFace2", fontsize=13)
        plt.tight_layout()
        fig.savefig(self.out / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Training curves disimpan")
