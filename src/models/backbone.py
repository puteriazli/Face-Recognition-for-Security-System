"""
backbone.py — InceptionResnetV1 pretrained VGGFace2 + classifier head.

Kenapa InceptionResnetV1 + VGGFace2:
  - Sudah dilatih di 3.3 juta foto wajah (9131 identitas)
  - Backbone sudah "mengerti" anatomi wajah: mata, hidung, rahang, kontur
  - Fine-tuning ke 105 kelas jauh lebih mudah → konvergen cepat
  - Terbukti mencapai 98%+ accuracy di penelitian face recognition

Strategi fine-tuning:
  Phase 1 (5 epoch): Backbone frozen → hanya classifier head belajar
    Tujuan: head mendapatkan arah gradient yang benar sebelum backbone ikut bergerak
  Phase 2 (25 epoch): Semua layer di-unfreeze → full fine-tuning
    Tujuan: seluruh model menyesuaikan diri ke 105 kelas spesifik ini
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1   # pretrained VGGFace2 built-in

log = logging.getLogger(__name__)


class FaceRecognitionModel(nn.Module):
    """
    InceptionResnetV1 (VGGFace2) + custom classifier head untuk N kelas.

    Forward mode 'embed': output embedding 512-d (untuk inference/verifikasi)
    Forward mode 'classify': output logits N-kelas (untuk training)
    """

    def __init__(self, num_classes: int, cfg: dict):
        super().__init__()
        m = cfg["model"]

        # ── Backbone: InceptionResnetV1 pretrained VGGFace2 ─────────────────
        # pretrained='vggface2' → download weight otomatis saat pertama kali
        # classify=False → output 512-d embedding (kita tambah head sendiri)
        self.backbone = InceptionResnetV1(
            pretrained=m["pretrained"],   # 'vggface2'
            classify=False,               # matikan classifier bawaan
            num_classes=None,
        )

        # ── Classifier head ──────────────────────────────────────────────────
        # Input: 512-d embedding dari backbone
        # Output: logits untuk num_classes kelas
        self.head = nn.Sequential(
            nn.BatchNorm1d(m["embedding_dim"]),   # normalisasi embedding
            nn.Dropout(p=m["dropout"]),            # regularisasi
            nn.Linear(m["embedding_dim"], num_classes),  # proyeksi ke N kelas
        )

        # Inisialisasi head
        nn.init.xavier_uniform_(self.head[2].weight)   # head[2] = Linear
        nn.init.zeros_(self.head[2].bias)

        self.num_classes = num_classes
        log.info(f"FaceRecognitionModel | backbone=InceptionResnetV1 "
                 f"pretrained={m['pretrained']} | head→{num_classes} kelas")

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """
        Args:
            x               : tensor (B, 3, 160, 160), range [-1, 1]
            return_embedding: True → output embedding 512-d (inference)
                              False → output logits N-kelas (training)
        Returns:
            Embedding L2-normalized (B, 512) atau logits (B, N)
        """
        emb = self.backbone(x)             # → (B, 512), sudah L2-normalized dari backbone

        if return_embedding:
            return F.normalize(emb, p=2, dim=1)   # pastikan unit-norm untuk cosine similarity

        return self.head(emb)              # → (B, num_classes) logits

    # ── Helper fine-tuning ────────────────────────────────────────────────────

    def freeze_backbone(self):
        """Phase 1: bekukan backbone, hanya head yang belajar."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True
        log.info("Backbone frozen (Phase 1: head-only training)")

    def unfreeze_all(self):
        """Phase 2: buka semua layer untuk full fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True
        log.info("Semua layer di-unfreeze (Phase 2: full fine-tuning)")

    def trainable_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
