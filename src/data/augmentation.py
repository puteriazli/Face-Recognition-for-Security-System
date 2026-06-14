"""
augmentation.py — Transform untuk training dan val/test/inference.

Aturan penting:
  train_transform : augmentasi random → model robust terhadap variasi foto nyata
  val_transform   : TANPA augmentasi → evaluasi deterministik & konsisten

Normalisasi yang dipakai: mean=0.5, std=0.5 → output [-1, 1]
Sama persis dengan MTCNN post_process=True → tidak ada distribusi gap.
"""

import torchvision.transforms as T


# Normalisasi identik dengan output MTCNN (post_process=True)
# MTCNN: (pixel/255 - 0.5) / 0.5 = pixel range [-1, 1]
_NORM = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def get_train_transform(cfg: dict) -> T.Compose:
    """
    Transform untuk training set.
    Augmentasi HANYA diaplikasikan ke gambar dari processed_dir
    (sudah 160×160 hasil MTCNN) → tidak ada resize ulang.
    """
    aug = cfg["augmentation"]
    return T.Compose([
        # ── Geometric ─────────────────────────────────────────────────────
        T.RandomHorizontalFlip(p=aug["horizontal_flip"]),  # cermin horizontal
        T.RandomRotation(degrees=aug["rotation"]),          # rotasi kecil ±10°
        # ── Warna ─────────────────────────────────────────────────────────
        T.ColorJitter(
            brightness=aug["brightness"],   # variasi kecerahan (pencahayaan)
            contrast=aug["contrast"],        # variasi kontras
            saturation=aug["saturation"],    # variasi saturasi warna
        ),
        # ── Tensor + Normalisasi ───────────────────────────────────────────
        T.ToTensor(),                       # PIL (H,W,C) uint8 → tensor (C,H,W) float [0,1]
        _NORM,                              # → [-1, 1], identik dengan MTCNN output
        # ── Regularisasi spasial ──────────────────────────────────────────
        T.RandomErasing(                    # hapus patch acak (simulasi wajah tertutup)
            p=aug["random_erasing"],
            scale=(0.02, 0.15),            # ukuran patch: 2%-15% area gambar
            ratio=(0.3, 3.0),              # rasio aspek patch
            value=0,                        # isi dengan 0 (hitam)
        ),
    ])


def get_val_transform(cfg: dict) -> T.Compose:
    """
    Transform untuk val, test, DAN inference.
    TANPA augmentasi → hasil deterministik dan konsisten.
    Gambar processed_dir sudah 160×160 → hanya perlu ToTensor + Normalize.
    """
    return T.Compose([
        T.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        # ↑ resize eksplisit untuk jaga-jaga gambar raw yang belum diproses MTCNN
        T.ToTensor(),   # → [0, 1]
        _NORM,          # → [-1, 1]
    ])


def denormalize(tensor):
    """
    Balik normalisasi untuk visualisasi.
    Input : tensor (C, H, W) range [-1, 1]
    Output: numpy (H, W, C) uint8 range [0, 255]
    """
    import numpy as np
    img = tensor.permute(1, 2, 0).numpy()   # (C,H,W) → (H,W,C)
    img = (img * 0.5 + 0.5) * 255.0        # [-1,1] → [0,255]
    return img.clip(0, 255).astype("uint8")
