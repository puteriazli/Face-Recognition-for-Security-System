"""
dataset_loader.py — Scan dataset, stratified split, buat DataLoader.

Dataset dibaca dari processed_dir (hasil MTCNN).
Split dilakukan SEBELUM augmentasi → tidak ada data leakage.
"""

import json, logging
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

log = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Dataset PyTorch ────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """Dataset minimal: list path + label → tensor gambar."""

    def __init__(self, paths: list, labels: list, transform=None):
        self.paths     = paths      # list path file gambar
        self.labels    = labels     # list integer label
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (160, 160))   # fallback gambar hitam

        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── Loader utama ───────────────────────────────────────────────────────────────

class DatasetLoader:
    """Scan processed_dir, split stratified, buat DataLoader."""

    def __init__(self, cfg: dict):
        self.cfg         = cfg
        self.class_to_idx: dict = {}   # {"Ronaldo": 0, "Messi": 1, ...}
        self.idx_to_class: dict = {}   # {0: "Ronaldo", 1: "Messi", ...}

    def scan(self, processed_dir: str):
        """
        Scan folder processed_dir, kumpulkan semua path gambar + label integer.
        Kelas dengan gambar < min_samples diabaikan.

        Returns:
            paths  : list[str] — path absolut setiap gambar
            labels : list[int] — label integer setiap gambar
        """
        root  = Path(processed_dir)
        pairs = []   # [(path, class_name)]

        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir(): continue
            imgs = [p for p in cls_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
            if len(imgs) < self.cfg["data"]["min_samples"]:
                continue   # abaikan kelas terlalu sedikit
            for p in imgs:
                pairs.append((str(p), cls_dir.name))

        # Buat mapping kelas → integer
        all_classes = sorted({c for _, c in pairs})
        self.class_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        paths  = [p for p, _ in pairs]
        labels = [self.class_to_idx[c] for _, c in pairs]

        counts = sorted(Counter(labels).values())
        log.info(f"Dataset: {len(all_classes)} kelas | {len(paths)} gambar | "
                 f"min/kelas={min(counts)} max/kelas={max(counts)}")
        return paths, labels

    def split(self, paths, labels):
        """
        Stratified split 70/15/15.
        Split berdasarkan PATH (bukan gambar yang sudah diproses) → zero leakage.
        """
        seed = self.cfg["project"]["seed"]
        tr   = self.cfg["data"]["train_ratio"]
        vr   = self.cfg["data"]["val_ratio"]

        # Split 1: train vs (val+test)
        Xtr, Xtmp, ytr, ytmp = train_test_split(
            paths, labels, test_size=(1-tr), stratify=labels, random_state=seed)

        # Split 2: val vs test
        Xval, Xte, yval, yte = train_test_split(
            Xtmp, ytmp, test_size=vr/(vr+self.cfg["data"]["test_ratio"]),
            stratify=ytmp, random_state=seed)

        # Balik: val_frac dari sisa = val/(val+test)
        # Di atas test_size = val/(val+test) → hasilnya val, sisanya test
        # Koreksi: test_size harusnya test/(val+test)
        Xval, Xte, yval, yte = train_test_split(
            Xtmp, ytmp,
            test_size=self.cfg["data"]["test_ratio"]/(vr+self.cfg["data"]["test_ratio"]),
            stratify=ytmp, random_state=seed)

        log.info(f"Split → train:{len(Xtr)} val:{len(Xval)} test:{len(Xte)}")
        return (Xtr,ytr), (Xval,yval), (Xte,yte)

    def save_meta(self, train_d, val_d, test_d, out_dir: str):
        """Simpan metadata split ke JSON agar reproducible lintas notebook."""
        meta = {
            "class_to_idx" : self.class_to_idx,
            "idx_to_class" : {str(k):v for k,v in self.idx_to_class.items()},
            "num_classes"  : len(self.class_to_idx),
            "sizes"        : {"train": len(train_d[0]), "val": len(val_d[0]), "test": len(test_d[0])},
            "seed"         : self.cfg["project"]["seed"],
        }
        out = Path(out_dir) / "split_meta.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f: json.dump(meta, f, indent=2)
        log.info(f"Metadata split disimpan → {out}")

    @staticmethod
    def _sample_weights(labels):
        """Hitung bobot per sampel (invers frekuensi kelas) untuk WeightedRandomSampler."""
        counts = Counter(labels)
        total  = len(labels)
        n_cls  = len(counts)
        # Kelas dengan sedikit data → bobot besar → dipilih lebih sering
        return torch.FloatTensor([total / (n_cls * counts[l]) for l in labels])

    def make_loaders(self, train_d, val_d, test_d, train_tf, val_tf):
        """
        Buat DataLoader untuk ketiga split.
        Train: WeightedRandomSampler untuk tangani class imbalance.
        Val/Test: shuffle=False → evaluasi deterministik.
        """
        bs  = self.cfg["training"]["batch_size"]
        nw  = self.cfg["data"]["num_workers"]

        Xtr,ytr   = train_d
        Xval,yval = val_d
        Xte,yte   = test_d

        train_ds = FaceDataset(Xtr,  ytr,  transform=train_tf)
        val_ds   = FaceDataset(Xval, yval, transform=val_tf)
        test_ds  = FaceDataset(Xte,  yte,  transform=val_tf)   # test pakai val_tf (no aug)

        sampler = WeightedRandomSampler(
            weights=self._sample_weights(ytr),
            num_samples=len(ytr),
            replacement=True,    # sampling with replacement → distribusi seimbang
        )

        tr_loader  = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                                num_workers=nw, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=True)
        te_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=True)

        log.info(f"DataLoader siap — train:{len(tr_loader)} val:{len(val_loader)} "
                 f"test:{len(te_loader)} batch (bs={bs})")
        return tr_loader, val_loader, te_loader
