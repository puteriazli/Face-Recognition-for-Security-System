"""
predictor.py — Engine inference untuk deployment.

Fitur utama:
  Auto-standardize: foto input APAPUN (ukuran, rasio, format)
  → diproses dengan pipeline IDENTIK dengan training
  → tidak ada distribusi gap antara training dan production

Dua mode inference:
  verify(img, identity)  : 1:1 — apakah foto ini cocok dengan identitas X?
  identify(img)          : 1:N — siapa orang di foto ini?
  enroll(identity, imgs) : daftarkan user baru tanpa retrain model
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.data.preprocessor import FacePreprocessor
from src.data.augmentation import get_val_transform

log = logging.getLogger(__name__)


class EmbeddingDB:
    """
    Database embedding wajah per identitas.
    Disimpan sebagai file .pkl — di produksi nyata, ganti dengan
    vector database (Faiss, Milvus, Pinecone) untuk skalabilitas.
    """

    def __init__(self, path: str = None):
        self.db   = {}     # {identity_name: np.ndarray (512,)}
        self.path = path
        if path and Path(path).exists():
            self.load(path)

    def add(self, identity: str, embeddings: np.ndarray):
        """
        Daftarkan satu identitas dari satu atau beberapa embedding.
        Jika sudah ada, update dengan rata-rata embedding.
        embeddings: shape (N, 512) atau (512,)
        """
        embs = np.atleast_2d(embeddings)
        mean = embs.mean(axis=0)
        mean /= np.linalg.norm(mean) + 1e-8   # L2 normalize mean embedding
        if identity in self.db:
            # Rata-rata dengan embedding lama → update bertahap
            old  = self.db[identity]
            mean = (old + mean) / 2
            mean /= np.linalg.norm(mean) + 1e-8
        self.db[identity] = mean
        log.info(f"Enrolled: '{identity}' | DB size: {len(self.db)}")

    def remove(self, identity: str):
        self.db.pop(identity, None)

    def get_matrix(self):
        """Kembalikan (names, matrix) untuk batch similarity search."""
        names  = list(self.db.keys())
        matrix = np.stack([self.db[n] for n in names])   # (N, 512)
        return names, matrix

    def save(self, path: str = None):
        p = path or self.path
        with open(p, "wb") as f: pickle.dump(self.db, f)
        log.info(f"DB disimpan → {p} ({len(self.db)} identitas)")

    def load(self, path: str = None):
        p = path or self.path
        with open(p, "rb") as f: self.db = pickle.load(f)
        log.info(f"DB di-load ← {p} ({len(self.db)} identitas)")

    def __len__(self): return len(self.db)
    def __contains__(self, x): return x in self.db


class FacePredictor:
    """
    Inference engine: auto-standardize input → embedding → verify/identify.

    Preprocessing pipeline identik dengan training:
      Input (apapun) → MTCNN crop+align 160×160 → normalize [-1,1]
      Jika wajah tidak terdeteksi → center-crop fallback
    """

    def __init__(self, model, preprocessor: FacePreprocessor,
                 db: EmbeddingDB, cfg: dict, device: torch.device):
        self.model  = model.eval()
        self.prep   = preprocessor     # FacePreprocessor: pipeline identik training
        self.val_tf = get_val_transform(cfg)  # transform val (no aug)
        self.db     = db
        self.cfg    = cfg
        self.device = device
        self.thr    = cfg["inference"]["threshold"]
        self.top_k  = cfg["inference"]["top_k"]

    # ── Core: gambar → embedding ──────────────────────────────────────────────

    @torch.no_grad()
    def get_embedding(self, image_input) -> np.ndarray:
        """
        Konversi input gambar (format apapun) → embedding 512-d.

        Auto-standardize pipeline:
          1. Terima PIL/numpy/path → konversi ke PIL RGB
          2. MTCNN: deteksi wajah + crop + align + normalize [-1,1]
          3. Jika gagal: center-crop fallback (tetap [-1,1])
          4. Masuk model → embedding 512-d L2-normalized

        Pipeline ini IDENTIK dengan preprocessing training.
        Foto ukuran apapun (1:1, 16:9, 4:3, potret, landscape, dll)
        distandarisasi menjadi tensor (3, 160, 160) yang sama.
        """
        # ── Step 1: normalisasi input ke PIL ──────────────────────────────
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Tipe input tidak dikenal: {type(image_input)}")

        # ── Step 2: preprocessing identik training (MTCNN) ────────────────
        face_tensor = self.prep.process_image(img)   # → (3,160,160) float32

        # ── Step 3: masuk model ───────────────────────────────────────────
        inp = face_tensor.unsqueeze(0).to(self.device)   # → (1,3,160,160)
        emb = self.model(inp, return_embedding=True)      # → (1,512)
        return emb.squeeze(0).cpu().numpy()               # → (512,)

    # ── 1:1 Verifikasi ────────────────────────────────────────────────────────

    def verify(self, image_input, claimed_identity: str) -> dict:
        """
        Verifikasi: "Apakah foto ini cocok dengan identitas yang diklaim?"
        Dipakai untuk: konfirmasi pembayaran e-wallet, login dengan wajah.

        Returns dict:
            match      : True/False
            similarity : float, 0–1 (cosine similarity)
            threshold  : batas yang dipakai
        """
        if claimed_identity not in self.db:
            return {"match": False, "similarity": 0.0,
                    "error": f"'{claimed_identity}' belum terdaftar di database"}

        query_emb  = self.get_embedding(image_input)
        stored_emb = self.db.db[claimed_identity]
        sim        = float(np.dot(query_emb, stored_emb))   # cosine sim (L2-normed)

        return {
            "match"            : sim >= self.thr,
            "similarity"       : round(sim, 4),
            "threshold"        : self.thr,
            "claimed_identity" : claimed_identity,
            "confidence"       : "HIGH" if sim > 0.75 else "MEDIUM" if sim > self.thr else "LOW",
        }

    # ── 1:N Identifikasi ─────────────────────────────────────────────────────

    def identify(self, image_input) -> dict:
        """
        Identifikasi: "Siapa orang di foto ini?"
        Dipakai untuk: sistem absensi, surveillance, verifikasi tanpa klaim identitas.

        Returns dict:
            top_k      : list of (identity, similarity)
            best_match : identitas dengan similarity tertinggi
            accepted   : True jika similarity > threshold
        """
        if len(self.db) == 0:
            return {"error": "Database kosong. Daftarkan user terlebih dahulu."}

        query_emb      = self.get_embedding(image_input)
        names, matrix  = self.db.get_matrix()
        sims           = matrix @ query_emb                    # (N,) cosine similarity
        top_idx        = np.argsort(sims)[::-1][:self.top_k]  # sort descending

        return {
            "top_k"        : [(names[i], round(float(sims[i]), 4)) for i in top_idx],
            "best_match"   : names[top_idx[0]],
            "best_sim"     : round(float(sims[top_idx[0]]), 4),
            "accepted"     : float(sims[top_idx[0]]) >= self.thr,
        }

    # ── Enrollment ───────────────────────────────────────────────────────────

    def enroll(self, identity: str, images: list, save: bool = True):
        """
        Daftarkan user baru dari 1–N foto. TIDAK perlu retrain model.
        Cukup hitung embedding → simpan ke database.

        Args:
            identity : nama/ID unik user
            images   : list PIL Image / path / numpy array
            save     : simpan database ke disk setelah enroll
        """
        embs = np.stack([self.get_embedding(img) for img in images])
        self.db.add(identity, embs)
        if save and self.db.path:
            self.db.save()
        log.info(f"Enrollment selesai: '{identity}' ({len(images)} foto)")

    # ── Build DB dari test loader ─────────────────────────────────────────────

    @torch.no_grad()
    def build_db_from_loader(self, loader, idx_to_class: dict):
        """Isi database dari DataLoader (untuk demo/testing)."""
        from collections import defaultdict
        self.model.eval()
        acc = defaultdict(list)

        for imgs, labels in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            embs = self.model(imgs, return_embedding=True).cpu().numpy()
            for emb, lbl in zip(embs, labels.numpy()):
                acc[idx_to_class[int(lbl)]].append(emb)

        for identity, emb_list in acc.items():
            self.db.add(identity, np.stack(emb_list))

        log.info(f"Database dibangun: {len(self.db)} identitas")
