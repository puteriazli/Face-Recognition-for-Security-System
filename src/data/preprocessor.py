"""
preprocessor.py — Pipeline preprocessing wajah yang IDENTIK untuk training & inference.

Prinsip utama:
  Foto input bisa berukuran APAPUN (1:1, 16:9, 4:3, potret, landscape, dst).
  Pipeline ini selalu menghasilkan tensor (3, 160, 160) yang ternormalisasi,
  sehingga distribusi data training == distribusi data inference → tidak ada gap.

Alur:
  Input (gambar apapun)
      │
      ▼  MTCNN
  Deteksi wajah → crop area wajah → align berdasarkan 5 landmark
      │
      │  Jika wajah tidak terdeteksi → fallback: center-crop + resize
      ▼
  Tensor (3, 160, 160), nilai range [-1, 1]
      │
      └─→ Siap masuk InceptionResnetV1
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN
from tqdm import tqdm

log = logging.getLogger(__name__)


class FacePreprocessor:
    """
    Wrapper MTCNN yang dipakai KONSISTEN di training dan inference.
    Satu instance ini = satu definisi preprocessing → tidak ada inkonsistensi.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, cfg: dict, device: torch.device):
        size = cfg["data"]["image_size"]   # 160

        # MTCNN: detektor wajah sekaligus preprocessor
        # post_process=True → output sudah ternormalisasi ke [-1, 1]
        # (sama persis dengan yang InceptionResnetV1 harapkan)
        self.mtcnn = MTCNN(
            image_size=size,       # output face crop = size × size
            margin=20,             # tambah 20px padding di sekeliling wajah
            min_face_size=20,      # abaikan wajah < 20px (terlalu kecil)
            thresholds=[0.6, 0.7, 0.8],  # confidence threshold P-Net, R-Net, O-Net
            factor=0.709,          # image pyramid scale factor
            post_process=True,     # normalisasi ke [-1, 1] otomatis
            keep_all=False,        # ambil 1 wajah paling prominent saja
            select_largest=True,   # jika ada >1 wajah, pilih yang terbesar
            device=device,
        )
        self.size   = size
        self.device = device
        self.proc   = Path(cfg["data"]["processed_dir"])
        self.raw    = Path(cfg["data"]["dataset_root"])

    # ── Core: satu gambar → tensor ────────────────────────────────────────────

    def process_image(self, img: Image.Image) -> torch.Tensor:
        """
        Proses satu gambar PIL (ukuran/rasio apapun) → tensor (3, 160, 160).
        Fungsi ini dipakai IDENTIK di training preprocessing dan inference.

        Returns:
            Tensor float32 shape (3, 160, 160), range [-1, 1]
        """
        img = img.convert("RGB")   # pastikan 3-channel, handle grayscale/RGBA

        # Coba deteksi wajah dengan MTCNN
        face = self.mtcnn(img)     # → tensor (3,160,160) atau None

        if face is None:
            # Fallback jika tidak ada wajah terdeteksi:
            # Center-crop ke persegi → resize → normalisasi manual [-1,1]
            face = self._center_crop_resize(img)

        return face   # shape: (3, 160, 160), dtype: float32

    def _center_crop_resize(self, img: Image.Image) -> torch.Tensor:
        """
        Fallback preprocessing: center crop ke persegi lalu resize.
        Normalisasi sama dengan MTCNN post_process=True → [-1, 1].
        """
        w, h  = img.size
        side  = min(w, h)                             # sisi terpendek sebagai acuan
        left  = (w - side) // 2                       # mulai crop dari tengah
        top   = (h - side) // 2
        img   = img.crop((left, top, left+side, top+side))   # crop ke persegi
        img   = img.resize((self.size, self.size), Image.LANCZOS)  # resize ke 160×160
        t     = TF.to_tensor(img)                     # → [0,1]
        t     = (t - 0.5) / 0.5                       # → [-1,1], sama dengan MTCNN
        return t

    # ── Batch preprocessing seluruh dataset ───────────────────────────────────

    def run(self, force: bool = False) -> Path:
        """
        Proses semua gambar di dataset_root → simpan ke processed_dir.

        Args:
            force: True = hapus processed_dir dulu dan proses ulang dari awal.
                   False = skip jika processed_dir sudah ada isinya.

        Returns:
            Path ke processed_dir.
        """
        import shutil

        # Skip jika sudah ada (hemat waktu 20-40 menit)
        if not force and self._already_done():
            n = sum(1 for _ in self.proc.rglob("*.jpg"))
            log.info(f"Processed folder sudah ada ({n} gambar). Skip MTCNN. "
                     "Gunakan force=True untuk proses ulang.")
            return self.proc

        if force and self.proc.exists():
            shutil.rmtree(self.proc)
            log.info("Processed folder dihapus, mulai dari awal.")

        self.proc.mkdir(parents=True, exist_ok=True)

        # Kumpulkan semua gambar dari dataset_root
        pairs = []   # list of (image_path, class_name)
        for cls_dir in sorted(self.raw.iterdir()):
            if not cls_dir.is_dir():
                continue
            # Hapus prefix 'pins_' jika ada (format dataset Kaggle)
            cls_name = cls_dir.name[5:] if cls_dir.name.lower().startswith("pins_") else cls_dir.name
            for img_path in cls_dir.rglob("*"):
                if img_path.suffix.lower() in self.IMG_EXTS:
                    pairs.append((img_path, cls_name))

        log.info(f"Memulai MTCNN preprocessing: {len(pairs)} gambar ...")
        ok = fail = 0

        for img_path, cls_name in tqdm(pairs, desc="MTCNN"):
            out_dir  = self.proc / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (img_path.stem + ".jpg")

            if out_path.exists():      # sudah diproses sebelumnya → skip
                ok += 1; continue

            try:
                img  = Image.open(img_path)
                face = self.process_image(img)   # pakai fungsi yang SAMA dengan inference!

                # Konversi tensor [-1,1] balik ke PIL untuk disimpan sebagai JPEG
                face_np  = ((face.permute(1,2,0).numpy() * 0.5 + 0.5) * 255).clip(0,255).astype(np.uint8)
                Image.fromarray(face_np).save(str(out_path), quality=95)
                ok += 1
            except Exception as e:
                log.debug(f"Gagal: {img_path} — {e}")
                fail += 1

        log.info(f"MTCNN selesai: {ok} berhasil, {fail} gagal.")
        return self.proc

    def _already_done(self) -> bool:
        return self.proc.exists() and sum(1 for _ in self.proc.rglob("*.jpg")) > 100
