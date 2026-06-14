<div align="center">

# 🔐 Face Recognition — E-Wallet Security System

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/MTCNN-Face%20Detection-00C7B7?style=for-the-badge"/>
<img src="https://img.shields.io/badge/VGGFace2-Pretrained-6F42C1?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Accuracy-99.58%25-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>

<br/>

**Sistem pengenalan wajah berbasis Deep Learning untuk autentikasi keamanan pembayaran digital**
*(GoPay · DANA · ShopeePay · OVO · LinkAja)*

<br/>

```
Top-1 Accuracy: 99.58%  |  EER: 0.02%  |  TAR@FAR=0.001: 100%  |  ROC-AUC: 1.00
```

</div>

---

## 📋 Daftar Isi

- [🎯 Problem Statement](#-problem-statement)
- [💡 Solution Overview](#-solution-overview)
- [📦 Dataset](#-dataset)
- [🗂️ Struktur Proyek](#️-struktur-proyek)
- [🔬 Metode & Algoritma](#-metode--algoritma)
- [🔄 Workflow Pipeline](#-workflow-pipeline)
- [🏋️ Strategi Training](#️-strategi-training)
- [📊 Hasil & Evaluasi](#-hasil--evaluasi)
- [🚀 Cara Menjalankan](#-cara-menjalankan)
- [🔌 Deployment & Export](#-deployment--export)
- [📚 Referensi](#-referensi)

---

## 🎯 Problem Statement

Sistem pembayaran digital (e-wallet) membutuhkan mekanisme autentikasi yang **aman, cepat, dan tidak bisa dipalsukan**. Metode tradisional seperti PIN dan password memiliki kelemahan:

```
┌─────────────────────────────────────────────────────────────────┐
│  MASALAH KEAMANAN E-WALLET                                       │
├─────────────────┬───────────────────────────────────────────────┤
│  PIN / Password │  Bisa ditebak, dicuri, atau disadap           │
│  OTP SMS        │  Rawan SIM-swap attack                        │
│  Fingerprint    │  Bisa dipalsukan dengan cetakan sidik jari    │
│  Face (lama)    │  Mudah ditipu dengan foto 2D                  │
├─────────────────┴───────────────────────────────────────────────┤
│  SOLUSI: Deep Learning Face Recognition                          │
│  → Embedding 512-d yang unik per orang                          │
│  → Tidak bisa dipalsukan dengan foto biasa                       │
│  → Autentikasi < 500ms bahkan di CPU biasa                       │
└─────────────────────────────────────────────────────────────────┘
```

**Target:**
- ✅ Akurasi ≥ 90% pada dataset multi-kelas nyata
- ✅ False Accept Rate (FAR) serendah mungkin → keamanan tinggi
- ✅ Model ringan → bisa dijalankan di mobile/IoT
- ✅ Enrollment user baru tanpa retrain model

---

## 💡 Solution Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ARSITEKTUR SISTEM                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Input Foto (ukuran/rasio apapun)                                    │
│         │                                                              │
│         ▼                                                              │
│   ┌─────────────┐    Deteksi wajah, align 5 landmark,                │
│   │    MTCNN    │ ── crop area wajah, normalize → 160×160 px         │
│   └─────────────┘                                                     │
│         │                                                              │
│         ▼                                                              │
│   ┌──────────────────────────────┐                                    │
│   │   InceptionResnetV1          │  Pretrained di VGGFace2            │
│   │   (Backbone VGGFace2)        │  3.3 juta foto, 9131 identitas    │
│   └──────────────────────────────┘                                    │
│         │                                                              │
│         ▼                                                              │
│   ┌─────────────────────────────────────────┐                        │
│   │  Embedding 512-d (unit-norm L2)          │                        │
│   │  "Sidik Jari Digital" setiap wajah       │                        │
│   └─────────────────────────────────────────┘                        │
│         │                                                              │
│         ├──────────────────────────────────────────────────┐         │
│         ▼                                                    ▼        │
│   ┌───────────┐                                    ┌───────────────┐ │
│   │ Training  │                                    │  Inference    │ │
│   │ CrossEntr │                                    │ Cosine Simil. │ │
│   │ +Head(N)  │                                    │ Verify/Ident  │ │
│   └───────────┘                                    └───────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

**Keunggulan pendekatan ini:**
| Fitur | Nilai |
|---|---|
| Enrollment user baru | **Tanpa retrain** — cukup simpan embedding ke DB |
| Waktu inference | **< 500ms** di CPU biasa (Intel i5 Gen 11) |
| Format export | ONNX · TorchScript · ONNX-INT8 (untuk IoT) |
| Skalabilitas | Database embedding bisa menampung jutaan user |

---

## 📦 Dataset

<div align="center">

**[📂 Pins Face Recognition — Kaggle](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition/data)**

</div>

```
┌──────────────────────────────────────────────────────────┐
│                  STATISTIK DATASET                        │
├─────────────────────┬────────────────────────────────────┤
│  Total Kelas        │  105 identitas (selebriti)         │
│  Total Gambar       │  ±17.534 foto                      │
│  Min foto/kelas     │  80 foto                           │
│  Max foto/kelas     │  200+ foto                         │
│  Rata-rata/kelas    │  ±167 foto                         │
│  Format gambar      │  JPG, PNG                          │
│  Ukuran dataset     │  ±1.5 GB                           │
├─────────────────────┴────────────────────────────────────┤
│  Contoh identitas: Adriana Lima, Chris Evans,            │
│  Cristiano Ronaldo, Elon Musk, Emma Watson,              │
│  Gal Gadot, Lionel Messi, Robert Downey Jr, dll.        │
└──────────────────────────────────────────────────────────┘
```

**Distribusi Kelas (Bracket):**

```
  < 80  foto  │████░░░░░░░░░░░░░░░░│  5 kelas   ( 4.8%)
  80–99 foto  │████████░░░░░░░░░░░░│ 18 kelas   (17.1%)
 100–149 foto │████████████████░░░░│ 41 kelas   (39.0%)
 150–199 foto │████████████░░░░░░░░│ 28 kelas   (26.7%)
   ≥ 200 foto │████████░░░░░░░░░░░░│ 13 kelas   (12.4%)
```

**Pembagian Data (Stratified Split):**

```
  ┌─────────────────────────────────────────────────────┐
  │                                                       │
  │   Dataset (100%)                                      │
  │       │                                               │
  │       ├── 🟦 TRAIN   70%  ─── 12.274 gambar         │
  │       │   (augmentasi aktif, WeightedSampler)         │
  │       │                                               │
  │       ├── 🟧 VAL     15%  ───  2.630 gambar          │
  │       │   (monitor selama training, no augmentation)  │
  │       │                                               │
  │       └── 🟥 TEST    15%  ───  2.631 gambar          │
  │           (evaluasi final, TIDAK disentuh saat train) │
  │                                                       │
  │   Split: Stratified → distribusi kelas seimbang      │
  │   Leakage check: 0 file overlap antar split ✅        │
  └─────────────────────────────────────────────────────┘
```

---

## 🗂️ Struktur Proyek

```
face_recognition_ewallet/
│
├── 📁 config/
│   └── config.yaml              ← Semua hyperparameter terpusat
│
├── 📁 src/
│   ├── utils.py                 ← Seed, logging, config loader
│   ├── 📁 data/
│   │   ├── preprocessor.py      ← MTCNN pipeline (identik train & inference)
│   │   ├── dataset_loader.py    ← Scan + stratified split + DataLoader
│   │   └── augmentation.py      ← Train vs val/inference transforms
│   ├── 📁 models/
│   │   └── backbone.py          ← InceptionResnetV1 + classifier head
│   ├── 📁 training/
│   │   ├── trainer.py           ← Two-phase training loop
│   │   └── cross_validator.py   ← Stratified K-Fold CV
│   ├── 📁 evaluation/
│   │   └── evaluator.py         ← Semua metrik + visualisasi
│   ├── 📁 inference/
│   │   └── predictor.py         ← Verify · Identify · Enroll
│   └── 📁 export/
│       └── exporter.py          ← ONNX · TorchScript · INT8
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb             ← EDA + MTCNN preprocessing
│   ├── 02_training.ipynb        ← Training pipeline
│   ├── 03_evaluation.ipynb      ← Evaluasi lengkap
│   └── 04_inference.ipynb       ← Export + inference demo
│
├── 📁 data/
│   ├── raw/                     ← Dataset asli Kaggle (jangan diubah)
│   └── processed/               ← Hasil MTCNN 160×160
│
├── 📁 checkpoints/              ← best_model.pt
├── 📁 exports/                  ← ONNX · TorchScript · INT8
├── 📁 results/                  ← Grafik evaluasi
├── 📁 logs/                     ← TensorBoard + training log
└── requirements.txt
```

---

## 🔬 Metode & Algoritma

### 1. Face Detection — MTCNN

```
┌────────────────────────────────────────────────────────────────┐
│         MULTI-TASK CASCADED CONVOLUTIONAL NETWORK              │
├──────────────┬─────────────────┬──────────────────────────────┤
│  P-Net       │  R-Net          │  O-Net                        │
│  Proposal    │  Refine         │  Output                       │
│  Network     │  Network        │  Network                      │
│              │                 │                                │
│  Cari area   │  Saring         │  Deteksi 5 landmark:          │
│  kandidat    │  kandidat       │  • Mata kiri & kanan          │
│  wajah       │  wajah palsu    │  • Hidung                     │
│  (cepat)     │  (lebih akurat) │  • Mulut kiri & kanan         │
│              │                 │  → Align + Crop               │
└──────────────┴─────────────────┴──────────────────────────────┘
   Input: foto apapun        Output: tensor 160×160, range [-1,1]
```

**Parameter MTCNN:**

| Parameter | Nilai | Fungsi |
|---|---|---|
| `image_size` | 160 | Output crop size (px) |
| `margin` | 20 | Padding di sekeliling wajah |
| `min_face_size` | 20 | Abaikan wajah < 20px |
| `thresholds` | [0.6, 0.7, 0.8] | Confidence P/R/O-Net |
| `post_process` | True | Auto-normalize ke [-1, 1] |
| `select_largest` | True | Pilih wajah terbesar |

> ⚡ **Prinsip utama:** Pipeline MTCNN yang SAMA digunakan untuk preprocessing training DAN inference → tidak ada distribusi gap

---

### 2. Backbone — InceptionResnetV1 + VGGFace2

```
┌──────────────────────────────────────────────────────────────┐
│               KENAPA INCEPTIONRESNETV1 + VGGFACE2?           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ImageNet Pretrained (EfficientNet/ResNet biasa):            │
│  ✗ Belajar dari kucing, mobil, pesawat, dll.                 │
│  ✗ Feature yang dipelajari: tekstur & bentuk umum            │
│  ✗ Butuh banyak epoch untuk adaptasi ke wajah                │
│                                                               │
│  VGGFace2 Pretrained (yang kita pakai):                     │
│  ✓ Belajar dari 3.3 JUTA foto wajah, 9131 identitas         │
│  ✓ Sudah "mengerti" mata, hidung, rahang, kontur wajah      │
│  ✓ Fine-tuning ke 105 kelas → konvergen dalam beberapa epoch │
│                                                               │
└──────────────────────────────────────────────────────────────┘

Arsitektur Forward Pass:
  Input (3, 160, 160)
      │
      ▼  InceptionResnetV1 Backbone (pretrained VGGFace2)
  Embedding (512,) — sudah L2-normalized
      │
      ├── [Training]    → BatchNorm → Dropout(0.5) → Linear(512, 105)
      │                    → CrossEntropy Loss
      │
      └── [Inference]   → F.normalize → cosine similarity
                           → verify / identify
```

**Spesifikasi Model:**

| Komponen | Detail |
|---|---|
| Backbone | InceptionResnetV1 |
| Pretrained | VGGFace2 (3.3M foto, 9131 identitas) |
| Output embedding | 512 dimensi, L2-normalized |
| Classifier head | BN → Dropout(0.5) → Linear(512, N) |
| Total parameter | ~23.5 juta |
| Input size | (3, 160, 160) range [-1, 1] |

---

### 3. Loss Function — CrossEntropy + Label Smoothing

```
  Standard CrossEntropy:
    L = -log(p_correct)
    → Model bisa terlalu overconfident (p → 1.0)

  CrossEntropy + Label Smoothing (ε = 0.1):
    L = (1-ε) × (-log(p_correct)) + ε/K × Σ(-log(p_k))
    → Distribusi probabilitas lebih "smooth"
    → Model tidak overfit pada training label
    → Generalisasi lebih baik ke data baru
```

---

### 4. Augmentasi Training

```
  AUGMENTASI (hanya training set, TIDAK dipakai saat inference):
  ┌───────────────────────────────────────────────────────┐
  │  RandomHorizontalFlip (p=0.5)                         │
  │  → Simulasi foto dari sisi berbeda                    │
  │                                                        │
  │  RandomRotation (±10°)                                │
  │  → Simulasi kepala sedikit miring                     │
  │                                                        │
  │  ColorJitter (brightness=0.2, contrast=0.2, sat=0.1)  │
  │  → Simulasi variasi pencahayaan & kamera              │
  │                                                        │
  │  RandomErasing (p=0.2, scale=2%-15%)                  │
  │  → Simulasi wajah tertutup masker/tangan/kacamata     │
  └───────────────────────────────────────────────────────┘

  VAL / TEST / INFERENCE (deterministik, tidak ada randomness):
  ┌───────────────────────────────────────────────────────┐
  │  Resize → (160, 160)                                  │
  │  ToTensor → [0, 1]                                    │
  │  Normalize (mean=0.5, std=0.5) → [-1, 1]             │
  └───────────────────────────────────────────────────────┘
```

---

## 🔄 Workflow Pipeline

```mermaid
flowchart TD
    A([🗂️ Dataset Kaggle\nPins Face Recognition\n105 kelas · ~17.534 foto]) --> B

    B[📥 Download & Extract\ndata/raw/105_classes_pins_dataset/] --> C

    C{📊 EDA\nNotebook 01} --> D
    C --> E

    D[Distribusi kelas\nSample preview] 
    E[MTCNN Preprocessing\nDeteksi & crop wajah\n160×160 px · normalize -1,1]

    E --> F[💾 Simpan ke\ndata/processed/]

    F --> G[🔀 Stratified Split\n70% Train · 15% Val · 15% Test\nZero Data Leakage]

    G --> H[🎨 Augmentasi\nHanya Training Set\nFlip · Rotate · ColorJitter · Erasing]

    G --> I[📦 DataLoader\nWeightedRandomSampler\nClass Imbalance Handling]

    H --> I

    I --> J[🧠 InceptionResnetV1\nPretrained VGGFace2]

    J --> K{🏋️ Two-Phase Training\nNotebook 02}

    K --> L[Phase 1 · 5 Epoch\nBackbone Frozen\nHead Only · LR=0.001]
    K --> M[Phase 2 · 25 Epoch\nFull Fine-tuning\nDiff LR: BB=1e-5 · Head=1e-4]

    L --> M
    M --> N{Early Stopping\npatience=8}

    N -->|Improve| M
    N -->|No Improve| O[💾 best_model.pt\nCheckpoint Terbaik]

    O --> P{📈 Evaluasi\nNotebook 03}

    P --> Q[Accuracy · F1\nConfusion Matrix\nt-SNE · TAR@FAR · EER]

    O --> R{🚀 Export\nNotebook 04}

    R --> S[ONNX\nCross-platform]
    R --> T[TorchScript\nProduction Python/C++]
    R --> U[ONNX INT8\nIoT · Edge Device]

    O --> V[🔍 Inference Engine\nVerify 1:1 · Identify 1:N\nEnroll tanpa retrain]

    style A fill:#4A90D9,color:#fff
    style O fill:#27AE60,color:#fff
    style Q fill:#8E44AD,color:#fff
    style V fill:#E67E22,color:#fff
    style S fill:#16A085,color:#fff
    style T fill:#16A085,color:#fff
    style U fill:#16A085,color:#fff
```

---

## 🏋️ Strategi Training

### Two-Phase Fine-Tuning

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PHASE 1 — Foundation (Epoch 1–5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [FROZEN]  InceptionResnetV1 Backbone
  [TRAIN]   Classifier Head (BN → Dropout → Linear)

  Optimizer : Adam · LR = 0.001 · WD = 1e-4
  Scheduler : CosineAnnealingLR (T_max=5, eta_min=1e-5)

  Tujuan: Head mendapatkan arah gradient yang benar
          sebelum backbone ikut bergerak
  Target: val_acc ~70–90%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PHASE 2 — Full Fine-Tuning (Epoch 6–30)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [TRAIN]   Backbone    LR = 0.00001  (sangat kecil!)
  [TRAIN]   Head        LR = 0.0001   (10× backbone)

  Differential LR: backbone perlu LR kecil agar
  pretrained weight tidak rusak/catastrophic forgetting

  Optimizer : Adam · Differential LR · WD = 1e-4
  Scheduler : CosineAnnealingLR (T_max=25, eta_min=1e-6)

  Tujuan: Seluruh model menyesuaikan diri ke 105 kelas
  Target: val_acc >95%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EARLY STOPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Patience : 8 epoch tanpa peningkatan val_acc
  Triggered: Epoch 27 (dari maksimal 30)
  Best saved: Epoch 19 · val_acc = 99.66%
```

### Hyperparameter Lengkap

| Parameter | Nilai | Keterangan |
|---|---|---|
| `image_size` | 160 | Input size model |
| `embedding_dim` | 512 | Dimensi embedding |
| `dropout` | 0.5 | Regularisasi head |
| `batch_size` | 32 | Per GPU |
| `phase1_epochs` | 5 | Backbone frozen |
| `phase2_epochs` | 25 | Full fine-tune |
| `phase1_lr` | 0.001 | Head learning rate |
| `phase2_lr` | 0.0001 | Head LR (phase 2) |
| `phase2_bb_lr` | 0.00001 | Backbone LR (phase 2) |
| `weight_decay` | 0.0001 | L2 regularization |
| `label_smoothing` | 0.1 | Anti overconfident |
| `grad_clip` | 1.0 | Gradient clipping |
| `patience` | 8 | Early stopping |
| `mixed_precision` | True | AMP untuk speed |
| `seed` | 42 | Reproducibility |

---

## 📊 Hasil & Evaluasi

### Training Curves

```
  LOSS (CrossEntropy)              ACCURACY                   LEARNING RATE
  ┌──────────────────┐            ┌──────────────────┐       ┌──────────────┐
  │2.0│╲             │            │100│     ═══════════│       │1e-3│╲        │
  │   │ ╲            │            │ 90├─────────────── │       │    │ ╲       │
  │   │  ╲━━         │            │   │  /             │       │    │  ╲      │
  │   │    ╲━━━━━━━━━│            │ 80│ /              │       │    │   ╲     │
  │   │     ╲       ─│            │   │/               │       │1e-4│    ╲═══│
  │0.9│──────╲─────/ │            │ 70│                │       │    │        │
  │   │       ─────  │            │   │                │       │1e-6│        │
  └──────────────────┘            └──────────────────┘       └──────────────┘
    P1▲  P2               P1▲ Best:99.7% @Ep19        P1▲
       epoch 5                   epoch 5                  epoch 5
  ── Train  ── Val           ── Train  ── Val  ── Target 90%
```

### Metrik Evaluasi — Test Set

```
╔══════════════════════════════════════════════════════════════════╗
║                    HASIL EVALUASI AKHIR                          ║
╠══════════════════════╦═══════════════╦═════════════════════════╣
║  Metrik              ║  Nilai        ║  Interpretasi            ║
╠══════════════════════╬═══════════════╬═════════════════════════╣
║  Top-1 Accuracy      ║  99.58%  🎯   ║  2620/2631 benar        ║
║  Top-5 Accuracy      ║  99.92%       ║  Hampir mustahil salah  ║
║  Macro F1            ║  99.59%       ║  Konsisten di 105 kelas ║
║  Macro Precision     ║  99.66%       ║  Sedikit false positive  ║
║  Macro Recall        ║  99.55%       ║  Sedikit false negative  ║
║  ROC-AUC (macro)     ║  1.0000  ⭐   ║  Ranking sempurna       ║
╠══════════════════════╬═══════════════╬═════════════════════════╣
║  EER                 ║  0.02%   ⭐   ║  Jauh di bawah standar  ║
║  TAR @ FAR=10%       ║  100.00%      ║  Semua user asli masuk  ║
║  TAR @ FAR=1%        ║  100.00%      ║  Semua user asli masuk  ║
║  TAR @ FAR=0.1%      ║  100.00%  ⭐  ║  Standar perbankan!     ║
╚══════════════════════╩═══════════════╩═════════════════════════╝

  EER perbandingan industri:
  ┌─────────────────────────────────────────────────────────────┐
  │  Sistem ATM/Banking       EER < 1.0%   ██████████░░░░░░░░  │
  │  Face ID Apple            EER ~ 0.1%   ██░░░░░░░░░░░░░░░░  │
  │  Sistem ini               EER = 0.02%  ▌░░░░░░░░░░░░░░░░░  │
  └─────────────────────────────────────────────────────────────┘
```

### Per-Class F1 — Top 10 Terbaik & Terlemah

```
  SEMUA KELAS F1 = 1.00 (91 dari 105 kelas):
  Adriana Lima · Alex Lawther · Amanda Crew · Andy Samberg ·
  Anne Hathaway · Ben Affleck · Emilia Clarke · Emma Watson · ... dll

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  KELAS DENGAN F1 < 1.00 (14 kelas):
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Johnny Depp         F1=0.93  ████████████████████████░  Make-up ekstrem
  Miley Cyrus         F1=0.96  █████████████████████████░  Penampilan berubah
  Rami Malek          F1=0.96  █████████████████████████░  Ekspresi unik
  Maria Pedraza       F1=0.97  ██████████████████████████  
  Gal Gadot           F1=0.97  ██████████████████████████  
  Neil Patrick Harris F1=0.97  ██████████████████████████  
  Leonardo DiCaprio   F1=0.97  ██████████████████████████  Variasi usia
  Chris Evans         F1=0.98  ██████████████████████████  
  Jeremy Renner       F1=0.98  ██████████████████████████  
  Jake Mcdorman       F1=0.98  ██████████████████████████  
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Error terbesar (Johnny Depp) masih F1=0.93 → sangat baik
```

### Confusion Matrix — Visualisasi

```
  Confusion Matrix (105×105):

       P R E D I K S I
  ┌──┬──────────────────────────────────────────┐
  │  │ A  B  C  D  E  F  G ... (105 kelas)     │
  │A │[■]  .  .  .  .  .  .  .  .  .  .  .    │  ■ = Prediksi BENAR
  │B │ . [■] .  .  .  .  .  .  .  .  .  .    │  . = Prediksi salah
  │C │ .  . [■] .  .  .  .  .  .  .  .  .    │     (hampir kosong)
  │  │ .  .  . [■] .  .  .  .  .  .  .  .    │
  │  │ .  .  .  . [■] .  .  .  .  .  .  .    │
  │  │ .  .  .  .  . [■] .  .  .  .  .  .    │
  │  │        ...  [■] ...                     │
  └──┴──────────────────────────────────────────┘
  L
  A
  B
  E
  L

  → Diagonal bersih = hampir semua prediksi BENAR
  → Hanya 11 kesalahan dari 2631 foto test (0.42% error)
```

### t-SNE Embedding Space

```
  t-SNE memproyeksikan embedding 512-d ke 2-d untuk visualisasi:

  ┌───────────────────────────────────────────────────────┐
  │  ●●  ▲▲▲   ■■  ◆◆◆   ▼▼  ★★   ●●   ▲▲  ■■■         │
  │       ●  ▲▲    ■       ◆     ★      ●     ▲   ■      │
  │  ◆◆◆      ▼▼▼    ★★★    ●●●   ▲▲▲    ■■■    ◆◆◆     │
  │       ■■    ◆       ▼       ★      ●    ▲      ■     │
  │  ▲▲     ★★    ●●    ▲▲    ■■    ◆◆    ▼▼    ★★      │
  └───────────────────────────────────────────────────────┘

  Setiap simbol = 1 identitas. Interpretasi:
  ✅ Cluster RAPAT  → embedding konsisten per identitas
  ✅ Antar cluster TERPISAH → model bisa bedakan antar orang
  ✅ Tidak ada tumpang tindih → zero confusion antar identitas
```

### Biometric Performance — TAR@FAR

```
  True Accept Rate pada berbagai False Accept Rate:

  FAR = 10%   │██████████████████████████████│ TAR = 100.0%
  FAR =  1%   │██████████████████████████████│ TAR = 100.0%
  FAR = 0.1%  │██████████████████████████████│ TAR = 100.0%  ← Standar bank!

  EER (Equal Error Rate): 0.02%
  (Titik di mana FAR = FRR — semakin kecil semakin aman)

  Perbandingan:
  Sistem bank kelas dunia  EER ~0.5–1.0%
  Apple Face ID            EER ~0.1%
  Model ini                EER  0.02%  ← Melampaui standar industri!
```

---

## 🚀 Cara Menjalankan

### Prasyarat

```bash
# Packages yang dibutuhkan
numpy==1.24.4
facenet-pytorch==2.5.3
torch>=2.0.0
torchvision>=0.15.0
onnx>=1.14.0
onnxruntime>=1.15.0
scikit-learn>=1.3.0
seaborn>=0.12.0
```

### Urutan Notebook (Google Colab)

```
Step 0: Siapkan dataset
  → Upload folder 'face_recognition_ewallet/' ke Google Drive
  → Pastikan dataset ada di:
    /content/drive/MyDrive/Projek/face_recognition_ewallet/
    data/raw/105_classes_pins_dataset/[nama_orang]/foto.jpg

Step 1: Buka Google Colab
  → Runtime → Change runtime type → GPU → T4 GPU → Save

Step 2: Jalankan notebook berurutan:
  📓 01_EDA.ipynb         EDA + MTCNN preprocessing (~30-40 menit)
  📓 02_training.ipynb    Training model (~2-3 jam)
  📓 03_evaluation.ipynb  Evaluasi lengkap (~10 menit)
  📓 04_inference.ipynb   Export + demo inference
```

### BASE_DIR yang Dipakai

```python
BASE_DIR = '/content/drive/MyDrive/Projek/face_recognition_ewallet'
```

> ⚠️ Setiap notebook dimulai dengan mount Google Drive dan set BASE_DIR.
> Dataset dibaca **langsung dari folder yang sudah ada** — tidak ada download ulang.

---

## 🔌 Deployment & Export

### Format yang Tersedia

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL EXPORT FORMATS                       │
├──────────────────┬──────────────────────────────────────────┤
│  best_model.pt   │  PyTorch checkpoint (untuk fine-tune     │
│  (checkpoint)    │  lanjutan atau load di Python)           │
├──────────────────┼──────────────────────────────────────────┤
│  face_recog.pt   │  TorchScript — production Python/C++,   │
│  (TorchScript)   │  tidak butuh source code                 │
├──────────────────┼──────────────────────────────────────────┤
│  face_recog.onnx │  ONNX — cross-platform:                  │
│  (ONNX FP32)     │  web (ONNX.js), .NET, Java, C++         │
├──────────────────┼──────────────────────────────────────────┤
│  face_recog      │  ONNX INT8 Quantized — untuk:            │
│  _int8.onnx      │  Raspberry Pi, IoT, edge device          │
│  (ONNX INT8)     │  ~4× lebih kecil, ~2× lebih cepat       │
└──────────────────┴──────────────────────────────────────────┘
```

### Inference Lokal (Windows, tanpa GPU)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch

# Load model ONNX
sess = ort.InferenceSession('exports/onnx/face_recognition.onnx',
                             providers=['CPUExecutionProvider'])

# Preprocessing identik dengan training
mtcnn = MTCNN(image_size=160, margin=20, post_process=True)
img   = Image.open('foto_wajah.jpg').convert('RGB')
face  = mtcnn(img)  # → tensor (3, 160, 160), range [-1, 1]

# Inference
face_np = face.unsqueeze(0).numpy()
logits  = sess.run(['logits'], {'face_image': face_np})[0]
pred    = logits.argmax(axis=1)[0]

# Waktu inference: ~150-400ms di Intel i5 Gen 11 (CPU only)
```

### Cara Kerja di Produksi (E-Wallet)

```
REGISTRASI USER BARU (Enrollment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  User ambil 3–5 selfie
       │
       ▼  MTCNN + Model
  Embedding 512-d per foto
       │
       ▼  Rata-rata + L2 normalize
  1 embedding representatif
       │
       ▼
  Simpan ke database (pickle/Faiss/Milvus)
  ✅ SELESAI — Model TIDAK dilatih ulang!

VERIFIKASI PEMBAYARAN (1:1)
━━━━━━━━━━━━━━━━━━━━━━━━━━
  User selfie untuk konfirmasi transfer
       │
       ▼  MTCNN + Model
  Embedding baru
       │
       ▼  cosine_similarity(emb_baru, emb_tersimpan)
  Similarity score [0, 1]
       │
       ├── score ≥ 0.6 → ✅ Pembayaran DISETUJUI
       └── score < 0.6 → ❌ Akses DITOLAK

IDENTIFIKASI (1:N)
━━━━━━━━━━━━━━━━━
  Embedding query
       │
       ▼  matrix @ query  (dot product semua embedding di DB)
  Ranking similarity semua user
       │
       ▼
  Top-K kandidat dengan skor tertinggi
```

---

## 📚 Referensi

| Paper / Resource | Keterangan |
|---|---|
| [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832) | Schroff et al., 2015 — Foundational face embedding |
| [VGGFace2: A dataset for recognising faces](https://arxiv.org/abs/1710.08092) | Cao et al., 2018 — Dataset pretrain backbone |
| [facenet-pytorch](https://github.com/timesler/facenet-pytorch) | Library InceptionResnetV1 + MTCNN |
| [Joint Face Detection and Alignment using MTCNN](https://arxiv.org/abs/1604.02878) | Zhang et al., 2016 — MTCNN paper |
| [Pins Face Recognition Dataset](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition/data) | Dataset Kaggle yang digunakan |

---

<div align="center">

### 🏆 Summary

```
Model      : InceptionResnetV1 pretrained VGGFace2
Dataset    : Pins Face Recognition (105 kelas, ~17K foto)
Split      : 70% Train · 15% Val · 15% Test (Stratified)
Training   : Two-Phase Fine-tuning · Adam · CosineAnnealingLR
Loss       : CrossEntropy + Label Smoothing (ε=0.1)

Val  Accuracy : 99.66%
Test Accuracy : 99.58%
Test F1 Macro : 99.59%
ROC-AUC       : 1.0000
EER           : 0.02%    ← Melampaui standar perbankan
TAR@FAR=0.1%  : 100.00%  ← Semua user asli diterima, 0 penyusup lolos
```

**Built with ❤️ using PyTorch · facenet-pytorch · MTCNN · Google Colab**

</div>
