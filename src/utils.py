"""
utils.py — Fungsi-fungsi helper yang dipakai di seluruh proyek.
Isi: set seed, setup logging, load config, resolve path, detect device.
"""

import os, random, logging
from pathlib import Path

import yaml
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Paksa semua library pakai seed yang sama → hasil training reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # GPU pakai algoritma deterministik
    torch.backends.cudnn.benchmark = False       # matikan auto-tuning non-deterministik
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Pilih GPU jika tersedia, fallback ke CPU."""
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        print(f"🚀 GPU: {dev.name}  |  VRAM: {dev.total_memory/1e9:.1f} GB")
        return torch.device("cuda")
    print("⚠️  GPU tidak ditemukan — pakai CPU (training akan lambat)")
    return torch.device("cpu")


def load_config(config_path: str) -> dict:
    """Baca file YAML dan kembalikan sebagai dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict, base_dir: str) -> dict:
    """
    Tambahkan base_dir ke semua path relatif di config.
    Dipanggil sekali di awal setiap notebook setelah BASE_DIR di-set.
    """
    b = str(base_dir)
    cfg["data"]["dataset_root"]  = os.path.join(b, cfg["data"]["dataset_root"])
    cfg["data"]["processed_dir"] = os.path.join(b, cfg["data"]["processed_dir"])
    for key in cfg["paths"]:                                  # checkpoints, logs, dst
        cfg["paths"][key] = os.path.join(b, cfg["paths"][key])
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["data"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def setup_logging(log_dir: str, name: str = "run") -> None:
    """Arahkan semua log ke console DAN file .log di log_dir."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    # Console
    ch = logging.StreamHandler(); ch.setFormatter(fmt); root.addHandler(ch)
    # File
    fh = logging.FileHandler(f"{log_dir}/{name}.log", encoding="utf-8")
    fh.setFormatter(fmt); root.addHandler(fh)
