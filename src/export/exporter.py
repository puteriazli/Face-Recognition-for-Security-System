"""
exporter.py — Export model ke berbagai format deployment.

Format:
  TorchScript (.pt)     → production Python/C++
  ONNX (.onnx)          → web, .NET, Java, C++, cross-platform
  ONNX INT8 (.onnx)     → IoT, Raspberry Pi, edge device (lebih kecil & cepat)
"""

import logging
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

log = logging.getLogger(__name__)


class Exporter:

    def __init__(self, model: torch.nn.Module, cfg: dict, device: torch.device):
        self.model  = model.eval().to(device)
        self.cfg    = cfg
        self.device = device
        self.size   = cfg["data"]["image_size"]          # 160
        self.out    = Path(cfg["paths"]["exports"])
        self.out.mkdir(parents=True, exist_ok=True)

    def _dummy(self, batch: int = 1) -> torch.Tensor:
        """Input dummy untuk tracing: (B, 3, 160, 160) range [-1, 1]."""
        return torch.randn(batch, 3, self.size, self.size, device=self.device)

    # ── TorchScript ──────────────────────────────────────────────────────────

    def to_torchscript(self) -> str:
        path = str(self.out / "torchscript" / "face_recognition.pt")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        traced = torch.jit.trace(self.model, self._dummy())
        traced.save(path)

        size_mb = Path(path).stat().st_size / 1e6
        log.info(f"TorchScript → {path}  ({size_mb:.1f} MB)")
        return path

    # ── ONNX ─────────────────────────────────────────────────────────────────

    def to_onnx(self, opset: int = 17) -> str:
        path = str(self.out / "onnx" / "face_recognition.onnx")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self.model, self._dummy(),
            path,
            opset_version=opset,
            input_names=["face_image"],           # nama input node
            output_names=["logits"],              # nama output node
            dynamic_axes={                        # batch size bisa berubah
                "face_image": {0: "batch"},
                "logits":     {0: "batch"},
            },
            do_constant_folding=True,             # optimasi konstanta saat export
        )

        # Verifikasi ONNX model valid
        onnx.checker.check_model(onnx.load(path))

        # Verifikasi output ONNX sama dengan PyTorch
        dummy_np = self._dummy().cpu().numpy()
        pt_out   = self.model(torch.tensor(dummy_np).to(self.device)).detach().cpu().numpy()
        sess     = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ort_out  = sess.run(["logits"], {"face_image": dummy_np})[0]
        max_diff = float(np.abs(pt_out - ort_out).max())

        if max_diff < 1e-4:
            log.info(f"✅ ONNX valid (max diff PyTorch vs ORT: {max_diff:.2e})")
        else:
            log.warning(f"⚠️  ONNX diff agak besar: {max_diff:.2e}")

        size_mb = Path(path).stat().st_size / 1e6
        log.info(f"ONNX → {path}  ({size_mb:.1f} MB)")
        return path

    # ── ONNX INT8 Quantization ────────────────────────────────────────────────

    def to_onnx_int8(self, onnx_path: str = None) -> str:
        """
        Quantize ONNX ke INT8 untuk deployment di edge device.
        Ukuran ~4× lebih kecil, inference ~2× lebih cepat di CPU.
        """
        if onnx_path is None:
            onnx_path = str(self.out / "onnx" / "face_recognition.onnx")

        out_path = str(self.out / "onnx" / "face_recognition_int8.onnx")

        quantize_dynamic(
            model_input=onnx_path,
            model_output=out_path,
            weight_type=QuantType.QInt8,
        )

        size_mb = Path(out_path).stat().st_size / 1e6
        log.info(f"ONNX INT8 → {out_path}  ({size_mb:.1f} MB)")
        return out_path

    def export_all(self) -> dict:
        """Export semua format sekaligus."""
        log.info("Memulai export semua format ...")
        paths = {}
        try:    paths["torchscript"] = self.to_torchscript()
        except Exception as e: log.error(f"TorchScript gagal: {e}")
        try:
            paths["onnx"]     = self.to_onnx()
            paths["onnx_int8"] = self.to_onnx_int8(paths["onnx"])
        except Exception as e: log.error(f"ONNX gagal: {e}")
        log.info("Export selesai.")
        return paths
