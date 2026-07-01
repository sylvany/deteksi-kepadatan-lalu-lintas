"""
src/detector.py

Wrapper deteksi kendaraan menggunakan YOLOv8 + ByteTrack.
Tugas 2 Visi Komputer — Universitas Mikroskil 2026
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class HasilDeteksi:
    """
    Satu objek deteksi/tracking per frame.

    Atribut:
        track_id   : ID unik dari ByteTrack (None kalau tracking gagal)
        class_id   : indeks kelas (0=motor, 1=mobil, 2=bus, 3=truk)
        class_name : nama kelas teks
        confidence : skor keyakinan model (0–1)
        x1, y1     : sudut kiri-atas bounding box (piksel)
        x2, y2     : sudut kanan-bawah bounding box (piksel)
    """
    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        """Titik tengah horizontal bounding box."""
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        """Titik tengah vertikal bounding box."""
        return (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        """Luas bounding box dalam piksel kuadrat."""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def bbox(self):
        """Kembalikan (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)


NAMA_KELAS = ["motor", "mobil", "bus", "truk"]


class VehicleDetector:
    """
    Detektor kendaraan berbasis YOLOv8 + ByteTrack.

    Params:
        model_path  : path ke file best.pt hasil fine-tuning
        confidence  : minimum confidence threshold (default 0.4)
        iou         : IoU threshold untuk NMS (default 0.5)
        device      : 'cpu', 'cuda', atau 'mps'
        imgsz       : ukuran input inferensi (default 416; lebih kecil = lebih cepat)
    """

    def __init__(
        self,
        model_path: str = "models/best.pt",
        confidence: float = 0.4,
        iou: float = 0.5,
        device: str = "cpu",
        imgsz: int = 416,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.nama_kelas = NAMA_KELAS

    def deteksi_dengan_tracking(self, frame: np.ndarray) -> List[HasilDeteksi]:
        """
        Jalankan YOLOv8 + ByteTrack pada satu frame.

        Args:
            frame: gambar BGR dari OpenCV

        Returns:
            List[HasilDeteksi]: semua objek yang terdeteksi dan dilacak
        """
        hasil_list: List[HasilDeteksi] = []

        try:
            # persist=True wajib agar ByteTrack ingat track antar frame
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                stream=False,
            )
        except Exception:
            return hasil_list

        if not results or results[0].boxes is None:
            return hasil_list

        boxes = results[0].boxes

        # Kalau tidak ada track ID (ByteTrack belum terinisialisasi), pakai fallback
        ids = boxes.id  # bisa None
        for i, box in enumerate(boxes):
            try:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                # pastikan class_id valid
                if cls_id >= len(self.nama_kelas):
                    continue

                track_id = None
                if ids is not None:
                    track_id = int(ids[i].cpu().numpy())

                hasil_list.append(HasilDeteksi(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=self.nama_kelas[cls_id],
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
            except Exception:
                continue

        return hasil_list
