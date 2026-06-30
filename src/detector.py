"""
detector.py
Modul deteksi kendaraan menggunakan YOLOv8 hasil fine-tuning.

Perbedaan dengan Tugas 1 (MOG2):
  - MOG2 hanya deteksi objek BERGERAK (kendaraan statis = hilang)
  - YOLOv8 mendeteksi kendaraan berdasarkan VISUAL APPEARANCE
    sehingga kendaraan berhenti di lampu merah tetap terdeteksi

Kelompok: Silvani Chayadi, Cindy Nathania, Gloria Apriyanti
Universitas Mikroskil 2026
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Deteksi:
    """
    Representasi satu kendaraan terdeteksi dalam satu frame.
    Menggunakan dataclass agar atribut jelas dan mudah diakses.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def cx(self) -> float:
        """Koordinat X centroid bounding box."""
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        """Koordinat Y centroid bounding box."""
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        """Luas bounding box dalam piksel."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def bbox_int(self):
        """Bounding box dalam integer untuk fungsi OpenCV."""
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


class VehicleDetector:
    """
    Wrapper YOLOv8 untuk deteksi kendaraan.

    Menggunakan best.pt hasil fine-tuning pada dataset
    Simpang Uniland Medan — bukan model COCO original.

    Cara kerja inference YOLOv8:
      1. Frame di-resize ke IMGSZ secara internal
      2. Backbone CSPDarknet mengekstrak fitur multi-skala
      3. Detection head menghasilkan prediksi anchor-free
      4. NMS memfilter bounding box yang overlap
      5. Output: list bbox + class + confidence
    """

    NAMA_KELAS = ["motor", "mobil", "bus", "truk"]

    # Warna per kelas untuk visualisasi (BGR)
    WARNA_KELAS = {
        0: (0, 220, 0),      # motor — hijau
        1: (255, 150, 0),    # mobil — biru muda
        2: (0, 165, 255),    # bus   — oranye
        3: (0, 0, 220),      # truk  — merah
    }

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.4,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "cpu"
    ):
        """
        Args:
            model_path : path ke best.pt hasil fine-tuning
            confidence : threshold minimum confidence (0.0–1.0)
            iou        : threshold IoU untuk NMS
            imgsz      : ukuran input model
            device     : "cpu" atau "0" untuk GPU
        """
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}\n"
                f"Pastikan best.pt sudah didownload dari Google Drive "
                f"dan diletakkan di folder models/"
            )

        self.model      = YOLO(model_path)
        self.confidence = confidence
        self.iou        = iou
        self.imgsz      = imgsz
        self.device     = device

        print(f"  ✅ Model dimuat: {model_path}")
        print(f"     Kelas  : {self.NAMA_KELAS}")
        print(f"     Conf   : {confidence} | IoU: {iou}")
        print(f"     Device : {device}")

    def deteksi_dengan_tracking(
        self,
        frame: np.ndarray,
        tracker: str = "bytetrack.yaml"
    ) -> List[Deteksi]:
        """
        Deteksi + tracking ByteTrack dalam satu panggilan.

        ByteTrack (Zhang et al., 2022):
          - Mengasosiasikan bbox antar frame menggunakan IoU
          - Setiap kendaraan mendapat track_id unik yang konsisten
          - Deteksi confidence rendah tetap dimanfaatkan untuk
            mempertahankan track saat kendaraan oklusi sementara
          - Mengatasi ID-switch yang umum pada tracker IoU sederhana

        Args:
            frame   : gambar BGR dari cv2.VideoCapture
            tracker : file konfigurasi tracker

        Returns:
            list Deteksi dengan track_id terisi
        """
        results = self.model.track(
            source  = frame,
            conf    = self.confidence,
            iou     = self.iou,
            imgsz   = self.imgsz,
            device  = self.device,
            tracker = tracker,
            persist = True,    # pertahankan state track antar frame
            verbose = False,
        )

        return self._parse_results(results)

    def _parse_results(self, results) -> List[Deteksi]:
        """Parse output YOLO menjadi list Deteksi."""
        deteksi_list = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                tid   = int(box.id[0]) if box.id is not None else -1
                nama  = (self.NAMA_KELAS[cls]
                         if cls < len(self.NAMA_KELAS)
                         else f"kelas_{cls}")

                deteksi_list.append(Deteksi(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_id=cls,
                    class_name=nama,
                    track_id=tid
                ))

        return deteksi_list