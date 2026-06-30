"""
visualizer.py
Fungsi visualisasi: bounding box, garis counting, panel info.

Kelompok: Silvani Chayadi, Cindy Nathania, Gloria Apriyanti
Universitas Mikroskil 2026
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from src.detector import Deteksi
from src.classifier import STATUS_INFO


NAMA_KELAS = ["motor", "mobil", "bus", "truk"]

WARNA_KELAS = {
    0: (0, 220, 0),
    1: (255, 150, 0),
    2: (0, 165, 255),
    3: (0, 0, 220),
}


def gambar_deteksi(
    frame: np.ndarray,
    deteksi_list: List[Deteksi],
    tampilkan_track_id: bool = True,
    tampilkan_confidence: bool = False
) -> np.ndarray:
    """
    Gambar bounding box + label untuk setiap kendaraan.
    """
    output = frame.copy()

    for det in deteksi_list:
        x1, y1, x2, y2 = det.bbox_int
        warna = WARNA_KELAS.get(det.class_id, (180, 180, 180))

        # Bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), warna, 2)

        # Susun label
        if tampilkan_track_id and det.track_id is not None and det.track_id >= 0:
            label = f"{det.class_name} #{det.track_id}"
        else:
            label = det.class_name

        if tampilkan_confidence:
            label += f" {det.confidence:.2f}"

        # Background teks
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        cv2.rectangle(
            output,
            (x1, y1 - th - 6),
            (x1 + tw + 4, y1),
            warna, -1
        )
        cv2.putText(
            output, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 0, 0), 1, cv2.LINE_AA
        )

        # Titik centroid
        cv2.circle(output, (int(det.cx), int(det.cy)), 3, warna, -1)

    return output


def gambar_garis_counting(
    frame: np.ndarray,
    line_y: int,
    warna: tuple = (0, 255, 255),
    tebal: int = 2
) -> np.ndarray:
    """
    Gambar garis virtual counting horizontal.
    """
    output = frame.copy()
    h, w   = output.shape[:2]

    cv2.line(output, (0, line_y), (w, line_y), warna, tebal)
    cv2.putText(
        output, "COUNTING LINE",
        (10, line_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        warna, 1, cv2.LINE_AA
    )
    return output


def gambar_panel_info(
    frame: np.ndarray,
    status_rule: str,
    count_aktif: int,
    count_total: Dict[str, int],
    frame_idx: int,
    fps_proses: float,
    status_kmeans: Optional[str] = None
) -> np.ndarray:
    """
    Panel info semi-transparan di pojok kiri atas.
    Menampilkan: status, jumlah kendaraan per kelas, FPS.
    """
    output  = frame.copy()
    info    = STATUS_INFO.get(status_rule, STATUS_INFO["LANCAR"])
    warna_s = info["warna_bgr"]

    panel_h = 210
    panel_w = 265

    # Background semi-transparan
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, output, 0.35, 0, output)

    # Status rule-based
    cv2.rectangle(output, (8, 8), (panel_w - 8, 40), warna_s, -1)
    cv2.putText(
        output, info["label"],
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (0, 0, 0), 2, cv2.LINE_AA
    )

    # Status K-Means
    if status_kmeans:
        info_km = STATUS_INFO.get(status_kmeans, STATUS_INFO["LANCAR"])
        cv2.putText(
            output, f"K-Means: {info_km['label']}",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (190, 190, 190), 1, cv2.LINE_AA
        )

    # Jumlah kendaraan aktif & total
    cv2.putText(
        output, f"Aktif  : {count_aktif} kendaraan",
        (10, 76),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        output, f"Melintas: {count_total.get('total', 0)} total",
        (10, 94),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (200, 200, 200), 1, cv2.LINE_AA
    )

    # Count per kelas
    y = 114
    for i, nama in enumerate(NAMA_KELAS):
        jumlah  = count_total.get(nama, 0)
        warna_k = WARNA_KELAS.get(i, (180, 180, 180))
        cv2.circle(output, (16, y - 4), 5, warna_k, -1)
        cv2.putText(
            output, f"{nama:6s}: {jumlah}",
            (26, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (210, 210, 210), 1, cv2.LINE_AA
        )
        y += 18

    # Frame & FPS
    cv2.putText(
        output, f"Frame {frame_idx}  |  {fps_proses:.1f} fps",
        (10, panel_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.36,
        (130, 130, 130), 1, cv2.LINE_AA
    )

    return output


def buat_frame_lengkap(
    frame: np.ndarray,
    deteksi_list: List[Deteksi],
    line_y: int,
    status_rule: str,
    count_aktif: int,
    count_total: Dict[str, int],
    frame_idx: int,
    fps_proses: float,
    status_kmeans: Optional[str] = None,
    tampilkan_track_id: bool = True
) -> np.ndarray:
    """
    Gabungkan semua layer anotasi menjadi satu frame.
    Urutan: garis counting → bbox deteksi → panel info.
    """
    hasil = gambar_garis_counting(frame, line_y)
    hasil = gambar_deteksi(hasil, deteksi_list, tampilkan_track_id)
    hasil = gambar_panel_info(
        hasil, status_rule, count_aktif,
        count_total, frame_idx, fps_proses, status_kmeans
    )
    return hasil