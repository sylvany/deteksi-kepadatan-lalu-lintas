"""
src/visualizer.py

Overlay anotasi pada frame video:
    - Bounding box berwarna per kelas kendaraan
    - Track ID dan confidence di atas bbox
    - Garis virtual counting (warna berubah saat kendaraan dekat)
    - Panel info kiri-atas (status, count aktif, count total)
    - Label status kepadatan kanan-atas

Tugas 2 Visi Komputer — Universitas Mikroskil 2026
"""

from typing import Dict, List, Optional
import cv2
import numpy as np

from src.detector import HasilDeteksi


# Warna per kelas (BGR)
WARNA_KELAS = {
    "motor": (0, 255, 0),      # hijau
    "mobil": (255, 165, 0),    # biru-muda (BGR: oranye → ubah ke biru)
    "bus":   (0, 128, 255),    # oranye-kebiruan
    "truk":  (0, 0, 255),      # merah
}
# fallback
WARNA_DEFAULT = (200, 200, 200)

# Warna status (BGR)
WARNA_STATUS = {
    "LENGANG": (39, 174, 96),    # hijau
    "LANCAR":  (52, 152, 219),   # biru
    "RAMAI":   (230, 126, 34),   # oranye
    "PADAT":   (231, 76, 60),    # merah
}


def buat_frame_lengkap(
    frame: np.ndarray,
    deteksi_list: List[HasilDeteksi],
    line_y: int,
    status_rule: str,
    count_aktif: int,
    count_total: Dict[str, int],
    frame_idx: int,
    fps_proses: float,
    status_kmeans: Optional[str] = None,
    tampilkan_track_id: bool = True,
) -> np.ndarray:
    """
    Gambarkan semua anotasi ke satu frame.

    Args:
        frame            : frame BGR asli (tidak dimodifikasi, di-copy dulu)
        deteksi_list     : list HasilDeteksi dari VehicleDetector
        line_y           : posisi garis counting (piksel y)
        status_rule      : status rule-based ('LENGANG' dst)
        count_aktif      : jumlah kendaraan aktif di frame ini
        count_total      : dict count kumulatif {'total', 'motor', ...}
        frame_idx        : nomor frame (untuk debug)
        fps_proses       : FPS pemrosesan (untuk debug)
        status_kmeans    : status K-Means (bisa None kalau belum fit)
        tampilkan_track_id: tampilkan ID tracking di atas bbox

    Returns:
        np.ndarray: frame BGR beranotasi
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # ------------------------------------------------------------------ #
    # 1. Bounding box + label per kendaraan                               #
    # ------------------------------------------------------------------ #
    for det in deteksi_list:
        warna = WARNA_KELAS.get(det.class_name, WARNA_DEFAULT)
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

        # Kotak
        cv2.rectangle(canvas, (x1, y1), (x2, y2), warna, 2)

        # Label teks
        if tampilkan_track_id and det.track_id is not None:
            teks = f"#{det.track_id} {det.class_name} {det.confidence:.2f}"
        else:
            teks = f"{det.class_name} {det.confidence:.2f}"

        (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(y1 - 4, th + 4)

        cv2.rectangle(canvas, (x1, label_y - th - 4), (x1 + tw + 4, label_y), warna, -1)
        cv2.putText(
            canvas, teks,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

        # Titik tengah
        cv2.circle(canvas, (det.cx, det.cy), 3, warna, -1)

    # ------------------------------------------------------------------ #
    # 2. Garis virtual counting                                           #
    # ------------------------------------------------------------------ #
    # Warna garis: merah kalau ada kendaraan dekat garis, kuning kalau tidak
    dekat_garis = any(abs(det.cy - line_y) < 40 for det in deteksi_list)
    warna_garis = (0, 0, 255) if dekat_garis else (0, 255, 255)

    cv2.line(canvas, (0, line_y), (w, line_y), warna_garis, 2)
    cv2.putText(
        canvas, "COUNTING LINE",
        (10, line_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        warna_garis, 1, cv2.LINE_AA,
    )

    # ------------------------------------------------------------------ #
    # 3. Panel info kiri-atas (semi-transparan)                           #
    # ------------------------------------------------------------------ #
    panel_w, panel_h = 260, 170
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    baris_info = [
        f"Frame  : {frame_idx}",
        f"FPS    : {fps_proses:.1f}",
        f"Aktif  : {count_aktif} kendaraan",
        f"Total  : {count_total.get('total', 0)} kendaraan",
        f"Motor  : {count_total.get('motor', 0)}  Mobil: {count_total.get('mobil', 0)}",
        f"Bus    : {count_total.get('bus', 0)}  Truk : {count_total.get('truk', 0)}",
    ]
    for i, baris in enumerate(baris_info):
        cv2.putText(
            canvas, baris,
            (14, 28 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (230, 230, 230), 1, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------ #
    # 4. Status kepadatan kanan-atas                                      #
    # ------------------------------------------------------------------ #
    warna_s = WARNA_STATUS.get(status_rule, (180, 180, 180))

    label_rule = f"RULE: {status_rule}"
    (rw, rh), _ = cv2.getTextSize(label_rule, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    rx = w - rw - 14

    cv2.rectangle(canvas, (rx - 6, 8), (w - 6, 8 + rh + 10), (20, 20, 20), -1)
    cv2.putText(
        canvas, label_rule,
        (rx, 8 + rh + 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        warna_s, 2, cv2.LINE_AA,
    )

    # Tampilkan K-Means status kalau ada
    if status_kmeans:
        warna_km = WARNA_STATUS.get(status_kmeans, (180, 180, 180))
        label_km = f"KM  : {status_kmeans}"
        (kmw, kmh), _ = cv2.getTextSize(label_km, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        kmx = w - kmw - 14

        cv2.putText(
            canvas, label_km,
            (kmx, 8 + rh + 10 + kmh + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            warna_km, 1, cv2.LINE_AA,
        )

    return canvas
