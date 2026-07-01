"""
src/counter.py

Penghitung kendaraan berbasis virtual line crossing.
Setiap track ID hanya dihitung SATU kali saat melewati garis,
mengatasi masalah double-counting per-frame pada Tugas 1.

Tugas 2 Visi Komputer — Universitas Mikroskil 2026
"""

from typing import Dict, List, Set
from src.detector import HasilDeteksi


class VehicleCounter:
    """
    Menghitung kendaraan yang melewati garis virtual horizontal.

    Logika:
        - Setiap frame, catat posisi cy (tengah-bawah) tiap track ID.
        - Kalau cy sebelumnya di satu sisi garis dan cy sekarang
          di sisi lain → kendaraan dihitung, track ID masuk set counted.
        - Track ID yang sudah counted tidak dihitung lagi.

    Params:
        line_position : posisi garis sebagai fraksi tinggi frame (default 0.5)
        frame_height  : tinggi frame dalam piksel
    """

    def __init__(self, line_position: float = 0.5, frame_height: int = 720):
        self.line_position = line_position
        self.frame_height = frame_height
        self.line_y_pos = int(frame_height * line_position)

        # track_id → posisi cy terakhir
        self._posisi_terakhir: Dict[int, int] = {}

        # track_id yang sudah melewati garis (sudah dihitung)
        self._sudah_dihitung: Set[int] = set()

        # akumulasi count per kelas
        self._count: Dict[str, int] = {
            "total": 0,
            "motor": 0,
            "mobil": 0,
            "bus": 0,
            "truk": 0,
        }

    @property
    def line_y(self) -> int:
        """Alias untuk kompatibilitas dengan visualizer."""
        return self.line_y_pos

    def update(self, deteksi_list: List[HasilDeteksi]) -> Dict[str, int]:
        """
        Update counter berdasarkan deteksi frame saat ini.

        Args:
            deteksi_list: list HasilDeteksi dari VehicleDetector

        Returns:
            dict count kumulatif {'total', 'motor', 'mobil', 'bus', 'truk'}
        """
        line_y = self.line_y_pos

        for det in deteksi_list:
            if det.track_id is None:
                continue

            tid = det.track_id
            cy_sekarang = det.cy

            if tid in self._posisi_terakhir:
                cy_sebelum = self._posisi_terakhir[tid]

                # Cek crossing: dari atas ke bawah ATAU dari bawah ke atas
                melewati = (
                    (cy_sebelum < line_y <= cy_sekarang) or
                    (cy_sebelum > line_y >= cy_sekarang)
                )

                if melewati and tid not in self._sudah_dihitung:
                    self._sudah_dihitung.add(tid)
                    kelas = det.class_name
                    self._count["total"] += 1
                    if kelas in self._count:
                        self._count[kelas] += 1

            self._posisi_terakhir[tid] = cy_sekarang

        return dict(self._count)

    def get_count_aktif(self, deteksi_list: List[HasilDeteksi]) -> int:
        """
        Hitung jumlah kendaraan yang TERLIHAT di frame saat ini (bukan kumulatif).

        Ini yang dipakai untuk klasifikasi kepadatan real-time.

        Args:
            deteksi_list: list HasilDeteksi frame saat ini

        Returns:
            int: jumlah bounding box aktif di frame
        """
        return len(deteksi_list)

    def reset(self):
        """Reset semua counter (untuk video baru)."""
        self._posisi_terakhir.clear()
        self._sudah_dihitung.clear()
        self._count = {
            "total": 0,
            "motor": 0,
            "mobil": 0,
            "bus": 0,
            "truk": 0,
        }
