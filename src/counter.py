"""
counter.py
Penghitungan kendaraan menggunakan Virtual Line Crossing.

Keunggulan vs counting per-frame (Tugas 1):
  Tugas 1: menghitung objek yang TERLIHAT di frame saat ini
           → kendaraan yang sama dihitung berulang tiap frame
  Tugas 2: menghitung kendaraan UNIK yang MELINTAS garis
           → setiap kendaraan dihitung tepat 1 kali

Cara kerja:
  1. Garis horizontal virtual di posisi tertentu (default: tengah)
  2. Setiap frame, catat posisi Y centroid per track_id
  3. Jika centroid berpindah melewati garis → hitung
  4. Set track_id yang sudah dihitung → tidak akan dihitung lagi

Kelompok: Silvani Chayadi, Cindy Nathania, Gloria Apriyanti
Universitas Mikroskil 2026
"""

from typing import Dict, Set, List
from src.detector import Deteksi


NAMA_KELAS = ["motor", "mobil", "bus", "truk"]


class VehicleCounter:
    """
    Counter kendaraan berbasis virtual line crossing.
    """

    def __init__(self, line_position: float = 0.5, frame_height: int = 480):
        """
        Args:
            line_position : posisi garis sebagai fraksi tinggi frame
                            0.5 = tengah, 0.3 = sepertiga dari atas
            frame_height  : tinggi frame dalam piksel
        """
        self.line_position = line_position
        self.frame_height  = frame_height
        self.line_y        = int(frame_height * line_position)

        # Posisi Y centroid frame sebelumnya per track_id
        # Format: {track_id: cy_frame_sebelumnya}
        self._posisi_prev: Dict[int, float] = {}

        # Set track_id yang sudah dihitung
        self._sudah_dihitung: Set[int] = set()

        # Counter kendaraan
        self._count: Dict[str, int] = {k: 0 for k in NAMA_KELAS}
        self._count["total"] = 0

    @property
    def line_y_pos(self) -> int:
        """Posisi Y garis virtual dalam piksel."""
        return self.line_y

    def update(self, deteksi_list: List[Deteksi]) -> Dict[str, int]:
        """
        Update counter dengan deteksi frame saat ini.

        Args:
            deteksi_list : hasil deteksi + tracking frame ini

        Returns:
            dict count kendaraan saat ini (per kelas + total)
        """
        for det in deteksi_list:
            # Skip jika tidak ada track_id valid
            if det.track_id is None or det.track_id < 0:
                continue

            tid        = det.track_id
            cy_skrg    = det.cy

            # Jika sudah pernah dihitung, skip tapi tetap update posisi
            if tid in self._sudah_dihitung:
                self._posisi_prev[tid] = cy_skrg
                continue

            # Cek line crossing
            if tid in self._posisi_prev:
                cy_prev = self._posisi_prev[tid]

                # Crossing: atas→bawah ATAU bawah→atas
                melintas = (
                    (cy_prev < self.line_y <= cy_skrg) or
                    (cy_prev > self.line_y >= cy_skrg)
                )

                if melintas:
                    # Tambah ke counter
                    if det.class_name in self._count:
                        self._count[det.class_name] += 1
                    self._count["total"] += 1

                    # Tandai sudah dihitung
                    self._sudah_dihitung.add(tid)

            self._posisi_prev[tid] = cy_skrg

        return self.get_count()

    def get_count(self) -> Dict[str, int]:
        """Ambil count saat ini (copy dict)."""
        return dict(self._count)

    def get_count_aktif(self, deteksi_list: List[Deteksi]) -> int:
        """
        Jumlah kendaraan AKTIF TERLIHAT frame ini.
        Digunakan untuk klasifikasi kepadatan real-time.
        """
        return len(deteksi_list)

    def update_frame_height(self, frame_height: int):
        """Update jika tinggi frame berubah."""
        self.frame_height = frame_height
        self.line_y       = int(frame_height * self.line_position)

    def reset(self):
        """Reset semua counter — panggil untuk video baru."""
        self._posisi_prev.clear()
        self._sudah_dihitung.clear()
        self._count = {k: 0 for k in NAMA_KELAS}
        self._count["total"] = 0