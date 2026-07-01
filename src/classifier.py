"""
src/classifier.py

Klasifikasi kepadatan lalu lintas dua jalur:
    1. Rule-based  — threshold jumlah kendaraan aktif per frame
    2. K-Means     — clustering fitur (count, density_pct, avg_area)

Keduanya dijalankan paralel untuk validasi silang,
konsisten dengan metodologi Tugas 1.

Tugas 2 Visi Komputer — Universitas Mikroskil 2026
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


# Label cluster → status (ditetapkan setelah fitting berdasarkan centroid count)
_LABEL_STATUS = ["LENGANG", "LANCAR", "RAMAI", "PADAT"]


class DensityClassifier:
    """
    Klasifier kepadatan lalu lintas.

    Params:
        thresholds : dict threshold per kategori, misalnya:
                     {
                       'lengang': {'max': 10},
                       'lancar':  {'max': 20},
                       'ramai':   {'max': 35},
                       'padat':   {'max': 9999},
                     }
        n_clusters : jumlah kluster K-Means (default 4)
    """

    def __init__(self, thresholds: Optional[Dict] = None, n_clusters: int = 4):
        self.thresholds = thresholds or {
            "lengang": {"max": 10},
            "lancar":  {"max": 20},
            "ramai":   {"max": 35},
            "padat":   {"max": 9999},
        }
        self.n_clusters = n_clusters

        # Buffer fitur untuk K-Means (dikumpulkan sepanjang video)
        self._fitur_buffer: List[List[float]] = []

        # Model K-Means (difit di akhir)
        self._kmeans = None
        self._scaler = None
        self._label_map: Dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # Rule-based                                                           #
    # ------------------------------------------------------------------ #

    def klasifikasi_rule_based(self, count_aktif: int) -> str:
        """
        Klasifikasikan status berdasarkan jumlah kendaraan aktif.

        Args:
            count_aktif: jumlah kendaraan terdeteksi di frame saat ini

        Returns:
            str: 'LENGANG' | 'LANCAR' | 'RAMAI' | 'PADAT'
        """
        t = self.thresholds
        if count_aktif <= t["lengang"]["max"]:
            return "LENGANG"
        elif count_aktif <= t["lancar"]["max"]:
            return "LANCAR"
        elif count_aktif <= t["ramai"]["max"]:
            return "RAMAI"
        else:
            return "PADAT"

    # ------------------------------------------------------------------ #
    # K-Means                                                              #
    # ------------------------------------------------------------------ #

    def tambah_fitur(
        self,
        count_aktif: int,
        density_ratio: float,
        avg_area: float,
    ):
        """
        Tambahkan satu baris fitur ke buffer untuk K-Means.

        Dipanggil setiap frame selama pemrosesan video.

        Args:
            count_aktif  : jumlah kendaraan aktif di frame
            density_ratio: rasio total area bbox / area frame (0–1)
            avg_area     : rata-rata luas bbox per kendaraan (piksel²)
        """
        self._fitur_buffer.append([
            float(count_aktif),
            float(density_ratio * 100),   # ubah ke persen agar skalanya serasi
            float(avg_area),
        ])

    def klasifikasi_kmeans(
        self,
        count_aktif: int,
        density_ratio: float,
        avg_area: float,
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Klasifikasikan satu frame dengan model K-Means yang sudah difit.

        Kalau model belum difit (terlalu sedikit data), return (None, None).

        Returns:
            (status_str, cluster_id) atau (None, None)
        """
        if self._kmeans is None or self._scaler is None:
            return None, None

        fitur = np.array([[
            float(count_aktif),
            float(density_ratio * 100),
            float(avg_area),
        ]])

        try:
            fitur_scaled = self._scaler.transform(fitur)
            cluster_id = int(self._kmeans.predict(fitur_scaled)[0])
            status = self._label_map.get(cluster_id, "LENGANG")
            return status, cluster_id
        except Exception:
            return None, None

    def fit_akhir(self) -> Dict:
        """
        Fit K-Means pada seluruh buffer fitur yang terkumpul.

        Dipanggil setelah seluruh frame selesai diproses.

        Returns:
            dict metrik evaluasi K-Means
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        buf = self._fitur_buffer
        if len(buf) < self.n_clusters * 3:
            return {}

        X = np.array(buf, dtype=np.float32)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(
            n_clusters=self.n_clusters,
            n_init=15,
            random_state=42,
        )
        labels = km.fit_predict(X_scaled)

        self._scaler = scaler
        self._kmeans = km

        # Petakan kluster → status berdasarkan urutan centroid count_aktif
        centroid_count = km.cluster_centers_[:, 0]  # fitur pertama = count
        urutan = np.argsort(centroid_count)  # kluster dari paling sepi ke padat
        status_list = ["LENGANG", "LANCAR", "RAMAI", "PADAT"]
        self._label_map = {
            int(urutan[i]): status_list[i]
            for i in range(min(len(urutan), len(status_list)))
        }

        # Hitung metrik evaluasi
        metrik = {"total_frame": len(buf)}
        try:
            if len(set(labels)) >= 2:
                metrik["silhouette_score"] = float(
                    silhouette_score(X_scaled, labels)
                )
                metrik["davies_bouldin_index"] = float(
                    davies_bouldin_score(X_scaled, labels)
                )
        except Exception:
            pass

        return metrik

    def reset(self):
        """Reset buffer dan model (untuk video baru)."""
        self._fitur_buffer.clear()
        self._kmeans = None
        self._scaler = None
        self._label_map = {}
