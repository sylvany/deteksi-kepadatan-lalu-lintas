"""
classifier.py
Klasifikasi kepadatan lalu lintas.

Dua metode dijalankan paralel — sama seperti Tugas 1
tapi input dari YOLO jauh lebih akurat:

  1. Rule-based : threshold dari config.yaml
                  cepat, deterministik, mudah dikonfigurasi
  2. K-Means    : clustering fitur multi-dimensi
                  sama persis dengan Tugas 1 untuk kontinuitas
                  input lebih akurat karena tidak ada contour merging

Kelompok: Silvani Chayadi, Cindy Nathania, Gloria Apriyanti
Universitas Mikroskil 2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Info tampilan per status
STATUS_INFO = {
    "LENGANG": {
        "label":     "🟢 LENGANG",
        "warna_bgr": (0, 200, 0),
        "warna_hex": "#2ECC71"
    },
    "LANCAR": {
        "label":     "🟡 LANCAR",
        "warna_bgr": (0, 200, 200),
        "warna_hex": "#F1C40F"
    },
    "RAMAI": {
        "label":     "🟠 RAMAI",
        "warna_bgr": (0, 130, 255),
        "warna_hex": "#E67E22"
    },
    "PADAT": {
        "label":     "🔴 PADAT",
        "warna_bgr": (0, 0, 220),
        "warna_hex": "#E74C3C"
    },
}

STATUS_URUTAN = ["LENGANG", "LANCAR", "RAMAI", "PADAT"]


class DensityClassifier:
    """
    Klasifikasi kepadatan lalu lintas.
    """

    def __init__(self, thresholds: dict):
        """
        Args:
            thresholds : dict dari config.yaml
                         {"lengang": {"max": 10}, "lancar": {"max": 20}, ...}
        """
        self.thresholds = thresholds

        # Buffer fitur untuk K-Means [count, density_pct, avg_area]
        # Buffer ini TERUS BERTAMBAH setiap frame, dipakai sebagai
        # riwayat data untuk fit ulang berkala
        self._buffer: List[List[float]] = []

        self._kmeans: Optional[KMeans]        = None
        self._scaler: Optional[StandardScaler] = None
        self._kmeans_ready                     = False

        self._buffer_saat_fit: Optional[np.ndarray] = None

    def klasifikasi_rule_based(self, count_aktif: int) -> str:
        """
        Klasifikasi berdasarkan threshold dari config.

        Args:
            count_aktif : jumlah kendaraan terlihat di frame ini

        Returns:
            str: "LENGANG" / "LANCAR" / "RAMAI" / "PADAT"
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

    def tambah_fitur(
        self,
        count: int,
        density_ratio: float,
        avg_box_area: float
    ):
        """
        Tambah fitur frame ini ke buffer K-Means.

        Fitur sama dengan Tugas 1:
          count        : jumlah kendaraan aktif
          density_pct  : rasio area bbox / area frame x 100
          avg_box_area : rata-rata luas bbox kendaraan

        Fit ulang K-Means setiap 50 frame terkumpul.
        """
        self._buffer.append([count, density_ratio * 100, avg_box_area])

        if len(self._buffer) >= 50 and len(self._buffer) % 50 == 0:
            self._fit_kmeans()

    def _fit_kmeans(self):
        """
        Fit KMeans dengan SELURUH data buffer saat ini, lalu simpan
        snapshot-nya agar konsisten dengan self._kmeans.labels_.
        """
        if len(self._buffer) < 4:
            return

        # Ambil snapshot PERSIS saat ini -> akan dipakai konsisten
        # bersamaan dengan self._kmeans.labels_ yang dihasilkan
        X = np.array(self._buffer)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._kmeans = KMeans(
            n_clusters   = 4,
            n_init       = 15,
            random_state = 42
        )
        self._kmeans.fit(X_scaled)
        self._kmeans_ready = True

        self._buffer_saat_fit = X

    def klasifikasi_kmeans(
        self,
        count: int,
        density_ratio: float,
        avg_box_area: float
    ) -> Tuple[str, float]:
        """
        Klasifikasi K-Means untuk frame saat ini.

        Returns:
            tuple: (status, confidence_proxy)
                   confidence adalah 1 - jarak_ke_centroid (proxy)
        """
        if not self._kmeans_ready or self._buffer_saat_fit is None:
            return self.klasifikasi_rule_based(count), 0.0

        fitur        = np.array([[count, density_ratio * 100, avg_box_area]])
        fitur_scaled = self._scaler.transform(fitur)

        cluster_id = int(self._kmeans.predict(fitur_scaled)[0])

        labels_arr = self._kmeans.labels_
        buffer_arr = self._buffer_saat_fit

        # Map cluster ke status berdasarkan urutan rata-rata count
        centroid_counts = []
        for k in range(4):
            mask = labels_arr == k
            # mask dan buffer_arr sekarang DIJAMIN ukurannya sama
            mean_count = buffer_arr[mask, 0].mean() if mask.sum() > 0 else 0
            centroid_counts.append((k, mean_count))

        centroid_counts.sort(key=lambda x: x[1])
        cluster_ke_status = {
            k: STATUS_URUTAN[i]
            for i, (k, _) in enumerate(centroid_counts)
        }

        status = cluster_ke_status.get(cluster_id, "LANCAR")

        # Hitung jarak ke centroid sebagai proxy confidence
        jarak      = self._kmeans.transform(fitur_scaled)[0][cluster_id]
        confidence = max(0.0, 1.0 - jarak / 10.0)

        return status, confidence

    def fit_akhir(self) -> dict:
        """
        Fit K-Means final setelah seluruh video selesai.
        Hitung Silhouette Score dan Davies-Bouldin Index.

        Returns:
            dict metrik kualitas klasterisasi
        """
        if len(self._buffer) < 8:
            return {"error": "Data tidak cukup untuk K-Means"}

        from sklearn.metrics import silhouette_score, davies_bouldin_score

        X = np.array(self._buffer)
        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(X)

        self._kmeans = KMeans(
            n_clusters   = 4,
            n_init       = 15,
            random_state = 42
        )
        labels = self._kmeans.fit_predict(X_scaled)
        self._kmeans_ready = True

        self._buffer_saat_fit = X

        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
        dbi = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0

        unique, counts = np.unique(labels, return_counts=True)

        metrik = {
            "silhouette_score":     float(sil),
            "davies_bouldin_index": float(dbi),
            "total_frame":          len(self._buffer),
            "distribusi_klaster":   dict(zip(unique.tolist(), counts.tolist()))
        }

        print(f"\n  K-Means Final (k=4):")
        print(f"  Silhouette Score : {sil:.4f} (mendekati 1 = bagus)")
        print(f"  Davies-Bouldin   : {dbi:.4f} (mendekati 0 = bagus)")
        print(f"  Total frame      : {len(self._buffer)}")

        return metrik

    def get_status_info(self, status: str) -> dict:
        """Ambil info tampilan (label, warna) untuk status tertentu."""
        return STATUS_INFO.get(status, STATUS_INFO["LANCAR"])
