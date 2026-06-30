# 🚦 Sistem Deteksi & Pelacakan Kepadatan Lalu Lintas Berbasis Deep Learning

<div align="center">

**YOLOv8 (Transfer Learning + Fine-tuning) · ByteTrack · K-Means Clustering**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://traffic-detection-app.streamlit.app/)
&nbsp;&nbsp;
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
&nbsp;&nbsp;
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-purple?logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
&nbsp;&nbsp;
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
&nbsp;&nbsp;
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

<br/>

> **Tugas 2 — Visi Komputer | Universitas Mikroskil 2025/2026**
>
> Kelanjutan dari Tugas 1 (MOG2 + K-Means). Sistem deteksi, pelacakan, dan klasifikasi kepadatan lalu lintas menggunakan YOLOv8 hasil **transfer learning + fine-tuning** pada dataset Simpang Uniland Medan yang dikumpulkan dan dilabel secara mandiri, dipadukan dengan ByteTrack untuk penghitungan kendaraan unik.

**🔗 [Akses Aplikasi Web](https://traffic-detection-app.streamlit.app/) &nbsp;|&nbsp; [Repositori GitHub](https://github.com/sylvany/traffic-detection)**

</div>

---

## 📌 Deskripsi Proyek

Sistem ini adalah pengembangan dari Tugas 1, di mana pendekatan klasik MOG2 + K-Means digantikan dengan model deep learning **YOLOv8n** hasil fine-tuning pada dataset kendaraan Simpang Uniland Medan yang dikumpulkan dan dilabel sendiri oleh kelompok. Berbeda dengan background subtraction yang hanya mendeteksi objek bergerak, YOLOv8 mendeteksi kendaraan berdasarkan ciri visual, sehingga kendaraan yang berhenti (misalnya di lampu merah) tetap terdeteksi dengan konsisten.

Setiap kendaraan yang terdeteksi dilacak lintas frame menggunakan **ByteTrack**, diberikan identitas unik (track ID), lalu dihitung secara akurat melalui metode **virtual line crossing** — setiap kendaraan hanya dihitung satu kali, mengatasi keterbatasan counting per-frame pada Tugas 1. Status kepadatan lalu lintas diklasifikasikan menggunakan dua metode paralel (rule-based dan K-Means Clustering) dan distabilkan melalui **temporal smoothing** agar tidak berubah-ubah cepat antar frame.

**Lokasi pengujian:** Simpang Jalan MT. Haryono (Simpang Uniland), Kota Medan, Sumatera Utara — lokasi yang sama dengan Tugas 1.

**Data pengujian:** Tiga sesi rekaman video berdurasi 60 detik pada pukul **09.00**, **12.00**, dan **17.00 WIB**.

---

## 👥 Tim Pengembang

| Nama | NIM | Peran & Kontribusi |
|------|-----|--------------------|
| **Silvani Chayadi** | 231112945 | Dataset preparation, frame extraction, auto-labeling, training pipeline |
| **Cindy Nathania Perangin Angin** | 231111567 | `app.py` — Antarmuka Streamlit, integrasi inference, temporal smoothing |
| **Gloria Apriyanti Siagian** | 231111304 | Evaluasi model, dokumentasi riset, analisis perbandingan dengan Tugas 1 |

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 🎯 **Deteksi Berbasis Deep Learning** | YOLOv8n hasil fine-tuning mendeteksi 4 kelas kendaraan (motor, mobil, bus, truk) berdasarkan ciri visual, bukan sekadar pergerakan piksel |
| 🆔 **Vehicle Tracking dengan ByteTrack** | Setiap kendaraan mendapat track ID unik yang konsisten antar frame, tahan terhadap oklusi sementara |
| 📏 **Vehicle Counting Akurat** | Penghitungan via virtual line crossing — setiap kendaraan unik dihitung tepat satu kali, bukan estimasi per-frame |
| 🤖 **Klasifikasi Ganda (Rule-based + K-Means)** | Status kepadatan dihitung dua arah dan divalidasi silang, konsisten dengan metodologi Tugas 1 |
| ⏱️ **Temporal Smoothing** | Voting mayoritas 15 frame terakhir menstabilkan status yang ditampilkan tanpa mengubah data deteksi mentah |
| 📹 **Video Hasil Kompatibel Browser** | Re-encode otomatis ke H.264 (via `imageio-ffmpeg`) agar video hasil dapat diputar langsung di browser |
| ⚙️ **Parameter Dapat Dikonfigurasi** | Confidence threshold, IoU threshold, posisi garis counting, dan threshold kepadatan dapat disesuaikan melalui sidebar |
| 📊 **Evaluasi Model Lengkap** | Precision, Recall, F1-Score, mAP@0.5, mAP@0.5:0.95, confusion matrix, dan AP per kelas |

---

## 🔬 Pipeline Sistem

Sistem mengimplementasikan delapan tahap pemrosesan, menggantikan pipeline tujuh tahap berbasis MOG2 pada Tugas 1 dengan pendekatan deep learning end-to-end.

```
Video Input (1280×720, 3 sesi: 09:00 / 12:00 / 17:00 WIB)
        │
        ▼
Extract Frame
      Interval 5 frame · filter blur (Laplacian variance) ·
      filter duplikat (korelasi histogram HSV) · resize 640px
        │
        ▼
Auto-Labeling + Koreksi Manual (LabelImg)
      YOLOv8 pretrained sebagai annotation assistant →
      koreksi manual untuk akurasi label akhir
        │
        ▼
Dataset Split  (Train 70% · Val 20% · Test 10%)
      216 gambar · 967 instance · 4 kelas
        │
        ▼
Transfer Learning + Fine-tuning  (Google Colab, GPU T4)
      Base: yolov8n.pt (COCO) → detection head diganti 4 kelas
      100 epoch · AdamW · mosaic + cosine LR scheduler
        │
        ▼
Evaluasi Model
      Precision 95.68% · Recall 66.32% · F1-Score 78.34%
      mAP@0.5 66.55% · mAP@0.5:0.95 56.66%
        │
        ▼
┌──────────────────────────────────────────────────────┐
│                  INFERENCE LOOP (per frame)            │
│                                                          │
│  Object Detection (YOLOv8n fine-tuned)                  │
│         │                                                │
│         ▼                                                │
│  Vehicle Tracking (ByteTrack) → track ID per kendaraan   │
│         │                                                │
│         ▼                                                │
│  Vehicle Counting (Virtual Line Crossing)                │
│         │                                                │
│         ▼                                                │
│  Klasifikasi Kepadatan (Rule-based + K-Means)             │
│         │                                                │
│         ▼                                                │
│  Temporal Smoothing (voting mayoritas 15 frame)           │
└──────────────────────────────────────────────────────┘
        │
        ▼
Output: Video Beranotasi (H.264) · Grafik · CSV · JSON · Aplikasi Web
```

---

## 🤖 Klasifikasi Kepadatan: Rule-based + K-Means

Sama seperti Tugas 1, klasifikasi kepadatan dijalankan dua arah secara paralel untuk validasi silang.

| Fitur K-Means | Deskripsi | Skala Tipikal |
|-------|-----------|---------------|
| `count` | Jumlah kendaraan aktif terdeteksi per frame | 0 – 60 |
| `density_pct` | Persentase area bounding box terhadap area frame | 0 – 100% |
| `avg_area` | Rata-rata luas bounding box kendaraan | 0 – 20.000 px² |

Fitur dinormalisasi dengan **StandardScaler** sebelum klasterisasi `k=4` (`n_init=15`). Berbeda dari Tugas 1, input K-Means pada sistem ini bebas dari distorsi *contour merging* karena bounding box berasal langsung dari deteksi YOLOv8, bukan kontur hasil background subtraction.

**Kategori kepadatan** (dapat dikonfigurasi via sidebar):

| Status | Threshold Default |
|--------|--------------------|
| 🟢 Lengang | ≤ 10 kendaraan |
| 🟡 Lancar | 11 – 20 kendaraan |
| 🟠 Ramai | 21 – 35 kendaraan |
| 🔴 Padat | > 35 kendaraan |

---

## 📁 Struktur Repositori

```
traffic-detection/
│
├── dataset/
│   ├── raw_videos/              ← Video sumber 3 sesi (tidak di-commit)
│   ├── frames/                  ← Hasil ekstraksi frame
│   ├── auto_labeled/            ← Hasil auto-labeling + koreksi LabelImg
│   └── yolo_dataset/             ← Dataset final (train/val/test + data.yaml)
│
├── models/
│   └── best.pt                  ← Model hasil fine-tuning (download dari Drive)
│
├── src/
│   ├── frame_extractor.py       ← Ekstraksi frame dari video
│   ├── auto_labeler.py          ← Auto-labeling pakai YOLOv8 pretrained
│   ├── merge_dataset.py         ← Gabung + split dataset
│   ├── detector.py              ← Wrapper deteksi YOLOv8 + ByteTrack
│   ├── counter.py               ← Vehicle counting (line crossing)
│   ├── classifier.py            ← Klasifikasi kepadatan (rule-based + K-Means)
│   └── visualizer.py            ← Overlay bounding box & panel info
│
├── notebooks/
│   └── training_colab.ipynb     ← Notebook training (Google Colab, GPU T4)
│
├── config/
│   └── config.yaml              ← Konfigurasi model & threshold
│
├── results/
│   └── training/                ← Grafik, confusion matrix, metrik training
│
├── app.py                       ← Aplikasi Streamlit
├── requirements.txt
└── README.md
```

---

## 🛠 Instalasi & Menjalankan

### Prasyarat

- Python 3.11 atau lebih baru
- pip (sudah termasuk dalam instalasi Python standar)

### Langkah Instalasi

**1. Clone repositori**
```bash
git clone https://github.com/sylvany/traffic-detection.git
cd traffic-detection
```

**2. Buat dan aktifkan virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependensi**
```bash
pip install -r requirements.txt
pip install imageio-ffmpeg
```

> `imageio-ffmpeg` membawa binary ffmpeg sendiri agar video hasil deteksi bisa langsung diputar di browser — tidak perlu instalasi ffmpeg terpisah ke sistem.

**4. Download model**

Unduh `best.pt` hasil fine-tuning dari Google Drive kelompok, letakkan di folder `models/`.

**5. Jalankan aplikasi**
```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`. Atau akses langsung versi deploy di **[traffic-detection-app.streamlit.app](https://traffic-detection-app.streamlit.app/)** tanpa instalasi.

---

## 🧠 Melatih Ulang Model

```bash
# 1. Ekstraksi frame dari video baru
python src/frame_extractor.py

# 2. Auto-labeling pakai YOLOv8 pretrained sebagai annotation assistant
python src/auto_labeler.py

# 3. Koreksi manual label yang salah
labelImg

# 4. Gabung dan split dataset
python src/merge_dataset.py

# 5. Upload dataset/yolo_dataset/ ke Google Drive

# 6. Buka notebooks/training_colab.ipynb di Google Colab (GPU T4)
#    jalankan seluruh cell secara berurutan

# 7. Download best.pt hasil training, taruh di models/
```

---

## 🖥 Panduan Penggunaan Aplikasi

**Upload Video** — Unggah video lalu lintas (MP4, AVI, MOV), klik **🚀 Jalankan Deteksi**. Sistem akan menjalankan deteksi, tracking, counting, dan klasifikasi secara otomatis.

**Sidebar Konfigurasi** — Sesuaikan confidence threshold, IoU threshold, posisi garis counting, dan ambang batas kategori kepadatan tanpa mengubah kode.

**Hasil Deteksi** — Menampilkan total kendaraan per kelas, status kepadatan dominan, video hasil beranotasi (bounding box + track ID + garis counting), grafik kendaraan per waktu (mentah vs. smoothed), grafik status kepadatan, serta evaluasi K-Means (Silhouette Score, Davies-Bouldin Index).

**Download Hasil** — Video hasil deteksi (MP4 H.264), data statistik per frame (CSV), dan ringkasan statistik (JSON).

---

## ⚙️ Referensi Parameter

| Parameter | Default | Pengaruh |
|-----------|---------|---------|
| `confidence_threshold` | 0.4 | Semakin tinggi → deteksi lebih ketat, recall menurun, precision naik |
| `iou_threshold` | 0.5 | Threshold IoU untuk Non-Maximum Suppression antar bounding box |
| `line_position` | 0.5 | Posisi garis virtual counting (0.5 = tengah frame) |
| Jendela smoothing | 15 frame | Jumlah frame untuk voting mayoritas status; lebih besar → lebih stabil tapi kurang responsif |
| `track_buffer` (ByteTrack) | 30 frame | Berapa lama track dipertahankan saat kendaraan sementara tidak terdeteksi |

---

## 📊 Hasil Evaluasi (Test Set)

| Metrik | Tugas 1 (MOG2 + K-Means) | Tugas 2 (YOLOv8n Fine-tuned) | Δ |
|---|---|---|---|
| Precision | 73.4% | **95.68%** | +22.3% |
| Recall | 67.8% | 66.32% | −1.5% |
| F1-Score | 70.3% | **78.34%** | +8.0% |
| mAP@0.5 | — | 66.55% | — |
| mAP@0.5:0.95 | — | 56.66% | — |

AP@0.5 per kelas: motor 73.5% · mobil 89.7% · bus 39.5% · truk 63.5%

Detail lengkap dan analisis ada di `Laporan_Tugas2_VisiKomputer.md` BAB IV.

---

## ⚠️ Keterbatasan yang Diketahui

1. **Ukuran dataset relatif kecil.** 216 gambar dan 967 instance, lebih kecil dibanding dataset standar object detection, membatasi kemampuan generalisasi model terutama pada kelas minoritas.
2. **Recall kelas bus lebih rendah.** AP@0.5 bus pada test set (39.5%) jauh di bawah validation set (92.5%) akibat representasi instance yang masih terbatas pada test split.
3. **Selisih performa val vs. test.** mAP@0.5 validation (92.8%) dan test (66.55%) menunjukkan gap yang wajar untuk dataset berskala kecil, di mana satu-dua kasus sulit dapat menggeser metrik agregat secara signifikan.
4. **Kebutuhan GPU untuk training.** Fine-tuning yang efisien memerlukan akselerasi GPU; inferensi tetap dapat berjalan pada CPU dengan kecepatan memadai untuk analisis non-real-time.

---

## 🔭 Rencana Pengembangan

- [ ] Penambahan data, khususnya kelas bus dan truk, untuk meningkatkan recall kelas minoritas
- [ ] Augmentasi lanjutan (copy-paste, mixup) yang menyasar khusus kelas dengan instance sedikit
- [ ] Eksplorasi YOLOv8s/m apabila ukuran dataset bertambah signifikan
- [ ] Validasi pada lebih banyak lokasi persimpangan dan kondisi (malam hari, hujan)
- [ ] Integrasi penyimpanan data jangka panjang untuk analisis pola lalu lintas historis

---

<div align="center">

**🚦 Traffic Detection System v2 — YOLOv8 Fine-tuned**

Tugas 2 — Visi Komputer | Universitas Mikroskil 2025/2026

Silvani Chayadi · Cindy Nathania Perangin Angin · Gloria Apriyanti Siagian

🔗 **[Live Demo](https://traffic-detection-app.streamlit.app/)** &nbsp;|&nbsp; **[Source Code](https://github.com/sylvany/traffic-detection)**

</div>
