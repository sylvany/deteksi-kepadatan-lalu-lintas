"""
app.py

Aplikasi Streamlit — Deteksi & Klasifikasi Kepadatan Lalu Lintas
Tugas 2 Visi Komputer — Universitas Mikroskil 2026

Kelompok 3:
    Silvani Chayadi         NIM 231112945
    Cindy Nathania P.A.     NIM 231111567
    Gloria Apriyanti S.     NIM 231111304

Cara menjalankan:
    streamlit run app.py
"""

import os
import json
import shutil
import subprocess
import tempfile
import time

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.detector import VehicleDetector
from src.counter import VehicleCounter
from src.classifier import DensityClassifier
from src.visualizer import buat_frame_lengkap

# ================================================================
# KONFIGURASI HALAMAN
# ================================================================

st.set_page_config(
    page_title="Deteksi Kepadatan Lalu Lintas",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    font-size: 2rem; font-weight: 700; color: #2C3E50;
    text-align: center; padding: 1rem 0 0.5rem 0;
}
.sub-title {
    font-size: 0.95rem; color: #7F8C8D;
    text-align: center; margin-bottom: 1.5rem; line-height: 1.6;
}
.status-lengang { background:#D5F5E3; color:#1E8449;
    padding:10px 18px; border-radius:8px;
    font-weight:bold; text-align:center; font-size:1.1rem; }
.status-lancar  { background:#FCF3CF; color:#B7950B;
    padding:10px 18px; border-radius:8px;
    font-weight:bold; text-align:center; font-size:1.1rem; }
.status-ramai   { background:#FAE5D3; color:#BA4A00;
    padding:10px 18px; border-radius:8px;
    font-weight:bold; text-align:center; font-size:1.1rem; }
.status-padat   { background:#FADBD8; color:#943126;
    padding:10px 18px; border-radius:8px;
    font-weight:bold; text-align:center; font-size:1.1rem; }
</style>
""", unsafe_allow_html=True)


# ================================================================
# FIX VIDEO — RE-ENCODE KE H.264 AGAR TAMPIL DI BROWSER
# ================================================================

def cari_ffmpeg() -> str:
    """
    Cari ffmpeg yang bisa dipakai, urutan prioritas:
    1. imageio-ffmpeg (binary Python, tidak bergantung PATH sistem)
    2. ffmpeg sistem (PATH)
    """
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
    except ImportError:
        pass

    path_sistem = shutil.which("ffmpeg")
    if path_sistem:
        return path_sistem

    return None


def reencode_untuk_browser(input_path: str) -> str:
    """
    Re-encode video hasil OpenCV (codec mp4v) ke H.264 agar bisa
    diputar di browser. OpenCV menulis mp4v yang tidak didukung
    tag <video> HTML5 di Chrome/Firefox/Edge.

    Returns path video H.264 baru, atau path asli jika ffmpeg tidak ada.
    """
    ffmpeg_path = cari_ffmpeg()
    if ffmpeg_path is None:
        st.warning(
            "⚠️ ffmpeg tidak ditemukan. Video mungkin tidak tampil di browser.\n\n"
            "Jalankan: `pip install imageio-ffmpeg` lalu restart aplikasi."
        )
        return input_path

    output_path = os.path.join(tempfile.gettempdir(), "output_traffic_h264.mp4")
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    perintah = [
        ffmpeg_path,
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-movflags", "+faststart",
        "-y",
        output_path,
    ]

    try:
        hasil = subprocess.run(perintah, capture_output=True, text=True, timeout=300)
        if hasil.returncode != 0:
            st.warning(f"⚠️ Re-encode gagal: {hasil.stderr[-300:]}")
            return input_path
        if not os.path.exists(output_path):
            st.warning("⚠️ File hasil re-encode tidak ditemukan.")
            return input_path
        return output_path
    except subprocess.TimeoutExpired:
        st.warning("⚠️ Re-encode video timeout (video terlalu panjang).")
        return input_path
    except Exception as e:
        st.warning(f"⚠️ Error re-encode: {e}")
        return input_path


# ================================================================
# LOAD KONFIGURASI & MODEL
# ================================================================

@st.cache_resource
def load_config():
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {
        "model": {
            "weights": "models/best.pt",
            "confidence_threshold": 0.4,
            "iou_threshold": 0.5,
            "image_size": 416,
        },
        "counting": {"line_position": 0.5},
        "density": {
            "lengang": {"max": 10},
            "lancar":  {"max": 20},
            "ramai":   {"max": 35},
            "padat":   {"max": 9999},
        },
    }


@st.cache_resource
def load_model(model_path: str, conf: float, iou: float, imgsz: int):
    """
    Load YOLOv8 model. Di-cache oleh Streamlit agar tidak reload setiap
    kali slider di sidebar diubah.
    """
    try:
        return VehicleDetector(
            model_path=model_path,
            confidence=conf,
            iou=iou,
            device="cpu",
            imgsz=imgsz,
        )
    except FileNotFoundError:
        return None


# ================================================================
# PROSES VIDEO
# ================================================================

def proses_video(
    video_path: str,
    detector: VehicleDetector,
    line_pos: float,
    thresholds: dict,
    frame_skip: int,
    progress_bar,
    status_text,
) -> tuple:
    """
    Pipeline lengkap: detection (YOLOv8) → tracking (ByteTrack)
    → counting (line crossing) → klasifikasi (rule-based + K-Means).

    frame_skip: proses 1 frame, lalu duplikasi frame terakhir sebanyak
                frame_skip sebelum lanjut deteksi berikutnya.
                frame_skip=0 → semua frame diproses.
                frame_skip=1 → 2x lebih cepat (proses tiap 2 frame).
                frame_skip=2 → 3x lebih cepat, dst.

    Returns:
        tuple: (output_video_path, statistik_dict, dataframe_per_frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {}, pd.DataFrame()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_frame      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tulis video mentah dulu (codec mp4v), di-re-encode setelah selesai
    out_path_mentah = os.path.join(tempfile.gettempdir(), "output_traffic_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path_mentah, fourcc, fps_video, (w_frame, h_frame))

    counter    = VehicleCounter(line_position=line_pos, frame_height=h_frame)
    classifier = DensityClassifier(thresholds=thresholds)

    data_frames       = []
    history_count     = []
    history_status    = []
    frame_idx         = 0
    waktu_mulai       = time.time()
    fps_proses        = 0.0

    # State terakhir — dipakai untuk frame yang di-skip
    deteksi_terakhir      = []
    frame_anotasi_terakhir = None
    count_aktif_terakhir   = 0
    count_total_terakhir   = {"total": 0, "motor": 0, "mobil": 0, "bus": 0, "truk": 0}
    status_rule_terakhir   = "LENGANG"
    status_km_terakhir     = "LENGANG"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Frame skip: proses deteksi hanya tiap (frame_skip+1) frame ──
        # Frame yang di-skip: tulis ulang frame beranotasi terakhir ke video
        # agar FPS output tetap sesuai video asli (tidak jadi lambat)
        harus_deteksi = (frame_skip == 0) or (frame_idx % (frame_skip + 1) == 0)

        if harus_deteksi:
            t0 = time.time()

            try:
                deteksi_terakhir = detector.deteksi_dengan_tracking(frame)
            except Exception:
                deteksi_terakhir = []

            count_total_terakhir  = counter.update(deteksi_terakhir)
            count_aktif_terakhir  = counter.get_count_aktif(deteksi_terakhir)
            status_rule_terakhir  = classifier.klasifikasi_rule_based(count_aktif_terakhir)

            frame_area     = w_frame * h_frame
            total_bbox_area = sum(d.area for d in deteksi_terakhir)
            density_ratio  = total_bbox_area / max(frame_area, 1)
            avg_box_area   = total_bbox_area / max(len(deteksi_terakhir), 1)

            classifier.tambah_fitur(count_aktif_terakhir, density_ratio, avg_box_area)

            status_km, _ = classifier.klasifikasi_kmeans(
                count_aktif_terakhir, density_ratio, avg_box_area
            )
            if status_km:
                status_km_terakhir = status_km

            fps_proses = 1.0 / max(time.time() - t0, 1e-6)

            frame_anotasi_terakhir = buat_frame_lengkap(
                frame=frame,
                deteksi_list=deteksi_terakhir,
                line_y=counter.line_y_pos,
                status_rule=status_rule_terakhir,
                count_aktif=count_aktif_terakhir,
                count_total=count_total_terakhir,
                frame_idx=frame_idx,
                fps_proses=fps_proses,
                status_kmeans=status_km_terakhir,
                tampilkan_track_id=True,
            )

            # Simpan data per frame (hanya frame yang dideteksi)
            waktu_detik = frame_idx / fps_video
            data_frames.append({
                "frame":       frame_idx,
                "waktu_detik": round(waktu_detik, 2),
                "count_aktif": count_aktif_terakhir,
                "count_total": count_total_terakhir.get("total", 0),
                "motor":       count_total_terakhir.get("motor", 0),
                "mobil":       count_total_terakhir.get("mobil", 0),
                "bus":         count_total_terakhir.get("bus", 0),
                "truk":        count_total_terakhir.get("truk", 0),
                "status_rule": status_rule_terakhir,
                "density_pct": round(density_ratio * 100, 2),
            })
            history_count.append(count_aktif_terakhir)
            history_status.append(status_rule_terakhir)

        # Tulis frame ke video output (duplikasi kalau frame di-skip)
        if frame_anotasi_terakhir is not None:
            writer.write(frame_anotasi_terakhir)
        else:
            writer.write(frame)

        frame_idx += 1

        # Update progress bar setiap 5 frame agar tidak terlalu sering rerender
        if frame_idx % 5 == 0 and total_frames > 0:
            persen = frame_idx / total_frames
            progress_bar.progress(min(persen, 1.0))
            status_text.text(
                f"⏳ Frame {frame_idx}/{total_frames} "
                f"({persen*100:.0f}%) | {fps_proses:.1f} fps deteksi"
            )

    cap.release()
    writer.release()

    # ── Re-encode ke H.264 agar tampil di browser ──
    status_text.text("🎞️ Mengonversi video ke format browser (H.264)...")
    out_path = reencode_untuk_browser(out_path_mentah)

    metrik_kmeans = classifier.fit_akhir()
    durasi        = time.time() - waktu_mulai

    status_counts = {}
    for s in history_status:
        status_counts[s] = status_counts.get(s, 0) + 1

    statistik = {
        "total_frame":      frame_idx,
        "durasi_proses":    round(durasi, 2),
        "fps_rata":         round(frame_idx / max(durasi, 1), 1),
        "kend_total":       count_total_terakhir.get("total", 0),
        "kend_motor":       count_total_terakhir.get("motor", 0),
        "kend_mobil":       count_total_terakhir.get("mobil", 0),
        "kend_bus":         count_total_terakhir.get("bus", 0),
        "kend_truk":        count_total_terakhir.get("truk", 0),
        "avg_count_aktif":  round(np.mean(history_count), 1) if history_count else 0,
        "max_count_aktif":  max(history_count) if history_count else 0,
        "status_dominan":   max(status_counts, key=status_counts.get) if status_counts else "LENGANG",
        "distribusi_status": status_counts,
        "kmeans":           metrik_kmeans,
        "output_path":      out_path,
    }

    return out_path, statistik, pd.DataFrame(data_frames)


# ================================================================
# GRAFIK
# ================================================================

def buat_grafik_count(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(df["waktu_detik"], df["count_aktif"], alpha=0.3, color="#3498DB")
    ax.plot(df["waktu_detik"], df["count_aktif"],
            color="#2980B9", linewidth=1.5, label="Kendaraan aktif")
    if len(df) > 10:
        ma = df["count_aktif"].rolling(window=15, min_periods=1).mean()
        ax.plot(df["waktu_detik"], ma,
                color="#E74C3C", linewidth=2, linestyle="--", label="Rata-rata (15 sampel)")
    ax.set_xlabel("Waktu (detik)", fontsize=10)
    ax.set_ylabel("Jumlah Kendaraan", fontsize=10)
    ax.set_title("Jumlah Kendaraan Aktif per Frame (yang Dideteksi)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def buat_grafik_status(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 2.2))
    warna_map = {
        "LENGANG": "#2ECC71", "LANCAR": "#F1C40F",
        "RAMAI":   "#E67E22", "PADAT":  "#E74C3C",
    }
    for _, row in df.iterrows():
        warna = warna_map.get(row["status_rule"], "#BDC3C7")
        ax.bar(row["waktu_detik"], 1, width=1 / 30, color=warna, edgecolor="none")
    patches = [
        mpatches.Patch(color=warna_map[s], label=s)
        for s in ["LENGANG", "LANCAR", "RAMAI", "PADAT"]
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=4)
    ax.set_xlabel("Waktu (detik)", fontsize=10)
    ax.set_yticks([])
    ax.set_title("Status Kepadatan per Waktu", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    return fig


def buat_pie_kelas(statistik: dict):
    labels = ["Motor", "Mobil", "Bus", "Truk"]
    values = [
        statistik["kend_motor"], statistik["kend_mobil"],
        statistik["kend_bus"],   statistik["kend_truk"],
    ]
    warna = ["#2ECC71", "#3498DB", "#E67E22", "#E74C3C"]

    label_f = [l for l, v in zip(labels, values) if v > 0]
    value_f = [v for v in values if v > 0]
    warna_f = [c for c, v in zip(warna, values) if v > 0]

    fig, ax = plt.subplots(figsize=(5, 4))
    if sum(value_f) > 0:
        wedges, texts, autotexts = ax.pie(
            value_f, labels=label_f, colors=warna_f,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for at in autotexts:
            at.set_fontsize(9)
    else:
        ax.text(0.5, 0.5, "Belum ada data", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=12)
    ax.set_title("Distribusi Kelas Kendaraan\n(Total Melintas)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def html_status(status: str) -> str:
    kelas_map = {
        "LENGANG": "status-lengang", "LANCAR": "status-lancar",
        "RAMAI":   "status-ramai",   "PADAT":  "status-padat",
    }
    emoji_map = {"LENGANG": "🟢", "LANCAR": "🟡", "RAMAI": "🟠", "PADAT": "🔴"}
    kelas = kelas_map.get(status, "status-lancar")
    emoji = emoji_map.get(status, "⚪")
    return f'<div class="{kelas}">{emoji} {status}</div>'


# ================================================================
# MAIN
# ================================================================

def main():
    config = load_config()

    st.markdown('<div class="main-title">🚦 Deteksi Kepadatan Lalu Lintas</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">YOLOv8 Fine-tuned + ByteTrack | '
        'Studi Kasus: Simpang Uniland Medan<br>'
        'Silvani Chayadi · Cindy Nathania · Gloria Apriyanti | '
        'Universitas Mikroskil 2026</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── SIDEBAR ──────────────────────────────────────────────────────── #
    with st.sidebar:
        st.header("⚙️ Konfigurasi")

        st.subheader("Model")
        model_path = st.text_input(
            "Path model (best.pt)",
            value=config["model"].get("weights", "models/best.pt"),
        )
        conf_thr = st.slider(
            "Confidence threshold", 0.1, 0.9,
            float(config["model"].get("confidence_threshold", 0.4)),
            step=0.05,
            help="Semakin tinggi = deteksi lebih ketat (precision naik, recall turun)",
        )
        iou_thr = st.slider(
            "IoU threshold (NMS)", 0.1, 0.9,
            float(config["model"].get("iou_threshold", 0.5)),
            step=0.05,
        )
        imgsz = st.select_slider(
            "Ukuran inferensi (imgsz)",
            options=[320, 416, 512, 640],
            value=int(config["model"].get("image_size", 416)),
            help="Lebih kecil = lebih cepat, sedikit pengaruh akurasi. "
                 "416 direkomendasikan untuk Streamlit Cloud.",
        )

        st.subheader("⚡ Kecepatan")
        frame_skip = st.select_slider(
            "Frame skip",
            options=[0, 1, 2, 3],
            value=1,
            format_func=lambda x: {
                0: "0 — Proses semua frame (lambat)",
                1: "1 — Proses tiap 2 frame (2× lebih cepat) ✅",
                2: "2 — Proses tiap 3 frame (3× lebih cepat)",
                3: "3 — Proses tiap 4 frame (4× lebih cepat, kasar)",
            }[x],
            help="Skip frame mengurangi beban CPU. Counting & grafik tetap akurat "
                 "karena frame yang di-skip menggunakan hasil deteksi terakhir.",
        )

        st.subheader("Counting Line")
        line_pos = st.slider(
            "Posisi garis counting", 0.2, 0.9,
            float(config["counting"].get("line_position", 0.5)),
            step=0.05,
            help="0.5 = tengah frame vertikal",
        )

        st.subheader("Threshold Kepadatan")
        t_lengang = st.number_input("Maks kend. Lengang", 1, 50,
                                    int(config["density"]["lengang"]["max"]))
        t_lancar  = st.number_input("Maks kend. Lancar",  1, 100,
                                    int(config["density"]["lancar"]["max"]))
        t_ramai   = st.number_input("Maks kend. Ramai",   1, 200,
                                    int(config["density"]["ramai"]["max"]))

        thresholds = {
            "lengang": {"max": t_lengang},
            "lancar":  {"max": t_lancar},
            "ramai":   {"max": t_ramai},
            "padat":   {"max": 9999},
        }

        st.divider()
        st.caption("ℹ️ Lengang → Lancar → Ramai → Padat")
        st.caption("Kelas: motor, mobil, bus, truk")

    # ── LOAD MODEL ───────────────────────────────────────────────────── #
    with st.spinner("Memuat model YOLOv8..."):
        detector = load_model(model_path, conf_thr, iou_thr, imgsz)

    if detector is None:
        st.error(
            f"❌ Model tidak ditemukan di `{model_path}`.\n\n"
            "Download `best.pt` hasil training dari Google Drive, "
            "letakkan di folder `models/`."
        )
        st.stop()

    st.success(f"✅ Model dimuat dari `{model_path}` (imgsz={imgsz})")

    # ── UPLOAD VIDEO ─────────────────────────────────────────────────── #
    st.header("📤 Upload Video")
    uploaded = st.file_uploader(
        "Upload video lalu lintas (.mp4 / .avi / .mov)",
        type=["mp4", "avi", "mov"],
    )

    if uploaded is None:
        st.info("👆 Upload video untuk memulai deteksi.")
        with st.expander("ℹ️ Cara kerja sistem"):
            st.markdown("""
**Pipeline Deteksi (Tugas 2):**
1. **Object Detection** — YOLOv8 fine-tuned mendeteksi motor, mobil, bus, truk
2. **ByteTrack** — Setiap kendaraan mendapat ID unik antar frame
3. **Line Crossing** — Kendaraan dihitung saat melewati garis virtual
4. **Klasifikasi** — Status Lengang/Lancar/Ramai/Padat (rule-based + K-Means)
""")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.video(tmp_path)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        mulai = st.button("🚀 Jalankan Deteksi", use_container_width=True, type="primary")

    if not mulai:
        return

    # ── PROSES ───────────────────────────────────────────────────────── #
    st.header("🔄 Memproses...")
    progress_bar = st.progress(0)
    status_text  = st.empty()

    with st.spinner("Deteksi sedang berjalan..."):
        out_path, statistik, df_frames = proses_video(
            video_path=tmp_path,
            detector=detector,
            line_pos=line_pos,
            thresholds=thresholds,
            frame_skip=frame_skip,
            progress_bar=progress_bar,
            status_text=status_text,
        )

    progress_bar.progress(1.0)
    status_text.text("✅ Selesai!")
    st.success(
        f"✅ Pemrosesan selesai dalam {statistik['durasi_proses']:.1f} detik "
        f"({statistik['fps_rata']} fps rata-rata)"
    )

    # ── HASIL ────────────────────────────────────────────────────────── #
    st.header("📊 Hasil Deteksi")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🚗 Total Kendaraan", statistik["kend_total"])
    col2.metric("🏍️ Motor",           statistik["kend_motor"])
    col3.metric("🚙 Mobil",           statistik["kend_mobil"])
    col4.metric("🚌 Bus",             statistik["kend_bus"])
    col5.metric("🚚 Truk",            statistik["kend_truk"])

    st.divider()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Status Kepadatan Dominan**")
        st.markdown(html_status(statistik["status_dominan"]), unsafe_allow_html=True)
    with col_b:
        st.metric("Rata-rata Kendaraan/Frame", f"{statistik['avg_count_aktif']}")
    with col_c:
        st.metric("Maks Kendaraan dalam 1 Frame", f"{statistik['max_count_aktif']}")

    st.divider()

    col_vid, col_pie = st.columns([2, 1])
    with col_vid:
        st.subheader("🎬 Video Hasil")
        if out_path and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                st.video(f.read())
        else:
            st.warning("Video output tidak tersedia.")
    with col_pie:
        st.subheader("🥧 Distribusi Kelas")
        fig_pie = buat_pie_kelas(statistik)
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)

    if not df_frames.empty:
        st.subheader("📈 Grafik Kendaraan per Waktu")
        fig_count = buat_grafik_count(df_frames)
        st.pyplot(fig_count, use_container_width=True)
        plt.close(fig_count)

        st.subheader("🎨 Status Kepadatan per Waktu")
        fig_status = buat_grafik_status(df_frames)
        st.pyplot(fig_status, use_container_width=True)
        plt.close(fig_status)

    if statistik.get("kmeans") and "silhouette_score" in statistik["kmeans"]:
        st.subheader("🔬 Evaluasi K-Means Clustering")
        km = statistik["kmeans"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette Score",       f"{km['silhouette_score']:.4f}")
        c2.metric("Davies-Bouldin Index",   f"{km['davies_bouldin_index']:.4f}")
        c3.metric("Total Frame Dianalisis", km.get("total_frame", 0))

    if statistik.get("distribusi_status"):
        st.subheader("📋 Distribusi Status Kepadatan")
        dist    = statistik["distribusi_status"]
        total_f = sum(dist.values())
        cols_st = st.columns(len(dist))
        for col, (status, jumlah) in zip(cols_st, dist.items()):
            persen = jumlah / total_f * 100 if total_f > 0 else 0
            col.metric(status, f"{jumlah} frame", f"{persen:.1f}%")

    st.divider()

    # ── DOWNLOAD ─────────────────────────────────────────────────────── #
    st.subheader("⬇️ Download Hasil")
    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        if out_path and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                st.download_button(
                    label="🎬 Download Video Hasil",
                    data=f,
                    file_name="hasil_deteksi_lalu_lintas.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
    with col_dl2:
        if not df_frames.empty:
            csv_data = df_frames.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📊 Download CSV (per frame)",
                data=csv_data,
                file_name="statistik_per_frame.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with col_dl3:
        stat_export = {k: v for k, v in statistik.items() if k != "output_path"}
        json_data   = json.dumps(stat_export, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            label="📋 Download Statistik (JSON)",
            data=json_data,
            file_name="statistik_ringkasan.json",
            mime="application/json",
            use_container_width=True,
        )

    # Hapus file temp upload
    try:
        os.unlink(tmp_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
