"""
app.py
Aplikasi Streamlit — Deteksi & Klasifikasi Kepadatan Lalu Lintas
Tugas 2 Visi Komputer — Universitas Mikroskil 2026

Kelompok:
  Silvani Chayadi      NIM 231112945
  Cindy Nathania P.A.  NIM 231111567
  Gloria Apriyanti S.  NIM 231111304

Cara menjalankan:
  streamlit run app.py
"""
import shutil
import subprocess
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import yaml
import time
import tempfile
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 1.5rem;
        line-height: 1.6;
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


NAMA_KELAS = ["motor", "mobil", "bus", "truk"]


# ================================================================
# FIX VIDEO — RE-ENCODE KE H.264 AGAR TAMPIL DI BROWSER
# ================================================================
def cari_ffmpeg() -> str:
    """
    Cari ffmpeg yang bisa dipakai, dengan urutan prioritas:
    1. imageio-ffmpeg (binary bawaan paket Python, PALING DIANDALKAN
       karena tidak bergantung pada instalasi sistem / PATH)
    2. ffmpeg yang ter-install di sistem (PATH)

    Returns:
        str: path ke executable ffmpeg, atau None jika tidak ada sama sekali
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
    Re-encode video hasil OpenCV ke H.264 agar bisa diputar di browser.

    OpenCV menulis video dengan codec "mp4v" (MPEG-4 Part 2) yang
    TIDAK didukung tag <video> HTML5 di Chrome/Firefox/Edge, sehingga
    video tampil sebagai kotak hitam dengan durasi 0:00.

    Args:
        input_path: path video hasil tulisan OpenCV (codec lama)

    Returns:
        str: path video baru yang sudah H.264, atau path asli jika
             ffmpeg tidak ditemukan sama sekali
    """
    ffmpeg_path = cari_ffmpeg()

    if ffmpeg_path is None:
        st.warning(
            "⚠️ ffmpeg tidak ditemukan. Video mungkin tidak tampil di browser.\n\n"
            "Jalankan: `pip install imageio-ffmpeg` lalu restart aplikasi Streamlit."
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
        output_path
    ]

    try:
        hasil = subprocess.run(
            perintah,
            capture_output=True,
            text=True,
            timeout=180
        )

        if hasil.returncode != 0:
            st.warning(f"⚠️ Re-encode video gagal: {hasil.stderr[-300:]}")
            return input_path

        if not os.path.exists(output_path):
            st.warning("⚠️ File hasil re-encode tidak ditemukan.")
            return input_path

        return output_path

    except subprocess.TimeoutExpired:
        st.warning("⚠️ Re-encode video timeout (video terlalu panjang).")
        return input_path
    except Exception as e:
        st.warning(f"⚠️ Error saat re-encode: {e}")
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
            "image_size": 640
        },
        "counting": {"line_position": 0.5},
        "density": {
            "lengang": {"max": 10},
            "lancar":  {"max": 20},
            "ramai":   {"max": 35},
            "padat":   {"max": 9999}
        }
    }


@st.cache_resource
def load_model(model_path: str, conf: float, iou: float):
    try:
        return VehicleDetector(
            model_path=model_path,
            confidence=conf,
            iou=iou,
            device="cpu"
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
    progress_bar,
    status_text
) -> tuple:
    """
    Pipeline lengkap: detection (YOLOv8) → tracking (ByteTrack)
    → counting (line crossing) → klasifikasi (rule-based + K-Means).

    Returns:
        tuple: (output_video_path, statistik, dataframe_per_frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {}, pd.DataFrame()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_frame      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tulis video mentah dulu dengan OpenCV (codec mp4v, belum bisa
    # diputar di browser -- akan di-re-encode setelah selesai)
    out_path_mentah = os.path.join(tempfile.gettempdir(), "output_traffic_raw.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path_mentah, fourcc, fps_video, (w_frame, h_frame))

    counter    = VehicleCounter(line_position=line_pos, frame_height=h_frame)
    classifier = DensityClassifier(thresholds=thresholds)

    data_frames     = []
    history_count   = []
    history_status  = []

    frame_idx  = 0
    waktu_mulai = time.time()
    fps_proses  = 0.0

    status_rule_terakhir   = "LENGANG"
    status_kmeans_terakhir = "LENGANG"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        try:
            deteksi_list = detector.deteksi_dengan_tracking(frame)
        except Exception:
            deteksi_list = []

        count_total = counter.update(deteksi_list)
        count_aktif = counter.get_count_aktif(deteksi_list)

        status_rule_terakhir = classifier.klasifikasi_rule_based(count_aktif)

        frame_area      = w_frame * h_frame
        total_bbox_area = sum(d.area for d in deteksi_list)
        density_ratio   = total_bbox_area / max(frame_area, 1)
        avg_box_area    = total_bbox_area / max(len(deteksi_list), 1)

        classifier.tambah_fitur(count_aktif, density_ratio, avg_box_area)
        status_km, _ = classifier.klasifikasi_kmeans(
            count_aktif, density_ratio, avg_box_area
        )
        if status_km:
            status_kmeans_terakhir = status_km

        fps_proses = 1.0 / max(time.time() - t0, 1e-6)

        frame_anotasi = buat_frame_lengkap(
            frame         = frame,
            deteksi_list  = deteksi_list,
            line_y        = counter.line_y_pos,
            status_rule   = status_rule_terakhir,
            count_aktif   = count_aktif,
            count_total   = count_total,
            frame_idx     = frame_idx,
            fps_proses    = fps_proses,
            status_kmeans = status_kmeans_terakhir,
            tampilkan_track_id = True
        )

        writer.write(frame_anotasi)

        waktu_detik = frame_idx / fps_video
        data_frames.append({
            "frame":       frame_idx,
            "waktu_detik": round(waktu_detik, 2),
            "count_aktif": count_aktif,
            "count_total": count_total.get("total", 0),
            "motor":       count_total.get("motor", 0),
            "mobil":       count_total.get("mobil", 0),
            "bus":         count_total.get("bus", 0),
            "truk":        count_total.get("truk", 0),
            "status_rule": status_rule_terakhir,
            "density_pct": round(density_ratio * 100, 2),
        })

        history_count.append(count_aktif)
        history_status.append(status_rule_terakhir)
        frame_idx += 1

        if total_frames > 0:
            persen = frame_idx / total_frames
            progress_bar.progress(min(persen, 1.0))
            status_text.text(
                f"⏳ Frame {frame_idx}/{total_frames} "
                f"({persen*100:.0f}%) | {fps_proses:.1f} fps"
            )

    cap.release()
    writer.release()

    # ── FIX: re-encode ke H.264 agar video tampil di browser ──
    status_text.text("🎞️ Mengonversi video agar kompatibel browser...")
    out_path = reencode_untuk_browser(out_path_mentah)

    metrik_kmeans = classifier.fit_akhir()
    durasi = time.time() - waktu_mulai

    status_counts = {}
    for s in history_status:
        status_counts[s] = status_counts.get(s, 0) + 1

    statistik = {
        "total_frame":      frame_idx,
        "durasi_proses":    round(durasi, 2),
        "fps_rata":         round(frame_idx / max(durasi, 1), 1),
        "kend_total":       count_total.get("total", 0),
        "kend_motor":       count_total.get("motor", 0),
        "kend_mobil":       count_total.get("mobil", 0),
        "kend_bus":         count_total.get("bus", 0),
        "kend_truk":        count_total.get("truk", 0),
        "avg_count_aktif":  round(np.mean(history_count), 1) if history_count else 0,
        "max_count_aktif":  max(history_count) if history_count else 0,
        "status_dominan":   max(status_counts, key=status_counts.get) if status_counts else "LENGANG",
        "distribusi_status":status_counts,
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
    ax.plot(df["waktu_detik"], df["count_aktif"], color="#2980B9",
            linewidth=1.5, label="Kendaraan aktif")

    if len(df) > 10:
        ma = df["count_aktif"].rolling(window=15, min_periods=1).mean()
        ax.plot(df["waktu_detik"], ma, color="#E74C3C", linewidth=2,
                linestyle="--", label="Rata-rata (15 frame)")

    ax.set_xlabel("Waktu (detik)", fontsize=10)
    ax.set_ylabel("Jumlah Kendaraan", fontsize=10)
    ax.set_title("Jumlah Kendaraan Aktif per Frame", fontsize=11, fontweight="bold")
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
        "RAMAI": "#E67E22", "PADAT": "#E74C3C",
    }
    for _, row in df.iterrows():
        warna = warna_map.get(row["status_rule"], "#BDC3C7")
        ax.bar(row["waktu_detik"], 1, width=1/30, color=warna, edgecolor="none")

    patches = [mpatches.Patch(color=warna_map[s], label=s)
               for s in ["LENGANG", "LANCAR", "RAMAI", "PADAT"]]
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
        statistik["kend_bus"], statistik["kend_truk"],
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
            wedgeprops={"edgecolor": "white", "linewidth": 2}
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
        "RAMAI": "status-ramai", "PADAT": "status-padat",
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

    st.markdown(
        '<div class="main-title">🚦 Deteksi Kepadatan Lalu Lintas</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-title">YOLOv8 Fine-tuned + ByteTrack | '
        'Studi Kasus: Simpang Uniland Medan<br>'
        'Silvani Chayadi · Cindy Nathania · Gloria Apriyanti | '
        'Universitas Mikroskil 2026</div>',
        unsafe_allow_html=True
    )
    st.divider()

    with st.sidebar:
        st.header("⚙️ Konfigurasi")

        st.subheader("Model")
        model_path = st.text_input(
            "Path model (best.pt)",
            value=config["model"].get("weights", "models/best.pt")
        )
        conf_thr = st.slider(
            "Confidence threshold", 0.1, 0.9,
            float(config["model"].get("confidence_threshold", 0.4)),
            step=0.05,
            help="Semakin tinggi = deteksi lebih ketat"
        )
        iou_thr = st.slider(
            "IoU threshold (NMS)", 0.1, 0.9,
            float(config["model"].get("iou_threshold", 0.5)),
            step=0.05
        )

        st.subheader("Counting Line")
        line_pos = st.slider(
            "Posisi garis counting", 0.2, 0.9,
            float(config["counting"].get("line_position", 0.5)),
            step=0.05,
            help="0.5 = tengah frame"
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

    with st.spinner("Memuat model..."):
        detector = load_model(model_path, conf_thr, iou_thr)

    if detector is None:
        st.error(
            f"❌ Model tidak ditemukan di `{model_path}`.\n\n"
            "Download `best.pt` hasil training dari Google Drive, "
            "letakkan di folder `models/`."
        )
        st.stop()

    st.success(f"✅ Model dimuat dari `{model_path}`")

    st.header("📤 Upload Video")
    uploaded = st.file_uploader(
        "Upload video lalu lintas (.mp4 / .avi / .mov)",
        type=["mp4", "avi", "mov"]
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

            **Keunggulan vs Tugas 1 (MOG2 + K-Means):**
            - Deteksi kendaraan statis (berhenti di lampu merah) ✅
            - Tidak ada contour merging saat kepadatan tinggi ✅
            - Counting akurat per kendaraan unik, bukan per frame ✅
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

    st.header("🔄 Memproses...")
    progress_bar = st.progress(0)
    status_text  = st.empty()

    with st.spinner("Deteksi sedang berjalan..."):
        out_path, statistik, df_frames = proses_video(
            video_path   = tmp_path,
            detector     = detector,
            line_pos     = line_pos,
            thresholds   = thresholds,
            progress_bar = progress_bar,
            status_text  = status_text
        )

    progress_bar.progress(1.0)
    status_text.text("✅ Selesai!")
    st.success(
        f"✅ Pemrosesan selesai dalam {statistik['durasi_proses']:.1f} detik "
        f"({statistik['fps_rata']} fps rata-rata)"
    )

    st.header("📊 Hasil Deteksi")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🚗 Total Kendaraan", statistik["kend_total"])
    col2.metric("🏍️ Motor", statistik["kend_motor"])
    col3.metric("🚙 Mobil", statistik["kend_mobil"])
    col4.metric("🚌 Bus", statistik["kend_bus"])
    col5.metric("🚚 Truk", statistik["kend_truk"])

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
                video_bytes = f.read()
            st.video(video_bytes)
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
        c1.metric("Silhouette Score", f"{km['silhouette_score']:.4f}")
        c2.metric("Davies-Bouldin Index", f"{km['davies_bouldin_index']:.4f}")
        c3.metric("Total Frame Dianalisis", km.get("total_frame", 0))

    if statistik.get("distribusi_status"):
        st.subheader("📋 Distribusi Status Kepadatan")
        dist = statistik["distribusi_status"]
        total_f = sum(dist.values())
        cols_st = st.columns(len(dist))
        for col, (status, jumlah) in zip(cols_st, dist.items()):
            persen = jumlah / total_f * 100 if total_f > 0 else 0
            col.metric(status, f"{jumlah} frame", f"{persen:.1f}%")

    st.divider()
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
                    use_container_width=True
                )

    with col_dl2:
        if not df_frames.empty:
            csv_data = df_frames.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📊 Download CSV (per frame)",
                data=csv_data,
                file_name="statistik_per_frame.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col_dl3:
        stat_export = {k: v for k, v in statistik.items() if k != "output_path"}
        json_data = json.dumps(stat_export, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            label="📋 Download Statistik (JSON)",
            data=json_data,
            file_name="statistik_ringkasan.json",
            mime="application/json",
            use_container_width=True
        )

    try:
        os.unlink(tmp_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()