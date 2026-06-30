"""
frame_extractor.py
Ekstraksi frame dari video untuk dataset training YOLOv8.

Kenapa interval 5?
  Video 30fps → ambil 1 per 5 frame = 6fps tersimpan
  1800 frame / 5 = 360 frame per video 60 detik
  Cukup variasi, tidak terlalu banyak redundansi

Kelompok: Silvani Chayadi, Cindy Nathania, Gloria Apriyanti
Universitas Mikroskil 2026
"""

import cv2
import os
import numpy as np
from pathlib import Path


def hitung_blur_score(frame: np.ndarray) -> float:
    """
    Laplacian variance — semakin tinggi semakin tajam.
    Frame blur (dari kompresi video) dibuang karena
    akan membingungkan model saat training.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def hitung_kemiripan(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Korelasi histogram HSV antara dua frame.
    Nilai mendekati 1.0 = sangat mirip = duplikat.
    Duplikat tidak berguna untuk training.
    """
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    h2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


def extract_frames(
    video_path: str,
    output_folder: str,
    frame_interval: int = 5,
    min_blur_score: float = 80.0,
    max_similarity: float = 0.97,
    resize_width: int = 640,
    sesi_label: str = ""
) -> dict:
    """
    Ekstrak frame dari satu video dengan filter kualitas.

    Args:
        video_path     : path ke file video
        output_folder  : folder tujuan simpan frame
        frame_interval : ambil 1 frame per N frame
        min_blur_score : buang frame dengan blur score di bawah ini
        max_similarity : buang frame yang terlalu mirip frame sebelumnya
        resize_width   : lebar output (tinggi menyesuaikan aspek ratio)
        sesi_label     : prefix nama file ("sesi_0900", dll)

    Returns:
        dict statistik hasil ekstraksi
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {video_path}")

    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
    durasi        = total_frames / fps
    lebar_asli    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    tinggi_asli   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skala         = resize_width / lebar_asli
    tinggi_output = int(tinggi_asli * skala)

    print(f"\n{'='*55}")
    print(f"  EKSTRAKSI: {sesi_label or video_path}")
    print(f"{'='*55}")
    print(f"  Resolusi asli  : {lebar_asli}×{tinggi_asli}")
    print(f"  Resolusi output: {resize_width}×{tinggi_output}")
    print(f"  Total frame    : {total_frames}")
    print(f"  FPS            : {fps:.0f}")
    print(f"  Durasi         : {durasi:.0f} detik")
    print(f"  Interval       : setiap {frame_interval} frame")
    print(f"  Estimasi output: ~{total_frames // frame_interval} frame")
    print(f"{'='*55}")

    frame_idx          = 0
    frame_disimpan     = 0
    buang_blur         = 0
    buang_duplikat     = 0
    frame_sebelumnya   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Filter interval
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        # Resize
        frame_resized = cv2.resize(
            frame,
            (resize_width, tinggi_output),
            interpolation=cv2.INTER_AREA
        )

        # Filter blur
        if hitung_blur_score(frame_resized) < min_blur_score:
            buang_blur += 1
            frame_idx += 1
            continue

        # Filter duplikat
        if frame_sebelumnya is not None:
            if hitung_kemiripan(frame_resized, frame_sebelumnya) > max_similarity:
                buang_duplikat += 1
                frame_idx += 1
                continue

        # Simpan
        prefix    = f"{sesi_label}_" if sesi_label else ""
        nama_file = f"{prefix}frame_{frame_disimpan:04d}.jpg"
        cv2.imwrite(
            os.path.join(output_folder, nama_file),
            frame_resized,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        frame_sebelumnya = frame_resized.copy()
        frame_disimpan  += 1
        frame_idx       += 1

        if frame_disimpan % 50 == 0:
            print(f"  Tersimpan: {frame_disimpan} | "
                  f"Blur dibuang: {buang_blur} | "
                  f"Duplikat dibuang: {buang_duplikat}")

    cap.release()

    stat = {
        "sesi": sesi_label,
        "frame_disimpan": frame_disimpan,
        "buang_blur": buang_blur,
        "buang_duplikat": buang_duplikat,
        "output_folder": output_folder
    }

    print(f"\n  ✅ Selesai: {frame_disimpan} frame disimpan")
    print(f"  ❌ Dibuang blur: {buang_blur} | duplikat: {buang_duplikat}")
    return stat


def extract_semua_sesi(
    config_sesi: list,
    base_output_dir: str = "dataset/frames"
) -> list:
    """
    Ekstrak frame dari semua sesi video.

    Args:
        config_sesi: list of dict
                     [{"video_path": "...", "sesi_label": "sesi_0900"}, ...]
        base_output_dir: folder utama

    Returns:
        list statistik per sesi
    """
    semua_stat = []

    for sesi in config_sesi:
        output_folder = os.path.join(base_output_dir, sesi["sesi_label"])
        try:
            stat = extract_frames(
                video_path    = sesi["video_path"],
                output_folder = output_folder,
                frame_interval= 5,
                min_blur_score= 80.0,
                max_similarity= 0.97,
                resize_width  = 640,
                sesi_label    = sesi["sesi_label"]
            )
            semua_stat.append(stat)
        except Exception as e:
            print(f"❌ Error {sesi['sesi_label']}: {e}")

    total = sum(s["frame_disimpan"] for s in semua_stat)
    print(f"\n{'='*55}")
    print(f"  TOTAL SEMUA SESI: {total} frame")
    for s in semua_stat:
        print(f"  {s['sesi']:15s}: {s['frame_disimpan']} frame")
    print(f"{'='*55}")

    return semua_stat


# ── Jalankan langsung ─────────────────────────────────────────────
if __name__ == "__main__":

    # Root proyek = satu level di atas src/
    ROOT = Path(__file__).parent.parent

    SESI_CONFIG = [
        {"video_path": str(ROOT / "dataset/raw_videos/sesi_0900.mp4"),
         "sesi_label": "sesi_0900"},
        {"video_path": str(ROOT / "dataset/raw_videos/sesi_1200.mp4"),
         "sesi_label": "sesi_1200"},
        {"video_path": str(ROOT / "dataset/raw_videos/sesi_1700.mp4"),
         "sesi_label": "sesi_1700"},
    ]

    extract_semua_sesi(SESI_CONFIG, str(ROOT / "dataset/frames"))