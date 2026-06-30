"""
train.py
Script training YOLOv8n Fine-tuning — satu file, langsung jalankan.

Cara pakai:
  1. Pastikan dataset sudah ada di dataset/yolo_dataset/
  2. pip install -r requirements.txt
  3. python train.py

Kelompok: Silvani Chayadi, Cindy Nathania, Gloria Apriyanti
Universitas Mikroskil 2026
"""

import os
import sys
import time
import json
import shutil
import yaml
import functools
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Tidak perlu display window


# ================================================================
# KONFIGURASI — ubah di sini sesuai kebutuhan
# ================================================================

DATASET_DIR  = "dataset/yolo_dataset"
DATA_YAML    = os.path.join(DATASET_DIR, "data.yaml")
MODEL_OUTPUT = "models/best.pt"

BASE_MODEL   = "yolov8n.pt"
PROJECT_NAME = "results/training"
RUN_NAME     = "yolov8n-simpanguniland-v1"

EPOCHS        = 100
BATCH         = 8        # Turunkan ke 4 jika RAM tidak cukup
IMGSZ         = 640
PATIENCE      = 20       # Early stopping
SEED          = 42
WORKERS       = 2

LR0           = 0.01
LRF           = 0.01
WARMUP_EPOCHS = 3.0
WEIGHT_DECAY  = 0.0005
MOMENTUM      = 0.937

MOSAIC        = 1.0
FLIPLR        = 0.5
FLIPUD        = 0.0
HSV_H         = 0.015
HSV_S         = 0.7
HSV_V         = 0.4
DEGREES       = 0.0
TRANSLATE     = 0.1
SCALE         = 0.5
CLOSE_MOSAIC  = 10

NAMA_KELAS = ["motor", "mobil", "bus", "truk"]


# ================================================================
# STEP 0 — FIX PYTORCH COMPATIBILITY
# ================================================================

def fix_pytorch_compatibility():
    """
    Fix kompatibilitas PyTorch >= 2.6 dengan Ultralytics.
    PyTorch 2.6 mengubah default weights_only=True di torch.load
    yang menyebabkan error saat load model Ultralytics.
    Solusi: patch torch.load agar default weights_only=False.
    """
    major, minor = [int(x) for x in torch.__version__.split(".")[:2]]

    if major > 2 or (major == 2 and minor >= 6):
        print(f"  ⚠️  PyTorch {torch.__version__} terdeteksi")
        print(f"      Menerapkan fix kompatibilitas...")

        _original_load = torch.load

        @functools.wraps(_original_load)
        def patched_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        torch.load = patched_load
        print(f"  ✅ Fix diterapkan")
    else:
        print(f"  ✅ PyTorch {torch.__version__} — tidak perlu fix")


# ================================================================
# STEP 1 — FIX DATA.YAML
# ================================================================

def fix_data_yaml():
    """
    Fix path di data.yaml ke path absolut sistem saat ini.
    Mencegah error jika data.yaml dibuat di OS lain (misal Windows)
    tapi dijalankan di Linux/Mac atau sebaliknya.
    """
    print(f"\n{'='*55}")
    print(f"  STEP 1 — FIX DATA.YAML")
    print(f"{'='*55}")

    if not os.path.exists(DATA_YAML):
        print(f"  ❌ data.yaml tidak ditemukan: {DATA_YAML}")
        print(f"     Jalankan src/merge_dataset.py terlebih dahulu")
        sys.exit(1)

    # Baca isi lama
    with open(DATA_YAML, "r") as f:
        isi_lama = yaml.safe_load(f)

    print(f"  Path lama: {isi_lama.get('path', '-')}")

    # Tulis ulang dengan path absolut sistem saat ini
    abs_dataset = os.path.abspath(DATASET_DIR)

    isi_baru = {
        "path":  abs_dataset,
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    isi_lama.get("nc", len(NAMA_KELAS)),
        "names": isi_lama.get("names", NAMA_KELAS)
    }

    with open(DATA_YAML, "w") as f:
        yaml.dump(isi_baru, f, default_flow_style=False, sort_keys=False)

    print(f"  Path baru: {abs_dataset}")

    # Verifikasi folder
    print(f"\n  Verifikasi folder:")
    semua_ok = True
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(abs_dataset, "images", split)
        lbl_dir = os.path.join(abs_dataset, "labels", split)

        n_img = len([f for f in os.listdir(img_dir)
                     if f.endswith((".jpg", ".jpeg", ".png"))]) \
                if os.path.exists(img_dir) else 0
        n_lbl = len([f for f in os.listdir(lbl_dir)
                     if f.endswith(".txt")]) \
                if os.path.exists(lbl_dir) else 0

        ok = "✅" if n_img > 0 else "❌"
        print(f"    {ok} {split:5s}: {n_img} gambar | {n_lbl} label")

        if n_img == 0:
            semua_ok = False
            print(f"       Path: {img_dir}")

    if not semua_ok:
        print(f"\n  ❌ Ada folder yang kosong!")
        print(f"     Pastikan merge_dataset.py sudah dijalankan dengan benar")
        sys.exit(1)

    print(f"\n  ✅ data.yaml siap")
    return DATA_YAML


# ================================================================
# STEP 2 — CEK DEVICE
# ================================================================

def cek_device():
    """Cek ketersediaan GPU."""
    print(f"\n{'='*55}")
    print(f"  STEP 2 — CEK DEVICE")
    print(f"{'='*55}")
    print(f"  PyTorch  : {torch.__version__}")

    if torch.cuda.is_available():
        nama_gpu = torch.cuda.get_device_name(0)
        vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU      : {nama_gpu}")
        print(f"  VRAM     : {vram:.1f} GB")
        device = "0"
        print(f"  ✅ GPU digunakan untuk training")
    else:
        device = "cpu"
        print(f"  ⚠️  GPU tidak tersedia — pakai CPU")
        print(f"      Training akan lebih lambat (~5-10x)")
        print(f"      Pertimbangkan mengurangi EPOCHS ke 30-50")

    return device


# ================================================================
# STEP 3 — TRAINING
# ================================================================

def jalankan_training(device: str, yaml_path: str):
    """
    Fine-tuning YOLOv8n pada dataset Simpang Uniland.

    Proses yang terjadi:
    1. Load yolov8n.pt — pretrained COCO (80 kelas, ~3.2M parameter)
    2. Ganti detection head → 4 kelas kita (motor/mobil/bus/truk)
    3. Training: forward pass → hitung loss → backpropagation
    4. Early stopping jika val mAP@0.5 tidak improve selama PATIENCE epoch
    5. Simpan best.pt = bobot dengan val mAP@0.5 tertinggi
    """
    from ultralytics import YOLO

    print(f"\n{'='*55}")
    print(f"  STEP 3 — TRAINING")
    print(f"{'='*55}")
    print(f"  Base model   : {BASE_MODEL}")
    print(f"  Epochs       : {EPOCHS} (early stop: {PATIENCE})")
    print(f"  Batch size   : {BATCH}")
    print(f"  Image size   : {IMGSZ}")
    print(f"  Device       : {device}")
    print(f"  Optimizer    : AdamW")
    print(f"  Augmentasi   : mosaic={MOSAIC}, fliplr={FLIPLR}")
    print(f"{'='*55}\n")

    model = YOLO(BASE_MODEL)
    t0    = time.time()

    hasil = model.train(
        data          = yaml_path,
        epochs        = EPOCHS,
        batch         = BATCH,
        imgsz         = IMGSZ,
        device        = device,
        workers       = WORKERS,
        seed          = SEED,
        patience      = PATIENCE,
        lr0           = LR0,
        lrf           = LRF,
        warmup_epochs = WARMUP_EPOCHS,
        weight_decay  = WEIGHT_DECAY,
        momentum      = MOMENTUM,
        optimizer     = "AdamW",
        cos_lr        = True,
        mosaic        = MOSAIC,
        fliplr        = FLIPLR,
        flipud        = FLIPUD,
        hsv_h         = HSV_H,
        hsv_s         = HSV_S,
        hsv_v         = HSV_V,
        degrees       = DEGREES,
        translate     = TRANSLATE,
        scale         = SCALE,
        close_mosaic  = CLOSE_MOSAIC,
        project       = PROJECT_NAME,
        name          = RUN_NAME,
        exist_ok      = True,
        save          = True,
        save_period   = 10,
        plots         = True,
        verbose       = True,
        rect          = False,
    )

    durasi = (time.time() - t0) / 60
    map50  = hasil.results_dict.get("metrics/mAP50(B)", 0)

    print(f"\n{'='*55}")
    print(f"  ✅ TRAINING SELESAI!")
    print(f"  Durasi       : {durasi:.1f} menit")
    print(f"  Best mAP@0.5 : {map50*100:.1f}%")
    print(f"{'='*55}")

    # Path ke model hasil training
    best_pt_training = os.path.join(
        PROJECT_NAME, RUN_NAME, "weights", "best.pt"
    )

    return best_pt_training, durasi, map50


# ================================================================
# STEP 4 — SIMPAN MODEL
# ================================================================

def simpan_model(best_pt_training: str):
    """Copy best.pt ke folder models/."""
    print(f"\n{'='*55}")
    print(f"  STEP 4 — SIMPAN MODEL")
    print(f"{'='*55}")

    if not os.path.exists(best_pt_training):
        print(f"  ❌ best.pt tidak ditemukan: {best_pt_training}")
        return False

    os.makedirs("models", exist_ok=True)
    shutil.copy2(best_pt_training, MODEL_OUTPUT)

    ukuran = os.path.getsize(MODEL_OUTPUT) / 1e6
    print(f"  ✅ best.pt disimpan ke: {MODEL_OUTPUT}")
    print(f"     Ukuran: {ukuran:.1f} MB")

    # Juga simpan last.pt untuk resume jika perlu
    last_pt = best_pt_training.replace("best.pt", "last.pt")
    if os.path.exists(last_pt):
        shutil.copy2(last_pt, "models/last.pt")
        print(f"  ✅ last.pt disimpan ke: models/last.pt")

    return True


# ================================================================
# STEP 5 — EVALUASI MODEL
# ================================================================

def evaluasi_model(device: str, yaml_path: str):
    """
    Evaluasi model pada test set.
    Menghasilkan: Precision, Recall, F1, mAP, Confusion Matrix.
    """
    from ultralytics import YOLO

    print(f"\n{'='*55}")
    print(f"  STEP 5 — EVALUASI MODEL PADA TEST SET")
    print(f"{'='*55}")

    if not os.path.exists(MODEL_OUTPUT):
        print(f"  ❌ Model tidak ditemukan: {MODEL_OUTPUT}")
        return {}

    model = YOLO(MODEL_OUTPUT)

    hasil = model.val(
        data     = yaml_path,
        split    = "test",
        conf     = 0.4,
        iou      = 0.5,
        imgsz    = IMGSZ,
        device   = device,
        verbose  = True,
        plots    = True,
        project  = PROJECT_NAME,
        name     = "evaluation",
        exist_ok = True,
    )

    mp      = float(hasil.box.mp)
    mr      = float(hasil.box.mr)
    map50   = float(hasil.box.map50)
    map5095 = float(hasil.box.map)
    f1      = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0

    # AP per kelas
    ap_per_kelas = {}
    if hasattr(hasil.box, "ap50"):
        for i, nama in enumerate(NAMA_KELAS):
            if i < len(hasil.box.ap50):
                ap_per_kelas[nama] = float(hasil.box.ap50[i])

    print(f"\n  {'─'*50}")
    print(f"  HASIL EVALUASI TEST SET")
    print(f"  {'─'*50}")
    print(f"  Precision (mean) : {mp*100:6.2f}%")
    print(f"  Recall (mean)    : {mr*100:6.2f}%")
    print(f"  F1-Score         : {f1*100:6.2f}%")
    print(f"  mAP@0.5          : {map50*100:6.2f}%")
    print(f"  mAP@0.5:0.95     : {map5095*100:6.2f}%")

    if ap_per_kelas:
        print(f"\n  AP@0.5 per kelas:")
        for nama, ap in ap_per_kelas.items():
            bar = "█" * int(ap * 20)
            print(f"    {nama:8s}: {ap*100:5.1f}% {bar}")

    # Perbandingan dengan Tugas 1
    print(f"\n  {'─'*50}")
    print(f"  PERBANDINGAN DENGAN TUGAS 1 (MOG2 + K-Means)")
    print(f"  {'─'*50}")
    t1 = {"Precision": 0.734, "Recall": 0.678, "F1-Score": 0.703}
    t2 = {"Precision": mp,    "Recall": mr,    "F1-Score": f1}

    print(f"  {'Metrik':12s} | {'Tugas 1':9s} | {'Tugas 2':9s} | Δ")
    print(f"  {'-'*45}")
    for m in ["Precision", "Recall", "F1-Score"]:
        delta = t2[m] - t1[m]
        tanda = "+" if delta >= 0 else ""
        print(f"  {m:12s} | {t1[m]*100:7.1f}% | {t2[m]*100:7.1f}% | "
              f"{tanda}{delta*100:.1f}%")

    metrik = {
        "model":         MODEL_OUTPUT,
        "precision":     mp,
        "recall":        mr,
        "f1_score":      f1,
        "mAP50":         map50,
        "mAP50_95":      map5095,
        "ap_per_kelas":  ap_per_kelas,
        "conf_thr":      0.4,
        "iou_thr":       0.5,
    }

    return metrik


# ================================================================
# STEP 6 — BUAT GRAFIK PERBANDINGAN
# ================================================================

def buat_grafik_perbandingan(metrik: dict):
    """
    Buat grafik batang perbandingan Tugas 1 vs Tugas 2.
    Disimpan ke results/training/perbandingan.png
    """
    print(f"\n{'='*55}")
    print(f"  STEP 6 — GRAFIK PERBANDINGAN")
    print(f"{'='*55}")

    labels   = ["Precision", "Recall", "F1-Score"]
    nilai_t1 = [0.734, 0.678, 0.703]
    nilai_t2 = [
        metrik.get("precision", 0),
        metrik.get("recall", 0),
        metrik.get("f1_score", 0)
    ]

    x     = np.arange(len(labels))
    lebar = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bar1 = ax.bar(
        x - lebar/2,
        [v*100 for v in nilai_t1],
        lebar,
        label="Tugas 1: MOG2 + K-Means",
        color="#95A5A6",
        edgecolor="white"
    )
    bar2 = ax.bar(
        x + lebar/2,
        [v*100 for v in nilai_t2],
        lebar,
        label="Tugas 2: YOLOv8n Fine-tuned",
        color="#2ECC71",
        edgecolor="white"
    )

    # Angka di atas bar
    for bar in bar1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., h + 0.5,
            f"{h:.1f}%",
            ha="center", va="bottom",
            fontsize=9, color="#555"
        )
    for bar in bar2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., h + 0.5,
            f"{h:.1f}%",
            ha="center", va="bottom",
            fontsize=9, color="#27AE60",
            fontweight="bold"
        )

    ax.set_ylabel("Nilai (%)", fontsize=11)
    ax.set_title(
        "Perbandingan Performa Tugas 1 vs Tugas 2\n"
        "MOG2 + K-Means  vs  YOLOv8n Fine-tuned\n"
        "Studi Kasus: Simpang Uniland Medan",
        fontsize=11, fontweight="bold", pad=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path = os.path.join(PROJECT_NAME, RUN_NAME,
                               "perbandingan_t1_vs_t2.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✅ Grafik disimpan: {output_path}")
    return output_path


# ================================================================
# STEP 7 — SIMPAN METRIK JSON
# ================================================================

def simpan_metrik(metrik: dict, durasi: float, map50_train: float):
    """Simpan ringkasan metrik ke JSON untuk laporan."""
    metrik_lengkap = {
        **metrik,
        "durasi_training_menit": round(durasi, 1),
        "map50_best_train":      round(map50_train * 100, 2),
        "base_model":            BASE_MODEL,
        "epochs_config":         EPOCHS,
        "batch":                 BATCH,
        "imgsz":                 IMGSZ,
        "optimizer":             "AdamW",
        "dataset":               "Simpang Uniland Medan",
        "kelas":                 NAMA_KELAS,
    }

    json_path = os.path.join(PROJECT_NAME, RUN_NAME, "metrics_final.json")
    with open(json_path, "w") as f:
        json.dump(metrik_lengkap, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Metrik disimpan: {json_path}")
    print(f"     Buka file ini untuk salin angka ke laporan BAB 4")


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "="*55)
    print("  TRAINING YOLOv8n — TUGAS 2 VISI KOMPUTER")
    print("  Kelompok: Silvani, Cindy, Gloria")
    print("  Universitas Mikroskil 2026")
    print("="*55)

    # Step 0: Fix PyTorch compatibility
    print(f"\n{'='*55}")
    print(f"  STEP 0 — FIX PYTORCH COMPATIBILITY")
    print(f"{'='*55}")
    fix_pytorch_compatibility()

    # Step 1: Fix data.yaml
    yaml_path = fix_data_yaml()

    # Step 2: Cek device
    device = cek_device()

    # Step 3: Training
    best_pt_path, durasi, map50_train = jalankan_training(device, yaml_path)

    # Step 4: Simpan model
    berhasil = simpan_model(best_pt_path)

    if not berhasil:
        print("❌ Gagal menyimpan model. Cek folder results/training/")
        sys.exit(1)

    # Step 5: Evaluasi
    metrik = evaluasi_model(device, yaml_path)

    # Step 6: Grafik perbandingan
    if metrik:
        buat_grafik_perbandingan(metrik)

    # Step 7: Simpan metrik
    if metrik:
        simpan_metrik(metrik, durasi, map50_train)

    # Ringkasan akhir
    print(f"\n{'='*55}")
    print(f"  🎉 SEMUA SELESAI!")
    print(f"{'='*55}")
    print(f"  Model     : {MODEL_OUTPUT}")
    print(f"  F1-Score  : {metrik.get('f1_score', 0)*100:.1f}%")
    print(f"  mAP@0.5   : {metrik.get('mAP50', 0)*100:.1f}%")
    print(f"  Hasil     : {os.path.join(PROJECT_NAME, RUN_NAME)}/")
    print(f"\n  Langkah selanjutnya:")
    print(f"  1. Cek folder results/training/{RUN_NAME}/")
    print(f"     → confusion_matrix.png")
    print(f"     → results.png (grafik loss)")
    print(f"     → perbandingan_t1_vs_t2.png")
    print(f"     → metrics_final.json")
    print(f"  2. Jalankan app: streamlit run app.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()