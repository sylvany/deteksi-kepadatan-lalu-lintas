"""
auto_labeler.py
Auto-labeling frame menggunakan YOLOv8 pretrained COCO.

Kelompok: Silvani Chayadi, Cindy Nathania, Gloria Apriyanti
Universitas Mikroskil 2026
"""

import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO


COCO_KE_KITA = {3: 0, 2: 1, 5: 2, 7: 3}
NAMA_KELAS   = ["motor", "mobil", "bus", "truk"]
WARNA_KELAS  = {
    0: (0, 255, 0),
    1: (255, 150, 0),
    2: (0, 165, 255),
    3: (0, 0, 255),
}


def auto_label_folder(
    input_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    preview_dir: str = None,
    confidence: float = 0.35,
    iou: float = 0.45
) -> dict:
    """
    Auto-label semua gambar dalam folder.

    Confidence 0.35 (lebih rendah dari default 0.5) karena:
      Lebih baik punya sedikit false positive (bisa dihapus manual)
      daripada banyak missed detection (kendaraan tidak terlabel).

    Args:
        input_dir         : folder berisi gambar .jpg
        output_images_dir : tujuan copy gambar
        output_labels_dir : tujuan simpan file .txt label
        preview_dir       : folder preview dengan bbox overlay (opsional)
        confidence        : threshold confidence
        iou               : threshold IoU NMS

    Returns:
        dict statistik
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    print("\n  Memuat YOLOv8 pretrained...")
    print("  [Hanya untuk auto-labeling — bukan inferensi final]")
    model = YOLO("yolov8n.pt")

    semua_gambar = sorted([
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not semua_gambar:
        print(f"  ⚠️  Tidak ada gambar di: {input_dir}")
        return {}

    print(f"\n{'='*55}")
    print(f"  AUTO-LABELING: {Path(input_dir).name}")
    print(f"  Jumlah gambar: {len(semua_gambar)}")
    print(f"  Confidence   : {confidence}")
    print(f"{'='*55}")

    total_bbox      = 0
    gambar_kosong   = 0
    distribusi      = {i: 0 for i in range(4)}

    for idx, img_path in enumerate(semua_gambar):

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Jalankan deteksi — hanya kelas kendaraan
        results = model(
            img,
            conf    = confidence,
            iou     = iou,
            classes = list(COCO_KE_KITA.keys()),
            verbose = False
        )

        label_lines = []
        img_preview = img.copy() if preview_dir else None

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                coco_id  = int(box.cls[0])
                if coco_id not in COCO_KE_KITA:
                    continue

                our_id   = COCO_KE_KITA[coco_id]
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Format YOLO (normalized)
                xc = max(0.0, min(1.0, (x1 + x2) / 2 / w))
                yc = max(0.0, min(1.0, (y1 + y2) / 2 / h))
                bw = max(0.001, min(1.0, (x2 - x1) / w))
                bh = max(0.001, min(1.0, (y2 - y1) / h))

                label_lines.append(
                    f"{our_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                )
                distribusi[our_id] += 1
                total_bbox += 1

                # Preview
                if img_preview is not None:
                    warna = WARNA_KELAS[our_id]
                    cv2.rectangle(
                        img_preview,
                        (int(x1), int(y1)), (int(x2), int(y2)),
                        warna, 2
                    )
                    cv2.putText(
                        img_preview,
                        f"{NAMA_KELAS[our_id]} {conf_val:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, warna, 1
                    )

        # Simpan label
        label_path = os.path.join(
            output_labels_dir, img_path.stem + ".txt"
        )
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        if not label_lines:
            gambar_kosong += 1

        # Copy gambar
        shutil.copy2(str(img_path),
                     os.path.join(output_images_dir, img_path.name))

        # Simpan preview
        if preview_dir and img_preview is not None:
            cv2.imwrite(
                os.path.join(preview_dir, f"prev_{img_path.name}"),
                img_preview
            )

        # Progress
        if (idx + 1) % 50 == 0 or (idx + 1) == len(semua_gambar):
            print(f"  [{idx+1}/{len(semua_gambar)}] "
                  f"bbox: {total_bbox} | kosong: {gambar_kosong}")

    print(f"\n  ✅ Selesai: {len(semua_gambar)} gambar diproses")
    print(f"  📦 Total bbox: {total_bbox}")
    print(f"  📭 Gambar tanpa kendaraan: {gambar_kosong}")
    print(f"\n  Distribusi kelas:")
    for i, nama in enumerate(NAMA_KELAS):
        pct = distribusi[i] / max(total_bbox, 1) * 100
        bar = "█" * int(pct / 3)
        print(f"    {nama:8s}: {distribusi[i]:5d} ({pct:5.1f}%) {bar}")

    if preview_dir:
        print(f"\n  🖼  Preview: {preview_dir}")
        print(f"      Buka folder ini, cek visual, tandai yang perlu dikoreksi")

    return {
        "total_gambar": len(semua_gambar),
        "total_bbox":   total_bbox,
        "distribusi":   distribusi
    }


def auto_label_semua_sesi(
    frames_base_dir: str = "dataset/frames",
    output_base_dir: str = "dataset/auto_labeled"
) -> list:
    """
    Auto-label semua sesi sekaligus.

    Struktur input yang diharapkan:
      frames_base_dir/
        sesi_0900/  ← folder per sesi dari frame_extractor.py
        sesi_1200/
        sesi_1700/

    Output:
      output_base_dir/
        images_raw/sesi_0900/
        labels_raw/sesi_0900/
        preview/sesi_0900/
        (dst untuk sesi lain)
    """
    sesi_folders = sorted([
        d for d in Path(frames_base_dir).iterdir() if d.is_dir()
    ])

    if not sesi_folders:
        # Jika tidak ada subfolder, anggap langsung berisi gambar
        sesi_folders = [Path(frames_base_dir)]

    print(f"  Ditemukan {len(sesi_folders)} sesi: "
          f"{[s.name for s in sesi_folders]}")

    semua_stat = []

    for sesi_folder in sesi_folders:
        nama = sesi_folder.name
        stat = auto_label_folder(
            input_dir         = str(sesi_folder),
            output_images_dir = os.path.join(output_base_dir, "images_raw", nama),
            output_labels_dir = os.path.join(output_base_dir, "labels_raw", nama),
            preview_dir       = os.path.join(output_base_dir, "preview", nama),
            confidence        = 0.35,
            iou               = 0.45
        )
        stat["sesi"] = nama
        semua_stat.append(stat)

    total_bbox   = sum(s.get("total_bbox", 0) for s in semua_stat)
    total_gambar = sum(s.get("total_gambar", 0) for s in semua_stat)

    print(f"\n{'='*55}")
    print(f"  RINGKASAN AUTO-LABELING")
    print(f"{'='*55}")
    print(f"  Total gambar : {total_gambar}")
    print(f"  Total bbox   : {total_bbox}")
    for s in semua_stat:
        print(f"  {s['sesi']:15s}: {s.get('total_bbox',0)} bbox")

    return semua_stat


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    auto_label_semua_sesi(
        frames_base_dir = str(ROOT / "dataset/frames"),
        output_base_dir = str(ROOT / "dataset/auto_labeled")
    )