"""
merge_dataset.py
Gabungkan label dari semua sesi → split train/val/test → data.yaml

Kelompok: Silvani Chayadi, Cindy Nathania, Gloria Apriyanti
Universitas Mikroskil 2026
"""

import os
import shutil
import random
import yaml
from pathlib import Path


NAMA_KELAS = ["motor", "mobil", "bus", "truk"]


def merge_dan_split(
    labeled_base_dir: str = "dataset/auto_labeled",
    output_dir: str       = "dataset/yolo_dataset",
    split_ratio: tuple    = (0.70, 0.20, 0.10),
    seed: int             = 42
) -> str:
    """
    Gabungkan semua sesi + split + generate data.yaml.

    Rasio 70/20/10:
      70% train → cukup untuk fine-tuning
      20% val   → monitor overfitting selama training
      10% test  → evaluasi akhir yang tidak disentuh selama training

    Args:
        labeled_base_dir : folder hasil auto_labeler.py
        output_dir       : folder output dataset final
        split_ratio      : (train, val, test)
        seed             : random seed untuk reprodusibilitas

    Returns:
        str: path ke data.yaml
    """
    assert abs(sum(split_ratio) - 1.0) < 1e-6, \
        "split_ratio harus berjumlah 1.0"

    random.seed(seed)

    images_raw = os.path.join(labeled_base_dir, "images_raw")
    labels_raw = os.path.join(labeled_base_dir, "labels_raw")

    # Kumpulkan semua pasangan (gambar, label) dari semua sesi
    pasangan = []

    for sesi_folder in sorted(Path(images_raw).iterdir()):
        if not sesi_folder.is_dir():
            continue

        sesi_nama    = sesi_folder.name
        label_folder = Path(labels_raw) / sesi_nama

        for img_path in sorted(sesi_folder.glob("*.jpg")):
            lbl_path = label_folder / (img_path.stem + ".txt")

            # Hanya include jika label ada dan TIDAK kosong
            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                # Rename dengan prefix sesi untuk hindari nama duplikat
                nama_baru = f"{sesi_nama}_{img_path.name}"
                pasangan.append((img_path, lbl_path, nama_baru))

    if not pasangan:
        raise ValueError(
            "Tidak ada pasangan gambar+label ditemukan!\n"
            "Pastikan auto_labeler.py dan LabelImg sudah dijalankan."
        )

    random.shuffle(pasangan)
    total   = len(pasangan)
    n_train = int(total * split_ratio[0])
    n_val   = int(total * split_ratio[1])

    splits = {
        "train": pasangan[:n_train],
        "val":   pasangan[n_train:n_train + n_val],
        "test":  pasangan[n_train + n_val:]
    }

    print(f"\n{'='*55}")
    print(f"  MERGE & SPLIT DATASET")
    print(f"{'='*55}")
    print(f"  Total pasangan : {total}")
    print(f"  Train (70%)    : {len(splits['train'])}")
    print(f"  Val   (20%)    : {len(splits['val'])}")
    print(f"  Test  (10%)    : {len(splits['test'])}")

    # Buat folder output
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Copy file ke split masing-masing
    for split, pasangan_list in splits.items():
        for img_path, lbl_path, nama_baru in pasangan_list:
            nama_label = os.path.splitext(nama_baru)[0] + ".txt"

            shutil.copy2(
                str(img_path),
                os.path.join(output_dir, "images", split, nama_baru)
            )
            shutil.copy2(
                str(lbl_path),
                os.path.join(output_dir, "labels", split, nama_label)
            )

        print(f"  ✅ {split:5s}: {len(pasangan_list)} file disalin")

    # Generate data.yaml
    abs_output = os.path.abspath(output_dir)
    data_yaml  = {
        "path":  abs_output,
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(NAMA_KELAS),
        "names": NAMA_KELAS
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n  ✅ data.yaml dibuat: {yaml_path}")
    print(f"\n  Isi data.yaml:")
    print(f"  {'─'*40}")
    with open(yaml_path) as f:
        for line in f:
            print(f"  {line}", end="")
    print(f"\n  {'─'*40}")

    # Tampilkan distribusi kelas
    print(f"\n  Distribusi bbox per kelas per split:")
    print(f"  {'Kelas':8s} | {'Train':6s} | {'Val':6s} | {'Test':6s} | Total")
    print(f"  {'-'*45}")

    for split in ["train", "val", "test"]:
        pass  # hitung di bawah

    dist = {split: {i: 0 for i in range(4)} for split in ["train","val","test"]}
    for split in ["train", "val", "test"]:
        ldir = os.path.join(output_dir, "labels", split)
        for lf in Path(ldir).glob("*.txt"):
            for line in lf.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    if cid < 4:
                        dist[split][cid] += 1

    for i, nama in enumerate(NAMA_KELAS):
        tr = dist["train"][i]
        vl = dist["val"][i]
        ts = dist["test"][i]
        print(f"  {nama:8s} | {tr:6d} | {vl:6d} | {ts:6d} | {tr+vl+ts}")

    print(f"\n  ✅ Dataset siap! Upload folder ke Google Drive:")
    print(f"     {abs_output}")
    print(f"{'='*55}")

    return yaml_path


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    merge_dan_split(
        labeled_base_dir = str(ROOT / "dataset/auto_labeled"),
        output_dir       = str(ROOT / "dataset/yolo_dataset"),
        split_ratio      = (0.70, 0.20, 0.10),
        seed             = 42
    )