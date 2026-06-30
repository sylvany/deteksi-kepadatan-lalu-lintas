"""
restratify_split.py
Split ulang dataset dengan menjamin kelas minoritas (bus, truk)
muncul di val dan test set, bukan murni acak.

Jalankan ini jika distribusi kelas sudah cukup (>30 instance per kelas)
tapi split acak kebetulan membuat satu kelas hilang dari test/val.

Kelompok: Silvani Chayadi, Cindy Nathania, Gloria Apriyanti
Universitas Mikroskil 2026
"""

import os
import shutil
import random
import yaml
from pathlib import Path


NAMA_KELAS = ["motor", "mobil", "bus", "truk"]

# Kelas yang harus dijamin punya representasi di val & test
# (index sesuai NAMA_KELAS: 2=bus, 3=truk)
KELAS_DIJAMIN = [2, 3]

# Minimal berapa gambar berisi kelas ini yang dipaksa masuk
# ke masing-masing val dan test
MIN_GAMBAR_PER_SPLIT = 2


def kumpulkan_semua_item(dataset_dir: Path):
    """Gabungkan kembali semua gambar+label dari train/val/test."""
    semua_item = []  # list of [img_path, lbl_path, set_kelas]

    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            continue

        for img_path in img_dir.glob("*.jpg"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            kelas_di_gambar = set()

            if lbl_path.exists():
                isi = lbl_path.read_text().strip()
                if isi:
                    for line in isi.split("\n"):
                        parts = line.strip().split()
                        if len(parts) == 5:
                            kelas_di_gambar.add(int(parts[0]))

            semua_item.append([img_path, lbl_path, kelas_di_gambar])

    return semua_item


def restratify(
    dataset_dir: str = "dataset/yolo_dataset",
    split_ratio: tuple = (0.70, 0.20, 0.10),
    seed: int = 42
):
    """
    Split ulang dataset dengan stratifikasi untuk kelas minoritas.
    """
    random.seed(seed)
    dataset_path = Path(dataset_dir)

    # ── Backup dulu ──────────────────────────────────────────
    backup_dir = dataset_path.parent / "yolo_dataset_backup_sebelum_restratify"
    if not backup_dir.exists():
        print(f"Membuat backup ke: {backup_dir}")
        shutil.copytree(dataset_path, backup_dir)
        print("✅ Backup selesai\n")

    # ── Kumpulkan semua item dari backup (bukan dataset_path,
    #    karena dataset_path akan dihapus lalu dibuat ulang) ──
    semua_item = kumpulkan_semua_item(backup_dir)
    random.shuffle(semua_item)

    print(f"{'='*55}")
    print(f"  RESTRATIFIKASI SPLIT")
    print(f"{'='*55}")
    print(f"  Total gambar dikumpulkan: {len(semua_item)}")

    assigned = {"train": [], "val": [], "test": []}
    sudah_dipakai = set()

    # ── Step 1: jamin kelas minoritas masuk val & test ───────
    for cid in KELAS_DIJAMIN:
        kandidat = [
            item for item in semua_item
            if cid in item[2] and id(item) not in sudah_dipakai
        ]
        random.shuffle(kandidat)

        nama_kelas = NAMA_KELAS[cid]
        print(f"\n  Kelas '{nama_kelas}': {len(kandidat)} gambar tersedia")

        for split_name in ["test", "val"]:
            jumlah_assigned = 0
            for item in kandidat:
                if id(item) in sudah_dipakai:
                    continue
                if jumlah_assigned >= MIN_GAMBAR_PER_SPLIT:
                    break
                assigned[split_name].append(item)
                sudah_dipakai.add(id(item))
                jumlah_assigned += 1

            print(f"    → dipaksa masuk {split_name}: {jumlah_assigned} gambar")

    # ── Step 2: sisanya di-split normal 70/20/10 ─────────────
    sisa = [item for item in semua_item if id(item) not in sudah_dipakai]
    random.shuffle(sisa)

    n = len(sisa)
    n_train = int(n * split_ratio[0])
    n_val   = int(n * split_ratio[1])

    assigned["train"] += sisa[:n_train]
    assigned["val"]   += sisa[n_train:n_train + n_val]
    assigned["test"]  += sisa[n_train + n_val:]

    print(f"\n  Sisa gambar di-split normal: {n}")
    print(f"  Train: {len(assigned['train'])} | "
          f"Val: {len(assigned['val'])} | "
          f"Test: {len(assigned['test'])}")

    # ── Hapus folder lama, buat ulang ────────────────────────
    for split in ["train", "val", "test"]:
        shutil.rmtree(dataset_path / "images" / split, ignore_errors=True)
        shutil.rmtree(dataset_path / "labels" / split, ignore_errors=True)
        os.makedirs(dataset_path / "images" / split, exist_ok=True)
        os.makedirs(dataset_path / "labels" / split, exist_ok=True)

    # ── Copy ulang sesuai assignment baru ────────────────────
    for split, item_list in assigned.items():
        for img_path, lbl_path, _ in item_list:
            shutil.copy2(img_path, dataset_path / "images" / split / img_path.name)
            shutil.copy2(lbl_path, dataset_path / "labels" / split / lbl_path.name)

    # ── Verifikasi distribusi baru ────────────────────────────
    print(f"\n{'='*55}")
    print(f"  DISTRIBUSI SETELAH RESTRATIFIKASI")
    print(f"{'='*55}")

    for split in ["train", "val", "test"]:
        labels_dir = dataset_path / "labels" / split
        distribusi = {i: 0 for i in range(len(NAMA_KELAS))}
        n_gambar = 0

        for lbl_file in labels_dir.glob("*.txt"):
            n_gambar += 1
            isi = lbl_file.read_text().strip()
            if not isi:
                continue
            for line in isi.split("\n"):
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    if cid in distribusi:
                        distribusi[cid] += 1

        print(f"\n  {split.upper()} ({n_gambar} gambar):")
        for i, nama in enumerate(NAMA_KELAS):
            jumlah = distribusi[i]
            flag = " ⚠️  MASIH KOSONG" if jumlah == 0 else " ✅"
            print(f"    {nama:8s}: {jumlah:4d} instance{flag}")

    print(f"\n  ✅ Dataset sudah di-restratifikasi")
    print(f"     Upload ulang folder {dataset_dir} ke Google Drive")


if __name__ == "__main__":
    restratify(
        dataset_dir = "dataset/yolo_dataset",
        split_ratio = (0.70, 0.20, 0.10),
        seed = 42
    )