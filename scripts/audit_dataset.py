from pathlib import Path
from collections import defaultdict
import statistics

ROOT = Path(__file__).resolve().parent.parent
LABEL_ROOT = ROOT / "dataset" / "yolo_expanded" / "labels"

CLASSES = {
    0: "Bottle",
    1: "Can",
    2: "Cup",
    3: "Packaging",
    4: "Straw",
    5: "Cigarette",
}

SPLITS = ["train", "val", "test"]

print("=" * 65)
print("LITTERVISION EXPANDED DATASET AUDIT")
print("=" * 65)

for split in SPLITS:
    folder = LABEL_ROOT / split
    counts = defaultdict(int)
    areas = defaultdict(list)
    total_images = 0
    empty_labels = 0
    invalid_labels = 0

    if not folder.exists():
        print(f"\nFolder tidak ditemukan: {folder}")
        continue

    for file in sorted(folder.glob("*.txt")):
        total_images += 1
        lines = file.read_text(encoding="utf-8").splitlines()

        if not lines:
            empty_labels += 1

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                invalid_labels += 1
                continue

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid_labels += 1
                continue

            eps = 1e-6

            valid = (
                cls in CLASSES
                and -eps <= x <= 1 + eps
                and -eps <= y <= 1 + eps
                and 0 < w <= 1 + eps
                and 0 < h <= 1 + eps
                and x - w / 2 >= -eps
                and y - h / 2 >= -eps
                and x + w / 2 <= 1 + eps
                and y + h / 2 <= 1 + eps
            )

            if not valid:
                invalid_labels += 1
                continue

            counts[cls] += 1
            areas[cls].append(w * h * 100)

    print(f"\n=== {split.upper()} ===")
    print("Jumlah file label:", total_images)
    print("File label kosong:", empty_labels)
    print("Baris label tidak valid:", invalid_labels)
    print("-" * 65)
    print(f"{'Class':<15} {'Count':>8} {'Mean %':>12} {'Median %':>12} {'<0.1%':>10}")

    for cls, name in CLASSES.items():
        values = areas[cls]

        if values:
            mean_area = statistics.mean(values)
            median_area = statistics.median(values)
            tiny = sum(v < 0.1 for v in values)

            print(
                f"{name:<15} {counts[cls]:>8} "
                f"{mean_area:>12.4f} {median_area:>12.4f} {tiny:>10}"
            )
        else:
            print(f"{name:<15} {0:>8} {'-':>12} {'-':>12} {0:>10}")

    print("Total objek:", sum(counts.values()))

print("\nAUDIT SELESAI")
