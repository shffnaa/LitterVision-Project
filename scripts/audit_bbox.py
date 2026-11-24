
from pathlib import Path
from collections import defaultdict
import statistics

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset/yolo"

CLASS_NAMES = [
    "Bottle",
    "Can",
    "Cup",
    "Packaging",
    "Straw",
    "Cigarette",
]

for split in ["train", "val", "test"]:
    areas = defaultdict(list)
    widths = defaultdict(list)
    heights = defaultdict(list)

    label_dir = DATASET / "labels" / split

    for file in label_dir.glob("*.txt"):
        for line in file.read_text().splitlines():
            values = line.split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            _, _, _, w, h = map(float, values)

            areas[class_id].append(w * h * 100)
            widths[class_id].append(w * 100)
            heights[class_id].append(h * 100)

    print(f"\n=== {split.upper()} ===")

    for class_id, name in enumerate(CLASS_NAMES):
        if not areas[class_id]:
            continue

        print(
            f"{name:12} | "
            f"Jumlah: {len(areas[class_id]):4} | "
            f"Rata-rata area: "
            f"{statistics.mean(areas[class_id]):.3f}% | "
            f"Median area: "
            f"{statistics.median(areas[class_id]):.3f}%"
        )
