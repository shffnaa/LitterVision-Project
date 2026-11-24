from pathlib import Path
from collections import Counter

LABEL_DIR = Path("dataset/yolo/labels/train")

CLASS_NAMES = {
    0: "Bottle",
    1: "Can",
    2: "Cup",
    3: "Packaging",
    4: "Straw",
    5: "Cigarette"
}

counts = Counter()
image_counts = Counter()

for label_file in LABEL_DIR.glob("*.txt"):
    seen_classes = set()

    for line in label_file.read_text().splitlines():
        parts = line.split()

        if not parts:
            continue

        class_id = int(parts[0])

        counts[class_id] += 1
        seen_classes.add(class_id)

    for class_id in seen_classes:
        image_counts[class_id] += 1

print("=== TRAINING ANNOTATION AUDIT ===")

for class_id, name in CLASS_NAMES.items():
    print(
        f"{name:12} | "
        f"objects: {counts[class_id]:4} | "
        f"images: {image_counts[class_id]:4}"
    )
