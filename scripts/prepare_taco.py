import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_FILE = ROOT / "TACO-source/data/annotations.json"
SOURCE_DIR = ROOT / "TACO-source/data"
OUTPUT_DIR = ROOT / "dataset/yolo"

SEED = 42

CLASS_MAPPING = {
    "Bottle": [
        "Clear plastic bottle",
        "Other plastic bottle",
        "Glass bottle",
    ],
    "Can": [
        "Drink can",
        "Food Can",
    ],
    "Cup": [
        "Paper cup",
        "Disposable plastic cup",
    ],
    "Packaging": [
        "Plastic film",
        "Other plastic wrapper",
        "Crisp packet",
    ],
    "Straw": ["Plastic straw"],
    "Cigarette": ["Cigarette"],
}

CLASS_NAMES = list(CLASS_MAPPING.keys())
CATEGORY_TO_CLASS = {
    category: class_id
    for class_id, class_name in enumerate(CLASS_NAMES)
    for category in CLASS_MAPPING[class_name]
}

with open(ANNOTATION_FILE, encoding="utf-8") as f:
    coco = json.load(f)

categories = {c["id"]: c["name"] for c in coco["categories"]}
images = {img["id"]: img for img in coco["images"]}

# Ambil anotasi dari enam kelas terpilih
selected = defaultdict(list)

for ann in coco["annotations"]:
    category_name = categories.get(ann["category_id"])
    if category_name not in CATEGORY_TO_CLASS:
        continue

    bbox = ann.get("bbox", [])
    if len(bbox) != 4:
        continue

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        continue

    selected[ann["image_id"]].append(ann)

# Hanya gunakan gambar yang memiliki objek dari enam kelas
image_ids = [
    image_id for image_id in selected
    if image_id in images
]

rng = random.Random(SEED)
rng.shuffle(image_ids)

n = len(image_ids)
n_train = int(n * 0.70)
n_val = int(n * 0.20)

splits = {
    "train": image_ids[:n_train],
    "val": image_ids[n_train:n_train + n_val],
    "test": image_ids[n_train + n_val:],
}

for split in splits:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

class_counts = Counter()
missing = []
total_annotations = 0

for split, ids in splits.items():
    for image_id in ids:
        image_info = images[image_id]
        source = SOURCE_DIR / image_info["file_name"]

        if not source.exists():
            missing.append(str(source))
            continue

        width = image_info["width"]
        height = image_info["height"]

        image_name = f"{image_id}_{Path(image_info["file_name"]).name}"
        destination = OUTPUT_DIR / "images" / split / image_name
        label_path = OUTPUT_DIR / "labels" / split / f"{destination.stem}.txt"

        shutil.copy2(source, destination)

        lines = []
        for ann in selected[image_id]:
            class_name = categories[ann["category_id"]]
            class_id = CATEGORY_TO_CLASS[class_name]

            x, y, w, h = ann["bbox"]

            # Convert COCO xywh to YOLO normalized xywh
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height

            # Skip invalid boxes outside image bounds
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1
                    and 0 < norm_w <= 1 and 0 < norm_h <= 1):
                continue

            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{norm_w:.6f} {norm_h:.6f}"
            )
            class_counts[CLASS_NAMES[class_id]] += 1
            total_annotations += 1

        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8"
        )

yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
test: images/test

names:
"""
for i, name in enumerate(CLASS_NAMES):
    yaml_content += f"  {i}: {name}\n"

(OUTPUT_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")

print("\n=== LITTERVISION DATASET PREPARATION ===")
print("Selected images:", n)
print("Selected annotations:", total_annotations)
print("\nSplit:")
for split, ids in splits.items():
    actual = len(list((OUTPUT_DIR / "images" / split).glob("*")))
    print(f"{split}: {actual} images")

print("\nClass counts:")
for name in CLASS_NAMES:
    print(f"{name}: {class_counts[name]}")

print("\nMissing images:", len(missing))
if missing:
    print("First missing files:", missing[:10])

print("\nYAML:", OUTPUT_DIR / "data.yaml")
print("Done!")
