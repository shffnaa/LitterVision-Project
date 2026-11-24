import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_FILE = ROOT / "TACO-source/data/annotations.json"
SOURCE_DIR = ROOT / "TACO-source/data"
OUTPUT_DIR = ROOT / "dataset/yolo_expanded"

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
        "Foam cup",
        "Glass cup",
        "Other plastic cup",
    ],
    "Packaging": [
        "Plastic film",
        "Other plastic wrapper",
        "Crisp packet",
        "Plastic bottle cap",
        "Other carton",
        "Metal bottle cap",
        "Plastic lid",
        "Corrugated carton",
        "Aluminium foil",
        "Single-use carrier bag",
        "Drink carton",
        "Disposable food container",
        "Meal carton",
        "Paper bag",
        "Foam food container",
        "Wrapping paper",
        "Egg carton",
        "Metal lid",
        "Spread tub",
        "Other plastic container",
        "Six pack rings",
        "Tupperware",
        "Polypropylene bag",
        "Pizza box",
        "Carded blister pack",
    ],
    "Straw": [
        "Plastic straw",
        "Paper straw",
    ],
    "Cigarette": [
        "Cigarette",
    ],
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

# Bersihkan output lama agar tidak ada file sisa
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for split in splits:
    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

class_counts = Counter()
clipped_counts = Counter()
skipped_counts = Counter()
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

        image_name = (
            f"{image_id}_{Path(image_info['file_name']).name}"
        )

        destination = OUTPUT_DIR / "images" / split / image_name
        label_path = (
            OUTPUT_DIR / "labels" / split /
            f"{destination.stem}.txt"
        )

        shutil.copy2(source, destination)

        lines = []

        for ann in selected[image_id]:
            class_name = categories[ann["category_id"]]
            class_id = CATEGORY_TO_CLASS[class_name]

            x, y, w, h = map(float, ann["bbox"])

            x1, y1 = x, y
            x2, y2 = x + w, y + h

            clipped_x1 = max(0.0, min(x1, width))
            clipped_y1 = max(0.0, min(y1, height))
            clipped_x2 = max(0.0, min(x2, width))
            clipped_y2 = max(0.0, min(y2, height))

            if (
                clipped_x2 <= clipped_x1
                or clipped_y2 <= clipped_y1
            ):
                skipped_counts[CLASS_NAMES[class_id]] += 1
                continue

            if (
                clipped_x1 != x1 or clipped_y1 != y1
                or clipped_x2 != x2 or clipped_y2 != y2
            ):
                clipped_counts[CLASS_NAMES[class_id]] += 1

            box_w = clipped_x2 - clipped_x1
            box_h = clipped_y2 - clipped_y1

            x_center = (clipped_x1 + clipped_x2) / 2 / width
            y_center = (clipped_y1 + clipped_y2) / 2 / height
            norm_w = box_w / width
            norm_h = box_h / height

            lines.append(
                f"{class_id} {x_center:.8f} {y_center:.8f} "
                f"{norm_w:.8f} {norm_h:.8f}"
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

(OUTPUT_DIR / "data.yaml").write_text(
    yaml_content, encoding="utf-8"
)

print("\n=== LITTERVISION DATASET PREPARATION ===")
print("Selected images:", n)
print("Selected annotations:", total_annotations)

print("\nSplit:")
for split, ids in splits.items():
    actual = len(
        list((OUTPUT_DIR / "images" / split).glob("*"))
    )
    print(f"{split}: {actual} images")

print("\nClass counts:")
for name in CLASS_NAMES:
    print(f"{name}: {class_counts[name]}")

print("\nBounding boxes clipped:")
for name in CLASS_NAMES:
    print(f"{name}: {clipped_counts[name]}")

print("\nAnnotations skipped:")
for name in CLASS_NAMES:
    print(f"{name}: {skipped_counts[name]}")

print("\nMissing images:", len(missing))

if missing:
    print("First missing files:", missing[:10])

print("\nYAML:", OUTPUT_DIR / "data.yaml")
print("Done!")
