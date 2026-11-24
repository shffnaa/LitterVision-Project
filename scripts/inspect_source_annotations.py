import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_FILE = ROOT / "TACO-source/data/annotations.json"

TARGET_IDS = [732, 1153]

CLASS_MAPPING = {
    "Bottle": [
        "Clear plastic bottle",
        "Other plastic bottle",
        "Glass bottle",
    ],
    "Can": ["Drink can", "Food Can"],
    "Cup": ["Paper cup", "Disposable plastic cup"],
    "Packaging": [
        "Plastic film",
        "Other plastic wrapper",
        "Crisp packet",
    ],
    "Straw": ["Plastic straw"],
    "Cigarette": ["Cigarette"],
}

MAPPED_CATEGORIES = {
    category
    for categories in CLASS_MAPPING.values()
    for category in categories
}

with open(ANNOTATION_FILE, encoding="utf-8") as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}
categories = {cat["id"]: cat["name"] for cat in data["categories"]}

for image_id in TARGET_IDS:
    print("\n" + "=" * 65)
    print("IMAGE ID:", image_id)

    image = images.get(image_id)

    if image is None:
        print("Image ID tidak ditemukan.")
        continue

    print("Source file:", image["file_name"])
    print("Dimensions:", image["width"], "x", image["height"])

    annotations = [
        ann for ann in data["annotations"]
        if ann["image_id"] == image_id
    ]

    print("Total source annotations:", len(annotations))

    for ann in annotations:
        category = categories.get(ann["category_id"], "UNKNOWN")
        status = "MAPPED" if category in MAPPED_CATEGORIES else "NOT MAPPED"

        print(
            f"Annotation {ann['id']} | "
            f"{category} | {status} | "
            f"bbox={ann.get('bbox')} | "
            f"area={ann.get('area')}"
        )
