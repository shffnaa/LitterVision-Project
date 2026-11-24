import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_FILE = ROOT / "TACO-source/data/annotations.json"

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

MAPPED = {
    category
    for categories in CLASS_MAPPING.values()
    for category in categories
}

with open(ANNOTATION_FILE, encoding="utf-8") as f:
    data = json.load(f)

categories = {
    item["id"]: item["name"]
    for item in data["categories"]
}

counts = Counter()
image_counts = {}

for ann in data["annotations"]:
    name = categories.get(ann["category_id"], "UNKNOWN")

    if name not in MAPPED:
        counts[name] += 1
        image_counts.setdefault(name, set()).add(ann["image_id"])

print("=== UNMAPPED TACO CATEGORIES ===")
print("Total unmapped annotations:", sum(counts.values()))
print("Total unmapped categories:", len(counts))

print("\n=== CATEGORY FREQUENCY ===")

for name, count in counts.most_common():
    print(
        f"{name:<35} | "
        f"objects: {count:>5} | "
        f"images: {len(image_counts[name]):>5}"
    )

print("\n=== CURRENT MAPPING ===")

for class_name, mapped_categories in CLASS_MAPPING.items():
    print(f"\n{class_name}:")
    for category in mapped_categories:
        print(" -", category)
