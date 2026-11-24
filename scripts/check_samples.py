from pathlib import Path
from PIL import Image

ROOT = Path("dataset/yolo")

SAMPLES = [
    "1153_000023",
    "732_000054",
    "514_000015"
]

for name in SAMPLES:
    print(f"\n===== {name} =====")

    labels = list((ROOT / "labels").rglob(name + ".txt"))
    images = [
        p for p in (ROOT / "images").rglob("*")
        if p.stem == name and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    print("Label files:", labels)
    print("Image files:", images)

    if not labels or not images:
        print("File belum lengkap, dilewati.")
        continue

    image_path = images[0]

    with Image.open(image_path) as image:
        width, height = image.size

    print("Image size:", width, "x", height)

    for line in labels[0].read_text().splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        x1 = (x - w / 2) * width
        y1 = (y - h / 2) * height
        x2 = (x + w / 2) * width
        y2 = (y + h / 2) * height

        print(
            f"Class {class_id} | "
            f"Pixel bbox: "
            f"({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )
