from pathlib import Path
import random
import cv2
import math

ROOT = Path("dataset/yolo_expanded")
IMAGE_DIR = ROOT / "images" / "train"
LABEL_DIR = ROOT / "labels" / "train"

CLASS_NAMES = {
    0: "Bottle",
    1: "Can",
    2: "Cup",
    3: "Packaging",
    4: "Straw",
    5: "Cigarette"
}

TARGET_CLASSES = list(CLASS_NAMES.keys())
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

SAMPLES_PER_CLASS = 4
TILE_WIDTH = 400
TILE_HEIGHT = 300

random.seed(42)

selected = {class_id: [] for class_id in TARGET_CLASSES}

for label_file in LABEL_DIR.glob("*.txt"):
    lines = label_file.read_text().splitlines()

    class_ids = set()

    for line in lines:
        parts = line.split()
        if parts:
            class_ids.add(int(parts[0]))

    for class_id in TARGET_CLASSES:
        if class_id in class_ids:
            selected[class_id].append(label_file)

for class_id in TARGET_CLASSES:
    random.shuffle(selected[class_id])
    selected[class_id] = selected[class_id][:SAMPLES_PER_CLASS]

tiles = []

for class_id in TARGET_CLASSES:
    for label_file in selected[class_id]:

        image_path = None

        for ext in IMAGE_EXTENSIONS:
            candidate = IMAGE_DIR / (label_file.stem + ext)
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print("Image not found:", label_file.stem)
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        h, w = image.shape[:2]

        for line in label_file.read_text().splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            cid = int(parts[0])
            x, y, bw, bh = map(float, parts[1:])

            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            name = CLASS_NAMES.get(cid, str(cid))

            cv2.putText(
                image, name,
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

        image = cv2.resize(image, (TILE_WIDTH, TILE_HEIGHT))

        title = f"{CLASS_NAMES[class_id]} | {image_path.name}"

        cv2.putText(
            image, title,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2
        )

        tiles.append(image)

if not tiles:
    raise RuntimeError("Tidak ada gambar yang berhasil ditemukan.")

columns = 3
rows = math.ceil(len(tiles) / columns)

canvas = cv2.vconcat([
    cv2.hconcat(
        tiles[i:i + columns] +
        [255 * (tiles[0] * 0)]
        * (columns - len(tiles[i:i + columns]))
    )
    for i in range(0, len(tiles), columns)
])

output = Path("results/visual_audit_expanded.jpg")
output.parent.mkdir(exist_ok=True)

cv2.imwrite(str(output), canvas)

print("Audit image saved:", output)
print("Bottle samples:", len(selected[0]))
print("Packaging samples:", len(selected[3]))
