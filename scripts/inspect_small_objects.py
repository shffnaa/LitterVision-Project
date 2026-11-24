from pathlib import Path
from PIL import Image, ImageDraw
import math

ROOT = Path(__file__).resolve().parent.parent

IMG_DIR = ROOT / "dataset" / "yolo_expanded" / "images" / "train"
LBL_DIR = ROOT / "dataset" / "yolo_expanded" / "labels" / "train"
OUT_DIR = ROOT / "runs" / "littervision" / "dataset_audit_expanded"

OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {
    4: "Straw",
    5: "Cigarette",
}

def find_image(stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = IMG_DIR / (stem + ext)
        if path.exists():
            return path
    return None

def collect_objects(class_id):
    objects = []

    for label_file in LBL_DIR.glob("*.txt"):
        image_file = find_image(label_file.stem)

        if image_file is None:
            continue

        for line in label_file.read_text().splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            cls, x, y, w, h = map(float, parts)

            if int(cls) == class_id:
                objects.append({
                    "image": image_file,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": w * h,
                })

    return sorted(objects, key=lambda obj: obj["area"])

def create_sheet(class_id, name):
    objects = collect_objects(class_id)

    if not objects:
        print(f"Tidak ada objek {name}")
        return

    selected = objects[:16]

    tile_w = 240
    tile_h = 240
    columns = 4
    rows = math.ceil(len(selected) / columns)

    sheet = Image.new(
        "RGB",
        (columns * tile_w, rows * tile_h),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, obj in enumerate(selected):
        img = Image.open(obj["image"]).convert("RGB")
        W, H = img.size

        x1 = (obj["x"] - obj["w"] / 2) * W
        y1 = (obj["y"] - obj["h"] / 2) * H
        x2 = (obj["x"] + obj["w"] / 2) * W
        y2 = (obj["y"] + obj["h"] / 2) * H

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        margin = max(24, int(max(bw, bh) * 2))

        left = max(0, int(x1 - margin))
        top = max(0, int(y1 - margin))
        right = min(W, int(x2 + margin))
        bottom = min(H, int(y2 + margin))

        crop = img.crop((left, top, right, bottom))

        max_w = tile_w - 12
        max_h = tile_h - 45

        scale = min(
            max_w / crop.width,
            max_h / crop.height
        )

        new_w = max(1, int(crop.width * scale))
        new_h = max(1, int(crop.height * scale))

        crop = crop.resize((new_w, new_h))

        col = i % columns
        row = i // columns

        px = col * tile_w + 6
        py = row * tile_h + 6

        sheet.paste(crop, (px, py))

        # Gambar bounding box pada crop
        box = (
            px + (x1 - left) * scale,
            py + (y1 - top) * scale,
            px + (x2 - left) * scale,
            py + (y2 - top) * scale
        )

        draw.rectangle(box, outline="red", width=2)

        draw.text(
            (px, row * tile_h + tile_h - 32),
            f"{name} | Area {obj['area'] * 100:.4f}%",
            fill="black"
        )

    output = OUT_DIR / f"{name.lower()}_small_objects.jpg"
    sheet.save(output, quality=95)

    print(f"{name}: {len(objects)} objek ditemukan")
    print("Contact sheet:", output)

for class_id, name in CLASSES.items():
    create_sheet(class_id, name)

print("\nINSPEKSI SELESAI")
