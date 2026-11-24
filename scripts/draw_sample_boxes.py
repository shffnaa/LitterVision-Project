from pathlib import Path
import cv2

ROOT = Path("dataset/yolo")
OUTPUT = Path("results/annotation_check")
OUTPUT.mkdir(parents=True, exist_ok=True)

SAMPLES = ["1153_000023", "732_000054"]

CLASS_NAMES = {
    0: "Bottle",
    1: "Can",
    2: "Cup",
    3: "Packaging",
    4: "Straw",
    5: "Cigarette"
}

for name in SAMPLES:
    image_files = [
        p for p in (ROOT / "images").rglob("*")
        if p.stem == name and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    label_files = list((ROOT / "labels").rglob(name + ".txt"))

    if not image_files or not label_files:
        print("File tidak lengkap:", name)
        continue

    image = cv2.imread(str(image_files[0]))
    h, w = image.shape[:2]

    for line in label_files[0].read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        cid = int(parts[0])
        x, y, bw, bh = map(float, parts[1:])

        x1 = round((x - bw / 2) * w)
        y1 = round((y - bh / 2) * h)
        x2 = round((x + bw / 2) * w)
        y2 = round((y + bh / 2) * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv2.putText(
            image,
            CLASS_NAMES[cid],
            (x1, max(35, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    output_path = OUTPUT / f"{name}_checked.jpg"
    cv2.imwrite(str(output_path), image)
    print("Saved:", output_path)
