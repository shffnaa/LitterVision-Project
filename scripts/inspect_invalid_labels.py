from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent.parent
LABEL_ROOT = ROOT / "dataset" / "yolo_expanded" / "labels"

print("=== INVALID LABEL INSPECTION ===")

total_invalid = 0

for split in ["train", "val", "test"]:
    folder = LABEL_ROOT / split
    print(f"\n--- {split.upper()} ---")

    for file in sorted(folder.glob("*.txt")):
        for line_number, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.split()
            reason = None

            if len(parts) != 5:
                reason = "Format bukan 5 kolom"
            else:
                try:
                    cls_value = float(parts[0])
                    x, y, w, h = map(float, parts[1:])

                    if not all(math.isfinite(v) for v in [cls_value, x, y, w, h]):
                        reason = "Nilai NaN atau infinity"
                    elif not cls_value.is_integer():
                        reason = "Class ID bukan bilangan bulat"
                    elif int(cls_value) not in range(6):
                        reason = "Class ID di luar 0-5"
                    elif not (0 <= x <= 1 and 0 <= y <= 1):
                        reason = "Titik tengah di luar gambar"
                    elif not (0 < w <= 1 and 0 < h <= 1):
                        reason = "Ukuran bbox tidak valid"
                    elif (
                        x - w / 2 < -1e-6
                        or x + w / 2 > 1 + 1e-6
                        or y - h / 2 < -1e-6
                        or y + h / 2 > 1 + 1e-6
                    ):
                        reason = "Bounding box melewati batas gambar"

                except ValueError:
                    reason = "Nilai bukan angka"

            if reason:
                total_invalid += 1
                print(f"{file.name}:{line_number}")
                print(f"  Alasan: {reason}")
                print(f"  Label : {line}")

print(f"\nTotal label bermasalah: {total_invalid}")
