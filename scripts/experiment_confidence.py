
from ultralytics import YOLO
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "best_finetune_30epoch.pt"
DATA_PATH = ROOT / "dataset" / "yolo" / "data.yaml"

THRESHOLDS = [0.15, 0.25, 0.35, 0.45]

print("=== LITTERVISION CONFIDENCE EXPERIMENT ===")
print("Model:", MODEL_PATH)
print("Dataset: Validation")
print("Thresholds:", THRESHOLDS)

model = YOLO(str(MODEL_PATH))

results = []

for conf in THRESHOLDS:
    print(f"\n{'='*45}")
    print(f"Testing confidence: {conf}")
    print(f"{'='*45}")

    metrics = model.val(
        data=str(DATA_PATH),
        split="val",
        imgsz=640,
        batch=4,
        conf=conf,
        iou=0.7,
        device="cpu",
        workers=0,
        plots=False,
        project=str(ROOT / "runs" / "littervision"),
        name=f"confidence_{int(conf*100)}",
        exist_ok=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    results.append({
        "confidence": conf,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": map50,
        "mAP50-95": map5095,
    })

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"mAP50     : {map50:.4f}")
    print(f"mAP50-95  : {map5095:.4f}")

output = ROOT / "runs" / "littervision" / "confidence_comparison.csv"
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n=== FINAL COMPARISON ===")
for result in results:
    print(
        f"Conf {result['confidence']:.2f} | "
        f"P {result['precision']:.4f} | "
        f"R {result['recall']:.4f} | "
        f"F1 {result['f1']:.4f} | "
        f"mAP50 {result['mAP50']:.4f}"
    )

print("\nCSV saved:", output)
print("EXPERIMENT COMPLETE!")
