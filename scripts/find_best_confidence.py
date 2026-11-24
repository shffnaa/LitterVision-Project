from ultralytics import YOLO
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

MODEL = (
    ROOT / "runs" / "littervision" /
    "yolo26n_expanded_640_30epoch" / "weights" / "best.pt"
)

DATA = ROOT / "dataset" / "yolo_expanded" / "data.yaml"

if not MODEL.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL}")

if not DATA.exists():
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA}")

model = YOLO(str(MODEL))

print("=== EXPANDED CONFIDENCE CURVE ANALYSIS ===")
print("Model:", MODEL)
print("Dataset:", DATA)

metrics = model.val(
    data=str(DATA),
    split="val",
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    conf=0.001,
    plots=True,
    project=str(ROOT / "runs" / "littervision"),
    name="expanded_confidence_analysis",
    exist_ok=False,
)

# Ambil kurva confidence dan F1
px, f1_curve, _, _ = metrics.box.curves_results[1]

mean_f1 = np.asarray(f1_curve).mean(axis=0)

best_idx = int(mean_f1.argmax())
best_conf = float(px[best_idx])

precision_curve = np.asarray(metrics.box.p_curve).mean(axis=0)
recall_curve = np.asarray(metrics.box.r_curve).mean(axis=0)

print("\n=== EXPANDED OPTIMAL CONFIDENCE ===")
print(f"Best confidence : {best_conf:.4f}")
print(f"Mean F1-score   : {mean_f1[best_idx]:.4f}")
print(f"Precision       : {precision_curve[best_idx]:.4f}")
print(f"Recall          : {recall_curve[best_idx]:.4f}")

print("\n=== VALIDATION mAP ===")
print(f"mAP50    : {metrics.box.map50:.4f}")
print(f"mAP50-95 : {metrics.box.map:.4f}")

print("\nKurva tersimpan di:")
print(ROOT / "runs" / "littervision" / "expanded_confidence_analysis")
