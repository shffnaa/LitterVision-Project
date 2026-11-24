from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL = (
    ROOT / "runs" / "littervision" /
    "yolo26n_expanded_640_30epoch" / "weights" / "best.pt"
)

DATA = ROOT / "dataset" / "yolo_expanded" / "data.yaml"

CONFIDENCE = 0.1982

if not MODEL.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL}")

if not DATA.exists():
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA}")

model = YOLO(str(MODEL))

print("=== LITTERVISION FINAL TEST EVALUATION ===")
print("Model:", MODEL)
print("Dataset:", DATA)
print("Operational confidence:", CONFIDENCE)

# 1. Standard test evaluation
print("\n=== 1. STANDARD TEST mAP ===")

metrics = model.val(
    data=str(DATA),
    split="test",
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    conf=0.001,
    plots=True,
    project=str(ROOT / "runs" / "littervision"),
    name="expanded_final_test_map",
    exist_ok=False,
)

print("\n=== STANDARD TEST RESULTS ===")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")

# 2. Operational evaluation at validation-selected threshold
print("\n=== 2. OPERATIONAL TEST EVALUATION ===")

operational = model.val(
    data=str(DATA),
    split="test",
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    conf=CONFIDENCE,
    plots=False,
    project=str(ROOT / "runs" / "littervision"),
    name="expanded_operational_test",
    exist_ok=False,
)

precision = operational.box.mp
recall = operational.box.mr

if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0.0

print("\n=== OPERATIONAL TEST RESULTS ===")
print(f"Confidence : {CONFIDENCE:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1-score   : {f1:.4f}")

print("\n=== EVALUATION COMPLETE ===")
print("Results saved in:")
print(ROOT / "runs" / "littervision")