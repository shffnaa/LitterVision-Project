
from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "runs" / "littervision" /
    "yolo26n_resolution_960_10epoch" /
    "weights" / "best.pt"
)

DATA_PATH = ROOT / "dataset" / "yolo" / "data.yaml"

print("=== LITTERVISION TEST EVALUATION ===")

model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data=str(DATA_PATH),
    split="test",
    imgsz=960,
    batch=2,
    device="cpu",
    workers=0,
    plots=True,
    project=str(ROOT / "runs" / "littervision"),
    name="resolution_960_test",
    exist_ok=True,
)

print("\n=== TEST METRICS ===")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")

print("\nTest evaluation selesai!")
