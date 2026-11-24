
from ultralytics import YOLO
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent

print("=== LITTERVISION EXPERIMENT 1 ===")
print("Experiment: Image Resolution 960")

model = YOLO(str(ROOT / "models" / "best_finetune_30epoch.pt"))

start = time.time()

model.train(
    data=str(ROOT / "dataset" / "yolo" / "data.yaml"),
    epochs=10,
    imgsz=960,
    batch=2,
    device="cpu",
    workers=0,
    patience=5,
    project=str(ROOT / "runs" / "littervision"),
    name="yolo26n_resolution_960_10epoch",
    exist_ok=False,
    save=True,
    save_period=5,
    plots=True,
    verbose=True,
    degrees=10,
    translate=0.1,
    scale=0.4,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=3,
)

elapsed = time.time() - start

print("\n=== EXPERIMENT COMPLETE ===")
print(f"Training time: {elapsed / 60:.2f} minutes")
print("Best model:", ROOT / "runs/littervision/yolo26n_resolution_960_10epoch/weights/best.pt")
