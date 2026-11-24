
from ultralytics import YOLO
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent

print("=== LITTERVISION IMPROVED TRAINING ===")

model = YOLO(str(ROOT / "models" / "best.pt"))

start = time.time()

model.train(
    data=str(ROOT / "dataset" / "yolo" / "data.yaml"),
    epochs=30,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    patience=10,

    project=str(ROOT / "runs" / "littervision"),
    name="yolo26n_finetune_640_30epoch",

    exist_ok=False,
    save=True,
    save_period=5,
    plots=True,
    verbose=True,

    # Augmentasi moderat
    degrees=10,
    translate=0.1,
    scale=0.4,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=10,
)

elapsed = time.time() - start

print("\n=== TRAINING COMPLETE ===")
print(f"Training time: {elapsed / 60:.2f} minutes")
print("Best:", ROOT / "runs/littervision/yolo26n_finetune_640_30epoch/weights/best.pt")
print("Last:", ROOT / "runs/littervision/yolo26n_finetune_640_30epoch/weights/last.pt")
