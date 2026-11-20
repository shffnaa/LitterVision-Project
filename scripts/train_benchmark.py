from ultralytics import YOLO
from pathlib import Path
import time

print("=== LITTERVISION CPU TRAINING BENCHMARK ===")

model = YOLO("yolo26n.pt")

start = time.time()

model.train(
    data="dataset/yolo/data.yaml",
    epochs=1,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=2,
    project="runs/littervision",
    name="cpu_benchmark_640",
    exist_ok=True,
    plots=False,
    save=True,
    verbose=True
)

elapsed = time.time() - start

print("\n=== BENCHMARK COMPLETE ===")
print(f"Training time: {elapsed / 60:.2f} minutes")
print("Checkpoint:", Path(
    "runs/littervision/cpu_benchmark_640/weights/best.pt"
).resolve())
