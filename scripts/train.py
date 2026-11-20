from ultralytics import YOLO
from pathlib import Path
import time

print("=== LITTERVISION MAIN TRAINING ===")

model = YOLO("yolo26n.pt")

start = time.time()

model.train(
    data="dataset/yolo/data.yaml",
    epochs=15,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=2,
    project="/workspaces/LitterVision-Project/runs/littervision",
    name="yolo26n_640_rebuild_15epoch",
    exist_ok=True,
    save=True,
    save_period=1,
    plots=True,
    verbose=True
)

elapsed = time.time() - start

print("\n=== TRAINING COMPLETE ===")
print(f"Total time: {elapsed / 60:.2f} minutes")

weights = Path(
    "runs/littervision/yolo26n_640_rebuild_15epoch/weights"
)

print("Best checkpoint:", weights / "best.pt")
print("Last checkpoint:", weights / "last.pt")
