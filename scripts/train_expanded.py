from ultralytics import YOLO
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "dataset" / "yolo_expanded" / "data.yaml"
CHECKPOINT = ROOT / "models" / "best_finetune_30epoch.pt"

print("=== LITTERVISION EXPANDED TRAINING ===")
print("Dataset:", DATA_YAML)
print("Checkpoint:", CHECKPOINT)

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML tidak ditemukan: {DATA_YAML}")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint tidak ditemukan: {CHECKPOINT}")

model = YOLO(str(CHECKPOINT))

start = time.time()

model.train(
    data=str(DATA_YAML),
    epochs=30,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    patience=10,

    project=str(ROOT / "runs" / "littervision"),
    name="yolo26n_expanded_640_30epoch",

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

print("\n=== EXPANDED TRAINING COMPLETE ===")
print(f"Training time: {elapsed / 60:.2f} minutes")

print(
    "Best:",
    ROOT / "runs" / "littervision" /
    "yolo26n_expanded_640_30epoch" / "weights" / "best.pt"
)

print(
    "Last:",
    ROOT / "runs" / "littervision" /
    "yolo26n_expanded_640_30epoch" / "weights" / "last.pt"
)
