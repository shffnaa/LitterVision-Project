from pathlib import Path

# Root directory project
BASE_DIR = Path(__file__).resolve().parent.parent

# Model configuration
MODEL_PATH = BASE_DIR / "models" / "best.pt"

IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.1982

# YOLO class names
CLASS_NAMES = [
    "Bottle",
    "Can",
    "Cup",
    "Packaging",
    "Straw",
    "Cigarette"
]

# Cleanliness Score configuration
CLEANLINESS_PENALTY = 5

# Output directory
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
