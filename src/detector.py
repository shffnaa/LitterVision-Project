from ultralytics import YOLO
from collections import Counter

from src.config import (
    MODEL_PATH,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    CLASS_NAMES
)


class LitterDetector:

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {MODEL_PATH}"
            )

        self.model = YOLO(str(MODEL_PATH))

    def detect(self, image):
        """
        Melakukan deteksi sampah pada gambar,
        menghitung luas bounding box, dan
        proporsi area objek terhadap gambar.
        """

        results = self.model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

        # Gambar hasil deteksi
        annotated_image = result.plot()

        # Konversi BGR ke RGB
        annotated_image = annotated_image[:, :, ::-1]

        # Ukuran gambar asli
        image_height, image_width = result.orig_shape
        image_area = image_width * image_height

        # Ambil hasil prediksi
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coordinates = box.xyxy[0].tolist()

            x1, y1, x2, y2 = coordinates

            # Menghitung luas bounding box
            bbox_width = max(0, x2 - x1)
            bbox_height = max(0, y2 - y1)
            bbox_area = bbox_width * bbox_height

            # Proporsi luas objek terhadap seluruh gambar
            area_ratio = (
                bbox_area / image_area
                if image_area > 0 else 0
            )

            class_name = (
                CLASS_NAMES[class_id]
                if class_id < len(CLASS_NAMES)
                else f"Class {class_id}"
            )

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [
                    round(value, 2)
                    for value in coordinates
                ],
                "bbox_area": round(bbox_area, 2),
                "area_ratio": round(area_ratio, 6)
            })

        # Hitung jumlah objek per kelas
        counts = Counter(
            item["class_name"] for item in detections
        )

        return {
            "image": annotated_image,
            "detections": detections,
            "counts": dict(counts),
            "total_objects": len(detections)
        }
