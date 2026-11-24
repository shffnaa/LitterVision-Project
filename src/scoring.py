# Parameter normalisasi
MAX_OBJECTS = 10
MAX_AREA_RATIO = 0.30

# Bobot awal setiap kelas
# Nilai sama agar tidak mengasumsikan tingkat bahaya sampah.
CLASS_WEIGHTS = {
    "Bottle": 1.0,
    "Can": 1.0,
    "Cup": 1.0,
    "Packaging": 1.0,
    "Straw": 1.0,
    "Cigarette": 1.0
}


def calculate_cleanliness_score(detections):
    """
    Menghitung Cleanliness Score berdasarkan:
    1. Jumlah objek berbobot kelas.
    2. Proporsi luas bounding box terhadap gambar.

    Skala skor: 0-100.
    """

    if not detections:
        return 100

    # Komponen jumlah objek berbobot
    weighted_count = sum(
        CLASS_WEIGHTS.get(item["class_name"], 1.0)
        for item in detections
    )

    count_penalty = min(
        weighted_count / MAX_OBJECTS, 1.0
    ) * 50

    # Komponen luas area objek
    total_area_ratio = sum(
        item.get("area_ratio", 0)
        for item in detections
    )

    area_penalty = min(
        total_area_ratio / MAX_AREA_RATIO, 1.0
    ) * 50

    # Total penalti
    total_penalty = count_penalty + area_penalty

    score = 100 - total_penalty

    return round(max(0, min(100, score)), 2)


def get_cleanliness_category(score):
    """
    Mengubah skor menjadi kategori kebersihan.
    """

    if score >= 80:
        return "Sangat Bersih"

    elif score >= 60:
        return "Bersih"

    elif score >= 40:
        return "Cukup Kotor"

    elif score >= 20:
        return "Kotor"

    else:
        return "Sangat Kotor"
