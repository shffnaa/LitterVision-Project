import cv2
import av

from streamlit_webrtc import VideoProcessorBase
from src.detector import LitterDetector


class LitterVideoProcessor(VideoProcessorBase):

    def __init__(self):
        # Memuat model YOLO saat kamera mulai dijalankan
        self.detector = LitterDetector()

    def recv(self, frame):
        # Mengambil frame kamera dalam format BGR
        img = frame.to_ndarray(format="bgr24")

        # Deteksi objek menggunakan model YOLO yang sudah ada
        result = self.detector.detect(img)

        # Detector mengembalikan gambar dalam format RGB
        annotated_image = cv2.cvtColor(
            result["image"],
            cv2.COLOR_RGB2BGR
        )

        # Mengirim frame hasil deteksi ke tampilan kamera
        return av.VideoFrame.from_ndarray(
            annotated_image,
            format="bgr24"
        )
