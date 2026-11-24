import sys
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_geolocation import streamlit_geolocation
from streamlit_webrtc import webrtc_streamer, RTCConfiguration


# =========================
# PROJECT PATH
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.realtime import LitterVideoProcessor
from src.detector import LitterDetector
from src.scoring import (
    calculate_cleanliness_score,
    get_cleanliness_category
)


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="LitterVision | Campus Monitoring",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# SESSION STATE
# =========================

if "gps_location" not in st.session_state:
    st.session_state["gps_location"] = None

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None


# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>
.stApp {
    background-color: #F5F8F5;
    color: #173A2B;
}

[data-testid="stAppViewContainer"] {
    background-color: #F5F8F5;
}

[data-testid="stHeader"] {
    background-color: transparent;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EAF3EB, #F5F8F5);
    border-right: 1px solid #DCE9DE;
}

[data-testid="stSidebar"] * {
    color: #173A2B;
}

.stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp p, .stApp label {
    color: #173A2B;
}

.hero {
    background: linear-gradient(110deg, #10452F, #26734D);
    border-radius: 20px;
    padding: 32px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 38px;
    margin-bottom: 5px;
}

.hero p {
    color: #E2F3E7 !important;
    font-size: 16px;
}

.hero-tag {
    color: #D8F3A5;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}

.section-heading {
    font-size: 23px;
    font-weight: 750;
    color: #174D36;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #668071;
    font-size: 14px;
    margin-bottom: 18px;
}

.feature-card {
    background: linear-gradient(135deg, #E1F2E5, #F2F9F3);
    border: 1px solid #D3E8D8;
    border-radius: 16px;
    padding: 23px;
    min-height: 220px;
}

.feature-card h3 {
    color: #174D36 !important;
    font-size: 19px;
}

.feature-card p {
    color: #315B42 !important;
    font-size: 14px;
}

.info-card {
    background: white;
    border: 1px solid #E0EAE1;
    border-radius: 16px;
    padding: 20px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E0EAE1;
    border-radius: 14px;
    padding: 17px;
}

div[data-testid="stMetric"] label {
    color: #607A69 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #174D36 !important;
}

div.stButton > button[kind="primary"] {
    background: #176B45;
    color: white !important;
    border: 1px solid #176B45;
    border-radius: 10px;
    font-weight: 650;
    min-height: 45px;
}

div.stButton > button[kind="primary"]:hover {
    background: #10452F;
    border-color: #10452F;
}

div.stButton > button {
    border-radius: 10px;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #A9C7B0;
    border-radius: 14px;
    padding: 12px;
}

.stAlert {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #718B79;
    font-size: 13px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_detector():
    return LitterDetector()


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:15px 5px 20px;">
        <div style="font-size:48px;">🌱</div>
        <h2 style="color:#174D36; margin:0;">
            LitterVision
        </h2>
        <p style="color:#668071; font-size:13px;">
            Clean Campus, Green Future
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚙️ Pengaturan")

    st.divider()

    # GPS
    st.markdown("### 📍 Lokasi Otomatis")

    location = streamlit_geolocation()

    if location and location.get("latitude") is not None:

        st.session_state["gps_location"] = {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "accuracy": location.get("accuracy")
        }

        st.success("Lokasi berhasil terdeteksi!")

        st.write(f"Latitude: {location['latitude']}")
        st.write(f"Longitude: {location['longitude']}")

        if location.get("accuracy") is not None:
            st.caption(
                f"Akurasi GPS: ±{location['accuracy']:.1f} meter"
            )

    else:
        st.info(
            "Klik tombol lokasi dan izinkan akses lokasi pada browser."
        )

    st.divider()

    st.markdown("""
    <div style="
        background:#DDEFE1;
        padding:15px;
        border-radius:12px;
        margin-top:10px;
        text-align:center;
    ">
        <div style="font-size:30px;">♻️</div>
        <b style="color:#174D36;">Keep Our Campus Clean</b>
        <p style="font-size:12px;color:#52745D;">
            Small actions, greener environment.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# HEADER / HERO
# =========================

st.markdown("""
<div class="hero">
    <div class="hero-tag">SMART ENVIRONMENTAL MONITORING</div>
    <h1>🌱 LitterVision</h1>
    <p style="font-weight:600;">
        Campus Environmental Cleanliness Monitoring System
    </p>
    <p>
        Sistem deteksi dan monitoring sampah lingkungan
        kampus berbasis YOLO26n.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# INPUT SECTION
# =========================

st.markdown(
    '<div class="section-heading">📸 Input Gambar</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Unggah gambar, ambil foto, atau lakukan deteksi langsung '
    'menggunakan kamera perangkat.'
    '</div>',
    unsafe_allow_html=True
)

input_col, feature_col = st.columns([2.1, 1])

with input_col:

    input_method = st.radio(
        "Pilih metode input:",
        [
            "Upload Gambar",
            "Ambil Foto dari Kamera",
            "Real-Time Detection"
        ],
        horizontal=True
    )

    input_file = None

    # =========================
    # UPLOAD IMAGE
    # =========================

    if input_method == "Upload Gambar":

        input_file = st.file_uploader(
            "Pilih gambar lingkungan kampus",
            type=["jpg", "jpeg", "png"],
            help="Format gambar: JPG, JPEG, atau PNG."
        )

    # =========================
    # CAMERA PHOTO
    # =========================

    elif input_method == "Ambil Foto dari Kamera":

        st.info(
            "Izinkan akses kamera pada browser jika muncul permintaan izin."
        )

        input_file = st.camera_input(
            "Ambil foto lingkungan kampus"
        )

    # =========================
    # REAL-TIME DETECTION
    # =========================

    else:

        st.info(
            "Arahkan kamera ke lingkungan kampus. "
            "YOLO26n akan mendeteksi sampah secara langsung."
        )

        rtc_configuration = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        })

        webrtc_streamer(
            key="littervision-realtime",
            video_processor_factory=LitterVideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": {
                    "facingMode": {"ideal": "environment"}
                },
                "audio": False
            },
            async_processing=True
        )


with feature_col:

    st.markdown("""
    <div class="feature-card">
        <h3>🌿 Fitur LitterVision</h3>
        <p>♻️ Deteksi berbagai jenis sampah</p>
        <p>📊 Analisis Cleanliness Score</p>
        <p>📍 Monitoring lokasi kampus</p>
        <p>📷 Input melalui kamera</p>
        <p>🎥 Deteksi objek secara realtime</p>
        <p>🤖 Analisis menggunakan YOLO26n</p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# IMAGE PREVIEW
# =========================

if input_file is not None:

    image = Image.open(input_file).convert("RGB")

    st.divider()

    st.markdown(
        '<div class="section-heading">🖼️ Pratinjau Gambar</div>',
        unsafe_allow_html=True
    )

    preview_col1, preview_col2 = st.columns([1.5, 1])

    with preview_col1:

        st.image(
            image,
            caption="Gambar pengamatan lingkungan kampus",
            use_container_width=True
        )

    with preview_col2:

        gps_preview = st.session_state.get("gps_location")

        if gps_preview:
            location_preview = "GPS berhasil terdeteksi"
        else:
            location_preview = "Menunggu akses GPS"

        st.markdown(f"""
        <div class="info-card">
            <h3 style="color:#174D36;">📋 Detail Pengamatan</h3>
            <p style="color:#52745D;">Lokasi pengamatan</p>
            <h4>📍 {location_preview}</h4>
            <p style="color:#174D36;font-weight:700;">
                Pengamatan melalui input gambar
            </p>
            <hr>
            <p style="color:#52745D;">Model: YOLO26n</p>
            <p style="color:#52745D;">Kategori: 6 jenis sampah</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button(
        "🔍 Mulai Deteksi",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "YOLO26n sedang menganalisis gambar..."
            ):

                detector = load_detector()

                # Deteksi gambar
                result = detector.detect(image)

                detections = result["detections"]

                # Perhitungan Cleanliness Score
                score = calculate_cleanliness_score(detections)

                category = get_cleanliness_category(score)

                # Ambil data GPS terakhir
                current_gps = st.session_state.get("gps_location")

                # Simpan hasil analisis
                st.session_state["analysis_result"] = {
                    "result": result,
                    "score": score,
                    "category": category,
                    "latitude": (
                        current_gps["latitude"]
                        if current_gps else None
                    ),
                    "longitude": (
                        current_gps["longitude"]
                        if current_gps else None
                    ),
                    "accuracy": (
                        current_gps.get("accuracy")
                        if current_gps else None
                    ),
                    "input_method": input_method
                }

            st.success("Analisis berhasil diselesaikan!")

        except Exception as e:

            st.error(
                f"Terjadi kesalahan saat analisis: {e}"
            )


# =========================
# DISPLAY ANALYSIS RESULT
# =========================

if (
    st.session_state["analysis_result"] is not None
    and input_method != "Real-Time Detection"
):

    analysis = st.session_state["analysis_result"]

    result = analysis["result"]
    score = analysis["score"]
    category = analysis["category"]

    st.divider()

    st.markdown(
        '<div class="section-heading">📊 Hasil Analisis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Visualisasi hasil deteksi dan penilaian kebersihan lingkungan.'
        '</div>',
        unsafe_allow_html=True
    )

    # RESULT IMAGE
    result_col1, result_col2 = st.columns([1.5, 1])

    with result_col1:

        st.markdown("#### 🎯 Hasil Deteksi YOLO26n")

        st.image(
            result["image"],
            caption="Gambar dengan bounding box hasil deteksi",
            use_container_width=True
        )

    with result_col2:

        st.markdown("#### 🌿 Cleanliness Score")

        st.metric(
            label="Skor Kebersihan",
            value=f"{score}/100"
        )

        st.progress(max(0, min(score, 100)) / 100)

        st.success(f"Kategori: {category}")

        st.divider()

        st.markdown("#### 📍 Informasi Pengamatan")

        st.write(f"**Metode input:** {analysis['input_method']}")
        st.write("**Model:** YOLO26n")

        if analysis.get("latitude") is not None:

            latitude = analysis["latitude"]
            longitude = analysis["longitude"]

            st.write("**Koordinat GPS:**")

            st.write(
                f"{latitude:.6f}, {longitude:.6f}"
            )

            if analysis.get("accuracy") is not None:

                st.caption(
                    f"Akurasi GPS: ±{analysis['accuracy']:.1f} meter"
                )

            maps_url = (
                f"https://www.google.com/maps?q={latitude},{longitude}"
            )

            st.markdown(
                f"[Lihat lokasi di Google Maps]({maps_url})"
            )

        else:

            st.warning(
                "Koordinat GPS belum tersedia saat pengamatan. "
                "Aktifkan izin lokasi di browser, lalu ambil lokasi "
                "sebelum menekan tombol Mulai Deteksi."
            )

    # SUMMARY
    st.divider()

    st.markdown(
        '<div class="section-heading">📦 Ringkasan Deteksi</div>',
        unsafe_allow_html=True
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Total Objek Terdeteksi",
            result["total_objects"]
        )

    with metric2:

        st.metric(
            "Jenis Sampah",
            len(result["counts"])
        )

    with metric3:

        st.metric(
            "Cleanliness Score",
            f"{score}/100"
        )

    # CLASS COUNTS
    st.divider()

    st.markdown("### 📊 Jumlah Sampah per Kategori")

    if result["counts"]:

        counts_df = pd.DataFrame(
            list(result["counts"].items()),
            columns=["Kategori", "Jumlah"]
        )

        st.bar_chart(
            counts_df.set_index("Kategori"),
            horizontal=True
        )

        st.dataframe(
            counts_df,
            hide_index=True,
            use_container_width=True
        )

    else:

        st.success("Tidak ada sampah yang terdeteksi.")

    # DETECTION DETAILS
    st.divider()

    with st.expander("🔎 Lihat Detail Prediksi"):

        detections = result["detections"]

        if detections is not None and len(detections) > 0:

            st.dataframe(
                detections,
                use_container_width=True
            )

        else:

            st.write("Tidak ada objek yang terdeteksi.")

    st.caption(
        "Catatan: Cleanliness Score merupakan indikator estimasi "
        "berdasarkan jumlah objek dan luas relatif bounding box "
        "yang berhasil dideteksi, bukan pengukuran kebersihan absolut."
    )


# =========================
# EMPTY STATE
# =========================

elif input_method != "Real-Time Detection":

    st.divider()

    st.markdown("""
    <div style="
        background:linear-gradient(120deg,#E4F2E7,#F4F9F4);
        border:1px solid #D9EBDD;
        border-radius:18px;
        padding:25px;
        text-align:center;
    ">
        <div style="font-size:48px;">🌎</div>
        <h3 style="color:#174D36;">
            Together for a Cleaner Campus!
        </h3>
        <p style="color:#668071;">
            Mulai pengamatan untuk membantu menciptakan
            lingkungan kampus yang lebih bersih dan hijau.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# FOOTER
# =========================

st.markdown("""
<div class="footer">
    🌱 <b>LitterVision</b><br>
    YOLO26n-Based Campus Cleanliness Monitoring System<br>
    <small>Keep Campus Clean, Greener Tomorrow.</small>
</div>
""", unsafe_allow_html=True)
