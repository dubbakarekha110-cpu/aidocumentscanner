import streamlit as st
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import requests
from datetime import datetime

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuVision Pro",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #ddd;
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    color: #777;
    font-size: 17px;
}

.card {
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #ddd;
    min-height: 120px;
}

.metric-title {
    font-size: 14px;
    color: #777;
}

.metric-value {
    font-size: 25px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #777;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>📄 DocuVision Pro</h1>

<p>
Professional Document Scanner & Image Enhancement System
</p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Scanner Settings")

st.sidebar.markdown("### 📥 Input")

input_type = st.sidebar.radio(
    "Select input",
    [
        "Upload Document",
        "Camera Scanner"
    ]
)

st.sidebar.markdown("### 🎨 Enhancement")

processing_mode = st.sidebar.selectbox(
    "Processing Mode",
    [
        "Professional",
        "Document Clean",
        "Grayscale",
        "Black & White",
        "High Contrast",
        "Edge Scanner"
    ]
)

brightness = st.sidebar.slider(
    "Brightness",
    -100,
    100,
    0
)

contrast = st.sidebar.slider(
    "Contrast",
    50,
    200,
    100
)

sharpness = st.sidebar.slider(
    "Sharpness",
    0,
    5,
    1
)

noise_reduction = st.sidebar.checkbox(
    "Noise Reduction",
    True
)

auto_crop = st.sidebar.checkbox(
    "Auto Crop",
    True
)

rotation = st.sidebar.selectbox(
    "Rotation",
    [0, 90, 180, 270]
)

# ============================================================
# INPUT
# ============================================================

input_file = None

if input_type == "Upload Document":

    input_file = st.file_uploader(
        "📤 Upload your document",
        type=["jpg", "jpeg", "png"]
    )

else:

    input_file = st.camera_input(
        "📷 Capture your document"
    )

# ============================================================
# EMPTY SCREEN
# ============================================================

if input_file is None:

    st.info(
        "Upload a document or capture one using the camera."
    )

    st.markdown("### 🚀 Professional Features")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="card">

        ### 📐 Smart Crop

        Automatically detects the
        main document area.

        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">

        ### ✨ Enhancement

        Improve brightness,
        contrast and sharpness.

        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">

        ### 📊 Analysis

        Check resolution,
        clarity and quality.

        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="card">

        ### 📥 Export

        Download professional
        PNG or JPG output.

        </div>
        """, unsafe_allow_html=True)

    st.stop()

# ============================================================
# READ IMAGE
# ============================================================

pil_image = Image.open(
    input_file
).convert("RGB")

original = np.array(
    pil_image
)

working = original.copy()

# ============================================================
# ROTATION
# ============================================================

if rotation == 90:

    working = cv2.rotate(
        working,
        cv2.ROTATE_90_CLOCKWISE
    )

elif rotation == 180:

    working = cv2.rotate(
        working,
        cv2.ROTATE_180
    )

elif rotation == 270:

    working = cv2.rotate(
        working,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )

# ============================================================
# AUTO DOCUMENT DETECTION
# ============================================================

document_detected = False

if auto_crop:

    gray_temp = cv2.cvtColor(
        working,
        cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray_temp,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    image_area = (
        working.shape[0] *
        working.shape[1]
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < image_area * 0.25:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w > 300 and h > 300:

            working = working[
                y:y+h,
                x:x+w
            ]

            document_detected = True

            break

# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

working = cv2.convertScaleAbs(
    working,
    alpha=contrast / 100,
    beta=brightness
)

# ============================================================
# NOISE REDUCTION
# ============================================================

if noise_reduction:

    working = cv2.fastNlMeansDenoisingColored(
        working,
        None,
        7,
        7,
        7,
        21
    )

# ============================================================
# SHARPENING
# ============================================================

for _ in range(sharpness):

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    working = cv2.filter2D(
        working,
        -1,
        kernel
    )

# ============================================================
# IMAGE MODES
# ============================================================

gray = cv2.cvtColor(
    working,
    cv2.COLOR_RGB2GRAY
)

black_white = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY +
    cv2.THRESH_OTSU
)[1]

clean_document = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    15,
    8
)

high_contrast = cv2.equalizeHist(
    gray
)

edge_result = cv2.Canny(
    gray,
    50,
    150
)

# ============================================================
# SELECT RESULT
# ============================================================

if processing_mode == "Professional":

    result = working

elif processing_mode == "Document Clean":

    result = clean_document

elif processing_mode == "Grayscale":

    result = gray

elif processing_mode == "Black & White":

    result = black_white

elif processing_mode == "High Contrast":

    result = high_contrast

else:

    result = edge_result

# ============================================================
# DOCUMENT INFORMATION
# ============================================================

height, width = gray.shape

aspect_ratio = width / height

if aspect_ratio > 1.3:

    orientation = "Landscape"

elif aspect_ratio < 0.8:

    orientation = "Portrait"

else:

    orientation = "Standard"

# ============================================================
# IMAGE QUALITY ANALYSIS
# ============================================================

brightness_value = float(
    np.mean(gray)
)

sharpness_value = float(
    cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()
)

# ============================================================
# QUALITY SCORE
# ============================================================

quality_score = 0

# Sharpness

if sharpness_value >= 800:
    quality_score += 50

elif sharpness_value >= 400:
    quality_score += 42

elif sharpness_value >= 150:
    quality_score += 32

elif sharpness_value >= 70:
    quality_score += 22

else:
    quality_score += 10

# Brightness

if 100 <= brightness_value <= 190:
    quality_score += 30

elif 80 <= brightness_value <= 210:
    quality_score += 20

else:
    quality_score += 10

# Resolution

if width >= 1600 and height >= 1200:
    quality_score += 20

elif width >= 1000 and height >= 800:
    quality_score += 15

else:
    quality_score += 8

# ============================================================
# QUALITY LABEL
# ============================================================

if quality_score >= 90:

    quality_label = "Excellent"

elif quality_score >= 75:

    quality_label = "Very Good"

elif quality_score >= 60:

    quality_label = "Good"

elif quality_score >= 40:

    quality_label = "Average"

else:

    quality_label = "Needs Improvement"

# ============================================================
# FILE INFORMATION
# ============================================================

file_size = len(
    input_file.getvalue()
) / 1024

file_name = input_file.name

scan_time = datetime.now().strftime(
    "%d %B %Y, %I:%M:%S %p"
)

# ============================================================
# MAIN DASHBOARD
# ============================================================

st.markdown("---")

st.subheader("📊 Scan Dashboard")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.metric(
        "⭐ Quality",
        f"{quality_score}/100"
    )

with c2:

    st.metric(
        "📐 Resolution",
        f"{width}×{height}"
    )

with c3:

    st.metric(
        "💡 Brightness",
        f"{brightness_value:.0f}"
    )

with c4:

    st.metric(
        "🔍 Sharpness",
        f"{sharpness_value:.0f}"
    )

with c5:

    st.metric(
        "📄 Orientation",
        orientation
    )

# ============================================================
# STATUS
# ============================================================

if quality_score >= 75:

    st.success(
        f"✅ Scan quality: {quality_label}"
    )

elif quality_score >= 50:

    st.warning(
        f"⚠️ Scan quality: {quality_label}"
    )

else:

    st.error(
        f"❌ Scan quality: {quality_label}"
    )

# ============================================================
# DOCUMENT STATUS
# ============================================================

if document_detected:

    st.success(
        "📐 Document area automatically detected."
    )

else:

    st.info(
        "ℹ️ Original image boundaries were retained."
    )

# ============================================================
# PREVIEW
# ============================================================

st.markdown("---")

st.subheader("🖼️ Professional Preview")

tab1, tab2, tab3 = st.tabs([
    "📷 Original",
    "✨ Final Scan",
    "🔎 Technical View"
])

with tab1:

    st.image(
        original,
        caption="Original Document",
        use_container_width=True
    )

with tab2:

    st.image(
        result,
        caption="Professional Processed Output",
        use_container_width=True
    )

with tab3:

    st.image(
        edge_result,
        caption="Document Edge Analysis",
        use_container_width=True
    )

# ============================================================
# DOCUMENT REPORT
# ============================================================

st.markdown("---")

st.subheader("📋 Document Report")

r1, r2 = st.columns(2)

with r1:

    st.markdown(f"""
    **📁 File Name:** {file_name}

    **📦 File Size:** {file_size:.2f} KB

    **📐 Resolution:** {width} × {height} pixels

    **📄 Orientation:** {orientation}

    **🎨 Processing Mode:** {processing_mode}
    """)

with r2:

    st.markdown(f"""
    **⭐ Quality Score:** {quality_score}/100

    **💡 Brightness:** {brightness_value:.2f}

    **🔍 Sharpness:** {sharpness_value:.2f}

    **✂️ Auto Crop:** {"Enabled" if auto_crop else "Disabled"}

    **🕒 Scan Time:** {scan_time}
    """)

# ============================================================
# SMART RECOMMENDATIONS
# ============================================================

st.markdown("---")

st.subheader("💡 Smart Recommendations")

if brightness_value < 80:

    st.warning(
        "The document is dark. Use better lighting or increase brightness."
    )

elif brightness_value > 210:

    st.warning(
        "The document is overexposed. Reduce brightness."
    )

else:

    st.success(
        "Brightness level is suitable for document scanning."
    )

if sharpness_value < 150:

    st.warning(
        "Image may be blurry. Keep the camera steady and capture again."
    )

else:

    st.success(
        "Document sharpness is suitable."
    )

if width < 1000:

    st.warning(
        "Low resolution detected. A higher-resolution image is recommended."
    )

else:

    st.success(
        "Resolution is suitable for digital document use."
    )

# ============================================================
# EXPORT
# ============================================================

st.markdown("---")

st.subheader("📥 Export Center")

png_ok, png_data = cv2.imencode(
    ".png",
    result
)

jpg_ok, jpg_data = cv2.imencode(
    ".jpg",
    result,
    [cv2.IMWRITE_JPEG_QUALITY, 95]
)

e1, e2 = st.columns(2)

with e1:

    if png_ok:

        st.download_button(
            "🖼️ Download PNG",
            data=png_data.tobytes(),
            file_name="DocuVision_Scan.png",
            mime="image/png"
        )

with e2:

    if jpg_ok:

        st.download_button(
            "📸 Download JPG",
            data=jpg_data.tobytes(),
            file_name="DocuVision_Scan.jpg",
            mime="image/jpeg"
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    <b>DocuVision Pro</b><br>
    Professional Document Scanner<br><br>

    Streamlit • OpenCV • Pillow • Python-dotenv • Requests

    </div>
    """,
    unsafe_allow_html=True
)