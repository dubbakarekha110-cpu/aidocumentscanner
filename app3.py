import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Document Scanner Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# HEADER
# =========================================================

st.title("📄 AI Document Scanner Pro")

st.write(
    "Scan, correct, enhance and download documents "
    "using Python and OpenCV."
)

st.markdown("---")


# =========================================================
# DOCUMENT POINT ORDERING
# =========================================================

def order_points(points):

    points = np.array(points, dtype="float32")

    ordered = np.zeros((4, 2), dtype="float32")

    total = points.sum(axis=1)

    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]

    difference = np.diff(points, axis=1)

    ordered[1] = points[np.argmin(difference)]
    ordered[3] = points[np.argmax(difference)]

    return ordered


# =========================================================
# PERSPECTIVE CORRECTION
# =========================================================

def perspective_transform(image, points):

    rect = order_points(points)

    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)

    max_width = max(
        int(width_a),
        int(width_b)
    )

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_height = max(
        int(height_a),
        int(height_b)
    )

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )


# =========================================================
# DOCUMENT DETECTION
# =========================================================

def detect_document(image):

    original = image.copy()

    h, w = image.shape[:2]

    target_width = 900

    scale = target_width / w

    resized = cv2.resize(
        image,
        (
            target_width,
            int(h * scale)
        )
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    document = None

    image_area = resized.shape[0] * resized.shape[1]

    for contour in contours[:40]:

        area = cv2.contourArea(contour)

        if area < image_area * 0.10:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approximation) == 4:

            document = (
                approximation.reshape(4, 2)
                / scale
            )

            break

    if document is not None:

        scanned = perspective_transform(
            original,
            document
        )

        return scanned, True

    return original, False


# =========================================================
# IMAGE ENHANCEMENT
# =========================================================

def enhance_image(
    image,
    brightness,
    contrast,
    sharpness,
    denoise
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Noise reduction
    if denoise:

        gray = cv2.fastNlMeansDenoising(
            gray,
            None,
            10,
            7,
            21
        )

    # Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Brightness
    enhanced = cv2.convertScaleAbs(
        enhanced,
        alpha=contrast,
        beta=brightness
    )

    # Sharpness
    if sharpness > 0:

        blurred = cv2.GaussianBlur(
            enhanced,
            (0, 0),
            3
        )

        enhanced = cv2.addWeighted(
            enhanced,
            1 + sharpness,
            blurred,
            -sharpness,
            0
        )

    return enhanced


# =========================================================
# BLACK AND WHITE
# =========================================================

def black_white_scan(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    result = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return result


# =========================================================
# ROTATION
# =========================================================

def rotate_image(image, angle):

    if angle == 90:

        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    if angle == 180:

        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )

    if angle == 270:

        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    return image


# =========================================================
# RESIZE
# =========================================================

def resize_image(image, scale):

    h, w = image.shape[:2]

    new_width = int(w * scale)
    new_height = int(h * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# =========================================================
# IMAGE TO BYTES
# =========================================================

def image_to_png(image):

    success, encoded = cv2.imencode(
        ".png",
        image
    )

    if success:

        return encoded.tobytes()

    return None


# =========================================================
# IMAGE TO PDF
# =========================================================

def image_to_pdf(image):

    if len(image.shape) == 2:

        pil_image = Image.fromarray(
            image
        ).convert("RGB")

    else:

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            rgb
        )

    pdf_buffer = io.BytesIO()

    pil_image.save(
        pdf_buffer,
        format="PDF",
        resolution=150.0
    )

    return pdf_buffer.getvalue()


# =========================================================
# INPUT METHOD
# =========================================================

st.sidebar.header("📥 Input")

input_method = st.sidebar.radio(
    "Choose input",
    [
        "Upload Image",
        "Camera"
    ]
)

image = None

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "📤 Upload Document",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

else:

    camera_file = st.camera_input(
        "📷 Take a picture of your document"
    )

    if camera_file:

        image = Image.open(
            camera_file
        ).convert("RGB")


# =========================================================
# PROCESSING
# =========================================================

if image is not None:

    image_array = np.array(image)

    original = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    # =====================================================
    # SIDEBAR SETTINGS
    # =====================================================

    st.sidebar.markdown("---")

    st.sidebar.header(
        "⚙️ Scanner Settings"
    )

    auto_detect = st.sidebar.checkbox(
        "🔍 Auto Detect Document",
        True
    )

    scan_mode = st.sidebar.selectbox(
        "📄 Scan Mode",
        [
            "Enhanced",
            "Original Color",
            "Grayscale",
            "Black & White"
        ]
    )

    rotation = st.sidebar.selectbox(
        "🔄 Rotation",
        [
            0,
            90,
            180,
            270
        ]
    )

    brightness = st.sidebar.slider(
        "☀️ Brightness",
        -100,
        100,
        0
    )

    contrast = st.sidebar.slider(
        "🎚️ Contrast",
        0.5,
        3.0,
        1.2
    )

    sharpness = st.sidebar.slider(
        "🔎 Sharpness",
        0.0,
        2.0,
        0.5
    )

    denoise = st.sidebar.checkbox(
        "🧹 Noise Reduction",
        True
    )

    resize_scale = st.sidebar.select_slider(
        "📏 Output Size",
        options=[
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            2.0
        ],
        value=1.0
    )

    # =====================================================
    # DOCUMENT INFORMATION
    # =====================================================

    height, width = original.shape[:2]

    st.sidebar.markdown("---")

    st.sidebar.header(
        "📊 Image Information"
    )

    st.sidebar.write(
        f"Width: `{width}px`"
    )

    st.sidebar.write(
        f"Height: `{height}px`"
    )

    st.sidebar.write(
        f"Channels: `{original.shape[2]}`"
    )

    st.sidebar.write(
        f"Pixels: `{width * height:,}`"
    )

    # =====================================================
    # ORIGINAL IMAGE
    # =====================================================

    st.subheader("📷 Original Document")

    st.image(
        image,
        use_container_width=True
    )

    # =====================================================
    # SCAN BUTTON
    # =====================================================

    scan_button = st.button(
        "🚀 Scan Document",
        type="primary",
        use_container_width=True
    )

    if scan_button:

        with st.spinner(
            "AI Document Scanner is processing..."
        ):

            # ---------------------------------------------
            # Document Detection
            # ---------------------------------------------

            if auto_detect:

                scanned, detected = detect_document(
                    original
                )

            else:

                scanned = original.copy()
                detected = False

            # ---------------------------------------------
            # Rotation
            # ---------------------------------------------

            scanned = rotate_image(
                scanned,
                rotation
            )

            # ---------------------------------------------
            # Scan Mode
            # ---------------------------------------------

            if scan_mode == "Enhanced":

                result = enhance_image(
                    scanned,
                    brightness,
                    contrast,
                    sharpness,
                    denoise
                )

            elif scan_mode == "Grayscale":

                result = cv2.cvtColor(
                    scanned,
                    cv2.COLOR_BGR2GRAY
                )

            elif scan_mode == "Black & White":

                result = black_white_scan(
                    scanned
                )

            else:

                result = scanned

            # ---------------------------------------------
            # Resize
            # ---------------------------------------------

            if resize_scale != 1.0:

                result = resize_image(
                    result,
                    resize_scale
                )

        # =================================================
        # STATUS
        # =================================================

        if detected:

            st.success(
                "✅ Document detected and perspective corrected."
            )

        else:

            st.warning(
                "⚠️ Document boundary was not detected. "
                "The image was processed without perspective correction."
            )

        # =================================================
        # BEFORE / AFTER
        # =================================================

        st.markdown("---")

        st.subheader(
            "🔄 Before & After"
        )

        before, after = st.columns(2)

        with before:

            st.write("**Original**")

            st.image(
                image,
                use_container_width=True
            )

        with after:

            st.write("**Scanned Result**")

            if len(result.shape) == 2:

                st.image(
                    result,
                    use_container_width=True
                )

            else:

                result_rgb = cv2.cvtColor(
                    result,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    result_rgb,
                    use_container_width=True
                )

        # =================================================
        # RESULT INFORMATION
        # =================================================

        st.markdown("---")

        st.subheader(
            "📊 Scan Information"
        )

        result_height, result_width = result.shape[:2]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Width",
                f"{result_width}px"
            )

        with col2:

            st.metric(
                "Height",
                f"{result_height}px"
            )

        with col3:

            st.metric(
                "Mode",
                scan_mode
            )

        with col4:

            st.metric(
                "Rotation",
                f"{rotation}°"
            )

        # =================================================
        # DOWNLOAD SECTION
        # =================================================

        st.markdown("---")

        st.subheader(
            "📥 Download Scanned Document"
        )

        png_data = image_to_png(
            result
        )

        pdf_data = image_to_pdf(
            result
        )

        download1, download2 = st.columns(2)

        with download1:

            st.download_button(
                "🖼️ Download PNG",
                data=png_data,
                file_name="scanned_document.png",
                mime="image/png",
                use_container_width=True
            )

        with download2:

            st.download_button(
                "📄 Download PDF",
                data=pdf_data,
                file_name="scanned_document.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        # =================================================
        # PROCESSING SUMMARY
        # =================================================

        st.markdown("---")

        st.subheader(
            "📋 Processing Summary"
        )

        summary = {
            "Document Detection":
                "Successful" if detected else "Not detected",

            "Scan Mode":
                scan_mode,

            "Perspective Correction":
                "Applied" if detected else "Not applied",

            "Noise Reduction":
                "Enabled" if denoise else "Disabled",

            "Brightness":
                brightness,

            "Contrast":
                contrast,

            "Sharpness":
                sharpness,

            "Rotation":
                f"{rotation}°",

            "Output Scale":
                f"{resize_scale}x"
        }

        for key, value in summary.items():

            st.write(
                f"**{key}:** {value}"
            )

else:

    # =====================================================
    # WELCOME PAGE
    # =====================================================

    st.info(
        "📤 Upload a document or use your camera to begin."
    )

    st.markdown(
        """
        ## ✨ Features

        | Feature | Description |
        |---|---|
        | 📷 Camera | Capture document directly |
        | 📤 Upload | Upload JPG/PNG/BMP/WebP |
        | 🔍 Detection | Detect document boundary |
        | 📐 Correction | Straighten tilted documents |
        | ✨ Enhancement | Improve document readability |
        | 🖤 B&W | Create black-and-white scan |
        | 🌫️ Grayscale | Create grayscale scan |
        | ☀️ Brightness | Adjust brightness |
        | 🎚️ Contrast | Adjust contrast |
        | 🔎 Sharpness | Improve text clarity |
        | 🧹 Noise Reduction | Remove image noise |
        | 🔄 Rotation | Rotate document |
        | 📏 Resize | Change output resolution |
        | 📊 Statistics | Show document information |
        | 🖼️ PNG | Download image |
        | 📄 PDF | Download PDF |
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Document Scanner Pro | Python + Streamlit + OpenCV + NumPy + Pillow"
)