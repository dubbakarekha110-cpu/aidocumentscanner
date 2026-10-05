import streamlit as st
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Document Scanner",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Scanner")
st.write("Upload and process your document image.")

# Upload document
uploaded_file = st.file_uploader(
    "Upload Document",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Original Document")
    st.image(image, use_container_width=True)

    # Convert PIL image to OpenCV
    img = np.array(image)

    # Convert RGB to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    processed = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    st.subheader("🔧 Processed Document")
    st.image(processed, use_container_width=True)

    # Save processed image
    success, encoded_image = cv2.imencode(
        ".png",
        processed
    )

    if success:

        st.download_button(
            "⬇️ Download Processed Image",
            data=encoded_image.tobytes(),
            file_name="processed_document.png",
            mime="image/png"
        )

    st.success("✅ Document processing completed!")

st.markdown("---")

st.caption(
    "AI Document Scanner | "
    "Streamlit + OpenCV + Pillow + Requests + dotenv"
)