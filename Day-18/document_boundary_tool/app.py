import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="DocVision X",layout="wide")
st.title("📄 DocVision X")
st.write("Upload a document image to detect its boundary using Canny edge detection and morphological operations.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated_edges = cv2.dilate(closed_edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundary_img = original.copy()
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)
        cv2.drawContours(boundary_img, [approx], -1, (0, 255, 0), 3)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("Edge Detection (Canny)")
        st.image(edges, use_container_width=True)

    with col2:
        st.subheader("Morphological Processing")
        st.image(dilated_edges, use_container_width=True)

        st.subheader("Detected Boundary")
        st.image(cv2.cvtColor(boundary_img, cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    st.info("Please upload an image to begin.")