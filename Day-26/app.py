import io
import os
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from thresholding import apply_binary_threshold, apply_adaptive_threshold, apply_otsu_threshold, resize_max_side
from segmentation import remove_background, watershed_segmentation
from utils import score_segmentation


# convert PIL to opencv format
def pil_to_cv2(pil_img):
    rgb_array = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


# convert image to png bytes for download
def cv2_to_downloadable_bytes(cv2_img):
    is_success, buffer = cv2.imencode(".png", cv2_img)
    if not is_success:
        raise ValueError("could not encode image for download")
    return io.BytesIO(buffer)


st.set_page_config(page_title="🔬 Image Segmentation Studio", layout="wide")


st.markdown(
    """
    <style>
    [data-testid="stImageCaption"],
    [data-testid="stImageCaption"] p,
    div[data-testid="caption"],
    div[data-testid="caption"] p {
        font-size: 22px !important;
        font-weight: 900 !important;
        color: #111 !important;
        text-align: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Image Segmentation Studio")
st.write("Upload an image, choose a segmentation method, and download the result.")

with st.sidebar:
    st.header("Controls")

    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

    st.divider()

    st.subheader("2. Segmentation Method")
    method = st.selectbox(
        "Select method",
        ["Binary Thresholding", "Adaptive Thresholding", "Otsu Thresholding",
         "Background Removal", "Watershed (separate touching objects)"],
    )

    threshold_value = 127
    if method == "Binary Thresholding":
        st.subheader("3. Threshold Value")
        threshold_value = st.slider("Threshold", 0, 255, 127)

    st.divider()
    st.caption("Upload an image and pick a method to see results on the main panel.")

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    cv2_img = pil_to_cv2(pil_img)
    cv2_img = resize_max_side(cv2_img)
    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    orig_col1, orig_col2, orig_col3 = st.columns([1, 2, 1])
    with orig_col2:
        st.image(pil_img, caption="Original image", use_container_width=True)

    height, width = cv2_img.shape[:2]
    channels = cv2_img.shape[2] if len(cv2_img.shape) == 3 else 1
    file_size_kb = uploaded_file.size / 1024

    with st.expander("Image Properties"):
        prop_col1, prop_col2, prop_col3, prop_col4 = st.columns(4)
        with prop_col1:
            st.metric("Width", f"{width} px")
        with prop_col2:
            st.metric("Height", f"{height} px")
        with prop_col3:
            st.metric("Channels", channels)
        with prop_col4:
            st.metric("File Size", f"{file_size_kb:.1f} KB")
        st.caption(f"Format: {pil_img.format} | Mode: {pil_img.mode} | Filename: {uploaded_file.name}")

    temp_path = "temp_uploaded_image.png"
    cv2.imwrite(temp_path, cv2_img)

    binary_out = apply_binary_threshold(gray_img)
    adaptive_out = apply_adaptive_threshold(gray_img)
    otsu_out, otsu_val = apply_otsu_threshold(gray_img)

    compare_results = {"binary": binary_out, "adaptive": adaptive_out, "otsu": otsu_out}
    best_label = max(compare_results, key=lambda k: score_segmentation(compare_results[k]))

    st.divider()

    tab_compare, tab_selected = st.tabs(["Compare All Methods", "Selected Method"])

    with tab_compare:
        st.subheader("Comparison of all segmentation methods")

        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        grid_cols = [row1_col1, row1_col2, row1_col3, row2_col1, row2_col2, row2_col3]

        with grid_cols[0]:
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption="original",
                      use_container_width=True, clamp=True)

        with grid_cols[1]:
            st.image(gray_img, caption="grayscale", use_container_width=True, clamp=True)

        for col, (label, img) in zip(grid_cols[2:5], compare_results.items()):
            with col:
                caption = label if label != best_label else f"{label} (best)"
                st.image(img, caption=caption, use_container_width=True, clamp=True)

        with grid_cols[5]:
            try:
                fg_out, _, _ = remove_background(temp_path)
                st.image(cv2.cvtColor(fg_out, cv2.COLOR_BGR2RGB), caption="background removal",
                          use_container_width=True, clamp=True)
            except ValueError as e:
                st.warning(f"background removal failed: {e}")

        best_download_bytes = cv2_to_downloadable_bytes(compare_results[best_label])
        st.download_button(
            label=f"Download best result ({best_label})",
            data=best_download_bytes,
            file_name=f"best_segmentation_{best_label}.png",
            mime="image/png",
            key="best_result_download",
        )

    with tab_selected:
        st.subheader("Selected method (with download)")

        result_img = None
        is_grayscale_output = True

        if method == "Binary Thresholding":
            result_img = apply_binary_threshold(gray_img, threshold_value)

        elif method == "Adaptive Thresholding":
            result_img = apply_adaptive_threshold(gray_img)

        elif method == "Otsu Thresholding":
            result_img, otsu_val = apply_otsu_threshold(gray_img)
            st.caption(f"Otsu automatically picked threshold value: {otsu_val:.1f}")

        elif method == "Background Removal":
            try:
                result_img, _, _ = remove_background(temp_path)
                is_grayscale_output = False
            except ValueError as e:
                st.error(f"Could not segment foreground: {e}")

        elif method == "Watershed (separate touching objects)":
            result_img, _ = watershed_segmentation(temp_path)
            is_grayscale_output = False

        if result_img is not None:
            display_img = result_img if is_grayscale_output else cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
            with res_col2:
                st.image(display_img, caption=f"Result: {method}", use_container_width=True, clamp=True)

            download_bytes = cv2_to_downloadable_bytes(result_img)
            st.download_button(
                label="Download segmented image",
                data=download_bytes,
                file_name=f"segmented_{method.lower().replace(' ', '_')}.png",
                mime="image/png",
            )

    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    st.info("Upload an image above to get started.")