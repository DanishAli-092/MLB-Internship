import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import zipfile

from document_enhancer import fix_perspective, convert_to_gray, remove_noise, fix_brightness_contrast, sharpen

# Day 17 Mini Project - Document Image Enhancement Tool (Streamlit App)
# Two modes: Single Image (with full step-by-step breakdown) and Batch
# (process a whole folder of images at once and download as a zip).


st.set_page_config(
    page_title="Document Image Enhancer", 
    page_icon="📄", 
    layout="wide"
)

st.title("📄 Document Image Enhancer Pro")
st.write("Upload document photos and this app will straighten, clean, and sharpen them.")

# --------------=------------- Sidebar settings (shared by both---------------
st.sidebar.header("Pipeline Settings")

run_perspective = st.sidebar.checkbox("1. Perspective correction", value=True)
run_grayscale = st.sidebar.checkbox("2. Convert to grayscale", value=True)
run_denoise = st.sidebar.checkbox("3. Reduce noise", value=True)
run_brightness_contrast = st.sidebar.checkbox("4. Brightness / contrast", value=True)
run_sharpen = st.sidebar.checkbox("5. Sharpen", value=True)

brightness_value = st.sidebar.slider("Brightness", -100, 100, 20)
contrast_value = st.sidebar.slider("Contrast", 0.5, 3.0, 1.2)

st.sidebar.divider()
st.sidebar.write("Output size (leave as 0 to keep the natural size)")
output_width = st.sidebar.number_input("Output width", min_value=0, value=0, step=50)
output_height = st.sidebar.number_input("Output height", min_value=0, value=0, step=50)

st.sidebar.divider()
st.sidebar.caption(
    "Pipeline used: Canny edge detection + contour analysis for perspective "
    "correction, bilateral filtering for noise removal, and a sharpening kernel."
)


def process_one_image(img):
    """
    Runs the selected pipeline steps on a single image (in order), based on
    which checkboxes are turned on. Also keeps track of the image after each
    step, so we can show a step-by-step breakdown afterwards.
    Returns (final_image, stages_dict).
    """
    stages = {"0. Original": img}
    result = img

    if run_perspective:
        result = fix_perspective(result)
        stages["1. Perspective Corrected"] = result

    if run_grayscale:
        result = convert_to_gray(result)
        stages["2. Grayscale"] = result

    if run_denoise:
        result = remove_noise(result)
        stages["3. Noise Removed"] = result

    if run_brightness_contrast:
        result = fix_brightness_contrast(result, brightness_value, contrast_value)
        stages["4. Brightness/Contrast"] = result

    if run_sharpen:
        result = sharpen(result)
        stages["5. Sharpened"] = result

    if output_width > 0 and output_height > 0:
        result = cv2.resize(result, (output_width, output_height))

    return result, stages


def image_to_display_rgb(img):
    """Converts a BGR or grayscale OpenCV image into something st.image can show correctly."""
    if len(img.shape) == 2:
        return img, "GRAY"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "RGB"


# ---------------- Two tabs: single image and batch ----------------
tab1, tab2 = st.tabs(["Single Image", "Batch (Dataset of 10+)"])

with tab1:
    uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(pil_image)
        original_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # measure how long the pipeline actually takes - a real number, not a made-up claim
        start_time = time.time()
        final_img, stages = process_one_image(original_img)
        processing_time = time.time() - start_time

        perspective_applied = run_perspective and (
            stages.get("1. Perspective Corrected") is not None
            and stages["1. Perspective Corrected"].shape[:2] != original_img.shape[:2]
        )

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(pil_image, use_container_width=True)
            st.caption(f"Input size: {original_img.shape[1]} x {original_img.shape[0]} px")

        with col2:
            st.subheader("Enhanced")
            display_img, mode = image_to_display_rgb(final_img)
            st.image(display_img, use_container_width=True, channels=mode)
            st.caption(f"Output size: {final_img.shape[1]} x {final_img.shape[0]} px")

        # ---------------- Metrics row ----------------
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric("Input Size", f"{original_img.shape[1]} x {original_img.shape[0]} px")
        stat2.metric("Output Size", f"{final_img.shape[1]} x {final_img.shape[0]} px")
        stat3.metric("Processing Time", f"{processing_time:.2f} sec")
        stat4.metric("Perspective Correction", "Applied" if perspective_applied else "Not applied")

        success, buffer = cv2.imencode(".jpg", final_img)
        if success:
            st.download_button(
                label="Download Enhanced Image",
                data=io.BytesIO(buffer).getvalue(),
                file_name="enhanced_" + uploaded_file.name,
                mime="image/jpeg"
            )

        # ---------------- Pipeline stage viewer ----------------
        with st.expander("See each pipeline step"):
            stage_names = list(stages.keys())
            stage_columns = st.columns(len(stage_names))
            for col, name in zip(stage_columns, stage_names):
                stage_img = stages[name]
                with col:
                    st.caption(name)
                    display_stage, stage_mode = image_to_display_rgb(stage_img)
                    st.image(display_stage, use_container_width=True, channels=stage_mode)
    else:
        st.info("Upload an image to get started.")

with tab2:
    st.write("Upload 10 or more document images to process all of them at once and download as a single zip file.")

    uploaded_files = st.file_uploader(
        "Upload dataset images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} images uploaded.")

        if st.button("Process All Images"):
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                progress_bar = st.progress(0)

                for index, file in enumerate(uploaded_files):
                    pil_image = Image.open(file).convert("RGB")
                    image_array = np.array(pil_image)
                    original_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                    final_img, _ = process_one_image(original_img)

                    success, buffer = cv2.imencode(".jpg", final_img)
                    if success:
                        zip_file.writestr("enhanced_" + file.name, buffer.tobytes())

                    progress_bar.progress((index + 1) / len(uploaded_files))

            st.success("All images processed.")

            st.download_button(
                label="Download All Enhanced Images (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="enhanced_documents.zip",
                mime="application/zip"
            )
    else:
        st.info("Upload your dataset (own photos or images from Kaggle / Google Images / DocLayNet).")