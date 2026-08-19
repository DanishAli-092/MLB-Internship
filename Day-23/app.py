import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

from utils.preprocessing import preprocess_pipeline
from utils.ocr_engine import extract_text


# streamlit uploader gives a PIL imag but OpenCV needs a BGR numpy array
def pil_to_cv2(pil_image):
    rgb_array = np.array(pil_image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


# converts the processed image (grayscale/binary) back to PIL for display
def cv2_to_pil(cv2_image):
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image)
    rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


def main():
    st.set_page_config(page_title="Document OCR App", layout="wide")

    st.title("Document OCR Web Application")
    st.write(
        "Upload a document, receipt, invoice or form image. "
        "The app will preprocess it and extract readable text using EasyOCR or PaddleOCR."
    )

    # sidebar settings so user can control OCR engine, preprocessing mode and confidence
    st.sidebar.header("Settings")

    # engine selector: lets user pick which OCR library actually reads the text
    ocr_engine = st.sidebar.selectbox(
        "OCR Engine",
        options=["EasyOCR", "PaddleOCR"],
        help=(
            "EasyOCR: solid all-rounder, good default choice\n"
            "PaddleOCR: often stronger on dense/small text, good for comparison"
        )
    )

    preprocessing_mode = st.sidebar.selectbox(
        "Preprocessing Mode",
        options=["standard", "receipt", "low_light"],
        help=(
            "standard: normal printed documents, signs, clean photos\n"
            "receipt: thermal-printed receipts with low contrast\n"
            "low_light: photos taken in poor/dim lighting"
        )
    )

    # small live description so user does not have to hover the help icon
    mode_descriptions = {
        "standard": "Best for normal printed documents, signs, and clean photos.",
        "receipt": "Best for thermal-printed receipts with faded, low contrast text.",
        "low_light": "Best for photos taken in dim or uneven lighting."
    }
    st.sidebar.caption(mode_descriptions[preprocessing_mode])
    min_confidence = st.sidebar.slider(
        "Minimum OCR Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Text detected below this confidence will be filtered out."
    )
    show_processed_image = st.sidebar.checkbox("Show preprocessed image", value=True)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"]
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        cv2_image = pil_to_cv2(pil_image)

        # two columns so original and processed image sit side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(pil_image, use_container_width=True)

        processed_image = preprocess_pipeline(cv2_image, mode=preprocessing_mode)

        if show_processed_image:
            with col2:
                st.subheader("Preprocessed Image")
                st.image(cv2_to_pil(processed_image), use_container_width=True)

        st.divider()

        with st.spinner(f"Extracting text using {ocr_engine}, please wait..."):
            try:
                full_text, detailed_results = extract_text(
                    processed_image, engine=ocr_engine, min_confidence=min_confidence
                )
            except Exception as e:
                st.error(f"OCR failed: {e}")
                return

        st.subheader(f"Extracted Text ({ocr_engine})")

        if full_text.strip() == "":
            st.warning("No text detected. Try a different preprocessing mode, engine, or lower the confidence threshold.")
        else:
            st.text_area("Result", value=full_text, height=250)

            # letting user download the extracted text as a txt file
            text_bytes = io.BytesIO(full_text.encode("utf-8"))
            st.download_button(
                label="Download Extracted Text (.txt)",
                data=text_bytes,
                file_name="extracted_text.txt",
                mime="text/plain"
            )

            # extra section to show bounding box + confidence for each detection
            with st.expander("View detection details (bounding boxes & confidence)"):
                for bbox, text, confidence in detailed_results:
                    st.write(f"Text: `{text}` | Confidence: {confidence:.2f}")

    else:
        st.info("Please upload an image to get started.")


if __name__ == "__main__":
    main()