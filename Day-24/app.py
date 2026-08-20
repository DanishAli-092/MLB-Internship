import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.preprocessing import preprocess_pipeline
from utils.ocr_engine import extract_text, warm_up_engine


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


# does the actual preprocessing + OCR work for one uploaded file.

def process_one_document(filename, pil_image, mode, engine, min_confidence):
    cv2_image = pil_to_cv2(pil_image)
    processed_image = preprocess_pipeline(cv2_image, mode=mode)

    try:
        full_text, detailed_results, elapsed_time = extract_text(
            processed_image, engine=engine, min_confidence=min_confidence
        )
        return {
            "filename": filename,
            "pil_image": pil_image,
            "processed_image": processed_image,
            "full_text": full_text,
            "detailed_results": detailed_results,
            "elapsed_time": elapsed_time,
            "error": None,
        }
    except Exception as e:
        return {
            "filename": filename,
            "pil_image": pil_image,
            "processed_image": processed_image,
            "full_text": "",
            "detailed_results": [],
            "elapsed_time": 0.0,
            "error": str(e),
        }


def render_result(result, ocr_engine, show_processed_image):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(result["pil_image"], use_container_width=True)

    if show_processed_image:
        with col2:
            st.subheader("Preprocessed Image")
            st.image(cv2_to_pil(result["processed_image"]), use_container_width=True)

    st.divider()

    if result["error"]:
        st.error(f"OCR failed: {result['error']}")
        return

    st.subheader(f"Extracted Text ({ocr_engine})")

    # Day 24 requirement: show the data extraction time
    st.metric("Extraction Time", f"{result['elapsed_time']:.2f}s")

    full_text = result["full_text"]
    if full_text.strip() == "":
        st.warning("No text detected. Try a different preprocessing mode, engine, or lower the confidence threshold.")
    else:
        
        st.text_area("Result", value=full_text, height=250, key=f"text_{result['filename']}_{ocr_engine}")

        text_bytes = io.BytesIO(full_text.encode("utf-8"))
        st.download_button(
            label="Download Extracted Text (.txt)",
            data=text_bytes,
            file_name=f"{result['filename']}_{ocr_engine}_extracted.txt",
            mime="text/plain",
        
            key=f"download_{result['filename']}_{ocr_engine}"
        )

        with st.expander("View detection details (bounding boxes & confidence)"):
            for bbox, text, confidence in result["detailed_results"]:
                st.write(f"Text: `{text}` | Confidence: {confidence:.2f}")


def main():
    st.set_page_config(page_title="Document OCR App", layout="wide")

    st.title("Document OCR Web Application")
    st.write(
        "Upload one or more documents, receipts, invoices or form images. "
        "The app will preprocess them and extract readable text using "
        "EasyOCR, PaddleOCR, RapidOCR, Tesseract, or DocTR."
    )

    # sidebar settings so user can control OCR engine preprocessing mode and confidence
    st.sidebar.header("Settings")

    # engine selector: lets user pick which OCR library actually reads the text
    ocr_engine = st.sidebar.selectbox(
        "OCR Engine",
        options=["EasyOCR", "PaddleOCR", "RapidOCR", "Tesseract", "DocTR"],
        help=(
            "EasyOCR: solid all-rounder, good default choice\n"
            "PaddleOCR: often stronger on dense/small text, good for comparison\n"
            "RapidOCR: lightweight ONNX-based engine, fast on CPU\n"
            "Tesseract: classic engine, good on clean printed documents\n"
            "DocTR: deep-learning engine, strong on structured documents"
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

    st.sidebar.markdown("---")
    max_workers = st.sidebar.slider(
        "Parallel workers (multi-document mode)",
        min_value=1,
        max_value=8,
        value=4,
        help="How many documents to process at the same time when multiple files are uploaded."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload one or more images to get started.")
        return

    # warm up the selected engine's model ONCE on the main thread before any
    # worker threads start - avoids multiple threads racing to load the same
    # @st.cache_resource model simultaneously when several files are uploaded
    with st.spinner(f"Loading {ocr_engine}..."):
        warm_up_engine(ocr_engine)

    
    
    if len(uploaded_files) == 1:
        uploaded_file = uploaded_files[0]
        pil_image = Image.open(uploaded_file)

        with st.spinner(f"Extracting text using {ocr_engine}, please wait..."):
            result = process_one_document(
                uploaded_file.name, pil_image, preprocessing_mode, ocr_engine, min_confidence
            )

        render_result(result, ocr_engine, show_processed_image)
        return

    st.subheader(f"Processing {len(uploaded_files)} documents with {ocr_engine}")

    total_start_placeholder = st.empty()
    progress_bar = st.progress(0)
    results_by_filename = {}

    
    import time
    batch_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one_document,
                uploaded_file.name,
                Image.open(uploaded_file),
                preprocessing_mode,
                ocr_engine,
                min_confidence,
            ): uploaded_file.name
            for uploaded_file in uploaded_files
        }

        completed = 0
        for future in as_completed(futures):
            filename = futures[future]
            results_by_filename[filename] = future.result()
            completed += 1
            progress_bar.progress(completed / len(uploaded_files))

    batch_elapsed = time.perf_counter() - batch_start
    total_start_placeholder.metric(
        "Total batch time",
        f"{batch_elapsed:.2f}s",
        help=f"{len(uploaded_files)} documents processed with {max_workers} parallel workers"
    )

    # results shown in the original upload order not completion order
    for uploaded_file in uploaded_files:
        result = results_by_filename[uploaded_file.name]
        with st.expander(f"📄 {uploaded_file.name}", expanded=False):
            render_result(result, ocr_engine, show_processed_image)

    
    combined_text = "\n\n".join(
        f"===== {uploaded_file.name} =====\n{results_by_filename[uploaded_file.name]['full_text']}"
        for uploaded_file in uploaded_files
    )
    st.download_button(
        label="Download All Extracted Text (.txt)",
        data=combined_text.encode("utf-8"),
        file_name="all_extracted_text.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()