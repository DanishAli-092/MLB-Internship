# Day 22 - app.py (Streamlit Deployment Wrapper)
# Requirement: Convert your OCR application into a Gradio/Streamlit app
# and deploy it on Hugging Face Spaces/Streamlit

import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from ocr_document_reader import OCRDocumentReader

st.set_page_config(page_title="OCR Document Reader", page_icon="📄", layout="wide")


# Loads OCRDocumentReader once and keeps it cached
@st.cache_resource
def load_reader():
    return OCRDocumentReader(languages=["en"], use_gpu=False)


# Runs the Streamlit app
def main():
    st.title("📄 Simple OCR Document Reader")
    st.caption("Day 22 - MLB Internship | EasyOCR based text extraction")

    st.sidebar.header("Settings")
    smart_enhance = st.sidebar.checkbox(
        "Enable Smart Auto-Enhance",
        value=True,
        help="Tests both raw and enhanced versions in background and picks the higher confidence result.",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Supported cases:** printed documents, receipts/invoices, "
        "signboards, book pages, handwritten notes"
    )

    uploaded_file = st.file_uploader(
        "Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Upload an image to extract text.")
        return

    pil_image = Image.open(uploaded_file)
    temp_path = os.path.join(os.path.dirname(__file__), "_temp_upload.png")
    pil_image.save(temp_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(pil_image, use_container_width=True)

    with st.spinner("Extracting and analyzing text..."):
        reader = load_reader()
        try:
            result = reader.extract_text(temp_path, smart_enhance=smart_enhance)
        except Exception as e:
            st.error(f"Error occurred during OCR: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return

    if os.path.exists(temp_path):
        os.remove(temp_path)

    with col2:
        st.subheader("Extracted Text")

        # raw_confidence and proc_confidence are separate fields, don't mix them
        if smart_enhance:
            if result.get("was_enhanced"):
                st.success(
                    f"✨ Auto-enhanced! "
                    f"(Enhanced: {result['proc_confidence']:.1%} vs Raw: {result['raw_confidence']:.1%})"
                )
            else:
                st.info(
                    f"🔍 Raw image kept. "
                    f"(Raw: {result['raw_confidence']:.1%} vs Enhanced: {result['proc_confidence']:.1%})"
                )

        if result["text"].strip():
            st.text_area("Result (Corrected Reading Order)", result["text"], height=300)
        else:
            st.warning("No text detected. Try a different image.")

        col_a, col_b = st.columns(2)
        col_a.metric("Final Confidence", f"{result['avg_confidence']:.2%}")
        col_b.metric("Text Regions Detected", result["detections"])

    if result["text"].strip():
        st.download_button(
            label="Download Extracted Text (.txt)",
            data=result["text"],
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_extracted.txt",
            mime="text/plain",
        )

        save_locally = st.checkbox("Also save extracted text on server inside extracted_texts/ folder")
        if save_locally:
            output_path = os.path.join(
                os.path.dirname(__file__), "extracted_texts",
                f"{os.path.splitext(uploaded_file.name)[0]}_extracted.txt",
            )
            OCRDocumentReader.save_text_to_file(result["text"], output_path)
            st.success(f"Saved: {output_path}")


if __name__ == "__main__":
    main()