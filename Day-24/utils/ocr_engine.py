import os
import threading
import time

import cv2
import streamlit as st

# Global lock to ensure thread safety across C++ inference backends
inference_lock = threading.Lock()



# EasyOCR


@st.cache_resource
def load_easyocr_reader():
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    return reader


def extract_text_easyocr(image, min_confidence=0.3):
    reader = load_easyocr_reader()

    raw_results = reader.readtext(image)

    filtered_results = []
    for bbox, text, confidence in raw_results:
        if confidence >= min_confidence:
            filtered_results.append((bbox, text, confidence))

    paragraph_results = reader.readtext(image, paragraph=True)
    text_blocks = [text for bbox, text in paragraph_results]
    full_text = "\n\n".join(text_blocks)

    return full_text, filtered_results



# PaddleOCR


@st.cache_resource
def load_paddleocr_reader():
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    from paddleocr import PaddleOCR
    reader = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        lang="en"
    )
    return reader


def extract_text_paddleocr(image, min_confidence=0.3):
    reader = load_paddleocr_reader()

    if len(image.shape) == 2:
        paddle_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        paddle_input = image

    with inference_lock:
        results = reader.predict(paddle_input)

    filtered_results = []
    text_lines = []
    for res in results:
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        polys = res["rec_polys"]
        for bbox, text, confidence in zip(polys, texts, scores):
            if confidence >= min_confidence:
                filtered_results.append((bbox, text, confidence))
                text_lines.append(text)

    full_text = "\n".join(text_lines)

    return full_text, filtered_results



# RapidOCR


@st.cache_resource
def load_rapidocr_reader():
    from rapidocr_onnxruntime import RapidOCR
    reader = RapidOCR()
    return reader


def extract_text_rapidocr(image, min_confidence=0.3):
    reader = load_rapidocr_reader()

    if len(image.shape) == 2:
        rapid_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        rapid_input = image

    with inference_lock:
        result, _elapsed = reader(rapid_input)

    filtered_results = []
    text_lines = []
    if result:
        for bbox, text, confidence in result:
            if confidence >= min_confidence:
                filtered_results.append((bbox, text, confidence))
                text_lines.append(text)

    full_text = "\n".join(text_lines)

    return full_text, filtered_results



# Tesseract OCR


@st.cache_resource
def load_tesseract_engine():
    import pytesseract
    import sys
    
    if sys.platform.startswith('win'):
        
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise RuntimeError(
            "Tesseract binary not found. On Streamlit Cloud, add "
            "'tesseract-ocr' to packages.txt. Locally on Windows, install "
            "Tesseract and configure pytesseract_cmd."
        ) from e
    return True


def extract_text_tesseract(image, min_confidence=0.3):
    load_tesseract_engine()

    import pytesseract
    from pytesseract import Output

    if len(image.shape) == 2:
        tess_input = image
    else:
        tess_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(tess_input, output_type=Output.DICT)

    filtered_results = []
    lines = {}

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf_raw = int(data["conf"][i])

        if not text or conf_raw < 0:
            continue

        confidence = conf_raw / 100.0
        if confidence < min_confidence:
            continue

        left, top, width, height = (
            data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        )
        bbox = [
            [left, top],
            [left + width, top],
            [left + width, top + height],
            [left, top + height],
        ]
        filtered_results.append((bbox, text, confidence))

        line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(line_key, []).append(text)

    full_text = "\n".join(" ".join(words) for words in lines.values())

    return full_text, filtered_results


# DocTR


@st.cache_resource
def load_doctr_reader():
    os.environ.setdefault("USE_TORCH", "1")

    from doctr.models import ocr_predictor
    model = ocr_predictor(
        det_arch="db_mobilenet_v3_large",
        reco_arch="crnn_mobilenet_v3_small",
        pretrained=True,
    )
    return model


def extract_text_doctr(image, min_confidence=0.3):
    model = load_doctr_reader()

    from doctr.io import DocumentFile

    if len(image.shape) == 2:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = image

    success, encoded = cv2.imencode(".png", bgr_image)
    if not success:
        raise ValueError("Could not encode image for DocTR")
    doc = DocumentFile.from_images(encoded.tobytes())

    with inference_lock:
        result = model(doc)

    h, w = image.shape[:2]
    filtered_results = []
    block_texts = []

    for page in result.pages:
        for block in page.blocks:
            line_texts = []
            for line in block.lines:
                word_texts = []
                for word in line.words:
                    confidence = word.confidence
                    if confidence < min_confidence:
                        continue
                    (x_min, y_min), (x_max, y_max) = word.geometry
                    bbox = [
                        [x_min * w, y_min * h],
                        [x_max * w, y_min * h],
                        [x_max * w, y_max * h],
                        [x_min * w, y_max * h],
                    ]
                    filtered_results.append((bbox, word.value, confidence))
                    word_texts.append(word.value)
                if word_texts:
                    line_texts.append(" ".join(word_texts))
            if line_texts:
                block_texts.append("\n".join(line_texts))

    full_text = "\n\n".join(block_texts)

    return full_text, filtered_results




def extract_text(image, engine="EasyOCR", min_confidence=0.3):
    start_time = time.perf_counter()

    if engine == "PaddleOCR":
        full_text, filtered_results = extract_text_paddleocr(image, min_confidence)
    elif engine == "RapidOCR":
        full_text, filtered_results = extract_text_rapidocr(image, min_confidence)
    elif engine == "Tesseract":
        full_text, filtered_results = extract_text_tesseract(image, min_confidence)
    elif engine == "DocTR":
        full_text, filtered_results = extract_text_doctr(image, min_confidence)
    else:
        full_text, filtered_results = extract_text_easyocr(image, min_confidence)

    elapsed_time = time.perf_counter() - start_time

    return full_text, filtered_results, elapsed_time


def warm_up_engine(engine):
    if engine == "PaddleOCR":
        load_paddleocr_reader()
    elif engine == "RapidOCR":
        load_rapidocr_reader()
    elif engine == "Tesseract":
        load_tesseract_engine()
    elif engine == "DocTR":
        load_doctr_reader()
    else:
        load_easyocr_reader()