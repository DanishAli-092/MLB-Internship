import cv2
import streamlit as st


# load EasyOCR once per session
@st.cache_resource
def load_easyocr_reader():
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    return reader


# load PaddleOCR once per session
# extra pipeline models disabled (not needed, preprocess already
# straightens the image); mkldnn off to avoid a CPU inference bug
@st.cache_resource
def load_paddleocr_reader():
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    from paddleocr import PaddleOCR
    reader = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        lang="en"
    )
    return reader


# EasyOCR: text + detailed results
# paragraph=True used for reading order (plain readtext scrambles it)
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


# PaddleOCR: text + results, same shape as EasyOCR output
def extract_text_paddleocr(image, min_confidence=0.3):
    reader = load_paddleocr_reader()

    # PaddleOCR needs 3-channel input, our pipeline gives 2D grayscale
    if len(image.shape) == 2:
        paddle_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        paddle_input = image

    # .predict() is the 3.x API, returns rec_texts/rec_scores/rec_polys
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


# entry point for app.py
def extract_text(image, engine="EasyOCR", min_confidence=0.3):
    if engine == "PaddleOCR":
        return extract_text_paddleocr(image, min_confidence)
    else:
        return extract_text_easyocr(image, min_confidence)