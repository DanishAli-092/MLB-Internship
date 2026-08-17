# Day 21 - Computer Vision Image Processing Studio
# A Streamlit app where the user uploads an image applies one or more
# OpenCV operations previews the result and downloads it.
# MLB Summer Internship Day 21

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# page setup

st.set_page_config(page_title="CV Image Processing Studio", page_icon="🎨", layout="wide")


# convert PIL image (RGB) to opencv array (BGR)

def pil_to_cv(image: Image.Image) -> np.ndarray:
    rgb_array = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


# Convert an OpenCV BGR (or grayscale) array back into a PIL image
def cv_to_pil(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 2:  # it's grayscale so no color channel
        return Image.fromarray(image)
    rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


# Some operations turn the image into grayscale. If the next operation
# in the pipeline expects color then convert it back to BGR here
# otherwise it will crash because of a channel-count mismatch.
def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


# ---------------- basic operations --------------

def op_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def op_blur(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    if ksize % 2 == 0:  # kernel size needs to be odd for GaussianBlur
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def op_edge_detection(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low, high)


def op_rotate(img: np.ndarray, angle: float = 45) -> np.ndarray:
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (width, height))

# CLAHE on the L channel only (LAB color space)
    # doing it on all channels messes up the colors
def op_enhance(img: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# used by both contour detection and shape detection below
#     finds contours and removes duplicate/noise ones

def _get_clean_contours(img: np.ndarray) -> list:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img.shape[0] * img.shape[1]
    candidates = [c for c in contours if 150 < cv2.contourArea(c) < 0.5 * img_area]
    candidates.sort(key=cv2.contourArea, reverse=True)
            # Tells us how much of inner_box falls inside outer_box (0 to 1)

    def contained_ratio(inner_box, outer_box):
        xi1, yi1, wi, hi = inner_box
        xo1, yo1, wo, ho = outer_box
        xi = max(xi1, xo1)
        yi = max(yi1, yo1)
        xf = min(xi1 + wi, xo1 + wo)
        yf = min(yi1 + hi, yo1 + ho)
        intersection = max(0, xf - xi) * max(0, yf - yi)
        inner_area = wi * hi
        return intersection / inner_area if inner_area > 0 else 0
               # Standard IOU (intersection over union) - how much two boxes overlap

    def bbox_iou(box_a, box_b):
        xa1, ya1, wa, ha = box_a
        xb1, yb1, wb, hb = box_b
        xi = max(xa1, xb1)
        yi = max(ya1, yb1)
        xf = min(xa1 + wa, xb1 + wb)
        yf = min(ya1 + ha, yb1 + hb)
        intersection = max(0, xf - xi) * max(0, yf - yi)
        union = wa * ha + wb * hb - intersection
        return intersection / union if union > 0 else 0
      # sometimes canny gives 2 contours for the same shape (fill edge + stroke edge)
    # so here we skip a contour if it heavily overlaps one we already kept
    kept, kept_boxes = [], []
    for contour in candidates:
        box = cv2.boundingRect(contour)
        
        is_duplicate = any(
            contained_ratio(box, kb) > 0.75 or bbox_iou(box, kb) > 0.5
            for kb in kept_boxes
        )
        if is_duplicate:
            continue
        kept.append(contour)
        kept_boxes.append(box)

    return kept


def op_contour_detection(img: np.ndarray) -> np.ndarray:
    output = img.copy()
    cv2.drawContours(output, _get_clean_contours(img), -1, (0, 255, 0), 2)
    return output


def op_shape_detection(img: np.ndarray) -> np.ndarray:
    output = img.copy()
    for contour in _get_clean_contours(img):
        area = cv2.contourArea(contour)
        if area < 200:  # skip tiny noise contours
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertices = len(approx)

        # Decide the shape based on the number of vertices
        if vertices == 3:
            shape_name = "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif vertices == 5:
            shape_name = "Pentagon"
        elif vertices == 6:
            shape_name = "Hexagon"
        else:
            # more sides means it could be an actual circle
            # check circularity to tell circle apart from a polygon
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            shape_name = "Circle" if circularity > 0.87 else "Polygon"
            
            
    # put label at top center of the box, not at approx[0][0]
            # because that point can be anywhere on the shape depending on orientatio
        cv2.drawContours(output, [approx], -1, (255, 0, 0), 2)
        
        label_x, label_y, label_w, _ = cv2.boundingRect(approx)
        text_size = cv2.getTextSize(shape_name, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        text_x = label_x + (label_w - text_size[0]) // 2
        text_y = max(label_y - 10, text_size[1] + 2)  # keep text inside screen even if shape is near the top edge
        cv2.putText(output, shape_name, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    return output


# ---------------- challenge task operation ----------------

def op_brightness_contrast(img: np.ndarray, brightness: int = 30, contrast: float = 1.2) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def op_sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def op_flip(img: np.ndarray, direction: str = "Horizontal") -> np.ndarray:
    flip_code = 1 if direction == "Horizontal" else 0
    return cv2.flip(img, flip_code)


def op_threshold(img: np.ndarray, thresh_val: int = 127) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return result


#drop down  label
OPERATIONS = {
    "Grayscale": op_grayscale,
    "Blur": op_blur,
    "Edge Detection": op_edge_detection,
    "Rotation": op_rotate,
    "Enhancement (CLAHE)": op_enhance,
    "Contour Detection": op_contour_detection,
    "Shape Detection": op_shape_detection,
    "Brightness & Contrast": op_brightness_contrast,
    "Sharpen": op_sharpen,
    "Flip": op_flip,
    "Threshold": op_threshold,
}


# Runs one operation and passes along extra params if it needs any
def apply_operation(name: str, img: np.ndarray, params: dict) -> np.ndarray:
    img = ensure_bgr(img)

    if name == "Blur":
        return OPERATIONS[name](img, ksize=params.get("blur_ksize", 7))
    if name == "Rotation":
        return OPERATIONS[name](img, angle=params.get("angle", 45))
    if name == "Brightness & Contrast":
        return OPERATIONS[name](img, brightness=params.get("brightness", 30),
                                 contrast=params.get("contrast", 1.2))
    if name == "Flip":
        return OPERATIONS[name](img, direction=params.get("flip_dir", "Horizontal"))
    if name == "Threshold":
        return OPERATIONS[name](img, thresh_val=params.get("thresh_val", 127))

    return OPERATIONS[name](img)


# ---------------- UI ---------
st.title("🎨 Computer Vision Image Processing Studio")
st.caption("Day 21 - MLB Summer Internship | Upload an image, chain operations, download the result.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

with st.sidebar:
    st.header("Operations")

    # Base requirement: pick one operation from a dropdown
    pipeline_mode = st.checkbox(
        "Enable Pipeline Mode (challenge task - chain multiple operations)",
        value=False,
    )

    if pipeline_mode:
        selected_ops = st.multiselect(
            "Choose operations (applied in the order you pick them)",
            options=list(OPERATIONS.keys()),
            default=["Grayscale"],
        )
    else:
        single_op = st.selectbox("Choose an operation", options=list(OPERATIONS.keys()))
        selected_ops = [single_op]

    st.markdown("---")
    st.subheader("Parameters")
    params = {}

    if "Blur" in selected_ops:
        params["blur_ksize"] = st.slider("Blur kernel size", 3, 25, 7, step=2)
    if "Rotation" in selected_ops:
        params["angle"] = st.slider("Rotation angle", -180, 180, 45)
    if "Brightness & Contrast" in selected_ops:
        params["brightness"] = st.slider("Brightness", -100, 100, 30)
        params["contrast"] = st.slider("Contrast", 0.5, 3.0, 1.2)
    if "Flip" in selected_ops:
        params["flip_dir"] = st.radio("Flip direction", ["Horizontal", "Vertical"])
    if "Threshold" in selected_ops:
        params["thresh_val"] = st.slider("Threshold value", 0, 255, 127)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    original_cv = pil_to_cv(pil_image)

    try:
        processed_cv = original_cv.copy()
        for op_name in selected_ops:
            processed_cv = apply_operation(op_name, processed_cv, params)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(pil_image, use_container_width=True)
        with col2:
            st.subheader("Processed")
            st.image(cv_to_pil(processed_cv), use_container_width=True)

        # Convert the processed image to PNG bytes for the download button
        result_pil = cv_to_pil(processed_cv)
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        st.download_button(
            label="⬇️ Download Processed Image",
            data=buffer.getvalue(),
            file_name="processed_image.png",
            mime="image/png",
        )

    except Exception as error:
        st.error(f"Something went wrong while processing the image: {error}")

else:
    st.info("Upload an image from the sidebar-less uploader above to get started.")

st.markdown("---")
st.caption("Built with OpenCV + Streamlit | Danish ALi")