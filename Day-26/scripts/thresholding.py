import cv2
import numpy as np


# load original color image
def load_original(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    return img


# resize big images so they don't slow down processing
def resize_max_side(img, max_side=1000):
    h, w = img.shape[:2]
    longer_side = max(h, w)

    if longer_side <= max_side:
        return img  # already small enough

    scale = max_side / longer_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


# simple threshold, one value for whole image
def apply_binary_threshold(gray_img, threshold_value=127):
    _, binary = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    return binary


# threshold that adjusts locally, good for uneven lighting
def apply_adaptive_threshold(gray_img, block_size=11, c=2):
    if block_size % 2 == 0:
        block_size += 1  # block size must be odd

    adaptive = cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )
    return adaptive


# threshold auto picked by otsu method
def apply_otsu_threshold(gray_img):
    otsu_value, otsu_img = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return otsu_img, otsu_value


# run all threshold methods together for comparison
def compare_thresholding_methods(original_img):
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    binary = apply_binary_threshold(gray_img)
    adaptive = apply_adaptive_threshold(gray_img)
    otsu, otsu_val = apply_otsu_threshold(gray_img)

    results = {
        "original": original_img,
        "gray": gray_img,
        "binary": binary,
        "adaptive": adaptive,
        "otsu": otsu,
    }
    return results, otsu_val