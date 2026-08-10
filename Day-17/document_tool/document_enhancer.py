import cv2
import numpy as np
import os
from typing import Optional

def find_document_corners(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Finds the 4 corners of the document using edge detection.
    Includes an area threshold to prevent detecting small charts/boxes as the main document.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    total_image_area = img.shape[0] * img.shape[1]

    for contour in contours[:5]:
        contour_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        #  Must have 4 corners AND cover at least 10% of the image area
        if len(approx) == 4 and contour_area > (total_image_area * 0.10):
            return approx.reshape(4, 2)

    return None

def sort_corner_points(points: np.ndarray) -> np.ndarray:
    """Sorts corners into top-left, top-right, bottom-right, bottom-left order."""
    points = points.astype("float32")
    ordered_points = np.zeros((4, 2), dtype="float32")
    
    sums = points.sum(axis=1)
    ordered_points[0] = points[np.argmin(sums)]
    ordered_points[2] = points[np.argmax(sums)]
    
    diffs = np.diff(points, axis=1)
    ordered_points[1] = points[np.argmin(diffs)]
    ordered_points[3] = points[np.argmax(diffs)]
    
    return ordered_points

def fix_perspective(img: np.ndarray) -> np.ndarray:
    """Straightens a tilted document. Returns original if no corners found."""
    corners = find_document_corners(img)
    if corners is None:
        return img

    corners = sort_corner_points(corners)
    top_left, top_right, bottom_right, bottom_left = corners

    width_top = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    width_bottom = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    new_width = int(max(width_top, width_bottom))

    height_left = np.sqrt(((bottom_left[0] - top_left[0]) ** 2) + ((bottom_left[1] - top_left[1]) ** 2))
    height_right = np.sqrt(((bottom_right[0] - top_right[0]) ** 2) + ((bottom_right[1] - top_right[1]) ** 2))
    new_height = int(max(height_left, height_right))

    destination_points = np.float32([
        [0, 0], [new_width - 1, 0],
        [new_width - 1, new_height - 1], [0, new_height - 1]
    ])

    matrix = cv2.getPerspectiveTransform(corners, destination_points)
    return cv2.warpPerspective(img, matrix, (new_width, new_height))

def convert_to_gray(img: np.ndarray) -> np.ndarray:
    """Converts image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def remove_noise(img: np.ndarray) -> np.ndarray:
    """Removes noise using Bilateral filter to preserve sharp text edges."""
    if len(img.shape) == 2:
        color_version = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        denoised = cv2.bilateralFilter(color_version, 9, 75, 75)
        return cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    return cv2.bilateralFilter(img, 9, 75, 75)

def fix_brightness_contrast(img: np.ndarray, brightness: int = 20, contrast: float = 1.2) -> np.ndarray:
    """Adjusts overall image lighting."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

def sharpen(img: np.ndarray) -> np.ndarray:
    """Sharpens text using a custom high-pass filter kernel."""
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, sharpen_kernel)