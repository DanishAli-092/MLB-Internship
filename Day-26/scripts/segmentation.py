import cv2
import numpy as np


# check if mask looks clean (not too small, not too big, fills its box well)
def _mask_quality_score(mask):
    fg_ratio = np.count_nonzero(mask == 255) / mask.size
    if fg_ratio < 0.02 or fg_ratio > 0.95:
        return 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    bbox_area = w * h
    if bbox_area == 0:
        return 0.0

    # how much of bounding box the contour fills
    solidity = cv2.contourArea(largest) / bbox_area
    return solidity


# flip mask if corners are mostly white (means background got picked as foreground)
def _auto_invert_if_needed(thresh):
    h, w = thresh.shape
    corners = [thresh[0, 0], thresh[0, w - 1], thresh[h - 1, 0], thresh[h - 1, w - 1]]
    white_corners = sum(1 for c in corners if c == 255)
    if white_corners >= 3:
        return cv2.bitwise_not(thresh)
    return thresh


# get foreground mask using grayscale brightness
def _grayscale_mask(img, threshold_value=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if threshold_value is None:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    thresh = _auto_invert_if_needed(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, thresh

    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask, thresh


# get foreground mask using color saturation (catches cases grayscale misses)
def _saturation_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(saturation, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = _auto_invert_if_needed(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, thresh

    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(saturation.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask, thresh


# separate foreground object from background tries two methods and keeps the better one
def remove_background(image_path, threshold_value=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    gray_mask, gray_thresh = _grayscale_mask(img, threshold_value)
    sat_mask, sat_thresh = _saturation_mask(img)

    candidates = []
    if gray_mask is not None:
        candidates.append((gray_mask, gray_thresh, _mask_quality_score(gray_mask)))
    if sat_mask is not None:
        candidates.append((sat_mask, sat_thresh, _mask_quality_score(sat_mask)))

    if not candidates:
        raise ValueError("No contours found - try a different threshold value")

    # keep whichever mask scored higher
    best_mask, best_thresh, _ = max(candidates, key=lambda c: c[2])

    # apply mask, background turns black
    foreground = cv2.bitwise_and(img, img, mask=best_mask)

    return foreground, best_mask, best_thresh


# separate touching/overlapping objects using watershed algorithm
def watershed_segmentation(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # clean up small noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # grow white area to get sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # unknown region = between sure bg and sure fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # label each object separately
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  
    markers[unknown == 255] = 0  

    markers = cv2.watershed(img, markers)

    # boundaries marked -1, draw them red
    result = img.copy()
    result[markers == -1] = [0, 0, 255]

    return result, markers