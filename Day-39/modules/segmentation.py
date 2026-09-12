import cv2

#segmentation.py
# This function does simple global binary thresholding with fixed value
def binary_threshold(image, thresh_value=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    return result


# This function does adaptive thresholding for uneven lighting
def adaptive_threshold(image, block_size=11, c=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if block_size % 2 == 0:
        block_size += 1
    result = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )
    return result


# This function does otsu thresholding to auto calculate best threshold
def otsu_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, result = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


# This function calls the right segmentation method based on input
def apply_segmentation(image, method="Otsu", **kwargs):
    method = method.lower()
    if method == "binary":
        return binary_threshold(image, kwargs.get("thresh_value", 127))
    elif method == "adaptive":
        return adaptive_threshold(image, kwargs.get("block_size", 11), kwargs.get("c", 2))
    elif method == "otsu":
        return otsu_threshold(image)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")