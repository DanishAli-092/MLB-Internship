import cv2
import numpy as np

# Converts image to grayscale, OCR only needs shape not color
def convert_to_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


# Removes noise so OCR doesn't get confused
def remove_noise(image):
    denoised = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
    return denoised


# Black/white conversion using adaptive threshold (handles uneven lighting)
def apply_adaptive_threshold(image):
    thresh = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15
    )
    return thresh


# Boosts local contrast (CLAHE), helps faded text
def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced


# Straightens tilted image so OCR line-sorting works correctly
def deskew_image(gray_image):
    # invert + threshold so text becomes white on black
    temp_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # get coords of text pixels
    coords = np.column_stack(np.where(temp_thresh > 0))

    # skip if barely any text pixels found
    if coords.shape[0] < 20:
        return gray_image

    # find tilt angle from smallest rotated rect around text
    angle = cv2.minAreaRect(coords)[-1]

    # fix angle range to get actual rotation needed
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # ignore unreliable angles (multi-line handwriting can trick this)
    MAX_TRUSTED_ANGLE = 15
    if abs(angle) > MAX_TRUSTED_ANGLE:
        return gray_image

    # skip tiny tilts, not worth rotating
    if abs(angle) < 0.5:
        return gray_image

    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        gray_image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


# Runs the preprocessing steps based on selected mode
def preprocess_pipeline(image, mode="standard"):
    gray = convert_to_grayscale(image)

    # deskew disabled: background often gets picked up as "text" and
    # causes bad 90deg rotation. Re-enable after page is cropped.
    # gray = deskew_image(gray)

    if mode == "receipt":
        # receipts are low contrast (thermal print)
        contrast_boosted = enhance_contrast(gray)
        denoised = remove_noise(contrast_boosted)
        final = apply_adaptive_threshold(denoised)

    elif mode == "low_light":
        # needs contrast boost, not much denoising
        contrast_boosted = enhance_contrast(gray)
        final = apply_adaptive_threshold(contrast_boosted)

    else:
        # standard docs: light denoise + threshold
        denoised = remove_noise(gray)
        final = apply_adaptive_threshold(denoised)

    return final