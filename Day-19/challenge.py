
# Day 19 - Challenge Task Runs the full pipeline (contours + shape detection) on 10 images and saves original, contour result, and labeled shape result separately for each image inside challenge_output/. Both results include bounding boxes and area/perimeter drawn on the image.


import cv2
import os
import glob
import math

INPUT_FOLDER = "input_images"
CHALLENGE_OUTPUT = "challenge_output"

MIN_CONTOUR_AREA = 100
NUM_CHALLENGE_IMAGES = 10


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Warning: could not load {image_path}, skipping.")
    return image


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def detect_shape(contour, area, perimeter):
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corners = len(approx)

    if corners == 3:
        shape_name = "Triangle"

    elif corners == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.90 <= aspect_ratio <= 1.10:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"

    elif corners == 5:
        shape_name = "Pentagon"

    else:
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        shape_name = "Circle" if circularity > 0.8 else "Polygon"

    return shape_name

# Draws contours + bounding rectangles + area/perimeter (same as contour_detection.py).

def get_contour_image(image, thresh):
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)

        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        label = f"A:{int(area)} P:{int(perimeter)}"
        cv2.putText(output, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return output

# Draws contours with shape names, bounding box, and area/perimeter (same as shape_detection.py).
def get_labeled_shape_image(image, thresh):
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        shape_name = detect_shape(contour, area, perimeter)

        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(output, shape_name, (x, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        metrics_label = f"A:{int(area)} P:{int(perimeter)}"
        cv2.putText(output, metrics_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return output


def process_challenge_image(image_path, index):
    filename = os.path.basename(image_path)
    print(f"\nProcessing image {index}: {filename}")

    image = load_image(image_path)
    if image is None:
        return

    # each image gets its own folder, e.g. challenge_output/image1/
    image_folder = os.path.join(CHALLENGE_OUTPUT, f"image{index}")
    os.makedirs(image_folder, exist_ok=True)

    thresh = preprocess_image(image)
    contour_result = get_contour_image(image, thresh)
    shape_result = get_labeled_shape_image(image, thresh)

    cv2.imwrite(os.path.join(image_folder, "original.jpg"), image)
    cv2.imwrite(os.path.join(image_folder, "contours.jpg"), contour_result)
    cv2.imwrite(os.path.join(image_folder, "shapes_labeled.jpg"), shape_result)

    print(f"  Saved all 3 outputs -> {image_folder}")


def main():
    os.makedirs(CHALLENGE_OUTPUT, exist_ok=True)

    image_paths = (
        glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))
        + glob.glob(os.path.join(INPUT_FOLDER, "*.jpeg"))
        + glob.glob(os.path.join(INPUT_FOLDER, "*.png"))
    )

    if not image_paths:
        print(f"Error: no images found in {INPUT_FOLDER}")
        return

    if len(image_paths) < NUM_CHALLENGE_IMAGES:
        print(f"Warning: only {len(image_paths)} images available, "
              f"challenge task needs {NUM_CHALLENGE_IMAGES}.")

    # taking only first 10 images (or however many are available)
    selected_images = image_paths[:NUM_CHALLENGE_IMAGES]

    for index, image_path in enumerate(selected_images, start=1):
        process_challenge_image(image_path, index)


if __name__ == "__main__":
    main()