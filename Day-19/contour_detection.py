
# Day 19 - Contour Detection
# Reads all images from input_images, converts to grayscale, applies thresholding, finds contours, draws them, and calculates area/perimeter. Also draws bounding rectangle and minimum enclosing circle for each shape, and writes area/perimeter directly on the output image.


import cv2
import os
import glob

INPUT_FOLDER = "input_images"
OUTPUT_ORIGINAL = "output_images/original"
OUTPUT_CONTOURS = "output_images/contours"

# ignoring very small contours since they are usually just noise
MIN_CONTOUR_AREA = 100

    # returns None if image fails to load (corrupt file, wrong path, etc.)
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


def find_and_draw_contours(image, thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # skip tiny contours, they're basically noise from thresholding
        if area < MIN_CONTOUR_AREA:
            continue

        valid_contours.append(contour)
        perimeter = cv2.arcLength(contour, True)

        # green outline for the contour
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

        # blue box for the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # yellow circle - smallest circle that can fully cover the shape
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cv2.circle(output, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)

        # writing area and perimeter directly on the image, above the box
        label = f"A:{int(area)} P:{int(perimeter)}"
        cv2.putText(output, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        print(f"  Contour -> Area: {area:.2f}, Perimeter: {perimeter:.2f}")

    print(f"  Total valid contours found: {len(valid_contours)}")
    return output, valid_contours


def process_single_image(image_path, filename):
    print(f"\nProcessing: {filename}")

    image = load_image(image_path)
    if image is None:
        return

    thresh = preprocess_image(image)
    output_image, _ = find_and_draw_contours(image, thresh)

    # saving a copy of the original too, needed for the challenge task later
    cv2.imwrite(os.path.join(OUTPUT_ORIGINAL, filename), image)

    contour_save_path = os.path.join(OUTPUT_CONTOURS, filename)
    cv2.imwrite(contour_save_path, output_image)

    print(f"  Saved -> {contour_save_path}")


def main():
    os.makedirs(OUTPUT_ORIGINAL, exist_ok=True)
    os.makedirs(OUTPUT_CONTOURS, exist_ok=True)

    # grabbing all common image formats from the input folder
    image_paths = (
        glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))
        + glob.glob(os.path.join(INPUT_FOLDER, "*.jpeg"))
        + glob.glob(os.path.join(INPUT_FOLDER, "*.png"))
    )

    if not image_paths:
        print(f"Error: no images found in {INPUT_FOLDER}")
        return

    print(f"Total images found: {len(image_paths)}")

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        process_single_image(image_path, filename)


if __name__ == "__main__":
    main()