
# Day 19 - Shape Detection System (Mini Project) Detects basic shapes (triangle, square, rectangle, polygon, circle) inimages using contour approximation, labels them, draws bounding boxes and displays area/perimeter directly on the output image.


import cv2
import os
import glob
import math

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images/shapes_labeled"

MIN_CONTOUR_AREA = 100


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
    """
    Approximates the contour to a polygon and decides the shape name
    based on number of corners (vertices).
    """
    # epsilon controls how "loose" the approx     imation is
    
    
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corners = len(approx)

    if corners == 3:
        shape_name = "Triangle"
     # need to check aspect ratio to tell square apart from rectangle
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
        # if it has many vertices, it's most likely a circle
        
        # double checking with circularity formula: 4*pi*area / perimeter^2
        
        # a perfect circle has circularity close to 1
        
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        shape_name = "Circle" if circularity > 0.8 else "Polygon"

    return shape_name


def detect_and_label_shapes(image, thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    detected_shapes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        shape_name = detect_shape(contour, area, perimeter)

        # draw the contour outline
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

        # draw bounding box around the shape
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # shape name just above the bounding box, bigger font
        cv2.putText(output, shape_name, (x, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # area and perimeter just below the shape name, slightly smaller
        metrics_label = f"A:{int(area)} P:{int(perimeter)}"
        cv2.putText(output, metrics_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        detected_shapes.append({
            "name": shape_name,
            "area": area,
            "perimeter": perimeter
        })

        print(f"  {shape_name} -> Area: {area:.2f}, Perimeter: {perimeter:.2f}")

    return output, detected_shapes


def process_single_image(image_path, filename):
    print(f"\nProcessing: {filename}")

    image = load_image(image_path)
    if image is None:
        return

    thresh = preprocess_image(image)
    output_image, shapes = detect_and_label_shapes(image, thresh)

    if not shapes:
        print("  No shapes detected in this image.")

    save_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(save_path, output_image)
    print(f"  Saved -> {save_path}")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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