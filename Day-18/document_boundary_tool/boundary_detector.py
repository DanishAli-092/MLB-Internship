import cv2
import numpy as np
import os
import glob

def detect_document_boundary(image_path, output_dir):
    """
    Detects the boundary of a document in an image and draws it.
    Steps: grayscale -> blur -> canny -> morphology -> largest contour -> draw
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping {image_path}, could not read image.")
        return

    filename = os.path.splitext(os.path.basename(image_path))[0]
    original = img.copy()

    # Step 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian Blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 4: Morphological operations to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated_edges = cv2.dilate(closed_edges, kernel, iterations=1)

    # Step 5: Find contours and pick the largest one (assumed to be the document)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No contours found for {filename}")
        return

    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon (ideally 4 points for a document)
    perimeter = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

    # Fallback: if the polygon collapsed to a degenerate shape (fewer than 4
    # points), the document likely touches the image border and the contour
    # could not close properly. Use the bounding rectangle of the contour
    # instead so we still get a usable rectangular boundary.
    if len(approx) < 4:
        x, y, w, h = cv2.boundingRect(largest_contour)
        approx = np.array([
            [[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]
        ])

    # Step 6: Draw the detected boundary on the original image
    boundary_img = original.copy()
    cv2.drawContours(boundary_img, [approx], -1, (0, 255, 0), 3)

    # Save all intermediate + final outputs (needed for the challenge task)
    cv2.imwrite(f"{output_dir}/{filename}_original.jpg", original)
    cv2.imwrite(f"{output_dir}/{filename}_edges.jpg", edges)
    cv2.imwrite(f"{output_dir}/{filename}_morphology.jpg", dilated_edges)
    cv2.imwrite(f"{output_dir}/{filename}_boundary.jpg", boundary_img)

    print(f"Processed {filename}: {len(approx)} boundary points detected.")


def main():
    input_dir = "../input_images"
    output_dir = "../output_images/boundary_detection"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = (
        glob.glob(f"{input_dir}/*.jpg") +
        glob.glob(f"{input_dir}/*.jpeg") +
        glob.glob(f"{input_dir}/*.png")
    )

    if not image_paths:
        print("No images found in input_images folder.")
        return

    for image_path in image_paths:
        detect_document_boundary(image_path, output_dir)

    print(f"\nDone. Processed {len(image_paths)} images.")


if __name__ == "__main__":
    main()