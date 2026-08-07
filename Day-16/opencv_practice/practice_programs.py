"""
Day 16 - OpenCV Fundamentals & Basic Image Processing
Practice Programs
"""

import cv2
import os


INPUT_DIR = "../sample_images"
OUTPUT_DIR = "../output_images/practice"


def load_image(filename):
    """Load an image and validate it loaded correctly."""
    path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Could not load image at: {path}")

    return img


def show_image_info(filename):
    """Program 1: Display dimensions, channels, and file size."""
    img = load_image(filename)
    height, width, channels = img.shape
    file_size_kb = os.path.getsize(os.path.join(INPUT_DIR, filename)) / 1024

    print(f"\n--- Image Info: {filename} ---")
    print(f"Dimensions : {width}x{height}")
    print(f"Channels   : {channels}")
    print(f"File Size  : {file_size_kb:.2f} KB")


def convert_to_grayscale(filename):
    """Program 2: Convert color image to grayscale."""
    img = load_image(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    out_path = os.path.join(OUTPUT_DIR, f"gray_{filename}")
    cv2.imwrite(out_path, gray)
    print(f"Grayscale image saved to: {out_path}")
    return gray


def resize_image(filename):
    """Program 3: Resize image to different resolutions."""
    img = load_image(filename)
    resolutions = [(640, 480), (320, 240), (1280, 720)]

    for w, h in resolutions:
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(OUTPUT_DIR, f"resized_{w}x{h}_{filename}")
        cv2.imwrite(out_path, resized)
        print(f"Resized to {w}x{h} -> saved: {out_path}")


def crop_image(filename):
    """Program 4: Crop different regions of the image."""
    img = load_image(filename)
    h, w = img.shape[:2]

    regions = {
        "top_left": img[0:h // 2, 0:w // 2],
        "top_right": img[0:h // 2, w // 2:w],
        "bottom_left": img[h // 2:h, 0:w // 2],
        "bottom_right": img[h // 2:h, w // 2:w],
        "center": img[h // 4:3 * h // 4, w // 4:3 * w // 4],
    }

    for region_name, cropped in regions.items():
        out_path = os.path.join(OUTPUT_DIR, f"crop_{region_name}_{filename}")
        cv2.imwrite(out_path, cropped)
        print(f"Cropped region '{region_name}' saved: {out_path}")


def rotate_image(filename):
    """Program 5: Rotate the image by 90, 180, and 270 degrees."""
    img = load_image(filename)

    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    for angle, rotate_code in rotations.items():
        rotated = cv2.rotate(img, rotate_code)
        out_path = os.path.join(OUTPUT_DIR, f"rotated_{angle}_{filename}")
        cv2.imwrite(out_path, rotated)
        print(f"Rotated {angle} degrees -> saved: {out_path}")


def flip_image(filename):
    """Program 6: Flip the image horizontally and vertically."""
    img = load_image(filename)

    flip_h = cv2.flip(img, 1)
    flip_v = cv2.flip(img, 0)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"flip_horizontal_{filename}"), flip_h)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"flip_vertical_{filename}"), flip_v)
    print("Horizontal and vertical flips saved.")


def draw_shapes_and_text(filename, name="Danish Ali", date="2026-08-07"):
    """Draw rectangle, circle, line, polygon, and add custom text."""
    img = load_image(filename)
    h, w = img.shape[:2]

    # Rectangle
    cv2.rectangle(img, (50, 50), (w - 50, h - 50), (0, 255, 0), 3)

    # Circle
    cv2.circle(img, (w // 2, h // 2), 60, (255, 0, 0), 3)

    # Line
    cv2.line(img, (0, 0), (w, h), (0, 0, 255), 2)

    # Polygon (a simple triangle)
    points = [[w // 2, 30], [w // 2 - 60, 130], [w // 2 + 60, 130]]
    pts_array = cv2.UMat(cv2.convexHull(cv2.UMat(
        __import__("numpy").array(points, dtype="int32")
    ))) if False else None  # placeholder removed below

    import numpy as np
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    # Custom text
    text = f"{name} | {date}"
    cv2.putText(img, text, (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    out_path = os.path.join(OUTPUT_DIR, f"shapes_text_{filename}")
    cv2.imwrite(out_path, img)
    print(f"Shapes and text drawn -> saved: {out_path}")


def run_all_practice(filename):
    """Run all practice programs on a given image."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    show_image_info(filename)
    convert_to_grayscale(filename)
    resize_image(filename)
    crop_image(filename)
    rotate_image(filename)
    flip_image(filename)
    draw_shapes_and_text(filename)


if __name__ == "__main__":
    sample_filename = "sample1.jpg"   
    try:
        run_all_practice(sample_filename)
    except FileNotFoundError as e:
        print(f"Error: {e}")