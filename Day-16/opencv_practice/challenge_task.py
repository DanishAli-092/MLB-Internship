"""
Day 16 - Challenge Task
Reuses the ImageToolkit class (from image_toolkit/toolkit.py)
to apply all operations on 5 different category images.
"""

import os
import sys

# Allow importing ImageToolkit from the sibling image_toolkit folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "image_toolkit"))

from toolkit import ImageToolkit


INPUT_DIR = "../sample_images"
CHALLENGE_OUTPUT_DIR = "../output_images/challenge_task"

IMAGE_CATEGORIES = {
    "landscape.jpg": "landscape",
    "person.jpg": "person",
    "vehicle.jpg": "vehicle",
    "document.jpg": "document",
    "object.jpg": "object",
}


def process_category_image(filename, category):
    """Load one image and run all toolkit operations on it, saving into its category folder."""
    out_dir = os.path.join(CHALLENGE_OUTPUT_DIR, category)
    toolkit = ImageToolkit(output_dir=out_dir)

    input_path = os.path.join(INPUT_DIR, filename)
    if not toolkit.load_image(input_path):
        print(f"Skipping '{category}': image not found at {input_path}")
        return

    h, w = toolkit.image.shape[:2]

    # Grayscale
    toolkit.to_grayscale()
    toolkit.save("1_grayscale.jpg")
    toolkit.reset()

    # Resize
    toolkit.resize(640, 480)
    toolkit.save("2_resized_640x480.jpg")
    toolkit.reset()

    # Crop (center region)
    cy, cx = h // 2, w // 2
    size = min(h, w) // 4
    toolkit.crop(cx - size, cy - size, cx + size, cy + size)
    toolkit.save("3_cropped_center.jpg")
    toolkit.reset()

    # Rotate
    for angle in [90, 180, 270]:
        toolkit.rotate(angle)
        toolkit.save(f"4_rotated_{angle}.jpg")
        toolkit.reset()

    # Flip
    for direction in ["horizontal", "vertical"]:
        toolkit.flip(direction)
        toolkit.save(f"5_flip_{direction}.jpg")
        toolkit.reset()

    # Draw shapes + text
    # Font scale, text thickness, and circle radius are calculated relative
    # to image size, so everything stays visible on both small and large images.
    font_scale = max(1.0, w / 800)
    text_thickness = max(2, int(font_scale * 2))
    circle_radius = int(min(h, w) * 0.08)

    toolkit.draw_shape("rectangle", pt1=(50, 50), pt2=(w - 50, h - 50), color=(0, 255, 0))
    toolkit.draw_shape("circle", center=(w // 2, h // 2), radius=circle_radius, color=(255, 0, 0))
    toolkit.draw_shape("line", pt1=(0, 0), pt2=(w, h), color=(0, 0, 255))
    toolkit.add_text(
        f"Danish Ali - {category}",
        (30, h - 40),
        color=(0, 0, 255),
        scale=font_scale,
        thickness=text_thickness,
    )
    toolkit.save("6_shapes_text.jpg")
    toolkit.reset()

    # Bonus: brightness/contrast
    toolkit.adjust_brightness_contrast(brightness=30, contrast=20)
    toolkit.save("7_brightness_contrast.jpg")
    toolkit.reset()

    print(f"[{category}] Completed -> saved in {out_dir}")


def run_challenge_task():
    for filename, category in IMAGE_CATEGORIES.items():
        process_category_image(filename, category)


if __name__ == "__main__":
    run_challenge_task()