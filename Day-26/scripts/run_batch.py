import os
import sys

# so scripts in this folder can be imported
sys.path.append(os.path.dirname(__file__))

from thresholding import load_original, compare_thresholding_methods
from segmentation import remove_background
from utils import save_comparison_grid, save_individual_outputs, save_best_result

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_images")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_images")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def process_image(image_name):
    image_path = os.path.join(INPUT_DIR, image_name)

    # load original image
    original = load_original(image_path)

    # run all thresholding methods and compare
    results, otsu_val = compare_thresholding_methods(original)

    grid_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_name)[0]}_comparison.png")
    save_comparison_grid(results, grid_path, otsu_value=otsu_val)
    save_individual_outputs(results, OUTPUT_DIR, image_name)

    # save the best looking method separately
    best_label, best_score = save_best_result(results, OUTPUT_DIR, image_name)

    # foreground/background segmentation
    try:
        foreground, mask, _ = remove_background(image_path)
        fg_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_name)[0]}_foreground.png")
        mask_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_name)[0]}_mask.png")
        import cv2
        cv2.imwrite(fg_path, foreground)
        cv2.imwrite(mask_path, mask)
    except ValueError as e:
        # some images might not have a clear object to find
        print(f"  skipped background removal for {image_name}: {e}")

    print(f"processed: {image_name} | otsu threshold = {otsu_val:.1f} | best method = {best_label} (score {best_score:.2f})")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        print(f"sample_images folder not found at {INPUT_DIR}")
        return

    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXTENSIONS)]

    if not images:
        print("no images found in sample_images/. add at least 15 images before running.")
        return

    print(f"found {len(images)} images, starting batch processing...\n")
    for image_name in images:
        process_image(image_name)

    print(f"\ndone. outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()