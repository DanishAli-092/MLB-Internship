"""
Challenge Task (Mandatory) - Day 18
Process 10 different document images and compare results.
For EACH image, save: Original, Edge Detection Result,
Morphological Operation Result, Final Image with Detected Boundary.

This script reuses the same pipeline as boundary_detector.py but is kept
as a separate, dedicated script so the mandatory challenge deliverable
is explicit and easy to grade, instead of being buried inside the main tool.
"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def process_single_image(image_path):
    """
    Runs the full pipeline on one image and returns all intermediate
    stages so they can be saved individually AND shown in a comparison grid.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    original = img.copy()

    # Stage 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Stage 2: Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Stage 3: Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Stage 4: Morphological Operations (closing removes gaps, dilation strengthens edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_result = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph_result = cv2.dilate(morph_result, kernel, iterations=1)

    # Stage 5: Largest contour = document boundary
    contours, _ = cv2.findContours(morph_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundary_img = original.copy()
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)
        cv2.drawContours(boundary_img, [approx], -1, (0, 255, 0), 3)

    return {
        "original": original,
        "edges": edges,
        "morphology": morph_result,
        "boundary": boundary_img,
    }


def main():
    input_dir = "../input_images"
    output_dir = "../output_images/challenge_task"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(f"{input_dir}/*.jpg") +
        glob.glob(f"{input_dir}/*.jpeg") +
        glob.glob(f"{input_dir}/*.png")
    )

    if len(image_paths) < 10:
        print(f"Warning: Only {len(image_paths)} images found. Challenge task requires 10.")

    if not image_paths:
        print("No images found in input_images folder. Add images first.")
        return

    # Build one big comparison grid: rows = images, columns = 4 stages
    fig, axes = plt.subplots(len(image_paths), 4, figsize=(16, 4 * len(image_paths)))
    col_titles = ["Original", "Edge Detection", "Morphological Result", "Detected Boundary"]

    for row_idx, image_path in enumerate(image_paths):
        filename = os.path.splitext(os.path.basename(image_path))[0]
        result = process_single_image(image_path)

        if result is None:
            print(f"Could not read {image_path}, skipping.")
            continue

        # Save each stage individually per image (as required)
        cv2.imwrite(f"{output_dir}/{filename}_1_original.jpg", result["original"])
        cv2.imwrite(f"{output_dir}/{filename}_2_edges.jpg", result["edges"])
        cv2.imwrite(f"{output_dir}/{filename}_3_morphology.jpg", result["morphology"])
        cv2.imwrite(f"{output_dir}/{filename}_4_boundary.jpg", result["boundary"])

        # Add to comparison grid
        display_images = [
            cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB),
            result["edges"],
            result["morphology"],
            cv2.cvtColor(result["boundary"], cv2.COLOR_BGR2RGB),
        ]

        for col_idx, (title, disp_img) in enumerate(zip(col_titles, display_images)):
            ax = axes[row_idx, col_idx] if len(image_paths) > 1 else axes[col_idx]
            cmap = "gray" if disp_img.ndim == 2 else None
            ax.imshow(disp_img, cmap=cmap)
            if row_idx == 0:
                ax.set_title(title)
            ax.set_ylabel(filename, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        print(f"Processed ({row_idx + 1}/{len(image_paths)}): {filename}")

    plt.tight_layout()
    grid_path = f"{output_dir}/challenge_comparison_grid.png"
    plt.savefig(grid_path, dpi=150)
    plt.show()

    print(f"\nChallenge task complete. {len(image_paths)} images processed.")
    print(f"Individual stage images saved in: {output_dir}")
    print(f"Comparison grid saved as: {grid_path}")


if __name__ == "__main__":
    main()