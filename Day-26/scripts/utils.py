import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# put all results in one image side by side
def save_comparison_grid(results_dict, save_path, otsu_value=None):
    num_images = len(results_dict)
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))

    for ax, (label, img) in zip(axes, results_dict.items()):
        if label == "original":
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        title = label
        if label == "otsu" and otsu_value is not None:
            title = f"otsu (t={otsu_value:.1f})"
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# save each result as separate file
def save_individual_outputs(results_dict, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(image_name)[0]

    for label, img in results_dict.items():
        out_path = os.path.join(output_dir, f"{base_name}_{label}.png")
        cv2.imwrite(out_path, img)


# check how good a thresholded result looks (balance of black/white pixels)
def score_segmentation(thresholded_img):
    white_ratio = np.count_nonzero(thresholded_img == 255) / thresholded_img.size

    # bad score if almost all black or almost all white
    if white_ratio < 0.02 or white_ratio > 0.98:
        balance_score = 0.0
    else:
        # closer to 0.5 ratio = better balance
        balance_score = 1.0 - abs(0.5 - white_ratio) * 2

    return balance_score


# pick best method out of binary, adaptive, otsu and save it
def save_best_result(results_dict, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(image_name)[0]

    candidate_methods = {k: v for k, v in results_dict.items() if k not in ("original", "gray")}

    best_label = None
    best_score = -1
    for label, img in candidate_methods.items():
        score = score_segmentation(img)
        if score > best_score:
            best_score = score
            best_label = label

    best_img = candidate_methods[best_label]
    best_path = os.path.join(output_dir, f"{base_name}_best_{best_label}.png")
    cv2.imwrite(best_path, best_img)

    return best_label, best_score