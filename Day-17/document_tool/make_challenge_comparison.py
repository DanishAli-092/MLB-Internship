import cv2
import numpy as np
import os
from document_enhancer import fix_perspective, convert_to_gray, remove_noise, fix_brightness_contrast, sharpen

# Day 17 - Challenge Task
# Take 5 tilted document images, process them, and save a side-by-side comparison

input_folder = "../input_images"
output_folder = "../output_images/challenge_comparison"
os.makedirs(output_folder, exist_ok=True)

def resize_for_display(img, target_height=500):
    if img is None: return None
    # If grayscale, convert to BGR so it can be displayed with color images.
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if h == 0 or w == 0: return None
    new_width = int(w * (target_height / h))
    return cv2.resize(img, (new_width, target_height))

def add_title(img, title_text):
    if img is None: return None
    labeled_img = img.copy()
    cv2.rectangle(labeled_img, (0, 0), (labeled_img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(labeled_img, title_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return labeled_img

all_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
selected_files = all_files[:5]

if len(selected_files) < 5:
    print(f"Warning: found only {len(selected_files)} image files. Need 5.")

processed_count = 0

for file_name in selected_files:
    print(f"\n{'='*60}\nProcessing: {file_name}\n{'='*60}")
    file_path = os.path.join(input_folder, file_name)
    original_img = cv2.imread(file_path)

    if original_img is None:
        print(f"ERROR: Could not read image, skipping: {file_path}")
        continue

    print(f"Original image loaded successfully: {original_img.shape}")

    # 1. Perspective Correction
    corrected_img = fix_perspective(original_img)
    if corrected_img is None:
        print("WARNING: Perspective correction returned None. Using original image.")
        corrected_img = original_img
    else:
        print("Perspective correction completed.")

    # 2. Convert to Grayscale
    gray_img = convert_to_gray(corrected_img)
    if gray_img is None:
        print("WARNING: Grayscale conversion failed. Using corrected image.")
        gray_img = corrected_img

    # 3. Remove Noise
    denoised_img = remove_noise(gray_img)
    if denoised_img is None:
        print("WARNING: Noise removal returned None. Using grayscale image.")
        denoised_img = gray_img

    # 4. Fix Brightness and Contrast
    bright_contrast_img = fix_brightness_contrast(denoised_img, 20, 1.2)
    if bright_contrast_img is None:
        print("WARNING: Brightness/contrast correction failed. Using denoised image.")
        bright_contrast_img = denoised_img

    # 5. Sharpen
    final_img = sharpen(bright_contrast_img)
    if final_img is None:
        print("WARNING: Sharpening returned None. Using brightness/contrast image.")
        final_img = bright_contrast_img

    # Resize all 3 panels
    panel1 = resize_for_display(original_img, target_height=500)
    panel2 = resize_for_display(corrected_img, target_height=500)
    panel3 = resize_for_display(final_img, target_height=500)

    if panel1 is None or panel2 is None or panel3 is None:
        print(f"ERROR: Could not create comparison panels. Skipping: {file_name}")
        continue

    # Add titles
    panel1 = add_title(panel1, "Original")
    panel2 = add_title(panel2, "Perspective Corrected")
    panel3 = add_title(panel3, "Final Enhanced")

    # Make all panels same width by adding borders
    max_w = max(panel1.shape[1], panel2.shape[1], panel3.shape[1])
    panel1 = cv2.copyMakeBorder(panel1, 0, 0, 0, max_w - panel1.shape[1], cv2.BORDER_CONSTANT, value=(200, 200, 200))
    panel2 = cv2.copyMakeBorder(panel2, 0, 0, 0, max_w - panel2.shape[1], cv2.BORDER_CONSTANT, value=(200, 200, 200))
    panel3 = cv2.copyMakeBorder(panel3, 0, 0, 0, max_w - panel3.shape[1], cv2.BORDER_CONSTANT, value=(200, 200, 200))

    # Join the 3 panels side by side and save
    comparison_img = cv2.hconcat([panel1, panel2, panel3])
    output_path = os.path.join(output_folder, "comparison_" + file_name)
    
    if cv2.imwrite(output_path, comparison_img):
        print(f"Saved comparison for: {file_name}\nOutput: {output_path}")
        processed_count += 1
    else:
        print(f"ERROR: Could not save output: {output_path}")

print(f"\n{'='*60}\nChallenge task done.\nSuccessfully processed: {processed_count} image(s)\nOutput folder: {output_folder}\n{'='*60}")