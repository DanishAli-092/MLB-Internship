import cv2
import numpy as np
import os

# Day 17 - Image Enhancement
# In this file I am improving image quality using different OpenCV techniques
# Brightness, Contrast, Gaussian Blur, Median Blur, Bilateral Filter, Sharpening

image_path = "../input_images/sample_document_2.jpg"
output_folder = "../output_images/enhancement"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

img = cv2.imread(image_path)

if img is None:
    print("Image not found, check the path:", image_path)
else:

    # ---------------- 1. BRIGHTNESS ----------------
    
    brighter_img = cv2.convertScaleAbs(img, alpha=1, beta=50)
    darker_img = cv2.convertScaleAbs(img, alpha=1, beta=-50)
    cv2.imwrite(output_folder + "/brighter.jpg", brighter_img)
    cv2.imwrite(output_folder + "/darker.jpg", darker_img)
    print("Brightness adjustment done")

    # ---------------- 2. CONTRAST ----------------
    
    high_contrast_img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    low_contrast_img = cv2.convertScaleAbs(img, alpha=0.6, beta=0)
    cv2.imwrite(output_folder + "/high_contrast.jpg", high_contrast_img)
    cv2.imwrite(output_folder + "/low_contrast.jpg", low_contrast_img)
    print("Contrast adjustment done")

    # ---------------- 3. GAUSSIAN BLUR ----------------
    # this blurs the image smoothly, good for removing general noise
    # kernel size must be an odd number like (5,5) or (7,7)
    gaussian_blur_img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(output_folder + "/gaussian_blur.jpg", gaussian_blur_img)
    print("Gaussian blur done")

    # ---------------- 4. MEDIAN BLUR ----------------
    # this replaces each pixel with the middle value of its neighbours
    # very good for removing salt and pepper type noise (small black/white dots)
    median_blur_img = cv2.medianBlur(img, 5)
    cv2.imwrite(output_folder + "/median_blur.jpg", median_blur_img)
    print("Median blur done")

    # ---------------- 5. BILATERAL FILTER ----------------
    # this also removes noise but keeps the edges sharp, unlike normal blur
    # good for documents because text edges stay clear
    bilateral_img = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite(output_folder + "/bilateral_filter.jpg", bilateral_img)
    print("Bilateral filter done")

    # ---------------- 6. SHARPENING ----------------
    # sharpening makes edges and text look more clear and defined
    # this kernel gives more weight to the center pixel and subtracts
    # from the surrounding pixels, which makes edges pop out
    sharpen_kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
    sharpened_img = cv2.filter2D(img, -1, sharpen_kernel)
    cv2.imwrite(output_folder + "/sharpened.jpg", sharpened_img)
    print("Sharpening done")

    print("All enhancement outputs saved in:", output_folder)