import cv2
import matplotlib.pyplot as plt
import os

# Load image and convert to grayscale
image_path = "../input_images/doc9_Clean.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur before edge detection to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 1. Sobel Edge Detection
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = cv2.convertScaleAbs(sobel_combined)  # convert back to uint8

# 2. Laplacian Edge Detection
laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

# 3. Canny Edge Detection
# Using median-based automatic threshold selection
median_val = float(cv2.mean(blurred)[0])
low_thresh = int(max(0, 0.66 * median_val))
high_thresh = int(min(255, 1.33 * median_val))
canny = cv2.Canny(blurred, low_thresh, high_thresh)

# Compare outputs side by side
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ["Grayscale", "Sobel", "Laplacian", "Canny"]
images = [gray, sobel_combined, laplacian, canny]

for ax, title, image in zip(axes, titles, images):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
output_dir = "../output_images/edge_detection"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/comparison.png")
plt.show()

# Save individual outputs
cv2.imwrite(f"{output_dir}/sobel.jpg", sobel_combined)
cv2.imwrite(f"{output_dir}/laplacian.jpg", laplacian)
cv2.imwrite(f"{output_dir}/canny.jpg", canny)

print("Edge detection completed. Results saved in", output_dir)