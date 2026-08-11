import cv2
import matplotlib.pyplot as plt
import os

image_path = "../input_images/doc9_Clean.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binary threshold needed before most morphological operations
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Define structuring element (kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Apply each morphological operation
erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)

# Compare before and after
operations = {
    "Original (Binary)": binary,
    "Erosion": erosion,
    "Dilation": dilation,
    "Opening": opening,
    "Closing": closing,
    "Gradient": gradient,
    "Top Hat": tophat,
    "Black Hat": blackhat,
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for ax, (title, image) in zip(axes, operations.items()):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
output_dir = "../output_images/morphological_ops"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/comparison.png")
plt.show()

# Save each result individually
for title, image in operations.items():
    filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    cv2.imwrite(f"{output_dir}/{filename}.jpg", image)

print("Morphological operations completed. Results saved in", output_dir)