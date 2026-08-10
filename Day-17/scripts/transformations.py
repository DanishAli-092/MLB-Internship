import cv2
import numpy as np
import os

# Day 17 - Image Transformations
# In this file I am doing 5 basic transformations on an image using OpenCV.
# Translation, Rotation, Scaling, Affine Transform, Perspective Transform

image_path = "../input_images/tilted_document_9.png"
output_folder = "../output_images/transformations"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

img = cv2.imread(image_path)

if img is None:
    print("Image not found, check the path:", image_path)
else:
    height = img.shape[0]
    width = img.shape[1]

    # ---------------- 1. TRANSLATION ----------------
    tx = 60
    ty = 40
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv2.warpAffine(img, translation_matrix, (width, height))
    cv2.imwrite(output_folder + "/translated.jpg", translated_img)
    print("Translation done")
    print("Original size:", img.shape)
    print("Translated size:", translated_img.shape)

    # ---------------- 2. ROTATION ----------------
    center_point = (width // 2, height // 2)
    angles_to_test = [15, 45, 90, -30]

    for angle in angles_to_test:
        rotation_matrix = cv2.getRotationMatrix2D(center_point, angle, 1.0)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
        cv2.imwrite(output_folder + f"/rotated_{angle}.jpg", rotated_img)
    print("Rotation done for multiple angles")

    # ---------------- 3. SCALING ----------------
    scaled_up_img = cv2.resize(img, None, fx=1.5, fy=1.5)
    scaled_down_img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imwrite(output_folder + "/scaled_up.jpg", scaled_up_img)
    cv2.imwrite(output_folder + "/scaled_down.jpg", scaled_down_img)
    print("Scaling done")
    print("Original:", img.shape)
    print("Scaled Up:", scaled_up_img.shape)
    print("Scaled Down:", scaled_down_img.shape)

    # ---------------- 4. AFFINE TRANSFORMATION ----------------
    points_before = np.float32([[50, 50], [200, 50], [50, 200]])
    points_after = np.float32([[10, 100], [200, 50], [100, 250]])
    affine_matrix = cv2.getAffineTransform(points_before, points_after)
    affine_img = cv2.warpAffine(img, affine_matrix, (width, height))
    cv2.imwrite(output_folder + "/affine.jpg", affine_img)
    print("Affine transform done")

    # ---------------- 5. PERSPECTIVE TRANSFORMATION ----------------
    # NOTE: earlier I had hardcoded 4 corner points here, but that only
    # works for one specific image. A random new document image would
    # have its corners in completely different pixel positions, so the
    # transform would warp the wrong area.
    # Fix: find the document corners automatically using edge detection,
    # so this works on ANY document image, not just this one sample.

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_corners = None
    for contour in contours[:5]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            document_corners = approx.reshape(4, 2).astype("float32")
            break

    if document_corners is None:
        print("Could not automatically find document corners, skipping perspective demo")
    else:
        # put the 4 points in order: top-left, top-right, bottom-right, bottom-left
        ordered = np.zeros((4, 2), dtype="float32")
        point_sums = document_corners.sum(axis=1)
        ordered[0] = document_corners[np.argmin(point_sums)]
        ordered[2] = document_corners[np.argmax(point_sums)]
        point_diffs = np.diff(document_corners, axis=1)
        ordered[1] = document_corners[np.argmin(point_diffs)]
        ordered[3] = document_corners[np.argmax(point_diffs)]

        top_left, top_right, bottom_right, bottom_left = ordered

        width_top = np.sqrt(((top_right[0]-top_left[0])**2) + ((top_right[1]-top_left[1])**2))
        width_bottom = np.sqrt(((bottom_right[0]-bottom_left[0])**2) + ((bottom_right[1]-bottom_left[1])**2))
        new_width = int(max(width_top, width_bottom))

        height_left = np.sqrt(((bottom_left[0]-top_left[0])**2) + ((bottom_left[1]-top_left[1])**2))
        height_right = np.sqrt(((bottom_right[0]-top_right[0])**2) + ((bottom_right[1]-top_right[1])**2))
        new_height = int(max(height_left, height_right))

        destination_points = np.float32([
            [0, 0], [new_width - 1, 0],
            [new_width - 1, new_height - 1], [0, new_height - 1]
        ])

        perspective_matrix = cv2.getPerspectiveTransform(ordered, destination_points)
        perspective_img = cv2.warpPerspective(img, perspective_matrix, (new_width, new_height))
        cv2.imwrite(output_folder + "/perspective.jpg", perspective_img)
        print("Perspective transform done (auto-detected corners)")

    print("All transformations saved in:", output_folder)