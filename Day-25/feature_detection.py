
# Day 25 - Feature Detection
# Harris Corner Detection + ORB Keypoint Detection


import cv2 as cv
import numpy as np


def load_image(path):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img


def harris_corner_detection(image_path, block_size=2, ksize=3, k=0.04, threshold=0.01):
    # detects corners and returns the marked image + how many corners were found
    img = load_image(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    harris_response = cv.cornerHarris(gray_float, block_size, ksize, k)
    harris_response = cv.dilate(harris_response, None)  # makes corner dots more visible

    result = img.copy()
    corner_mask = harris_response > threshold * harris_response.max()
    result[corner_mask] = [0, 0, 255]  # mark corners in red

    num_corners = int(np.sum(corner_mask))
    return result, num_corners


def orb_keypoint_detection(image_path, n_features=1000):
    # detects ORB keypoints and returns the visualization + keypoints + descriptors
    img = load_image(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    result = cv.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return result, keypoints, descriptors


def compare_harris_vs_orb(image_path):
    # compares speed and output count between the two methods satisfies the
    # compare performance of Harris vs ORB requirement
    import time

    start = time.time()
    _, harris_count = harris_corner_detection(image_path)
    harris_time = time.time() - start

    start = time.time()
    _, keypoints, _ = orb_keypoint_detection(image_path)
    orb_time = time.time() - start

    print("Comparison: Harris vs ORB")
    print(f"Harris corners detected : {harris_count}   | time: {harris_time:.4f}s")
    print(f"ORB keypoints detected  : {len(keypoints)}   | time: {orb_time:.4f}s")
    print("Note: Harris only gives locations (no descriptor), ORB gives keypoints + binary descriptors usable for matching.")

    return {
        "harris_count": harris_count, "harris_time": harris_time,
        "orb_count": len(keypoints), "orb_time": orb_time
    }


if __name__ == "__main__":
    import os

    os.makedirs("outputs", exist_ok=True)

    harris_img, corner_count = harris_corner_detection("images/pair6/img1.jpg")
    cv.imwrite("outputs/harris_corners.jpg", harris_img)
    print(f"Harris corners detected: {corner_count}")

    orb_img, keypoints, descriptors = orb_keypoint_detection("images/pair6/img1.jpg")
    cv.imwrite("outputs/orb_keypoints.jpg", orb_img)
    print(f"ORB keypoints detected: {len(keypoints)}")

    print()
    compare_harris_vs_orb("images/pair6/img1.jpg")