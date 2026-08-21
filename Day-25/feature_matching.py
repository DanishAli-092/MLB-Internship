
# Day 25 - Feature Matching
# ORB keypoints + Brute Force / KNN Matcher + Good Match Filtering (Lowe's ratio test)


import cv2 as cv


def load_image(path):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img


def get_orb_features(image, n_features=1000):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def orb_bruteforce_matching(path1, path2):
    # simple brute force matching with crossCheck=True (keeps only mutual best matches)
    img1 = load_image(path1)
    img2 = load_image(path2)

    kp1, des1 = get_orb_features(img1)
    kp2, des2 = get_orb_features(img2)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    img_matches = cv.drawMatches(
        img1, kp1, img2, kp2, matches[:50], None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return img_matches, kp1, kp2, des1, des2, matches


def orb_knn_matching(path1, path2, ratio_thresh=0.75):
    # knn matching (k=2) + Lowe's ratio test to keep only the reliable matches
    img1 = load_image(path1)
    img2 = load_image(path2)

    kp1, des1 = get_orb_features(img1)
    kp2, des2 = get_orb_features(img2)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)  # crossCheck must be False for knnMatch
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    img_matches = cv.drawMatches(
        img1, kp1, img2, kp2, good_matches[:50], None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return img_matches, kp1, kp2, des1, des2, good_matches


if __name__ == "__main__":
    import os

    os.makedirs("outputs", exist_ok=True)

    matched_img, kp1, kp2, des1, des2, good_matches = orb_knn_matching(
        "images/pair6/img1.jpg", "images/pair6/img2.jpg"
    )
    cv.imwrite("outputs/matched_features.jpg", matched_img)

    print(f"Keypoints in image 1: {len(kp1)}")
    print(f"Keypoints in image 2: {len(kp2)}")
    print(f"Good matches (KNN + ratio test): {len(good_matches)}")