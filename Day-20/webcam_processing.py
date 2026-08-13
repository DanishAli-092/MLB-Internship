"""
Day 20 - Real-time Webcam Processing
Captures live video from the webcam, applies grayscale + Gaussian Blur +
Canny edge detection in real time, and saves the session as a video file.

Note: Gaussian Blur is added here (beyond the base Coding Practice spec)
because webcam sensors introduce noticeable sensor noise, especially in
average room lighting. Without blurring first, Canny edge detection picks
up this noise as false/flickering edges. Blurring first produces cleaner,
more stable edge output. This same technique is explicitly required in
the Mini Project section of Day 20.
"""

import cv2


def run_webcam_processing(output_path: str = "output/webcam_output.mp4") -> None:
    """Capture webcam feed, process each frame live, and save output."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Could not access the webcam. Check camera permissions/index.")

    # Webcams often don't report a reliable FPS, so we set a sensible default
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:  # handles 0 or NaN
        fps = 20.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Webcam started. Press 'q' to stop recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur before Canny: reduces sensor noise so edges come
        # out clean/stable instead of flickery. Kernel (5,5) is a mild
        # blur that smooths noise without losing real edges.
        blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

        edges = cv2.Canny(blurred, 100, 200)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Webcam - Original", frame)
        cv2.imshow("Webcam - Processed", edges_bgr)

        out.write(edges_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Webcam session saved to: {output_path}")


if __name__ == "__main__":
    run_webcam_processing()