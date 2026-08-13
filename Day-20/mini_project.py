
#   Day 20 - Mini Project: Real Time Video Processing ToolProcesses both a recorded video file and live webcam feed using:grayscale conversion -> Gaussian blur -> Canny edge detection.Displays original vs processed frames side by side and saves output.


import cv2
import os
import numpy as np


class VideoProcessor:
    #Handles frame-by-frame video processing for files and webcam.

    def __init__(self, blur_kernel: tuple = (5, 5), canny_thresholds: tuple = (100, 200)):
        self.blur_kernel = blur_kernel
        self.canny_low, self.canny_high = canny_thresholds

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply grayscale -> Gaussian blur -> Canny edge detection to a single frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, sigmaX=0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def _combine_side_by_side(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """Stack original and processed frames horizontally for a single display window."""
        return np.hstack((original, processed))

    def run_on_file(self, input_path: str, output_path: str) -> None:
        """Process a video file and save the processed output."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing: {input_path}")
        print(f"  FPS: {fps:.2f} | Size: {width}x{height} | Frames: {total_frames}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed = self.process_frame(frame)
            out.write(processed)

            display = self._combine_side_by_side(frame, processed)
            cv2.imshow("Original | Processed", display)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Saved processed video to: {output_path}\n")

    def run_on_webcam(self, output_path: str = "output/webcam_live.mp4") -> None:
        """Process live webcam feed and save the session."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Could not access webcam.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps:  # handles 0 and NaN
            fps = 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Webcam live processing started. Press 'q' to stop.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = self.process_frame(frame)
            out.write(processed)

            display = self._combine_side_by_side(frame, processed)
            cv2.imshow("Webcam: Original | Processed", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Saved webcam session to: {output_path}")


if __name__ == "__main__":
    processor = VideoProcessor(blur_kernel=(5, 5), canny_thresholds=(100, 200))

    # Process recorded video file
    processor.run_on_file("data/sample1_video.mp4", "output/mini_project_video.mp4")

    #  test webcam
    processor.run_on_webcam("output/mini_project_webcam.mp4")