import os
import time

import cv2
import pandas as pd
from ultralytics import YOLO

SAMPLE_VIDEOS_DIR = "sample_videos"
MODEL_NAME = "yolov8n.pt"

TRACKERS = {
    "ByteTrack": "config/bytetrack_custom.yaml",
    "BoT-SORT": "config/botsort_custom.yaml",
}

BASE_CONFIGS = [
    ("640px_no_skip", 640, 1),
    ("480px_no_skip", 480, 1),
    ("640px_skip2", 640, 2),
]

model = YOLO(MODEL_NAME)

# runs one config on a video and returns fps/timing metrics
def run_config(video_path, img_size, skip_rate, tracker_file):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    processed_frames = 0

    start_time = time.time()
    last_results = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        if frame_number % skip_rate == 0 or last_results is None:
            results = model.track(
                frame, imgsz=img_size, persist=True, verbose=False,
                tracker=tracker_file
            )[0]
            last_results = results
            processed_frames += 1

    elapsed = time.time() - start_time
    cap.release()

    avg_fps = frame_number / elapsed if elapsed > 0 else 0

    return {
        "total_frames": frame_number,
        "processed_frames": processed_frames,
        "elapsed_seconds": round(elapsed, 2),
        "avg_fps": round(avg_fps, 2),
    }

if __name__ == "__main__":
    video_files = [
        os.path.join(SAMPLE_VIDEOS_DIR, f)
        for f in os.listdir(SAMPLE_VIDEOS_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    if not video_files:
        print(f"No videos found in {SAMPLE_VIDEOS_DIR}/. Please add videos first.")
        exit()

    all_results = []

    for video_path in video_files:
        print(f"\n=== Testing video: {os.path.basename(video_path)} ===")
        for label, img_size, skip_rate in BASE_CONFIGS:
            for tracker_name, tracker_file in TRACKERS.items():
                full_label = f"{label}_{tracker_name}"
                print(f"Running config: {full_label} ...")
                metrics = run_config(video_path, img_size, skip_rate, tracker_file)
                row = {
                    "video": os.path.basename(video_path),
                    "config": full_label,
                    "img_size": img_size,
                    "skip_rate": skip_rate,
                    "tracker": tracker_name,
                    **metrics,
                }
                all_results.append(row)
                print(row)

    df = pd.DataFrame(all_results)
    df.to_csv("performance_results.csv", index=False)

    print("\n=== Full Comparison Table ===")
    print(df.to_string(index=False))

    print("\n=== Average FPS per config (across all videos) ===")
    avg_per_config = df.groupby("config")["avg_fps"].mean().sort_values(ascending=False)
    print(avg_per_config)

    best_config = avg_per_config.idxmax()
    print(f"\nBest performing configuration: {best_config}")

    print("\n=== Average FPS per tracker (across all videos & configs) ===")
    avg_per_tracker = df.groupby("tracker")["avg_fps"].mean().sort_values(ascending=False)
    print(avg_per_tracker)