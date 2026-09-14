import os
import cv2
import numpy as np
from ultralytics import YOLO
from src.tracker_utils import FPSCounter, point_in_roi, get_box_center, EntryExitTracker

SAMPLE_VIDEOS_DIR = "sample_videos"
MODEL_NAME = "yolov8n.pt"
ROI_POLYGON = [(200, 150), (500, 150), (500, 400), (200, 400)]

TRACKERS = {
    "ByteTrack": "config/bytetrack_custom.yaml",
    "BoT-SORT": "config/botsort_custom.yaml",
}

SHOW_WINDOW = True

model = YOLO(MODEL_NAME)


# runs detection + tracking on one video with one tracker and returns summary stats
def run_on_video(video_path, tracker_name, tracker_file):
    cap = cv2.VideoCapture(video_path)
    fps_counter = FPSCounter()
    ee_tracker = EntryExitTracker()
    unique_ids = set()
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        fps_counter.tick()

        results = model.track(
            frame, persist=True, verbose=False, tracker=tracker_file
        )[0]

        current_frame_count = 0

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            current_frame_count = len(track_ids)

            for box, track_id in zip(boxes, track_ids):
                unique_ids.add(track_id)
                cx, cy = get_box_center(box)
                inside = point_in_roi((cx, cy), ROI_POLYGON)
                ee_tracker.update(track_id, inside, frame_number)

                x1, y1, x2, y2 = box.astype(int)
                color = (0, 255, 0) if inside else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.polylines(frame, [np.array(ROI_POLYGON)], True, (255, 0, 0), 2)

        fps = fps_counter.get_fps()
        entries, exits = ee_tracker.get_totals()
        cv2.putText(frame, f"[{tracker_name}] FPS: {fps:.1f}  Current: {current_frame_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Unique: {len(unique_ids)}  In:{entries} Out:{exits}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if SHOW_WINDOW:
            cv2.imshow(f"Testing: {os.path.basename(video_path)} | {tracker_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    avg_fps = fps_counter.get_fps()
    entries, exits = ee_tracker.get_totals()
    return {
        "video": os.path.basename(video_path),
        "tracker": tracker_name,
        "unique_objects": len(unique_ids),
        "entries": entries,
        "exits": exits,
        "avg_fps": round(avg_fps, 2),
    }


if __name__ == "__main__":
    video_files = [
        os.path.join(SAMPLE_VIDEOS_DIR, f)
        for f in os.listdir(SAMPLE_VIDEOS_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    if len(video_files) < 3:
        print(f"Warning: fewer than 3 videos found in sample_videos/ ({len(video_files)} found).")

    results_summary = []
    for video_path in video_files:
        for tracker_name, tracker_file in TRACKERS.items():
            print(f"\nRunning on: {video_path} | Tracker: {tracker_name}")
            result = run_on_video(video_path, tracker_name, tracker_file)
            results_summary.append(result)
            print(result)

    print("\n=== Final Summary (all videos, both trackers) ===")
    for r in results_summary:
        print(r)