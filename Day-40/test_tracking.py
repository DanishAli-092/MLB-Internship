
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from src.tracker_utils import FPSCounter, point_in_roi, get_box_center, EntryExitTracker

SAMPLE_VIDEOS_DIR = "sample_videos"
MODEL_NAME = "yolov8n.pt"

TRACKERS = {
    "ByteTrack": "config/bytetrack_custom.yaml",
    "BoT-SORT": "config/botsort_custom.yaml",
}

SKIP_RATE = 1
SHOW_WINDOW = True

MAX_DISPLAY_WIDTH = 900
MAX_DISPLAY_HEIGHT = 700

AUTO_FINISH_DELAY = 1.5  # seconds of no new drawing before ROI selection auto-finishes

model = YOLO(MODEL_NAME)


# scales frame down to fit display box while keeping aspect ratio
def compute_display_scale(frame_w, frame_h, max_w=MAX_DISPLAY_WIDTH, max_h=MAX_DISPLAY_HEIGHT):
    scale = min(max_w / frame_w, max_h / frame_h, 1.0)
    display_w = max(int(frame_w * scale), 1)
    display_h = max(int(frame_h * scale), 1)
    return scale, display_w, display_h


# lets user draw one or more ROI rectangles, auto-finishes after a short idle pause
def select_rois_interactively(first_frame, window_title):
    frame_h, frame_w = first_frame.shape[:2]
    scale, disp_w, disp_h = compute_display_scale(frame_w, frame_h)
    display_frame = cv2.resize(first_frame, (disp_w, disp_h))

    boxes = []
    redo_stack = []
    drawing = {"active": False, "start": (0, 0), "current": (0, 0)}
    last_action = {"time": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["active"] = True
            drawing["start"] = (x, y)
            drawing["current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing["active"]:
            drawing["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing["active"]:
            drawing["active"] = False
            x1, y1 = drawing["start"]
            x2, y2 = x, y
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                redo_stack.clear()
                last_action["time"] = time.time()

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, on_mouse)

    print(f"\n[{window_title}] Drag to draw ROI(s). Processing starts automatically "
          f"{AUTO_FINISH_DELAY}s after your last box, or press 'q' to finish immediately.")

    while True:
        canvas = display_frame.copy()

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if drawing["active"]:
            x1, y1 = drawing["start"]
            x2, y2 = drawing["current"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(canvas, f"ROIs: {len(boxes)}  |  u=undo  r=redo  q=finish",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_title, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("u"), ord("U")) and boxes:
            redo_stack.append(boxes.pop())
            last_action["time"] = time.time()
        elif key in (ord("r"), ord("R")) and redo_stack:
            boxes.append(redo_stack.pop())
            last_action["time"] = time.time()
        elif key in (ord("q"), ord("Q"), 27):
            break

        if boxes and not drawing["active"] and last_action["time"] is not None:
            if time.time() - last_action["time"] > AUTO_FINISH_DELAY:
                break

    cv2.destroyWindow(window_title)

    roi_polygons = []
    for (x1, y1, x2, y2) in boxes:
        ox1, oy1 = int(x1 / scale), int(y1 / scale)
        ox2, oy2 = int(x2 / scale), int(y2 / scale)
        roi_polygons.append([(ox1, oy1), (ox2, oy1), (ox2, oy2), (ox1, oy2)])

    if not roi_polygons:
        print("No ROI was drawn — the whole frame will be treated as the ROI.")
        roi_polygons = [[(0, 0), (frame_w, 0), (frame_w, frame_h), (0, frame_h)]]

    return roi_polygons


# runs detection + tracking on one video with one tracker and returns summary stats
def run_on_video(video_path, tracker_name, tracker_file, roi_polygons):
    cap = cv2.VideoCapture(video_path)
    fps_counter = FPSCounter()
    ee_tracker = EntryExitTracker()
    unique_ids = set()
    frame_number = 0
    last_results = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        fps_counter.tick()

        if frame_number % SKIP_RATE == 0 or last_results is None:
            results = model.track(
                frame, persist=True, verbose=False, tracker=tracker_file
            )[0]
            last_results = results
        else:
            results = last_results

        current_frame_count = 0

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            current_frame_count = len(track_ids)

            for box, track_id in zip(boxes, track_ids):
                unique_ids.add(track_id)
                cx, cy = get_box_center(box)
                inside = any(point_in_roi((cx, cy), poly) for poly in roi_polygons)
                ee_tracker.update(track_id, inside, frame_number)

                x1, y1, x2, y2 = box.astype(int)
                color = (0, 255, 0) if inside else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for poly in roi_polygons:
            cv2.polylines(frame, [np.array(poly)], True, (255, 0, 0), 2)

        fps = fps_counter.get_fps()
        entries, exits = ee_tracker.get_totals()

        overlay_lines = [
            (f"[{tracker_name}] FPS: {fps:.1f}  Current: {current_frame_count}", 1.0),
            (f"Unique: {len(unique_ids)}  In:{entries} Out:{exits}", 0.85),
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        line_gap = 12
        padding = 10

        line_sizes = [cv2.getTextSize(text, font, scale, thickness)[0] for text, scale in overlay_lines]
        box_width = max(w for w, h in line_sizes) + padding * 2
        box_height = sum(h for w, h in line_sizes) + line_gap * (len(overlay_lines) - 1) + padding * 2

        overlay_bg = frame.copy()
        cv2.rectangle(overlay_bg, (5, 5), (5 + box_width, 5 + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bg, 0.5, frame, 0.5, 0, frame)

        y = 5 + padding
        for (text, scale), (w, h) in zip(overlay_lines, line_sizes):
            y += h
            cv2.putText(frame, text, (5 + padding, y), font, scale, (255, 255, 0), thickness)
            y += line_gap

        if SHOW_WINDOW:
            frame_h, frame_w = frame.shape[:2]
            _, disp_w, disp_h = compute_display_scale(frame_w, frame_h)
            preview = cv2.resize(frame, (disp_w, disp_h))
            cv2.imshow(f"Testing: {os.path.basename(video_path)} | {tracker_name}", preview)
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


# asks the user whether to process one specific video or every video in the folder
def choose_videos_to_process(video_files):
    print("\nAvailable videos:")
    for i, path in enumerate(video_files, start=1):
        print(f"  {i}. {os.path.basename(path)}")

    choice = input("\nEnter a video number to process just that one, or type 'all' to process every video: ").strip().lower()

    if choice == "all":
        return video_files

    try:
        index = int(choice) - 1
        if 0 <= index < len(video_files):
            return [video_files[index]]
    except ValueError:
        pass

    print("Invalid choice, defaulting to the first video.")
    return [video_files[0]]


if __name__ == "__main__":
    video_files = [
        os.path.join(SAMPLE_VIDEOS_DIR, f)
        for f in os.listdir(SAMPLE_VIDEOS_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    if len(video_files) < 3:
        print(f"Warning: fewer than 3 videos found in sample_videos/ ({len(video_files)} found).")

    selected_videos = choose_videos_to_process(video_files)

    results_summary = []
    for video_path in selected_videos:
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            print(f"Skipping unreadable video: {video_path}")
            continue

        roi_polygons = select_rois_interactively(first_frame, os.path.basename(video_path))

        for tracker_name, tracker_file in TRACKERS.items():
            print(f"\nRunning on: {video_path} | Tracker: {tracker_name}")
            result = run_on_video(video_path, tracker_name, tracker_file, roi_polygons)
            results_summary.append(result)
            print(result)

    print("\n=== Final Summary ===")
    for r in results_summary:
        print(r)