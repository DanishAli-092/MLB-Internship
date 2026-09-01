import os
import subprocess
import cv2
import argparse
import imageio_ffmpeg
from ultralytics import YOLO

# coco class ids that represent vehicles we care about mapped to readable names
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# line color and thickness used for drawing the counting line on the frame
LINE_COLOR = (0, 255, 255)
LINE_THICKNESS = 3
BOX_COLOR = (0, 200, 0)
TEXT_COLOR = (255, 255, 255)


class FFmpegWriter:
    # this class writes raw bgr frames straight to h264 mp4 using ffmpeg
    def __init__(self, output_path, width, height, fps):
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_path, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def write(self, frame):
        # frame must be a bgr numpy array matching the width and height given at init
        self.process.stdin.write(frame.tobytes())

    def release(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()


def draw_label(frame, text, x, y, box_color):
    # this function draws a filled background box behind the id label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    label_y1 = max(y - text_h - baseline - 6, 0)

    cv2.rectangle(frame, (x, label_y1), (x + text_w + 10, y), box_color, -1)
    cv2.putText(frame, text, (x + 5, y - 6), font, font_scale, (0, 0, 0), thickness)


def get_centroid(box):
    # this function returns center point of a bounding box
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def get_crossing_direction(prev_cy, curr_cy, line_y):
    # this function checks if a vehicle crossed the line going down or up
    if prev_cy is None:
        return None
    if prev_cy < line_y <= curr_cy:
        return "down"
    if prev_cy > line_y >= curr_cy:
        return "up"
    return None


def draw_counts_panel(frame, counts, direction_counts, total):
    # this function draws a small stats panel in the top left corner
    panel_h = 30 * (len(counts) + 4)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 10 + panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    y = 35
    cv2.putText(frame, f"Total Count: {total}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 2)
    y += 28
    cv2.putText(frame, f"Down (entering): {direction_counts['down']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    y += 25
    cv2.putText(frame, f"Up (exiting): {direction_counts['up']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    y += 30
    for cls_name, cnt in counts.items():
        cv2.putText(frame, f"{cls_name}: {cnt}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
        y += 25

    return frame


def process_video(input_path, output_path, model_path="yolov8n.pt",
                   line_position=0.6, conf=0.35):
    # this function is the main pipeline that runs detection tracking and counting
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * line_position)

    out = FFmpegWriter(output_path, width, height, fps)

    # tracks id to previous centroid y so we know when it crosses the line
    prev_positions = {}
    # ids that already got counted so we stop duplicate counting for same vehicle
    counted_ids = set()
    counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    # separate totals for each crossing direction down means entering up means exiting
    direction_counts = {"down": 0, "up": 0}
    total_count = 0

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # persist true keeps the same tracker state across frames for stable ids
        results = model.track(frame, persist=True, conf=conf,
                               classes=list(VEHICLE_CLASSES.keys()),
                               tracker="botsort.yaml", verbose=False)

        cv2.line(frame, (0, line_y), (width, line_y), LINE_COLOR, LINE_THICKNESS)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, classes):
                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                cx, cy = get_centroid(box)
                cls_name = VEHICLE_CLASSES[cls_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 3)
                draw_label(frame, f"{cls_name} #{track_id}", x1, max(y1 - 4, 30), BOX_COLOR)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                prev_cy = prev_positions.get(track_id)
                direction = get_crossing_direction(prev_cy, cy, line_y)

                if track_id not in counted_ids and direction is not None:
                    counted_ids.add(track_id)
                    counts[cls_name] += 1
                    direction_counts[direction] += 1
                    total_count += 1

                prev_positions[track_id] = cy

        frame = draw_counts_panel(frame, counts, direction_counts, total_count)
        out.write(frame)

    cap.release()
    out.release()

    return {
        "total": total_count,
        "by_class": counts,
        "by_direction": direction_counts,
        "frames_processed": frame_num,
    }


if __name__ == "__main__":
    # command line entry point so we can run the script directly on a sample video
    parser = argparse.ArgumentParser(description="Smart Vehicle Counting System")
    parser.add_argument("--input", required=True, help="Path to input traffic video")
    parser.add_argument("--output", default="output_videos/result.mp4", help="Path to save output video")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--line", type=float, default=0.6, help="Counting line position (0-1 of frame height)")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    summary = process_video(args.input, args.output, args.model, args.line)

    print("Processing complete.")
    print(f"Total vehicles counted: {summary['total']}")
    print(f"  Down (entering): {summary['by_direction']['down']}")
    print(f"  Up (exiting): {summary['by_direction']['up']}")
    for cls_name, cnt in summary["by_class"].items():
        print(f"  {cls_name}: {cnt}")