import os
import time
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

from vehicle_counting import (
    VEHICLE_CLASSES,
    get_centroid,
    get_crossing_direction,
    draw_counts_panel,
    draw_label,
    FFmpegWriter,
)

st.set_page_config(page_title="Smart Vehicle Counting System", page_icon="🚗", layout="wide")


@st.cache_resource
def load_model(model_name):
    # this function loads the YOLO model one time and keeps it cached
    return YOLO(model_name)


def run_counting(input_path, output_path, model, line_position, conf, tracker_name, progress_bar, status_text):
    # this function runs the main frame by frame counting loop
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    line_y = int(height * line_position)

    out = FFmpegWriter(output_path, width, height, fps)

    prev_positions = {}
    counted_ids = set()
    counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    direction_counts = {"down": 0, "up": 0}
    total_count = 0
    frame_num = 0

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = model.track(frame, persist=True, conf=conf,
                               classes=list(VEHICLE_CLASSES.keys()),
                               tracker=tracker_name, verbose=False)

        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 3)

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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 3)
                draw_label(frame, f"{cls_name} #{track_id}", x1, max(y1 - 4, 30), (0, 200, 0))

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

        progress_bar.progress(min(frame_num / total_frames, 1.0))
        status_text.text(f"Processing frame {frame_num} / {total_frames} ...")

    cap.release()
    out.release()

    elapsed_time = time.time() - start_time

    return {
        "total": total_count,
        "by_class": counts,
        "by_direction": direction_counts,
        "total_frames": total_frames,
        "processed_frames": frame_num,
        "elapsed_time": elapsed_time,
    }


def main():
    # this function builds the whole streamlit page and handles the app flow
    st.title("🚗 Smart Vehicle Counting System")
    st.write(
        "Upload a traffic video, detect and track vehicles with YOLO, "
        "and get a live count of cars, buses, trucks and motorcycles crossing a line."
    )

    # folder where the bundled sample traffic videos live
    SAMPLE_VIDEOS_DIR = str(Path(__file__).parent / "sample_videos")

    # maps the dropdown label shown to the user to the actual tracker config file
    TRACKER_OPTIONS = {
        "ByteTrack (faster, motion-based)": "bytetrack.yaml",
        "BoT-SORT (more accurate, handles occlusion better)": "botsort.yaml",
    }

    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt"], index=0)
        tracker_label = st.selectbox("Tracking Algorithm", list(TRACKER_OPTIONS.keys()), index=0)
        tracker_name = TRACKER_OPTIONS[tracker_label]
        line_position = st.slider("Counting Line Position", 0.1, 0.9, 0.6, 0.05,
                                   help="Position of the counting line as a fraction of frame height")
        conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)

    # lets the user either upload their own clip or pick from bundled sample videos
    source_choice = st.radio("Video Source", ["Upload Your Own Video", "Use Sample Video"])

    input_path = None

    if source_choice == "Upload Your Own Video":
        uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            # save the uploaded video to a temp file since opencv needs a file path
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            input_path = tfile.name
    else:
        if os.path.isdir(SAMPLE_VIDEOS_DIR):
            sample_files = sorted(
                f for f in os.listdir(SAMPLE_VIDEOS_DIR)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            )
        else:
            sample_files = []

        if sample_files:
            selected_sample = st.selectbox("Choose a sample video", sample_files)
            input_path = os.path.join(SAMPLE_VIDEOS_DIR, selected_sample)
        else:
            st.warning(f"No sample videos found in '{SAMPLE_VIDEOS_DIR}/'. Add some or switch to uploading your own.")

    if input_path is not None:
        st.video(input_path)

        if st.button("Run Vehicle Counting", type="primary"):
            output_path = os.path.join(tempfile.gettempdir(), "vehicle_count_output.mp4")

            with st.spinner("Loading model..."):
                model = load_model(model_name)

            progress_bar = st.progress(0)
            status = st.empty()
            status.text("Processing video, please wait...")

            summary = run_counting(input_path, output_path, model, line_position, conf, tracker_name, progress_bar, status)

            status.text(
                f"Processing complete! Processed {summary['processed_frames']} / {summary['total_frames']} frames "
                f"in {summary['elapsed_time']:.2f} seconds."
            )

            st.subheader("Results")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total", summary["total"])
            col2.metric("Car", summary["by_class"].get("car", 0))
            col3.metric("Truck", summary["by_class"].get("truck", 0))
            col4.metric("Bus", summary["by_class"].get("bus", 0))
            col5.metric("Motorcycle", summary["by_class"].get("motorcycle", 0))

            st.caption("Direction-wise crossing count")
            dcol1, dcol2 = st.columns(2)
            dcol1.metric("Down (entering)", summary["by_direction"]["down"])
            dcol2.metric("Up (exiting)", summary["by_direction"]["up"])

            st.caption("Processing stats")
            pcol1, pcol2 = st.columns(2)
            pcol1.metric("Frames Processed", f"{summary['processed_frames']} / {summary['total_frames']}")
            pcol2.metric("Time Taken", f"{summary['elapsed_time']:.2f} sec")

            st.subheader("Processed Video")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f,
                    file_name="vehicle_counting_output.mp4",
                    mime="video/mp4",
                )
    else:
        st.info("Upload a video or choose a sample video to get started.")


if __name__ == "__main__":
    main()