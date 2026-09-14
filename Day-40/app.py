import os
import time
import tempfile
import subprocess

import cv2
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from ultralytics import YOLO
import imageio_ffmpeg

from src.tracker_utils import FPSCounter, point_in_roi, get_box_center, EntryExitTracker
from src.roi_utils import draw_roi, highlight_roi_fill

# page setup and basic styling
st.set_page_config(page_title="Smart Video Analytics", page_icon="🎥", layout="wide")

st.markdown(
    """
    <style>
    .main-header {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(90deg, #1f2937, #374151);
        color: white;
        margin-bottom: 1.2rem;
    }
    .main-header h1 { margin: 0; font-size: 1.7rem; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
    <h1>🎥 Smart Video Analytics System</h1>
    <p>YOLO detection • Multi-tracker support (ByteTrack / BoT-SORT) • ROI entry-exit analytics • Day 40, ML Bench Internship • Developed By Danish Ali</p>
</div>
    """,
    unsafe_allow_html=True,
)

MAX_DISPLAY_WIDTH = 700
MAX_DISPLAY_HEIGHT = 600

TRACKER_CONFIGS = {
    "ByteTrack (faster, motion-based)": "config/bytetrack_custom.yaml",
    "BoT-SORT (slower, appearance-aware)": "config/botsort_custom.yaml",
}

# sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Detection")
    img_size = st.selectbox("Inference image size", [640, 480], index=0)
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.4)

    st.subheader("Tracking")
    tracker_choice = st.selectbox("Tracking algorithm", list(TRACKER_CONFIGS.keys()), index=0)
    tracker_file = TRACKER_CONFIGS[tracker_choice]
    st.caption(
        "ByteTrack: faster, purely motion-based — best for real-time/FPS-heavy use.\n\n"
        "BoT-SORT: uses appearance features too — can handle crossing objects better, at lower FPS."
    )

    st.subheader("Performance")
    skip_rate = st.slider("Frame skip (process every Nth frame)", 1, 5, 1)

    st.markdown("---")
    st.caption("📁 Sample test videos are available in `sample_videos/` for local testing with `test_tracking.py`.")

uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])

if "roi_points" not in st.session_state:
    st.session_state.roi_points = None


# loads the yolo model once and caches it
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


# scales frame down to fit display box while keeping aspect ratio
def compute_display_scale(frame_w, frame_h, max_w=MAX_DISPLAY_WIDTH, max_h=MAX_DISPLAY_HEIGHT):
    scale = min(max_w / frame_w, max_h / frame_h, 1.0)
    display_w = max(int(frame_w * scale), 1)
    display_h = max(int(frame_h * scale), 1)
    return scale, display_w, display_h


# re-encodes opencv output to h264 so browser can play it
def reencode_for_browser(raw_path, final_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", raw_path, "-vcodec", "libx264",
         "-pix_fmt", "yuv420p", final_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(raw_path)


model = load_model()

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.subheader("📼 Uploaded Video Preview")
    st.video(video_path)

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        st.error("Could not read the video. Please try another video.")
        st.stop()

    frame_h, frame_w = first_frame.shape[:2]

    display_scale, display_w, display_h = compute_display_scale(frame_w, frame_h)
    display_frame = cv2.resize(first_frame, (display_w, display_h))

    st.subheader("✏️ Step 1: Draw your ROI(s) on the first frame")
    st.caption("Draw one or more rectangles on the areas where you want to track entry/exit.")

    bg_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.2)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=bg_image,
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode="rect",
        key="roi_canvas",
    )

    roi_polygons = []
    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        for obj in canvas_result.json_data["objects"]:
            left = obj["left"]
            top = obj["top"]
            width = obj["width"] * obj["scaleX"]
            height = obj["height"] * obj["scaleY"]

            polygon = [
                (int(left / display_scale), int(top / display_scale)),
                (int((left + width) / display_scale), int(top / display_scale)),
                (int((left + width) / display_scale), int((top + height) / display_scale)),
                (int(left / display_scale), int((top + height) / display_scale)),
            ]
            roi_polygons.append(polygon)

        st.session_state.roi_points = roi_polygons

    if not st.session_state.roi_points:
        st.info("Draw one or more ROIs on the canvas above, then click 'Start Processing'.")
    else:
        st.success(f"✅ {len(st.session_state.roi_points)} ROI(s) drawn — ready to process.")

    start_button = st.button(
        "▶️ Start Processing",
        disabled=not st.session_state.roi_points,
        type="primary",
    )

    if start_button:
        roi_polygons = st.session_state.roi_points
        cap = cv2.VideoCapture(video_path)
        fps_input = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.join("outputs", "processed"), exist_ok=True)
        timestamp = int(time.time())
        raw_output_path = os.path.join("outputs", "processed", f"raw_{timestamp}.mp4")
        final_output_path = os.path.join("outputs", "processed", f"processed_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(raw_output_path, fourcc, fps_input, (frame_w, frame_h))

        fps_counter = FPSCounter()
        ee_tracker = EntryExitTracker()
        unique_ids = set()
        max_in_roi = 0
        current_frame_count = 0

        st.subheader("⏳ Processing")
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_display = st.empty()

        frame_number = 0
        last_results = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            fps_counter.tick()

            if frame_number % skip_rate == 0 or last_results is None:
                results = model.track(
                    frame, imgsz=img_size, conf=conf_thresh,
                    persist=True, verbose=False,
                    tracker=tracker_file
                )[0]
                last_results = results
            else:
                results = last_results

            current_in_roi = 0
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

                    if inside:
                        current_in_roi += 1

                    x1, y1, x2, y2 = box.astype(int)
                    color = (0, 255, 0) if inside else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{track_id}", (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            max_in_roi = max(max_in_roi, current_in_roi)

            for poly in roi_polygons:
                highlight_roi_fill(frame, poly)
                draw_roi(frame, poly)

            fps_now = fps_counter.get_fps()
            entries, exits = ee_tracker.get_totals()

            overlay_text = [
                f"Tracker: {tracker_choice.split(' ')[0]}",
                f"FPS: {fps_now:.1f}",
                f"Current objects: {current_frame_count}",
                f"Current in ROI: {current_in_roi}",
                f"Unique objects: {len(unique_ids)}",
                f"Entries: {entries}  Exits: {exits}",
            ]

            box_top = 10
            box_bottom = 20 + len(overlay_text) * 28
            box_right = 300
            overlay_bg = frame.copy()
            cv2.rectangle(overlay_bg, (5, box_top), (box_right, box_bottom), (0, 0, 0), -1)
            cv2.addWeighted(overlay_bg, 0.5, frame, 0.5, 0, frame)

            for i, text in enumerate(overlay_text):
                cv2.putText(frame, text, (10, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out_writer.write(frame)

            if frame_number % 5 == 0:
                preview_scale, preview_w, preview_h = compute_display_scale(frame_w, frame_h)
                preview_frame = cv2.resize(frame, (preview_w, preview_h))
                frame_display.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                progress_bar.progress(min(frame_number / total_frames, 1.0))
                status_text.text(f"Processing frame {frame_number}/{total_frames} — Tracker: {tracker_choice}")

        cap.release()
        out_writer.release()

        status_text.text("Finalizing video (encoding for playback)...")
        reencode_for_browser(raw_output_path, final_output_path)

        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")

        events_df = pd.DataFrame(ee_tracker.events)
        events_csv_path = "events.csv"
        events_df.to_csv(events_csv_path, index=False)

        avg_fps = fps_counter.get_fps()
        entries, exits = ee_tracker.get_totals()

        st.success("Video processing done!")

        st.subheader("📊 Summary")
        st.caption(f"Tracker used: **{tracker_choice}**  |  Image size: **{img_size}px**  |  Frame skip: **{skip_rate}**")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Objects (unique)", len(unique_ids))
        col1.metric("Total Entries", entries)
        col2.metric("Total Exits", exits)
        col2.metric("Max Objects in ROI", max_in_roi)
        col3.metric("Average FPS", f"{avg_fps:.1f}")
        col4.metric("Last frame object count", current_frame_count)

        st.subheader("🎬 Processed Video")
        st.video(final_output_path)

        st.subheader("📋 Events Log")
        if len(events_df) == 0:
            st.warning("No entry/exit event was detected in this video.")
        else:
            st.dataframe(events_df, use_container_width=True)

        with open(events_csv_path, "rb") as f:
            st.download_button("⬇️ Download events.csv", f, file_name="events.csv")
else:
    st.info("👆 Upload a video to start. Sample test videos are available in `sample_videos/` if you want to try locally with `test_tracking.py` first.")