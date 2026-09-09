"""
app.py
Smart People Counting System Streamlit App
Day 37 MLB Internship
Detects and tracks people in images and videos using YOLO.
Has line crossing mode and region based mode.
Also tracks peak number of people seen in video.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import imageio.v2 as imageio
from PIL import Image

from src.detector import PersonDetector
from src.tracker_counter import CrowdCounter
from src.utils import draw_boxes, draw_info_panel, draw_line, draw_region


st.set_page_config(
    page_title="Smart People Counting System",
    page_icon="🧍",
    layout="wide",
)

# custom css for metrics and buttons
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #f5f7fb;
        border: 1px solid #e0e4ec;
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #182848;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# header section
st.title("🧍‍♂️ Smart People Counting & Crowd Analysis")
st.caption("YOLO-powered detection, tracking, and real-time occupancy analytics for malls, offices, and public spaces.")
st.divider()

# sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    media_type = st.radio("Input Type", ["Image", "Video"], horizontal=True)
    confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.4, 0.05)

    counting_mode = "Simple Count"
    if media_type == "Video":
        st.markdown("---")
        st.subheader("📊 Counting Mode")
        counting_mode = st.selectbox(
            "Choose analysis mode",
            ["Simple Count", "Line Crossing (Entry/Exit)", "Region Based Counting"],
        )


# loads model once and caches it
@st.cache_resource
def load_detector(conf):
    return PersonDetector(model_path="yolov8n.pt", confidence=conf)


detector = load_detector(confidence)

# image mode
if media_type == "Image":
    uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting people..."):
            result = detector.detect(frame)
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

            display_ids = list(range(1, len(boxes) + 1))
            processed = draw_boxes(frame.copy(), boxes, display_ids, confidences)
            processed = draw_info_panel(processed, live_count=len(boxes), peak_count=len(boxes))

        st.metric("👥 People Detected", len(boxes))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Processed")
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)

        os.makedirs("outputs/images", exist_ok=True)
        output_path = os.path.join("outputs/images", "processed_" + uploaded_image.name)
        cv2.imwrite(output_path, processed)

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Processed Image", f, file_name=os.path.basename(output_path))

# video mode
else:
    uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        st.subheader("🎬 Uploaded Video")
        st.video(uploaded_video)

        # save upload to temp file then close it so windows releases lock
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.getbuffer())
        tfile.close()
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        # get first frame for setup preview
        ret, first_frame = cap.read()
        if not ret:
            st.error("Could not read this video file.")
            st.stop()
        # go back to start so full video gets processed later
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        line_start = line_end = None
        region_coords = None

        # setup line or region on first frame before processing
        if counting_mode == "Line Crossing (Entry/Exit)":
            st.markdown("### 📍 Position the Counting Line")
            line_y_percent = st.slider("Line Position (% of frame height)", 10, 90, 50)
            line_start = (0, int(frame_height * line_y_percent / 100))
            line_end = (frame_width, int(frame_height * line_y_percent / 100))

            preview = draw_line(first_frame.copy(), line_start, line_end)
            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                caption="Preview: this is exactly where the line will be during processing",
                use_container_width=True,
            )

        elif counting_mode == "Region Based Counting":
            st.markdown("### 📐 Position the Counting Region")

            col_a, col_b = st.columns(2)
            with col_a:
                region_width_pct = st.slider("Region Width (%)", 10, 90, 40)
                region_height_pct = st.slider("Region Height (%)", 10, 90, 40)
            with col_b:
                pos_x_pct = st.slider("Move Left ↔ Right (%)", 0, 100, 30)
                pos_y_pct = st.slider("Move Up ↕ Down (%)", 0, 100, 30)

            # region size stays same only position moves like dragging a box
            region_w = int(frame_width * region_width_pct / 100)
            region_h = int(frame_height * region_height_pct / 100)

            # keep region inside frame bounds
            max_x = frame_width - region_w
            max_y = frame_height - region_h
            region_x = int(max_x * pos_x_pct / 100)
            region_y = int(max_y * pos_y_pct / 100)

            region_coords = [
                (region_x, region_y),
                (region_x + region_w, region_y),
                (region_x + region_w, region_y + region_h),
                (region_x, region_y + region_h),
            ]

            preview = draw_region(first_frame.copy(), region_coords)
            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                caption="Preview: this is exactly where the region will be during processing",
                use_container_width=True,
            )

        process_clicked = st.button("▶️ Process Video")

        if process_clicked:
            os.makedirs("outputs/videos", exist_ok=True)
            output_path = os.path.join("outputs/videos", "processed_" + uploaded_video.name)

            # imageio with ffmpeg writes proper h264 mp4 that plays in browser
            writer = imageio.get_writer(
                output_path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p"
            )

            counter = CrowdCounter()

            metric_cols = st.columns(3) if counting_mode == "Line Crossing (Entry/Exit)" else st.columns(2)
            live_metric = metric_cols[0].empty()
            peak_metric = metric_cols[1].empty()
            if counting_mode == "Line Crossing (Entry/Exit)":
                flow_metric = metric_cols[2].empty()

            st.subheader("🔍 Live Processing Preview")
            stframe = st.empty()
            progress_bar = st.progress(0)
            frame_status = st.empty()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frame_index = 0

            # main loop that reads frame detects people and draws overlay
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = detector.track(frame)
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
                track_ids = (
                    result.boxes.id.cpu().numpy().astype(int)
                    if result.boxes is not None and result.boxes.id is not None
                    else list(range(len(boxes)))
                )

                live_count = len(boxes)

                if counting_mode == "Line Crossing (Entry/Exit)":
                    counter.update_line_count(boxes, track_ids, line_start, line_end)
                    processed = draw_line(frame.copy(), line_start, line_end)
                    processed = draw_boxes(processed, boxes, track_ids, confidences)
                    processed = draw_info_panel(
                        processed, live_count, counter.peak_count,
                        counter.entry_count, counter.exit_count
                    )
                    display_count = live_count

                elif counting_mode == "Region Based Counting":
                    inside_count = counter.update_region_count(boxes, track_ids, region_coords)
                    processed = draw_region(frame.copy(), region_coords)
                    processed = draw_boxes(processed, boxes, track_ids, confidences)
                    processed = draw_info_panel(processed, inside_count, counter.peak_count)
                    display_count = inside_count

                else:
                    counter.peak_count = max(counter.peak_count, live_count)
                    processed = draw_boxes(frame.copy(), boxes, track_ids, confidences)
                    processed = draw_info_panel(processed, live_count, counter.peak_count)
                    display_count = live_count

                # imageio needs rgb but opencv gives bgr so convert first
                writer.append_data(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

                frame_index += 1

                if frame_index % 3 == 0 or frame_index == total_frames:
                    live_metric.metric("👥 Live Count", display_count)
                    peak_metric.metric("📈 Peak Count", counter.peak_count)
                    if counting_mode == "Line Crossing (Entry/Exit)":
                        flow_metric.metric("🚪 Entries / Exits", f"{counter.entry_count} / {counter.exit_count}")
                    stframe.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
                    frame_status.write(f"Processed **{frame_index} / {total_frames}** frames")
                    progress_bar.progress(min(frame_index / total_frames, 1.0))

            cap.release()
            writer.close()

            st.success("✅ Video processing complete!")

            st.subheader("🎯 Processed Output")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download Processed Video", f, file_name=os.path.basename(output_path))

            try:
                os.remove(video_path)
            except PermissionError:
                # windows keeps a lock sometimes so ignore this
                pass