import os
import io
import subprocess
import tempfile

import cv2
import pandas as pd
import streamlit as st
import imageio_ffmpeg
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from modules.detector import PersonDetector
from modules.roi_manager import ROIManager
from modules.event_logger import EventLogger
from modules.segmentation import apply_segmentation
from utils.helpers import pil_to_cv2, cv2_to_pil, draw_detections

st.set_page_config(
    page_title="Intelligent Security Monitoring System",
    page_icon="🛡️",
    layout="wide"
)

TEMP_DIR = tempfile.gettempdir()

SAMPLE_VIDEOS_DIR = "data/sample_videos"
SAMPLE_IMAGES_DIR = "data/sample_images"
OUTPUT_IMAGES_DIR = "outputs/segmented_images"
OUTPUT_VIDEOS_DIR = "outputs/processed_videos"

for d in (SAMPLE_VIDEOS_DIR, SAMPLE_IMAGES_DIR, OUTPUT_IMAGES_DIR, OUTPUT_VIDEOS_DIR):
    os.makedirs(d, exist_ok=True)

st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 30px !important;
        font-weight: 900 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 75, 75, 0.08);
        border-radius: 8px 8px 0 0;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 26px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        font-size: 15px !important;
    }
    </style>
""", unsafe_allow_html=True)


# This function loads yolo model once and caches it for reuse
@st.cache_resource(show_spinner="Loading YOLO model (first run only)...")
def load_detector(confidence_value):
    return PersonDetector(confidence=confidence_value)


# This function reencodes video to h264 so it plays in browser
def reencode_to_h264(input_path, output_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-i", input_path,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.exists(output_path)


with st.sidebar:
    st.markdown("## 🛡️ Security Monitor")
    st.caption("Day 38 . ML Bench AI/ML Internship")
    st.markdown("---")

    st.markdown("### ⚙️ Detection Settings")
    confidence = st.slider(
        "YOLO confidence threshold", 0.1, 0.9, 0.4, 0.05,
        help="Higher = fewer but more confident detections."
    )

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown(
        "1. Upload a video\n"
        "2. Draw ROI(s) on the extracted frame\n"
        "3. Click **Process Video**\n"
        "4. Watch the annotated result + download logs"
    )

    st.markdown("---")
    if st.button("🔄 Reset session"):
        st.session_state.clear()
        st.rerun()

st.title("🛡️ Intelligent Security Monitoring & Image Segmenation System")

tab_monitor, tab_segment = st.tabs(["🎥 Security Monitoring", "🧩 Image Segmentation"])

with tab_monitor:
    st.subheader("Event-Based Video Analytics")

    video_file = st.file_uploader("Step 1 . Upload a video", type=["mp4", "avi", "mov"], key="video_uploader")

    if video_file is not None:
        if st.session_state.get("uploaded_video_name") != video_file.name:
            video_bytes = video_file.read()
            video_path = os.path.join(TEMP_DIR, f"day38_input_{video_file.name}")
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            sample_video_path = os.path.join(SAMPLE_VIDEOS_DIR, video_file.name)
            with open(sample_video_path, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(video_path)
            ret, first_frame = cap.read()
            cap.release()

            st.session_state["uploaded_video_name"] = video_file.name
            st.session_state["video_path"] = video_path
            st.session_state["first_frame"] = first_frame if ret else None

        video_path = st.session_state["video_path"]
        first_frame = st.session_state.get("first_frame")

        st.markdown("#### Preview")
        st.video(video_path)

        if first_frame is None:
            st.error("Could not extract a frame from this video. Please try a different video.")
        else:
            st.markdown("#### Step 2 . Draw your ROI(s) on the frame")
            st.caption("Draw a rectangle with your mouse (click-drag). You can draw more than one ROI. Undo/clear is available on the toolbar on hover.")

            frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            orig_w, orig_h = frame_pil.size

            canvas_width = 700
            scale = canvas_width / orig_w
            canvas_height = int(orig_h * scale)

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.25)",
                stroke_width=2,
                stroke_color="#FFA500",
                background_image=frame_pil,
                background_image_fit="stretch",
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                display_toolbar=True,
                key="roi_canvas",
            )

            roi_definitions = {}
            if canvas_result.json_data is not None:
                rects = [o for o in canvas_result.json_data.get("objects", []) if o.get("type") == "Rect"]
                for i, obj in enumerate(rects):
                    roi_id = f"ROI_{i + 1}"
                    left = obj["left"]
                    top = obj["top"]
                    width = obj["width"] * obj.get("scaleX", 1)
                    height = obj["height"] * obj.get("scaleY", 1)

                    x1 = int(left / scale)
                    y1 = int(top / scale)
                    x2 = int((left + width) / scale)
                    y2 = int((top + height) / scale)
                    roi_definitions[roi_id] = (x1, y1, x2, y2)

            with st.expander("🔧 Debug: raw canvas data (check here if ROI isn't detected)"):
                st.json(canvas_result.json_data if canvas_result.json_data else {"objects": []})

            if roi_definitions:
                st.success(f"{len(roi_definitions)} ROI(s) defined and ready.")
            else:
                st.info("Draw at least one ROI on the canvas above before processing.")

            run_button = st.button("🚀 Process Video", type="primary", disabled=not roi_definitions)

            if run_button and roi_definitions:
                log_path = "data/logs/event_log.csv"
                sessions_path = "data/logs/sessions.csv"
                for p in (log_path, sessions_path):
                    if os.path.exists(p):
                        os.remove(p)

                detector = load_detector(confidence)
                roi_managers = {
                    roi_id: ROIManager(ROIManager.rectangle_to_points(x1, y1, x2, y2))
                    for roi_id, (x1, y1, x2, y2) in roi_definitions.items()
                }
                event_logger = EventLogger(log_path=log_path, sessions_path=sessions_path)

                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

                raw_output_path = os.path.join(TEMP_DIR, "day38_raw_output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (frame_width, frame_height))

                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_number = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_number += 1
                    detections = detector.track_frame(frame)
                    annotated = draw_detections(frame, detections, roi_managers, event_logger, frame_number)
                    writer.write(annotated)

                    progress_bar.progress(min(frame_number / total_frames, 1.0))
                    status_text.text(f"Processing frame {frame_number}/{total_frames} — active: {event_logger.active_count()}")

                cap.release()
                writer.release()
                status_text.empty()

                with st.spinner("Encoding final video for playback..."):
                    final_output_path = os.path.join(TEMP_DIR, "day38_final_output.mp4")
                    encoded_ok = reencode_to_h264(raw_output_path, final_output_path)
                    display_path = final_output_path if encoded_ok else raw_output_path

                st.success(f"Processing complete — {frame_number} frames processed.")

                saved_video_path = os.path.join(OUTPUT_VIDEOS_DIR, f"processed_{video_file.name}")
                with open(display_path, "rb") as src, open(saved_video_path, "wb") as dst:
                    dst.write(src.read())

                st.markdown("#### Step 3 . Processed Video")
                st.video(display_path)

                with open(display_path, "rb") as f:
                    st.download_button(
                        "Download processed video",
                        data=f,
                        file_name="processed_output.mp4",
                        mime="video/mp4"
                    )

                st.markdown("### 📊 Logs & Report")
                col_a, col_b, col_c = st.columns(3)

                if os.path.exists(log_path):
                    log_df = pd.read_csv(log_path)
                    with col_a:
                        with st.container(border=True):
                            st.markdown("**📄 Raw Event Log**")
                            st.dataframe(log_df, use_container_width=True, height=200)
                            with open(log_path, "rb") as f:
                                st.download_button("Download event_log.csv", f, "event_log.csv", "text/csv")

                if os.path.exists(sessions_path):
                    sessions_df = pd.read_csv(sessions_path)
                    with col_b:
                        with st.container(border=True):
                            st.markdown("**⏱️ Sessions (Entry + Exit Time)**")
                            st.dataframe(sessions_df, use_container_width=True, height=200)
                            with open(sessions_path, "rb") as f:
                                st.download_button("Download sessions.csv", f, "sessions.csv", "text/csv")

                    with col_c:
                        with st.container(border=True):
                             st.markdown("**📝 Summary Report**")
                             summary_text = event_logger.generate_summary()
                             st.text_area("", summary_text, height=200,     label_visibility="collapsed")
                             st.download_button("Download summary_report.txt",       summary_text, "summary_report.txt", "text/plain")

with tab_segment:
    st.subheader("Image Segmentation")
    st.caption("Upload an image and compare thresholding-based segmentation methods.")

    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
    method = st.selectbox("Segmentation method", ["Binary", "Adaptive", "Otsu"])

    extra_kwargs = {}
    if method == "Binary":
        extra_kwargs["thresh_value"] = st.slider("Threshold value", 0, 255, 127)
    elif method == "Adaptive":
        extra_kwargs["block_size"] = st.slider("Block size (odd number)", 3, 51, 11, step=2)
        extra_kwargs["c"] = st.slider("Constant C", -10, 10, 2)

    if image_file is not None:
        pil_image = Image.open(image_file)
        cv2_image = pil_to_cv2(pil_image)

        sample_image_path = os.path.join(SAMPLE_IMAGES_DIR, image_file.name)
        pil_image.save(sample_image_path)

        segmented = apply_segmentation(cv2_image, method=method, **extra_kwargs)
        segmented_pil = cv2_to_pil(segmented)

        output_name = f"{os.path.splitext(image_file.name)[0]}_{method.lower()}.png"
        output_path = os.path.join(OUTPUT_IMAGES_DIR, output_name)
        segmented_pil.save(output_path)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original**")
            st.image(pil_image, use_container_width=True)
        with col_b:
            st.markdown(f"**Segmented ({method})**")
            st.image(segmented_pil, use_container_width=True)

        buf = io.BytesIO()
        segmented_pil.save(buf, format="PNG")
        st.download_button(
            "Download segmented image",
            data=buf.getvalue(),
            file_name=f"segmented_{method.lower()}.png",
            mime="image/png"
        )
    else:
        st.info("Upload an image to see the segmentation preview.")