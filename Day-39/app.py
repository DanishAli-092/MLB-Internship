#app.py
import os
import io
import subprocess
import tempfile

import cv2
import numpy as np
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
OUTPUT_MONITOR_IMAGES_DIR = "outputs/monitored_images"

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "webp")
VIDEO_EXTENSIONS = ("mp4", "avi", "mov")

for d in (SAMPLE_VIDEOS_DIR, SAMPLE_IMAGES_DIR, OUTPUT_IMAGES_DIR, OUTPUT_VIDEOS_DIR, OUTPUT_MONITOR_IMAGES_DIR):
    os.makedirs(d, exist_ok=True)

st.markdown("""
    <style>
        /* ---- Global font & background ---- */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', -apple-system, sans-serif;
        }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] p {
            font-size: 22px !important;
            font-weight: 700 !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(255, 75, 75, 0.08);
            border-radius: 8px 8px 0 0;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(150, 150, 150, 0.2);
        }
        section[data-testid="stSidebar"] h2 {
            font-size: 24px !important;
            margin-bottom: 0px;
        }
        section[data-testid="stSidebar"] h3 {
            font-size: 16px !important;
            font-weight: 700 !important;
            letter-spacing: 0.3px;
            text-transform: uppercase;
            opacity: 0.75;
            margin-top: 18px;
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            font-size: 13px !important;
            opacity: 0.7;
        }

        /* ---- Buttons ---- */
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
            font-weight: 600;
        }

        /* ---- Hero header ---- */
        .app-hero {
            padding: 18px 22px;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(255,75,75,0.10), rgba(75,140,255,0.08));
            border: 1px solid rgba(150,150,150,0.15);
            margin-bottom: 18px;
        }
        .app-hero h1 {
            margin: 0;
            font-size: 30px;
        }
        .app-hero p {
            margin: 4px 0 0 0;
            opacity: 0.75;
            font-size: 15px;
        }
        .badge-row { margin-top: 10px; }
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 6px;
            background: rgba(0, 150, 90, 0.12);
            color: #0a8f5f;
            border: 1px solid rgba(0,150,90,0.25);
        }

        /* ---- Section cards ---- */
        div[data-testid="stContainer"] {
            border-radius: 10px;
        }

        /* ---- Footer ---- */
        .app-footer {
            margin-top: 40px;
            padding-top: 14px;
            border-top: 1px solid rgba(150,150,150,0.2);
            font-size: 13px;
            opacity: 0.65;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


# This function loads yolo model once and caches it for reuse
@st.cache_resource(show_spinner="Loading YOLO model (first run only)...")
def load_detector(confidence_value, iou_value, infer_width_value, tracker_value):
    try:
        return PersonDetector(confidence=confidence_value, iou=iou_value,
                               infer_width=infer_width_value, tracker_config=tracker_value)
    except Exception as e:
        st.error(f"Could not load the YOLO model: {e}")
        st.stop()


# This function reencodes video to h264 so it plays in browser
def reencode_to_h264(input_path, output_path):
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", input_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"[reencode_to_h264] failed: {e}")
        return False


def get_extension(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


with st.sidebar:
    st.markdown("## 🛡️ Security Monitor")
    st.caption("Day 39 · ML Bench AI/ML Internship")
    st.markdown("---")

    st.markdown("### ⚙️ Detection Settings")
    confidence = st.slider(
        "YOLO confidence threshold", 0.1, 0.9, 0.4, 0.05,
        help="Higher = fewer but more confident detections."
    )
    iou_threshold = st.slider(
        "IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05,
        help="Lower = stricter overlap suppression (fewer duplicate boxes on the same person). "
             "Higher = allows more overlapping boxes to survive."
    )
    infer_width = st.select_slider(
        "Inference resize width (speed)", options=[320, 480, 640, 960, 1280], value=640,
        help="Frames wider than this are downscaled before running YOLO, then boxes are "
             "scaled back up. Lower = faster inference, slightly less accurate on tiny/far objects."
    )

    st.markdown("---")
    st.markdown("### 🎯 Tracker")
    tracker_choice_label = st.selectbox(
        "Tracking algorithm",
        options=["ByteTrack (fast)", "BoT-SORT + ReID (more consistent, slower)"],
        index=1,
        help="ByteTrack tracks people by motion/position only — fast, works well when people "
             "stay visible. BoT-SORT + ReID also compares appearance, so it holds onto the same "
             "ID more reliably through brief occlusions (e.g. one person crossing in front of "
             "another) — at the cost of extra processing time."
    )
    TRACKER_MAP = {
        "ByteTrack (fast)": "byte_track.yaml",
        "BoT-SORT + ReID (more consistent, slower)": "botsort_reid.yaml",
    }
    tracker_config = TRACKER_MAP[tracker_choice_label]

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown(
        "**1.** Upload a video or image  \n"
        "**2.** Draw ROI(s) on the frame  \n"
        "**3.** Click **Process**  \n"
        "**4.** View + download results"
    )

    st.markdown("---")
    if st.button("🔄 Reset session"):
        st.session_state.clear()
        st.rerun()

st.markdown("""
    <div class="app-hero">
        <h1>🛡️ Intelligent Security Monitoring & Image Segmentation</h1>
        <p>YOLOv8 person detection · configurable multi-tracker (ByteTrack / BoT-SORT+ReID) · ROI-based event analytics</p>
        <div class="badge-row">
            <span class="badge">✅ Image + Video</span>
            <span class="badge">✅ ROI Detection</span>
            <span class="badge">✅ CSV Reports</span>
            <span class="badge">✅ Downloadable Results</span>
        </div>
    </div>
""", unsafe_allow_html=True)

tab_monitor, tab_segment = st.tabs(["🎥 Security Monitoring", "🧩 Image Segmentation"])

with tab_monitor:
    st.subheader("Event-Based Detection Analytics")
    st.caption("Supports both video files (tracking + entry/exit events over time) and single images (one-shot ROI check).")

    media_file = st.file_uploader(
        "Step 1 . Upload a video or an image",
        type=list(VIDEO_EXTENSIONS + IMAGE_EXTENSIONS),
        key="media_uploader"
    )

    if media_file is not None:
        ext = get_extension(media_file.name)

        if ext not in VIDEO_EXTENSIONS + IMAGE_EXTENSIONS:
            st.error(f"Unsupported file type: .{ext}. Please upload one of {VIDEO_EXTENSIONS + IMAGE_EXTENSIONS}.")
            st.stop()

        is_video = ext in VIDEO_EXTENSIONS

        # VIDEO MODE
        if is_video:
            if st.session_state.get("uploaded_media_name") != media_file.name:
                try:
                    media_bytes = media_file.read()
                    video_path = os.path.join(TEMP_DIR, f"day39_input_{media_file.name}")
                    with open(video_path, "wb") as f:
                        f.write(media_bytes)

                    sample_video_path = os.path.join(SAMPLE_VIDEOS_DIR, media_file.name)
                    with open(sample_video_path, "wb") as f:
                        f.write(media_bytes)

                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Could not open this video file. It may be corrupted or in an unsupported codec.")
                        st.stop()

                    ret, first_frame = cap.read()
                    cap.release()

                    st.session_state["uploaded_media_name"] = media_file.name
                    st.session_state["media_kind"] = "video"
                    st.session_state["video_path"] = video_path
                    st.session_state["first_frame"] = first_frame if ret else None
                except Exception as e:
                    st.error(f"Failed to read uploaded video: {e}")
                    st.stop()

            video_path = st.session_state["video_path"]
            first_frame = st.session_state.get("first_frame")

            st.markdown("#### Preview")
            st.video(video_path)

            if first_frame is None:
                st.error("Could not extract a frame from this video. Please try a different video.")
            else:
                st.markdown("#### Step 2 . Draw your ROI(s) on the frame")
                st.caption("Draw a rectangle with your mouse (click-drag). You can draw more than one ROI.")

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
                    key="roi_canvas_video",
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

                    try:
                        detector = load_detector(confidence, iou_threshold, infer_width, tracker_config)
                        roi_managers = {
                            roi_id: ROIManager(ROIManager.rectangle_to_points(x1, y1, x2, y2))
                            for roi_id, (x1, y1, x2, y2) in roi_definitions.items()
                        }
                        event_logger = EventLogger(log_path=log_path, sessions_path=sessions_path)

                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            st.error("Could not reopen video for processing.")
                            st.stop()

                        fps = cap.get(cv2.CAP_PROP_FPS) or 25
                        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

                        raw_output_path = os.path.join(TEMP_DIR, "day39_raw_output.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (frame_width, frame_height))
                        if not writer.isOpened():
                            st.error("Could not initialize video writer for output.")
                            st.stop()

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        frame_number = 0
                        failed_frames = 0

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_number += 1
                            try:
                                detections = detector.track_frame(frame)
                                annotated = draw_detections(frame, detections, roi_managers, event_logger, frame_number)
                            except Exception as e:
                                failed_frames += 1
                                annotated = frame  # fall back to raw frame so output stays in sync

                            writer.write(annotated)

                            progress_bar.progress(min(frame_number / total_frames, 1.0))
                            status_text.text(
                                f"Processing frame {frame_number}/{total_frames} — active: {event_logger.active_count()}"
                            )

                        cap.release()
                        writer.release()
                        status_text.empty()

                        if failed_frames:
                            st.warning(f"{failed_frames} frame(s) failed detection and were passed through un-annotated.")

                        with st.spinner("Encoding final video for playback..."):
                            final_output_path = os.path.join(TEMP_DIR, "day39_final_output.mp4")
                            encoded_ok = reencode_to_h264(raw_output_path, final_output_path)
                            display_path = final_output_path if encoded_ok else raw_output_path

                        st.success(f"Processing complete — {frame_number} frames processed.")

                        saved_video_path = os.path.join(OUTPUT_VIDEOS_DIR, f"processed_{media_file.name}")
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
                                    st.dataframe(log_df, width="stretch", height=200)
                                    with open(log_path, "rb") as f:
                                        st.download_button("Download event_log.csv", f, "event_log.csv", "text/csv")

                        if os.path.exists(sessions_path):
                            sessions_df = pd.read_csv(sessions_path)
                            with col_b:
                                with st.container(border=True):
                                    st.markdown("**⏱️ Sessions (Entry + Exit Time)**")
                                    st.dataframe(sessions_df, width="stretch", height=200)
                                    with open(sessions_path, "rb") as f:
                                        st.download_button("Download sessions.csv", f, "sessions.csv", "text/csv")

                            with col_c:
                                with st.container(border=True):
                                    st.markdown("**📝 Summary Report**")
                                    summary_text = event_logger.generate_summary()
                                    st.text_area("Summary Report", summary_text, height=200, label_visibility="collapsed")
                                    st.download_button("Download summary_report.txt", summary_text, "summary_report.txt", "text/plain")

                    except Exception as e:
                        st.error(f"Video processing failed: {e}")

        # IMAGE MODE
        else:
            try:
                pil_image = Image.open(media_file)
                cv2_image = pil_to_cv2(pil_image)
            except Exception as e:
                st.error(f"Could not read this image: {e}")
                st.stop()

            sample_image_path = os.path.join(SAMPLE_IMAGES_DIR, media_file.name)
            pil_image.save(sample_image_path)

            st.markdown("#### Preview")
            st.image(pil_image, width="stretch")

            st.markdown("#### Step 2 . Draw your ROI(s) on the image")
            st.caption("Draw a rectangle with your mouse (click-drag). You can draw more than one ROI.")

            orig_w, orig_h = pil_image.size
            canvas_width = 700
            scale = canvas_width / orig_w
            canvas_height = int(orig_h * scale)

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.25)",
                stroke_width=2,
                stroke_color="#FFA500",
                background_image=pil_image,
                background_image_fit="stretch",
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                display_toolbar=True,
                key="roi_canvas_image",
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

            if roi_definitions:
                st.success(f"{len(roi_definitions)} ROI(s) defined and ready.")
            else:
                st.info("Draw at least one ROI on the image above before processing (or process without ROI to just view detections).")

            run_button = st.button("🚀 Process Image", type="primary")

            if run_button:
                try:
                    detector = load_detector(confidence, iou_threshold, infer_width, tracker_config)
                    roi_managers = {
                        roi_id: ROIManager(ROIManager.rectangle_to_points(x1, y1, x2, y2))
                        for roi_id, (x1, y1, x2, y2) in roi_definitions.items()
                    }
                    # A lightweight throwaway logger just to reuse draw_detections' counting logic
                    image_logger = EventLogger(
                        log_path=os.path.join(TEMP_DIR, "day39_image_event_log.csv"),
                        sessions_path=os.path.join(TEMP_DIR, "day39_image_sessions.csv"),
                    )

                    detections = detector.detect_image(cv2_image)
                    annotated = draw_detections(cv2_image.copy(), detections, roi_managers, image_logger, frame_number=1)
                    annotated_pil = cv2_to_pil(annotated)

                    st.success(f"Processing complete — {len(detections)} person(s) detected.")

                    st.markdown("#### Step 3 . Annotated Result")
                    st.image(annotated_pil, width="stretch")

                    output_name = f"monitored_{os.path.splitext(media_file.name)[0]}.png"
                    output_path = os.path.join(OUTPUT_MONITOR_IMAGES_DIR, output_name)
                    annotated_pil.save(output_path)

                    buf = io.BytesIO()
                    annotated_pil.save(buf, format="PNG")
                    st.download_button(
                        "Download annotated image",
                        data=buf.getvalue(),
                        file_name=output_name,
                        mime="image/png"
                    )

                    if roi_managers:
                        detections_df = pd.DataFrame([
                            {
                                "person_id": d["track_id"],
                                "confidence": round(d["conf"], 3),
                                "in_roi": any(rm.is_inside(d["center"]) for rm in roi_managers.values())
                            }
                            for d in detections
                        ])
                        st.markdown("**📄 Detections in this image**")
                        st.dataframe(detections_df, width="stretch")
                        csv_bytes = detections_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download detections.csv", csv_bytes, "detections.csv", "text/csv")

                except Exception as e:
                    st.error(f"Image processing failed: {e}")

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
        try:
            pil_image = Image.open(image_file)
            cv2_image = pil_to_cv2(pil_image)
        except Exception as e:
            st.error(f"Could not read this image: {e}")
            st.stop()

        sample_image_path = os.path.join(SAMPLE_IMAGES_DIR, image_file.name)
        pil_image.save(sample_image_path)

        try:
            segmented = apply_segmentation(cv2_image, method=method, **extra_kwargs)
            segmented_pil = cv2_to_pil(segmented)
        except Exception as e:
            st.error(f"Segmentation failed: {e}")
            st.stop()

        output_name = f"{os.path.splitext(image_file.name)[0]}_{method.lower()}.png"
        output_path = os.path.join(OUTPUT_IMAGES_DIR, output_name)
        segmented_pil.save(output_path)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original**")
            st.image(pil_image, width="stretch")
        with col_b:
            st.markdown(f"**Segmented ({method})**")
            st.image(segmented_pil, width="stretch")

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

st.markdown("""
    <div class="app-footer">
        🛡️ Intelligent Security Monitoring System · Day 39 · ML Bench AI/ML Internship<br>
        Built by Danish Ali — YOLOv8 + ByteTrack/BoT-SORT
    </div>
""", unsafe_allow_html=True)