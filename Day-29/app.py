import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import streamlit as st
import yaml
from ultralytics import YOLO
from ultralytics.utils import USER_CONFIG_DIR


st.set_page_config(page_title="Smart Object Tracking System", page_icon="🆔", layout="wide")

st.title("🆔 Smart Object Tracking System")
st.write(
    "Video upload karo, tracker select karo, aur YOLO based multi-object "
    "tracking dekho with consistent IDs across frames."
)


if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]

SESSION_TMP_DIR = os.path.join(tempfile.gettempdir(), f"tracking_{st.session_state.session_id}")
os.makedirs(SESSION_TMP_DIR, exist_ok=True)


# finds the base tracker config file
def _find_base_tracker_config(base_name: str) -> Path:
    base_config_path = USER_CONFIG_DIR.parent / "cfg" / "trackers" / base_name
    if base_config_path.exists():
        return base_config_path
    import ultralytics
    return Path(ultralytics.__file__).parent / "cfg" / "trackers" / base_name


# makes sure tracker config exists and builds tuned version if needed
def ensure_tracker_config(tracker_name: str) -> str:
    if tracker_name == "botsort_reid.yaml":
        custom_path = Path("botsort_reid.yaml")
        if custom_path.exists():
            return str(custom_path)
        with open(_find_base_tracker_config("botsort.yaml"), "r") as f:
            config = yaml.safe_load(f)
        config["with_reid"] = True
        config["gmc_method"] = "sparseOptFlow"
        config["track_buffer"] = 60
        with open(custom_path, "w") as f:
            yaml.safe_dump(config, f)
        return str(custom_path)

    if tracker_name == "bytetrack.yaml":
        custom_path = Path("bytetrack_tuned.yaml")
        if custom_path.exists():
            return str(custom_path)
        with open(_find_base_tracker_config("bytetrack.yaml"), "r") as f:
            config = yaml.safe_load(f)
        config["track_buffer"] = 60
        with open(custom_path, "w") as f:
            yaml.safe_dump(config, f)
        return str(custom_path)

    return tracker_name


# loads the yolo model once
@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


# resets tracker state before each run
def reset_tracker(model: YOLO) -> None:
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor.trackers[0].reset()


# finds ffmpeg path on system or bundled
def _get_ffmpeg_path():
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox(
    "Model",
    options=["yolov8n.pt", "yolov8s.pt"],
    index=0,
    help="Pre-trained COCO model. Apna custom trained .pt bhi rakh kar path change kar sakte ho.",
)

tracker_choice = st.sidebar.selectbox(
    "Tracker",
    options=["ByteTrack", "BoT-SORT (with ReID)"],
    index=0,
    help=(
        "ByteTrack: fast, motion-only matching. "
        "BoT-SORT (with ReID): also compares appearance, better at telling apart "
        "similar-looking objects (e.g. players in the same jersey) - a bit slower."
    ),
)
tracker_file = "bytetrack.yaml" if tracker_choice == "ByteTrack" else "botsort_reid.yaml"

confidence = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.6, 0.05)

SAMPLE_VIDEOS_DIR = "sample_videos"

video_source = st.radio(
    "Video Source", ["Upload Your Own Video", "Use Sample Video"], horizontal=True
)

uploaded_video = None
selected_sample_path = None

if video_source == "Upload Your Own Video":
    uploaded_video = st.file_uploader(
        "Upload a video (mp4, avi, mov)", type=["mp4", "avi", "mov", "mkv"]
    )
else:
    sample_files = (
        sorted(
            f for f in os.listdir(SAMPLE_VIDEOS_DIR)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        )
        if os.path.isdir(SAMPLE_VIDEOS_DIR)
        else []
    )
    if sample_files:
        chosen_sample = st.selectbox("Choose a sample video", sample_files)
        selected_sample_path = os.path.join(SAMPLE_VIDEOS_DIR, chosen_sample)
    else:
        st.warning(f"No sample videos found in '{SAMPLE_VIDEOS_DIR}/'. Add some video files there.")


# main function that runs tracking on the video
def process_video(video_path: str, model: YOLO, tracker: str, conf: float):
    reset_tracker(model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, set(), {}, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(SESSION_TMP_DIR, "tracked_output.mp4")

    ffmpeg_path = _get_ffmpeg_path()
    if ffmpeg_path is None:
        st.error(
            "ffmpeg not found (system or bundled). Add `imageio-ffmpeg` to "
            "requirements.txt and reinstall, then try again."
        )
        cap.release()
        return None, set(), {}, 0.0

    import imageio_ffmpeg
    writer = imageio_ffmpeg.write_frames(
        output_path,
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
    )
    writer.send(None)

    unique_ids = set()
    best_seen = {}

    progress_bar = st.progress(0, text="Processing video...")
    frame_count = 0
    is_first_frame = True
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                tracker=tracker,
                conf=conf,
                persist=not is_first_frame,
                verbose=False,
            )
            is_first_frame = False
            result = results[0]
            annotated_frame = result.plot()
            writer.send(annotated_frame.tobytes())

            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.int().tolist()
                classes = result.boxes.cls.int().tolist()
                confs = result.boxes.conf.tolist()
                names = model.names

                for obj_id, cls_id, conf_score in zip(ids, classes, confs):
                    unique_ids.add(obj_id)
                    class_name = names[cls_id]
                    prev = best_seen.get(obj_id)
                    if prev is None or conf_score > prev[1]:
                        best_seen[obj_id] = (class_name, conf_score)

            frame_count += 1
            if total_frames > 0:
                progress_bar.progress(
                    min(frame_count / total_frames, 1.0),
                    text=f"Processing frame {frame_count}/{total_frames}",
                )
    finally:
        cap.release()
        writer.close()
        progress_bar.empty()

    elapsed = time.time() - start_time

    if frame_count == 0:
        st.error("No frames were read from this video - the file may be corrupt or in an unsupported format.")
        return None, set(), {}, 0.0

    return output_path, unique_ids, best_seen, elapsed


# gets basic video info for display
def get_video_properties(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_sec = (total_frames / fps) if fps else 0
    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    return {
        "Resolution": f"{width} x {height}",
        "Duration": f"{duration_sec:.1f}s",
        "FPS": f"{fps:.1f}",
        "Total Frames": total_frames,
        "File Size": f"{file_size_mb:.2f} MB",
    }


# shows video properties on screen
def show_video_properties(path: str):
    props = get_video_properties(path)
    if not props:
        return
    cols = st.columns(len(props))
    for col, (label, value) in zip(cols, props.items()):
        col.metric(label, value)


if uploaded_video is not None:
    temp_input_path = os.path.join(SESSION_TMP_DIR, "input_video.mp4")
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_video.read())
elif selected_sample_path is not None:
    temp_input_path = selected_sample_path
else:
    temp_input_path = None

if temp_input_path is not None:

    st.video(temp_input_path)
    show_video_properties(temp_input_path)

    if st.button("Run Tracking", type="primary"):
        with st.spinner("Loading model..."):
            model = load_model(model_choice)
            resolved_tracker = ensure_tracker_config(tracker_file)

        output_path, unique_ids, best_seen, elapsed = process_video(
            temp_input_path, model, resolved_tracker, confidence
        )

        if output_path is None:
            st.error("Couldn't read this video file. Try a different format (mp4/avi/mov/mkv).")
        else:
            st.success("Tracking complete!")

            input_props = get_video_properties(temp_input_path)
            processed_fps = (
                float(input_props.get("Total Frames", 0) or 0) / elapsed if elapsed > 0 else 0
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Unique Objects Tracked", len(unique_ids))
            with col2:
                st.metric("Tracker Used", tracker_choice)
            with col3:
                st.metric("Processing Speed", f"{processed_fps:.1f} FPS", help=f"Took {elapsed:.1f}s total")

            class_counts = {}
            for class_name, _ in best_seen.values():
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if class_counts:
                st.write(" · ".join(f"**{count}** {name}" for name, count in sorted(class_counts.items())))

            st.subheader("Tracked Objects - ID, Class & Best Confidence")
            if best_seen:
                st.table(
                    {
                        "Object ID": list(best_seen.keys()),
                        "Class": [v[0] for v in best_seen.values()],
                        "Best Confidence": [f"{v[1]:.2f}" for v in best_seen.values()],
                    }
                )
            else:
                st.write("No objects detected in this video.")

            st.subheader("Tracked Output Video")
            st.video(output_path)

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download Tracked Video",
                        data=f,
                        file_name="tracked_output.mp4",
                        mime="video/mp4",
                    )
            with dl_col2:
                csv_lines = ["Object ID,Class,Best Confidence"]
                for obj_id, (class_name, conf_score) in best_seen.items():
                    csv_lines.append(f"{obj_id},{class_name},{conf_score:.2f}")
                st.download_button(
                    "Download Detections (CSV)",
                    data="\n".join(csv_lines),
                    file_name="tracked_objects.csv",
                    mime="text/csv",
                )
else:
    st.info("Upload a video or choose a sample video to get started.")