# Day 28 - Custom PPE Detection App
# This app loads my trained YOLO model and runs detection on
# images or videos that the user uploads.

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import shutil
import subprocess
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="PPE Detection - Day 28", page_icon="🦺", layout="wide")
st.title("🦺 Custom PPE Detection System")
st.write("Upload an image or video and the model will detect PPE items with confidence scores.")


@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

RESULTS_DIR = "Prediction Results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# Draws boxes and labels manually so overlapping labels don't collide.
def draw_detections(image_np, result):
    img = image_np.copy()
    h, w = img.shape[:2]
    drawn_label_boxes = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) / 1000)
    thickness = max(1, int(font_scale * 2))

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        class_name = model.names[cls_id]
        label = f"{class_name} {conf_score:.2f}"

        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_x, label_y = x1, y1 - 10
        if label_y - text_h < 0:
            label_y = y1 + text_h + 10

        label_box = (label_x, label_y - text_h, label_x + text_w, label_y)

        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
            overlap_found = False
            for prev_box in drawn_label_boxes:
                if not (label_box[2] < prev_box[0] or label_box[0] > prev_box[2] or
                        label_box[3] < prev_box[1] or label_box[1] > prev_box[3]):
                    overlap_found = True
                    break
            if not overlap_found:
                break
            label_y += text_h + 8
            label_box = (label_x, label_y - text_h, label_x + text_w, label_y)
            attempt += 1

        drawn_label_boxes.append(label_box)

        cv2.rectangle(img, (label_x, label_y - text_h - 6),
                       (label_x + text_w + 4, label_y + 4), color, -1)
        cv2.putText(img, label, (label_x + 2, label_y), font, font_scale,
                    (0, 0, 0), thickness, cv2.LINE_AA)

    return img


# Shows a bar chart of detected objects per class with confidence on hover.
def show_detection_summary(class_counts, class_confidences, title="Detection Summary"):
    if not class_counts:
        return

    st.subheader(title)

    summary_data = []
    for cls_name, count in class_counts.items():
        confs = class_confidences.get(cls_name, [])
        avg_conf = sum(confs) / len(confs) if confs else 0
        summary_data.append({"Class": cls_name, "Count": count, "Avg Confidence": avg_conf})

    df = pd.DataFrame(summary_data)

    fig = px.bar(
        df,
        x="Class",
        y="Count",
        color="Class",
        text="Count",
        hover_data={"Avg Confidence": ":.2%"},
        title="Detected Objects by Class",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=400)

    st.plotly_chart(fig, use_container_width=True)


# Finds a usable ffmpeg binary, either system-installed or bundled.
def get_ffmpeg_binary():
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# Re-encodes a video to H.264 so it plays back properly in the browser.
def reencode_for_browser(input_path, output_path):
    ffmpeg_bin = get_ffmpeg_binary()
    if ffmpeg_bin is None:
        st.info(
            "ffmpeg not found — processed video was saved but may not preview "
            "in-browser. Run `pip install imageio-ffmpeg` and rerun the app, "
            "or add a `packages.txt` file with `ffmpeg` in it if deploying on "
            "Streamlit Cloud."
        )
        return input_path

    cmd = [
        ffmpeg_bin, "-y", "-i", input_path,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not os.path.exists(output_path):
        return input_path
    return output_path


st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.3, 0.05)
file_type = st.sidebar.radio("Input type", ["Image", "Video"])

st.sidebar.divider()
st.sidebar.subheader("Detectable Classes")
CLASS_NAMES = ["Gloves", "Hard_hat", "Mask", "Person", "Safety_boots", "Vest"]
for cls in CLASS_NAMES:
    st.sidebar.write(f"- {cls}")

if file_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Running inference..."):
            results = model.predict(source=np.array(image), conf=confidence)
            result = results[0]
            annotated = draw_detections(np.array(image), result)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(annotated, caption="Detection Result", use_container_width=True)

            class_counts = {}
            class_confidences = {}
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf_score = float(box.conf[0])
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                class_confidences.setdefault(class_name, []).append(conf_score)

            if len(result.boxes) > 0:
                st.subheader("Detected Objects")
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    class_name = model.names[cls_id]
                    st.write(f"- **{class_name}**: {conf_score:.2%} confidence")
            else:
                st.warning("No objects detected above the confidence threshold.")

            show_detection_summary(class_counts, class_confidences)

            output_path = os.path.join(RESULTS_DIR, "output_image.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            with open(output_path, "rb") as f:
                st.download_button("Download Result", f, file_name="detection_result.jpg")

else:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        original_ext = os.path.splitext(uploaded_video.name)[1].lower()
        if original_ext not in (".mp4", ".avi", ".mov"):
            original_ext = ".mp4"

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=original_ext)
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()

        preview_path = os.path.join(RESULTS_DIR, "preview_upload.mp4")
        with st.spinner("Preparing preview..."):
            preview_path = reencode_for_browser(tfile.name, preview_path)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with open(preview_path, "rb") as vf:
                st.video(vf.read())

        if st.button("Run Detection on Video"):
            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            raw_output_path = os.path.join(RESULTS_DIR, "output_video_raw.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            all_class_counts = {}
            all_class_confidences = {}

            with st.spinner("Processing video frame by frame..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(source=frame, conf=confidence, verbose=False)
                    result = results[0]
                    annotated_frame = draw_detections(frame, result)
                    writer.write(annotated_frame)

                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        conf_score = float(box.conf[0])
                        all_class_counts[class_name] = all_class_counts.get(class_name, 0) + 1
                        all_class_confidences.setdefault(class_name, []).append(conf_score)

                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            writer.release()

            with st.spinner("Preparing video for playback..."):
                final_output_path = os.path.join(RESULTS_DIR, "output_video.mp4")
                final_output_path = reencode_for_browser(raw_output_path, final_output_path)

            if final_output_path != raw_output_path and os.path.exists(raw_output_path):
                os.remove(raw_output_path)

            st.success("Video processing complete.")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                with open(final_output_path, "rb") as vf:
                    st.video(vf.read())

            show_detection_summary(
                all_class_counts,
                all_class_confidences,
                title="Detection Summary (across all frames)",
            )

            with open(final_output_path, "rb") as f:
                st.download_button("Download Processed Video", f, file_name="detection_result.mp4")