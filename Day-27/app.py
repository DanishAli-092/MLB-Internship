import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path
import tempfile
import subprocess
import io
from ultralytics import YOLO
import imageio_ffmpeg

# Page setup
st.set_page_config(page_title="Smart Object Detection", page_icon="⚡", layout="wide")


st.markdown("""
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .sub-header {
        color: #9ca3af;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
    }
    .detection-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(99, 102, 241, 0.4);
        border-radius: 12px;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div style="font-size: 2.4rem; font-weight: 700; margin-bottom: 0rem;"> ⚡<span class="main-header">Smart Object Detection</span></div>', unsafe_allow_html=True)
st.markdown('<h4 class="sub-header">Powered by YOLOv8 — upload an image or video and let the model find what\'s in it.</h4>', unsafe_allow_html=True)


# Model loading
# cache_resource so the model only loads once not on every single interaction
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"Could not load YOLO model: {e}")
        return None


model = load_model()


if model is None:
    st.stop()


NAME_TO_ID = {name: class_id for class_id, name in model.names.items()}


CLASS_COLORS = {}


def get_color_for_class(class_id):
    
    if class_id not in CLASS_COLORS:
        np.random.seed(class_id)  
        color = tuple(int(c) for c in np.random.randint(50, 255, size=3))
        CLASS_COLORS[class_id] = color
    return CLASS_COLORS[class_id]


def bgr_to_hex(color_bgr):
    b, g, r = color_bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_detections(image_bgr, results, model_names):
    # draws boxes + labels manually so we control color per class and text visibility
    annotated = image_bgr.copy()
    detections_info = []

    # 🧠 PRO HACK: Dynamic scaling based on image resolution!
    # This ensures boxes/text are visible whether the image is 480p or 4K
    height, width = image_bgr.shape[:2]
    scale = min(width, height) / 800.0  # Base scale relative to a standard 800px image
    
    box_thickness = max(2, int(3 * scale))
    font_scale = max(0.6, 0.9 * scale)
    font_thickness = max(1, int(2 * scale))
    bg_padding = max(5, int(10 * scale))

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model_names[class_id]
        color = get_color_for_class(class_id)

        # Draw the thicker bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        # Calculate text size for the background label
        label = f"{class_name} {confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Draw the label background rectangle (so text is readable on any background)
        cv2.rectangle(annotated, (x1, y1 - text_h - bg_padding), (x1 + text_w + bg_padding, y1), color, -1)
        
        # Draw the bold text
        cv2.putText(
            annotated, label, (x1 + int(bg_padding/2), y1 - int(bg_padding/3)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
        )

        detections_info.append({
            "class": class_name,
            "confidence": confidence,
            "color_hex": bgr_to_hex(color),
        })

    return annotated, detections_info


# Sidebar controls
st.sidebar.markdown("### ⚙️ Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05,
    help="Detections below this confidence get filtered out."
)

st.sidebar.markdown("### 📥 Input Type")
mode = st.sidebar.radio("Choose input type", ["Image", "Video"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Model Info")
st.sidebar.markdown(f"""
- **Model:** YOLOv8 Nano
- **Classes:** {len(model.names)} (COCO dataset)
- **Threshold:** {confidence_threshold}
""")


# Image mode
if mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # read uploaded file into a PIL image then convert to OpenCV BGR format
            pil_image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(pil_image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            with st.spinner("Running detection..."):
                results = model(image_bgr, conf=confidence_threshold, verbose=False)[0]
                annotated_bgr, detections = draw_detections(image_bgr, results, model.names)

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.image(pil_image, use_container_width=True)
            with col2:
                st.markdown("**Detected Objects**")
                st.image(annotated_rgb, use_container_width=True)

            # show detection details
            if len(detections) == 0:
                st.warning("No objects detected above this confidence threshold. Try lowering it.")
            else:
                # quick stats row  count average confidence unique classes
                avg_conf = sum(d["confidence"] for d in detections) / len(detections)
                unique_classes = len(set(d["class"] for d in detections))

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value">{len(detections)}</div>
                        <div class="metric-label">Objects Found</div></div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value">{avg_conf:.0%}</div>
                        <div class="metric-label">Avg Confidence</div></div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value">{unique_classes}</div>
                        <div class="metric-label">Unique Classes</div></div>""", unsafe_allow_html=True)

                st.write("")  

                
                chips_html = ""
                for d in detections:
                    chips_html += (
                        f'<span class="detection-chip" style="background-color:{d["color_hex"]}">'
                        f'{d["class"]} · {d["confidence"]:.0%}</span>'
                    )
                st.markdown(chips_html, unsafe_allow_html=True)

                st.write("")

                
                detections_df = pd.DataFrame(detections)[["class", "confidence"]]
                detections_df["confidence"] = (detections_df["confidence"] * 100).round(1)

                chart_col, table_col = st.columns([3, 2])

                with chart_col:
                    st.markdown("**Class Distribution**")
                    
                    class_counts = detections_df["class"].value_counts().reset_index()
                    class_counts.columns = ["class", "count"]
                    fig = px.bar(
                        class_counts, x="class", y="count", color="class",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig.update_layout(
                        showlegend=False, height=280,
                        margin=dict(l=10, r=10, t=10, b=10),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with table_col:
                    st.markdown("**Detection Report**")
                    st.dataframe(detections_df, use_container_width=True, height=280)

                # per-class average confidence 
                with st.expander("📊 Model Confidence by Class"):
                    st.caption(
                        "These are the model's own confidence scores, not accuracy. "
                        "Real accuracy needs a labeled ground-truth dataset to compare against."
                    )
                    conf_summary = (
                        detections_df.groupby("class")["confidence"]
                        .agg(["mean", "min", "max", "count"])
                        .round(1)
                        .rename(columns={"mean": "avg_confidence", "count": "detections"})
                        .sort_values("avg_confidence", ascending=False)
                    )
                    st.dataframe(conf_summary, use_container_width=True)

                # CSV export of the raw detection data separate from the image download
                csv_bytes = detections_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📄 Download Detection Report (CSV)",
                    data=csv_bytes,
                    file_name="detections_" + Path(uploaded_file.name).stem + ".csv",
                    mime="text/csv",
                )

                # download button for the processed image
                result_pil = Image.fromarray(annotated_rgb)
                buffer = io.BytesIO()
                result_pil.save(buffer, format="PNG")
                st.download_button(
                    label="⬇️ Download Processed Image",
                    data=buffer.getvalue(),
                    file_name="detected_" + uploaded_file.name,
                    mime="image/png",
                )

        except Exception as e:
            st.error(f"Something went wrong while processing the image: {e}")


# Video mode (bonus challenge)
else:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        try:
          
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(uploaded_video.read())
            temp_input.close()

            cap = cv2.VideoCapture(temp_input.name)
            if not cap.isOpened():
                st.error("Could not open the uploaded video. Try a different file.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

             
                raw_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

                progress_bar = st.progress(0, text="Processing video frames...")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                frame_count = 0
                all_classes_seen = set()
                class_hit_counts = {}  # tracks how many frames each class showed up in
                class_conf_sums = {}   # running sum of confidence per class, for averaging later

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=confidence_threshold, verbose=False)[0]
                    annotated_frame, detections = draw_detections(frame, results, model.names)
                    writer.write(annotated_frame)

                    for d in detections:
                        all_classes_seen.add(d["class"])
                        class_hit_counts[d["class"]] = class_hit_counts.get(d["class"], 0) + 1
                        class_conf_sums[d["class"]] = class_conf_sums.get(d["class"], 0) + d["confidence"]

                    frame_count += 1
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()
                writer.release()
                progress_bar.empty()

                # re-encode the raw video to h264 with faststart so it actually plays
                # inline in the browser (imageio_ffmpeg bundles its own ffmpeg binary,
                
                temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

                with st.spinner("Finalizing video for playback..."):
                    encode_result = subprocess.run(
                        [
                            ffmpeg_path, "-y", "-i", raw_output_path,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart",
                            temp_output_path,
                        ],
                        capture_output=True,
                    )

                if encode_result.returncode != 0:
                    # ffmpeg failed for some reason - fall back to the raw file so the
                    # user at least gets a downloadable result even if it won't preview
                    st.warning("Couldn't optimize the video for browser playback, but the file below should still download and play in VLC/media players.")
                    temp_output_path = raw_output_path

                st.success(f"✅ Done — processed {frame_count} frames.")

                if all_classes_seen:
                    chips_html = "".join(
                        f'<span class="detection-chip" style="background-color:{bgr_to_hex(get_color_for_class(NAME_TO_ID[c]))}">{c}</span>'
                        for c in sorted(all_classes_seen)
                    )
                    st.markdown("**Classes seen across the video:**", unsafe_allow_html=True)
                    st.markdown(chips_html, unsafe_allow_html=True)

                    # class distribution chart same idea as the image mode one
                    # but here "count" means number of frames the class appeared in
                    counts_df = pd.DataFrame(
                        list(class_hit_counts.items()), columns=["class", "frame_count"]
                    )
                    # bring in the average confidence per class alongside the frame count
                    counts_df["avg_confidence"] = counts_df["class"].apply(
                        lambda c: round((class_conf_sums[c] / class_hit_counts[c]) * 100, 1)
                    )
                    counts_df = counts_df.sort_values("frame_count", ascending=False)

                    chart_col, table_col = st.columns([3, 2])
                    with chart_col:
                        st.markdown("**Class Distribution (by frame appearances)**")
                        fig = px.bar(
                            counts_df, x="class", y="frame_count", color="class",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                        )
                        fig.update_layout(
                            showlegend=False, height=280,
                            margin=dict(l=10, r=10, t=10, b=10),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with table_col:
                        st.markdown("**Detection Report**")
                        st.caption("avg_confidence = model's own confidence, not accuracy (no ground-truth labels to compare against).")
                        st.dataframe(counts_df, use_container_width=True, height=280)

                    csv_bytes = counts_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📄 Download Detection Report (CSV)",
                        data=csv_bytes,
                        file_name="detections_" + Path(uploaded_video.name).stem + ".csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No objects detected in this video above the current threshold.")

                
                vid_col1, vid_col2, vid_col3 = st.columns([1, 2, 1])
                with vid_col2:
                    st.video(temp_output_path)

                with open(temp_output_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=f.read(),
                        file_name="detected_" + uploaded_video.name,
                        mime="video/mp4",
                    )

        except Exception as e:
            st.error(f"Something went wrong while processing the video: {e}")


st.sidebar.markdown("---")
# Custom bold and visible caption for sidebar
st.sidebar.markdown(
    "<p style='font-weight: 600; font-size: 14px; color: #4B5563; line-height: 1.4;'>"
    "Day 27 - Object Detection with YOLOv8<br>"
    "Developed by Danish Ali<br>"
    "<span style='font-size: 12px; font-weight: 500; color: #6B7280;'>Machine Learning Bench Internship</span>"
    "</p>", 
    unsafe_allow_html=True
)


st.markdown("---")
# Custom bold and visible caption for main page
st.markdown(
    "<p style='text-align: center; font-weight: 600; font-size: 14px; color: #4B5563;'> Day 27 Mini Project  · Built with YOLOv8 + Streamlit</p>", 
    unsafe_allow_html=True
)