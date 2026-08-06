import plotly.express as px
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="PPE Object Detection Dashboard", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
    index=0
)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.write(f"**Model:** {model_choice}")
st.sidebar.write("**Trained on:** COCO Dataset (80 classes)")
st.sidebar.write("**Framework:** Ultralytics YOLO11")


@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)


model = load_model(model_choice)

# ---------------- Main Page ----------------
st.title("🎯 Object Detection Dashboard")
st.write("Upload an image or video to run object detection using YOLO.")

file_type = st.radio("Select input type:", ["Image", "Video"], horizontal=True)


def get_detection_dataframe(result, model):
    """Turns detection results into a table for display."""
    data = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        data.append({
            "Class": model.names[cls_id],
            "Confidence": round(float(box.conf[0]), 2)
        })
    return pd.DataFrame(data)


# ---------------- Img MODE -------=====---------
if file_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        if st.button("Run Detection", type="primary"):
            with st.spinner("Running detection..."):
                
                img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                results = model.predict(source=img_array, conf=confidence_threshold)
                result = results[0]
                annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("Detected Objects")
                st.image(annotated, use_container_width=True)

            df = get_detection_dataframe(result, model)

            st.markdown("---")

            # summary metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Objects", len(df))
            m2.metric("Unique Classes", df["Class"].nunique() if not df.empty else 0)
            m3.metric("Avg Confidence", f"{df['Confidence'].mean():.2f}" if not df.empty else "N/A")

            if not df.empty:
                st.markdown("### Detection Details")
                st.dataframe(df, use_container_width=True)

                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    st.markdown("### Objects by Class")
                    class_counts = df["Class"].value_counts().reset_index()
                    class_counts.columns = ["Class", "Count"]

                    fig1 = px.bar(
                        class_counts,
                        x="Class",
                        y="Count",
                        color="Class",
                        text="Count"
                    )
                    fig1.update_traces(textposition="outside")
                    fig1.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig1, use_container_width=True)

                with col_chart2:
                    st.markdown("### Confidence by Detection")
                    fig2 = px.bar(
                        df,
                        x=df.index,
                        y="Confidence",
                        color="Class",
                        text="Confidence",
                        labels={"x": "Detection #"}
                    )
                    fig2.update_traces(textposition="outside")
                    fig2.update_layout(height=350, yaxis_range=[0, 1])
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("### Class Distribution")
                fig3 = px.pie(
                    class_counts,
                    names="Class",
                    values="Count",
                    hole=0.4
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("No objects detected. Try lowering the confidence threshold.")

            # download button
            output_image = Image.fromarray(annotated)
            temp_path = os.path.join(tempfile.gettempdir(), "result.jpg")
            output_image.save(temp_path)

            with open(temp_path, "rb") as f:
                st.download_button("⬇️ Download Result Image", f, file_name="detected_image.jpg")

# ---------------- VIDEO MODE ----======--------------
else:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_input = os.path.join(tempfile.gettempdir(), "input_video.mp4")
        with open(temp_input, "wb") as f:
            f.write(uploaded_video.read())

        st.video(temp_input)

        if st.button("Run Detection on Video", type="primary"):
            with st.spinner("Processing video, this may take a moment..."):
                results = model.predict(
                    source=temp_input,
                    conf=confidence_threshold,
                    save=True,
                    project=tempfile.gettempdir(),
                    name="video_output",
                    exist_ok=True
                )

                # collect class counts across all frames for a summary chart
                all_classes = []
                for r in results:
                    for box in r.boxes:
                        all_classes.append(model.names[int(box.cls[0])])

                output_dir = os.path.join(tempfile.gettempdir(), "video_output")
                output_files = [f for f in os.listdir(output_dir) if f.endswith((".mp4", ".avi"))]

            if output_files:
                st.success("Detection complete!")

                if all_classes:
                    st.markdown("### Objects Detected Across Video")
                    video_counts = pd.Series(all_classes).value_counts().reset_index()
                    video_counts.columns = ["Class", "Count"]

                    fig_video = px.bar(
                        video_counts,
                        x="Class",
                        y="Count",
                        color="Class",
                        text="Count"
                    )
                    fig_video.update_traces(textposition="outside")
                    fig_video.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_video, use_container_width=True)

                output_path = os.path.join(output_dir, output_files[0])
                with open(output_path, "rb") as f:
                    st.download_button("⬇️ Download Result Video", f, file_name="detected_video.mp4")
            else:
                st.error("Something went wrong processing the video.")