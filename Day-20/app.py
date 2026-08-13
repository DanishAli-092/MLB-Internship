import streamlit as st
import cv2
import tempfile
import os

st.set_page_config(page_title="Video Edge Processor", layout="centered")

st.title("🎥 Video Processing Engine")
st.write(
    "Upload a video to apply Grayscale → Gaussian Blur → Canny Edge Detection, "
    "frame by frame."
)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

low_thresh = st.slider("Canny Lower Threshold", 0, 255, 100)
high_thresh = st.slider("Canny Upper Threshold", 0, 255, 200)


def get_video_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer


if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    input_path = tfile.name

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.write(f"**FPS:** {fps:.2f} | **Resolution:** {width}x{height} | **Frames:** {total_frames}")

    st.subheader("Original Video")
    st.video(input_path)

    output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
    out = get_video_writer(output_path, fps, width, height)

    if st.button("Process Video"):
        progress_bar = st.progress(0)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, low_thresh, high_thresh)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            out.write(edges_bgr)

            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        out.release()

        st.success("Processing complete!")
        st.subheader("Processed Video")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "Download Processed Video", f, file_name="processed_video.mp4"
            )

st.markdown("---")
st.subheader("📸 Example Results")

st.write("Sample frames showing original vs processed output:")

example_col1, example_col2 = st.columns(2)
with example_col1:
    if os.path.exists("examples/original_sample.jpg"):
        st.image("examples/original_sample.jpg", caption="Original Frame")
    else:
        st.info("Original sample image not found.")
with example_col2:
    if os.path.exists("examples/processed_sample.jpg"):
        st.image("examples/processed_sample.jpg", caption="Processed Frame (Edges)")
    else:
        st.info("Processed sample image not found.")