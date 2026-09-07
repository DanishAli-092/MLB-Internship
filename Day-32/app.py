# Day 32 Smart Parking Monitoring System streamlit app upload video pick variant run analysis and download output


import os
import time
import json
import tempfile

import cv2
import streamlit as st

from parking_detection import (
    load_model,
    load_aerial_model,
    load_parking_spaces,
    generate_grid_spaces,
    detect_vehicles,
    check_occupancy,
    draw_vehicle_boxes,
    draw_parking_spaces,
    VEHICLE_CLASS_IDS,
    AERIAL_CLASS_IDS,
)
from analytics import (
    OccupancyTracker,
    calculate_occupancy_percentage,
    draw_basic_stats,
    draw_analytics_panel,
)

st.set_page_config(page_title="Smart Parking Monitoring System", page_icon="🅿️", layout="wide")

MODEL_PATH = "yolov8n.pt"


# loads and caches the normal yolo model
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


# loads and caches the aerial model
@st.cache_resource
def get_aerial_model():
    return load_aerial_model()


# grabs one frame from video at given second for preview
def grab_reference_frame(video_path, second=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# builds parking space layout from json upload manual grid or auto detect
def get_layout(ref_frame, uploaded_layout, layout_mode, use_manual, grid_rows, grid_cols, x_range, y_range, x_nudge, y_nudge, brightness_thresh):
    if uploaded_layout is not None:
        try:
            uploaded_layout.seek(0)
            data = json.load(uploaded_layout)
        except Exception as e:
            return None, f"Could not read layout JSON: {e}"
        if not data:
            return None, "Uploaded layout JSON is empty."
        return data, None

    width, height = ref_frame.shape[1], ref_frame.shape[0]
    if use_manual:
        spaces_data = generate_grid_spaces(
            width, height, rows=grid_rows, cols=grid_cols,
            x_start_frac=x_range[0] / 100, x_end_frac=x_range[1] / 100,
            y_start_frac=y_range[0] / 100, y_end_frac=y_range[1] / 100,
            x_nudge=x_nudge, y_nudge=y_nudge,
        )
        return spaces_data, None

    from parking_detection import detect_columns_in_region 
    
    spaces_data = detect_columns_in_region(
        ref_frame,
        x_start_frac=x_range[0] / 100, x_end_frac=x_range[1] / 100,
        y_start_frac=y_range[0] / 100, y_end_frac=y_range[1] / 100,
        brightness_thresh=brightness_thresh,
    )
    if not spaces_data:
        return None, (
            "No reliable vertical dividers found in this region. Widen/move "
            "the region, lower the brightness threshold, or switch to "
            "manual grid / JSON upload for this band."
        )
    return spaces_data, None


# runs full video processing loop detection occupancy drawing and writing output
def process_video(input_path, output_path, variant, conf_threshold, overlap_threshold,
                  progress_bar, live_stats_placeholder, spaces_data, use_aerial_model=False):
    model = get_aerial_model() if use_aerial_model else get_model()
    class_ids = AERIAL_CLASS_IDS if use_aerial_model else VEHICLE_CLASS_IDS
    imgsz = 1280 if use_aerial_model else None
    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_shape = (height, width)

    confirm_frames = max(int(round(fps * 0.5)), 3)
    spaces = load_parking_spaces(default_spaces=spaces_data)

    out_width = width + 260 if variant == "Parking Analytics" else width

    import imageio_ffmpeg
    writer = imageio_ffmpeg.write_frames(
        output_path,
        (out_width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
    )
    writer.send(None)

    tracker = OccupancyTracker()
    frame_number = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1

        detections = detect_vehicles(model, frame, conf_threshold, class_ids=class_ids, imgsz=imgsz)
        spaces = check_occupancy(spaces, detections, frame_shape, overlap_threshold,
                                 frame_number=frame_number, confirm_frames=confirm_frames)

        occupied = sum(1 for s in spaces if s.occupied)
        total = len(spaces)
        available = total - occupied
        pct = tracker.update(occupied, total, detections)

        if variant == "Parking Monitor":
            frame = draw_parking_spaces(frame, spaces, style="simple")
            frame = draw_vehicle_boxes(frame, detections)
            frame = draw_basic_stats(frame, total, occupied, available)
        else:
            frame = draw_parking_spaces(frame, spaces, style="outline")
            frame = draw_vehicle_boxes(frame, detections)
            frame = draw_analytics_panel(
                frame, total, occupied, available, pct,
                tracker.history, frame_number, fps, tracker.unique_vehicle_count()
            )

        writer.send(frame.tobytes())

        if frame_number % 5 == 0:
            progress_bar.progress(min(frame_number / total_frames, 1.0))
            live_stats_placeholder.markdown(
                f"**Live stats** — Frame {frame_number}/{total_frames} &nbsp;|&nbsp; "
                f"Total: {total} &nbsp;|&nbsp; Occupied: {occupied} &nbsp;|&nbsp; "
                f"Available: {available} &nbsp;|&nbsp; Occupancy: {pct}%"
            )

    cap.release()
    writer.close()
    progress_bar.progress(1.0)


# main streamlit ui entry point sidebar settings upload run and download
def main():
    st.title("🅿️ Smart Parking Monitoring System")
    st.caption("Day 32 - Upload a parking lot video, detect vehicles, and analyze occupancy.")

    with st.sidebar:
        st.header("Settings")
        variant = st.selectbox("Demo Variant", ["Parking Monitor", "Parking Analytics"])
        detection_model = st.selectbox(
            "Detection model",
            ["yolov8n (fast)", "Aerial/drone specialized pretrained model"],
            help="Use Geo-trax if a normal model misses vehicles in your "
                 "top-down/drone footage (e.g. buses or trucks at odd angles). "
                 "It's slower and downloads extra weights on first use."
        )
        use_aerial_model = detection_model.startswith("Aerial")
        conf_threshold = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
        overlap_threshold = st.slider("Occupancy overlap threshold", 0.1, 0.9, 0.3, 0.05)

        st.markdown("---")

        layout_mode = st.radio(
            "How to define parking spaces",
            ["Auto-detect columns (per row/band)", "Manual equal-width grid", "Upload saved layout JSON"],
        )

        uploaded_layout = None
        use_manual = False
        grid_rows = grid_cols = 1
        x_range = y_range = (0, 100)
        x_nudge = y_nudge = 0
        brightness_thresh = 170

        if layout_mode == "Upload saved layout JSON":
            uploaded_layout = st.file_uploader(
                "Saved parking-space layout (JSON)", type=["json"]
            )
        elif layout_mode == "Manual equal-width grid":
            use_manual = True
            grid_rows = st.number_input("Rows", min_value=1, max_value=10, value=2)
            grid_cols = st.number_input("Columns", min_value=1, max_value=30, value=5)
            x_range = st.slider("Horizontal region (% of frame)", 0, 100, (0, 100))
            y_range = st.slider("Vertical region (% of frame)", 0, 100, (0, 100))
            x_nudge = st.slider("Nudge grid left/right (px)", -100, 100, 0, 2)
            y_nudge = st.slider("Nudge grid up/down (px)", -100, 100, 0, 2)
        else:
            st.caption(
                "Set the region for ONE row/band of stalls, detect it, add it to "
                "the layout, then repeat for the next row/band (e.g. top row, "
                "then bottom row) before running analysis."
            )
            x_range = st.slider("Horizontal region (% of frame)", 0, 100, (0, 100))
            y_range = st.slider("Vertical region - top/bottom of THIS row (% of frame)", 0, 100, (0, 40))
            brightness_thresh = st.slider("Line brightness threshold", 100, 255, 170, 5)

        st.markdown("---")
        st.markdown(
            "**Parking Monitor**: spaces + occupied/free status + basic counts.\n\n"
            "**Parking Analytics**: occupancy %, utilization level, mini trend graph, timestamps."
        )

    uploaded_file = st.file_uploader("Upload a parking lot video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        tmp_dir = tempfile.mkdtemp()
        input_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(input_path)

        preview_second = st.slider("Preview frame at (seconds)", 0.0, 15.0, 1.0, 0.5)

        layout_fingerprint = (
            uploaded_layout.name, uploaded_layout.size
        ) if uploaded_layout is not None else None

        grid_settings_fingerprint = (
            uploaded_file.name, preview_second, use_manual,
            grid_rows, grid_cols, x_range, y_range, x_nudge, y_nudge,
            layout_fingerprint,
        )

        if "accumulated_spaces" not in st.session_state:
            st.session_state["accumulated_spaces"] = []

        col1, col2 = st.columns(2)
        with col1:
            add_clicked = st.button("Detect + add this region/band")
        with col2:
            clear_clicked = st.button("Clear layout")

        if clear_clicked:
            st.session_state["accumulated_spaces"] = []

        if add_clicked and layout_mode != "Upload saved layout JSON":
            ref_frame = grab_reference_frame(input_path, second=preview_second)
            if ref_frame is None:
                st.error("Could not grab a frame at that timestamp.")
            else:
                new_spaces, err = get_layout(
                    ref_frame, uploaded_layout, layout_mode, use_manual, grid_rows, grid_cols,
                    x_range, y_range, x_nudge, y_nudge, brightness_thresh,
                )
                if err:
                    st.error(err)
                else:
                    existing = st.session_state["accumulated_spaces"]
                    start_id = len(existing) + 1
                    for i, s in enumerate(new_spaces):
                        s["id"] = start_id + i
                    st.session_state["accumulated_spaces"] = existing + new_spaces
                    st.success(f"Added {len(new_spaces)} spaces - total now {len(st.session_state['accumulated_spaces'])}")

        if layout_mode == "Upload saved layout JSON" and uploaded_layout is not None:
            final_spaces, err = get_layout(None, uploaded_layout, layout_mode, use_manual,
                                           grid_rows, grid_cols, x_range, y_range,
                                           x_nudge, y_nudge, brightness_thresh)
            if err:
                st.error(err)
            else:
                st.session_state["accumulated_spaces"] = final_spaces

        spaces_data = st.session_state["accumulated_spaces"] if st.session_state["accumulated_spaces"] else None

        if spaces_data:
            ref_frame_preview = grab_reference_frame(input_path, second=preview_second)
            if ref_frame_preview is not None:
                spaces = load_parking_spaces(default_spaces=spaces_data)
                preview_img = draw_parking_spaces(ref_frame_preview.copy(), spaces, style="simple")
                st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB),
                         caption=f"Layout so far: {len(spaces_data)} parking spaces - check before running")
                st.download_button(
                    "Save this layout as JSON (reuse next time for this camera angle)",
                    data=json.dumps(spaces_data, indent=2),
                    file_name="parking_layout.json",
                    mime="application/json",
                )

        if st.button("Run Analysis", type="primary"):
            err = None
            if not spaces_data:
                err = "No parking spaces defined. Please detect and add at least one region, or upload a JSON."

            if err:
                st.error(err)
            else:
                output_path = os.path.join(tmp_dir, f"output_{int(time.time())}.mp4")
                progress_bar = st.progress(0)
                live_stats_placeholder = st.empty()

                with st.spinner("Processing video, this can take a while for longer clips..."):
                    process_video(input_path, output_path, variant, conf_threshold,
                                  overlap_threshold, progress_bar, live_stats_placeholder,
                                  spaces_data, use_aerial_model=use_aerial_model)

                st.success("Done! Processed video is ready below.")
                st.video(output_path)

                variant_folder = "variant-1" if variant == "Parking Monitor" else "variant-2"
                saved_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", variant_folder)
                os.makedirs(saved_dir, exist_ok=True)
                saved_path = os.path.join(saved_dir, f"output_{int(time.time())}.mp4")
                with open(output_path, "rb") as src, open(saved_path, "wb") as dst:
                    dst.write(src.read())
                st.caption(f"Also saved to `{os.path.relpath(saved_path)}`")

                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download processed video",
                        data=f,
                        file_name=f"parking_{variant.replace(' ', '_').lower()}.mp4",
                        mime="video/mp4",
                    )
    else:
        st.info("Upload a video to get started. Sample videos can be placed in sample_videos/.")


if __name__ == "__main__":
    main()