import json
import math
import os
import tempfile
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import imageio_ffmpeg

from tracker import CentroidTracker
from traffic_violation import (
    VehicleDetector,
    DirectionAnalyzer,
    RestrictedZone,
    ViolationRecorder,
    draw_track_box,
    draw_direction_arrow,
    draw_normal_direction_indicator,
    draw_violation_counter,
)
from analytics import TrafficAnalytics


# reads zone points from uploaded json and scales them to video size
def load_zone_from_json(uploaded_json, target_width: int, target_height: int) -> Optional[List[Tuple[int, int]]]:
    try:
        data = json.load(uploaded_json)
        raw_points = data["zone_points"]
        saved_w = data.get("frame_width", target_width)
        saved_h = data.get("frame_height", target_height)

        scale_x = target_width / saved_w if saved_w else 1.0
        scale_y = target_height / saved_h if saved_h else 1.0

        return [(int(x * scale_x), int(y * scale_y)) for x, y in raw_points]
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        st.warning(f"Could not read zone file, using default zone instead. ({exc})")
        return None


# gives a default zone in bottom right if user didnt set one
def get_default_zone(frame_width: int, frame_height: int) -> List[Tuple[int, int]]:
    w, h = frame_width, frame_height
    return [
        (int(w * 0.55), int(h * 0.55)),
        (int(w * 0.95), int(h * 0.55)),
        (int(w * 0.95), int(h * 0.95)),
        (int(w * 0.55), int(h * 0.95)),
    ]


# builds zone rectangle from slider percent values
def get_zone_from_sliders(frame_width: int, frame_height: int,
                           x_pct: float, y_pct: float,
                           width_pct: float, height_pct: float) -> List[Tuple[int, int]]:
    x1 = int(frame_width * (x_pct / 100.0))
    y1 = int(frame_height * (y_pct / 100.0))
    x2 = int(frame_width * (min(x_pct + width_pct, 100.0) / 100.0))
    y2 = int(frame_height * (min(y_pct + height_pct, 100.0) / 100.0))

    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


# keeps a percent value inside zero to hundred range
def clamp_pct(value: float) -> float:
    return max(0.0, min(100.0, value))


# gives starting circle layout for polygon point sliders
def get_default_polygon_points(num_points: int) -> List[Tuple[int, int]]:
    center_x, center_y = 55, 55
    radius = 20
    points = []
    for i in range(num_points):
        angle = (2 * math.pi * i / num_points) - (math.pi / 2)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        points.append((int(round(x)), int(round(y))))
    return points


# builds polygon zone from list of percent points
def get_polygon_from_sliders(frame_width: int, frame_height: int,
                              points_pct: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    return [(int(frame_width * (x / 100.0)), int(frame_height * (y / 100.0)))
            for x, y in points_pct]


# builds thin rotated rectangle to act as a line zone
def get_line_zone_from_sliders(frame_width: int, frame_height: int,
                                x1_pct: float, y1_pct: float,
                                x2_pct: float, y2_pct: float,
                                thickness_pct: float) -> List[Tuple[int, int]]:
    x1 = frame_width * (x1_pct / 100.0)
    y1 = frame_height * (y1_pct / 100.0)
    x2 = frame_width * (x2_pct / 100.0)
    y2 = frame_height * (y2_pct / 100.0)

    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)

    reference_dim = min(frame_width, frame_height)
    half_thick = max(2.0, reference_dim * (thickness_pct / 100.0) / 2.0)

    if length < 1e-6:
        return [
            (int(x1 - half_thick), int(y1 - half_thick)),
            (int(x1 + half_thick), int(y1 - half_thick)),
            (int(x1 + half_thick), int(y1 + half_thick)),
            (int(x1 - half_thick), int(y1 + half_thick)),
        ]

    ux, uy = -dy / length, dx / length
    p1 = (x1 + ux * half_thick, y1 + uy * half_thick)
    p2 = (x2 + ux * half_thick, y2 + uy * half_thick)
    p3 = (x2 - ux * half_thick, y2 - uy * half_thick)
    p4 = (x1 - ux * half_thick, y1 - uy * half_thick)

    return [(int(p[0]), int(p[1])) for p in (p1, p2, p3, p4)]


# grabs first frame of video for zone preview
def get_first_frame(video_bytes: bytes) -> Optional[np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
        tmp_f.write(video_bytes)
        tmp_path = tmp_f.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    finally:
        os.remove(tmp_path)


# draws stats panel above frame for dashboard variant
def draw_dashboard_overlay(frame: np.ndarray, analytics: TrafficAnalytics,
                            recorder: ViolationRecorder) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_height = 110
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    summary = analytics.as_summary_dict()
    line1 = (f"Total Vehicles: {summary['total_vehicles']}   |   "
             f"Total Violations: {summary['total_violations']}")
    line2 = (f"Wrong-Way: {summary['wrong_way_violations']}   |   "
             f"Restricted Zone: {summary['restricted_zone_violations']}")

    vt = summary["vehicle_type_counts"]
    line3 = "  ".join(f"{k}: {v}" for k, v in vt.items()) if vt else "No vehicles yet"

    cv2.putText(panel, line1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(panel, line2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(panel, line3, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return np.vstack([panel, frame])


# main pipeline that does detection tracking and violation check
def run_traffic_analysis(video_path: str, output_path: str, variant: str,
                          normal_angle: float, zone_points: List[Tuple[int, int]] = None,
                          confidence: float = 0.35, model_path: str = "yolov8n.pt",
                          progress_callback=None):
    detector = VehicleDetector(model_path=model_path, confidence=confidence)
    tracker = CentroidTracker(max_missed=15, max_distance=120.0)
    direction_analyzer = DirectionAnalyzer(normal_direction_angle=normal_angle)
    recorder = ViolationRecorder()
    analytics = TrafficAnalytics()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    if zone_points is None:
        zone_points = get_default_zone(frame_width, frame_height)
    zone = RestrictedZone(zone_points)

    out_height = frame_height + (110 if variant == "dashboard" else 0)

    writer = imageio_ffmpeg.write_frames(
        output_path,
        (frame_width, out_height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
    )
    writer.send(None)

    known_track_ids = set()
    frame_number = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1

        raw_detections = detector.detect(frame)
        detections_for_tracker = [(box, cls) for box, cls, _ in raw_detections]
        tracks = tracker.update(detections_for_tracker)

        for track_id, track in tracks.items():
            if track_id not in known_track_ids:
                known_track_ids.add(track_id)
                analytics.register_vehicle(track.class_name)

        if variant == "wrong_way":
            draw_normal_direction_indicator(frame, normal_angle)

        for track_id, track in tracks.items():
            is_wrong_way, angle = direction_analyzer.is_wrong_way(track)
            in_zone = zone.contains_point(track.current_centroid)

            is_violation = False
            timestamp_sec = frame_number / fps

            if is_wrong_way:
                newly_recorded = recorder.record(track_id, track.class_name, "wrong_way",
                                                   frame_number, timestamp_sec)
                if newly_recorded:
                    analytics.add_violation(recorder.events[-1])
                is_violation = True

            if in_zone:
                newly_recorded = recorder.record(track_id, track.class_name, "restricted_zone",
                                                   frame_number, timestamp_sec)
                if newly_recorded:
                    analytics.add_violation(recorder.events[-1])
                is_violation = True

            draw_track_box(frame, track, is_violation)
            if variant == "wrong_way":
                draw_direction_arrow(frame, track, angle)

        zone.draw(frame)

        if variant == "wrong_way":
            draw_violation_counter(frame, recorder.total_violations())
            output_frame = frame
        else:
            output_frame = draw_dashboard_overlay(frame, analytics, recorder)

        writer.send(np.ascontiguousarray(output_frame, dtype=np.uint8).tobytes())

        if progress_callback and total_frames > 0:
            progress_callback(min(1.0, frame_number / total_frames))

    cap.release()
    writer.close()

    return analytics


st.set_page_config(page_title="Traffic Violation Monitoring System", page_icon="🚓", layout="wide")

st.title("🚓 Traffic Violation Monitoring System")
st.markdown("Day 36 · Vehicle Detection, Tracking, Direction & Restricted-Zone Violation Analytics")

with st.sidebar:
    st.header("⚙️ Settings")
    variant_label = st.selectbox(
        "Select Analysis Mode",
        ["Wrong-Way Detection", "Traffic Violation Analytics"],
    )
    variant = "wrong_way" if variant_label == "Wrong-Way Detection" else "dashboard"

    normal_angle = st.slider(
        "Normal Traffic Direction (degrees)",
        min_value=-180, max_value=180, value=0, step=5,
        help="0 = moving right, 90 = moving down, 180 = moving left, -90 = moving up",
    )

    confidence = st.slider("Detection Confidence Threshold", 0.1, 0.9, 0.35, 0.05)

    st.markdown("---")
    st.markdown("**Restricted Zone**")
    st.caption(
        "Choose a zone shape and control every part of it with sliders - "
        "no clicking or drawing on the video frame needed."
    )

    zone_shape = st.selectbox(
        "Zone Shape",
        ["Rectangle", "Polygon", "Line", "Custom (upload zone.json)"],
        index=0,
    )

    zone_json_file = None
    polygon_points_pct: List[Tuple[float, float]] = []

    if zone_shape == "Rectangle":
        zone_x_pct = st.slider("Zone X Position (%)", 0, 100, 55, 1)
        zone_y_pct = st.slider("Zone Y Position (%)", 0, 100, 55, 1)
        zone_w_pct = st.slider("Zone Width (%)", 5, 100, 40, 1)
        zone_h_pct = st.slider("Zone Height (%)", 5, 100, 40, 1)

    elif zone_shape == "Polygon":
        num_points = st.slider("Number of Points", 3, 8, 4, 1,
                                help="How many corners the zone should have")
        default_points = get_default_polygon_points(num_points)

        for i in range(num_points):
            st.caption(f"Point {i + 1}")
            col_x, col_y = st.columns(2)
            default_x, default_y = default_points[i]
            x_pct = col_x.slider(f"X% (P{i + 1})", 0, 100, default_x, 1, key=f"poly_x_{i}")
            y_pct = col_y.slider(f"Y% (P{i + 1})", 0, 100, default_y, 1, key=f"poly_y_{i}")
            polygon_points_pct.append((x_pct, y_pct))

        st.markdown("**Move Whole Shape**")
        poly_shift_x_pct = st.slider("Shift X (%)", -50, 50, 0, 1,
                                      help="Moves all points together left/right")
        poly_shift_y_pct = st.slider("Shift Y (%)", -50, 50, 0, 1,
                                      help="Moves all points together up/down")
        polygon_points_pct = [
            (clamp_pct(x + poly_shift_x_pct), clamp_pct(y + poly_shift_y_pct))
            for x, y in polygon_points_pct
        ]

    elif zone_shape == "Line":
        line_x1_pct = st.slider("Start X (%)", 0, 100, 20, 1)
        line_y1_pct = st.slider("Start Y (%)", 0, 100, 55, 1)
        line_x2_pct = st.slider("End X (%)", 0, 100, 80, 1)
        line_y2_pct = st.slider("End Y (%)", 0, 100, 55, 1)
        line_thickness_pct = st.slider(
            "Line Thickness (%)", 1, 20, 4, 1,
            help="A line has no 'inside', so it's drawn as a thin band "
                 "this many % of the frame wide",
        )

        st.markdown("**Move Whole Line**")
        line_shift_x_pct = st.slider("Shift X (%)", -50, 50, 0, 1,
                                      help="Moves both endpoints together left/right")
        line_shift_y_pct = st.slider("Shift Y (%)", -50, 50, 0, 1,
                                      help="Moves both endpoints together up/down")
        line_x1_pct = clamp_pct(line_x1_pct + line_shift_x_pct)
        line_y1_pct = clamp_pct(line_y1_pct + line_shift_y_pct)
        line_x2_pct = clamp_pct(line_x2_pct + line_shift_x_pct)
        line_y2_pct = clamp_pct(line_y2_pct + line_shift_y_pct)

    else:
        zone_json_file = st.file_uploader(
            "Upload zone.json (overrides sliders above)",
            type=["json"],
            help=(
                "For a shape none of the slider options cover, draw it locally "
                "with `python zone_selector.py --video your_video.mp4` (click-based), "
                "then upload the resulting zone.json here."
            ),
        )


# picks the right zone builder based on selected shape
def compute_zone_points(frame_width: int, frame_height: int) -> Optional[List[Tuple[int, int]]]:
    if zone_shape == "Rectangle":
        return get_zone_from_sliders(frame_width, frame_height,
                                      zone_x_pct, zone_y_pct, zone_w_pct, zone_h_pct)
    elif zone_shape == "Polygon":
        return get_polygon_from_sliders(frame_width, frame_height, polygon_points_pct)
    elif zone_shape == "Line":
        return get_line_zone_from_sliders(frame_width, frame_height,
                                           line_x1_pct, line_y1_pct,
                                           line_x2_pct, line_y2_pct, line_thickness_pct)
    else:
        if zone_json_file is None:
            return None
        points = load_zone_from_json(zone_json_file, frame_width, frame_height)
        zone_json_file.seek(0)
        return points


uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    st.subheader("🎬 Uploaded Video Preview")
    st.video(uploaded_file.getvalue())

    first_frame = get_first_frame(uploaded_file.getvalue())
    if first_frame is not None:
        preview_h, preview_w = first_frame.shape[:2]
        preview_zone_points = compute_zone_points(preview_w, preview_h)

        if preview_zone_points:
            preview_frame = first_frame.copy()
            RestrictedZone(preview_zone_points).draw(preview_frame)
            st.subheader("🟥 Restricted Zone Preview")
            st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB),
                      caption="This zone will be applied to every frame of the video",
                      use_container_width=True)

col_run, col_info = st.columns([1, 3])
run_clicked = col_run.button("▶ Run Analysis", type="primary", disabled=uploaded_file is None)

if uploaded_file is not None and run_clicked:
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "input_video.mp4")
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        output_path = os.path.join(tmp_dir, f"processed_{variant}.mp4")

        probe_cap = cv2.VideoCapture(input_path)
        vid_w = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe_cap.release()

        custom_zone_points = compute_zone_points(vid_w, vid_h)
        if custom_zone_points:
            st.caption(f"Using {zone_shape} restricted zone: {custom_zone_points}")

        progress_bar = st.progress(0.0, text="Starting analysis...")

        def update_progress(fraction: float) -> None:
            progress_bar.progress(fraction, text=f"Processing video... {int(fraction * 100)}%")

        start_time = time.time()
        analytics = run_traffic_analysis(
            video_path=input_path,
            output_path=output_path,
            variant=variant,
            normal_angle=float(normal_angle),
            zone_points=custom_zone_points,
            confidence=confidence,
            progress_callback=update_progress,
        )
        elapsed = time.time() - start_time
        progress_bar.progress(1.0, text=f"Done in {elapsed:.1f}s")

        st.success("Analysis complete!")

        st.subheader("📹 Processed Video")
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)

        st.download_button(
            "⬇ Download Processed Video",
            data=video_bytes,
            file_name=f"traffic_violation_{variant}.mp4",
            mime="video/mp4",
        )

        summary = analytics.as_summary_dict()

        st.subheader("📊 Violation Statistics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Vehicles", summary["total_vehicles"])
        m2.metric("Total Violations", summary["total_violations"])
        m3.metric("Wrong-Way Violations", summary["wrong_way_violations"])
        m4.metric("Restricted Zone Violations", summary["restricted_zone_violations"])

        if summary["vehicle_type_counts"]:
            st.markdown("**Vehicle Type Breakdown**")
            st.bar_chart(pd.Series(summary["vehicle_type_counts"]))

        st.info(f"**Traffic Status:** {summary['status_summary']}")

        events = analytics.events_table()
        if events:
            st.markdown("**Violation Events by Vehicle ID**")
            st.dataframe(pd.DataFrame(events), use_container_width=True)
        else:
            st.markdown("No violations were recorded in this video.")

elif uploaded_file is None:
    st.info("Upload a traffic video from the field above to get started.")