# Day 32 Parking Detection Module core logic for space definition vehicle detection and occupancy check

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# coco class ids we care about for normal vehicles
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# class ids for aerial drone model kept unchanged
AERIAL_CLASS_IDS = {0: "car", 1: "bus", 2: "truck", 3: "motorcycle"}


# class for single parking space polygon with hysteresis state
class ParkingSpace:
    def __init__(self, space_id, points):
        self.space_id = space_id
        self.points = np.array(points, dtype=np.int32)
        self.occupied = False
        self.pending_state = False
        self.pending_frames = 0
        self.last_change_frame = 0

    # returns center point of the polygon
    def get_center(self):
        M = cv2.moments(self.points)
        if M["m00"] == 0:
            return tuple(self.points.mean(axis=0).astype(int))
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    # returns area of the polygon
    def get_area(self):
        return cv2.contourArea(self.points)


# loads parking spaces from json file or from given default list
def load_parking_spaces(json_path=None, default_spaces=None):
    import json
    import os

    spaces_data = []
    if json_path and os.path.exists(json_path):
        with open(json_path, "r") as f:
            spaces_data = json.load(f)
    elif default_spaces:
        spaces_data = default_spaces

    return [ParkingSpace(item.get("id", i+1), item["points"]) for i, item in enumerate(spaces_data)]


# builds rectangular grid of spaces for advanced override mode in ui
def generate_grid_spaces(frame_width, frame_height, rows=2, cols=5, margin=20,
                          x_start_frac=0.0, x_end_frac=1.0,
                          y_start_frac=0.0, y_end_frac=1.0,
                          x_nudge=0, y_nudge=0):
    region_x1 = int(frame_width * x_start_frac) + margin + x_nudge
    region_x2 = int(frame_width * x_end_frac) - margin + x_nudge
    region_y1 = int(frame_height * y_start_frac) + margin + y_nudge
    region_y2 = int(frame_height * y_end_frac) - margin + y_nudge

    usable_w = region_x2 - region_x1
    usable_h = region_y2 - region_y1
    cell_w = usable_w // cols
    cell_h = usable_h // rows

    spaces = []
    space_id = 1
    for r in range(rows):
        for c in range(cols):
            x1 = region_x1 + c * cell_w
            y1 = region_y1 + r * cell_h
            x2 = x1 + cell_w - 10
            y2 = y1 + cell_h - 10
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            spaces.append({"id": space_id, "points": points})
            space_id += 1
    return spaces


# finds vertical divider lines in a region using brightness projection
def detect_columns_in_region(frame, x_start_frac=0.0, x_end_frac=1.0,
                              y_start_frac=0.0, y_end_frac=1.0,
                              brightness_thresh=170, min_coverage_frac=0.5,
                              min_gap_px=15):
    h, w = frame.shape[:2]
    x1, x2 = int(w * x_start_frac), int(w * x_end_frac)
    y1, y2 = int(h * y_start_frac), int(h * y_end_frac)
    if x2 <= x1 or y2 <= y1:
        return None

    region = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    bright_mask = gray > brightness_thresh
    coverage = bright_mask.sum(axis=0) / bright_mask.shape[0]

    line_cols = np.where(coverage >= min_coverage_frac)[0]
    if len(line_cols) == 0:
        return None

    groups = [[line_cols[0]]]
    for c in line_cols[1:]:
        if c - groups[-1][-1] <= min_gap_px:
            groups[-1].append(c)
        else:
            groups.append([c])
    x_bounds = [int(np.mean(g)) + x1 for g in groups]

    if len(x_bounds) < 3:
        return None

    spaces = []
    for i in range(len(x_bounds) - 1):
        cx1, cx2 = x_bounds[i], x_bounds[i + 1]
        points = [[cx1, y1], [cx2, y1], [cx2, y2], [cx1, y2]]
        spaces.append({"points": points})
    return spaces


# loads normal yolo model kept unchanged
def load_model(model_path="yolov8n.pt"):
    from ultralytics import YOLO
    return YOLO(model_path)


# loads geo trax aerial model kept unchanged
def load_aerial_model():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    weights_path = hf_hub_download(repo_id="rfonod/geo-trax",
                                   filename="geotrax_hbb_yolov8s_1920_v1.pt")
    return YOLO(weights_path)


# runs detection and tracking on frame kept unchanged
def detect_vehicles(model, frame, conf_threshold=0.35, class_ids=None, imgsz=None):
    class_ids = class_ids or VEHICLE_CLASS_IDS
    track_kwargs = {}
    if imgsz:
        track_kwargs["imgsz"] = imgsz

    results = model.track(
        frame,
        persist=True,
        conf=conf_threshold,
        classes=list(class_ids.keys()),
        verbose=False,
        **track_kwargs,
    )

    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            conf = float(box.conf[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class": class_ids.get(cls_id, "vehicle"),
                "track_id": track_id,
                "conf": conf,
            })
    return detections


# returns center point of a bbox
def box_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# computes overlap ratio between bbox and polygon using mask intersection
def box_overlap_ratio(bbox, polygon_points, frame_shape):
    x1, y1, x2, y2 = bbox
    box_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

    h, w = frame_shape[:2]
    canvas1 = np.zeros((h, w), dtype=np.uint8)
    canvas2 = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(canvas1, [box_poly], 1)
    cv2.fillPoly(canvas2, [polygon_points], 1)

    intersection = np.logical_and(canvas1, canvas2).sum()
    box_area = max(cv2.contourArea(box_poly), 1)

    return intersection / box_area


# builds overlap matrix between all detections and all spaces
def compute_overlap_matrix(spaces, detections, frame_shape):
    n_det = len(detections)
    n_sp = len(spaces)
    matrix = np.zeros((n_det, n_sp), dtype=np.float32)

    for i, det in enumerate(detections):
        dx1, dy1, dx2, dy2 = det["bbox"]
        for j, space in enumerate(spaces):
            sx1, sy1 = space.points[:, 0].min(), space.points[:, 1].min()
            sx2, sy2 = space.points[:, 0].max(), space.points[:, 1].max()
            if dx2 < sx1 or dx1 > sx2 or dy2 < sy1 or dy1 > sy2:
                continue
            matrix[i, j] = box_overlap_ratio(det["bbox"], space.points, frame_shape)
    return matrix


# matches vehicles to spaces one to one using hungarian algorithm
def match_vehicles_to_spaces(spaces, detections, frame_shape, min_overlap=0.3):
    if not detections or not spaces:
        return set()

    overlap = compute_overlap_matrix(spaces, detections, frame_shape)
    cost = 1.0 - overlap
    cost[overlap < min_overlap] = 10.0

    det_idx, sp_idx = linear_sum_assignment(cost)

    occupied_ids = set()
    for i, j in zip(det_idx, sp_idx):
        if overlap[i, j] >= min_overlap:
            occupied_ids.add(spaces[j].space_id)
    return occupied_ids


# updates each slot state only after it holds steady for confirm frames
def update_slot_states(spaces, occupied_ids, frame_number, confirm_frames=5):
    for space in spaces:
        raw = space.space_id in occupied_ids
        if raw == space.pending_state:
            space.pending_frames += 1
        else:
            space.pending_state = raw
            space.pending_frames = 1

        if space.pending_frames >= confirm_frames and space.occupied != space.pending_state:
            space.occupied = space.pending_state
            space.last_change_frame = frame_number
    return spaces


# main entry point that runs matching then hysteresis on all spaces
def check_occupancy(spaces, detections, frame_shape, overlap_threshold=0.3,
                     frame_number=0, confirm_frames=5):
    occupied_ids = match_vehicles_to_spaces(spaces, detections, frame_shape, overlap_threshold)
    return update_slot_states(spaces, occupied_ids, frame_number, confirm_frames)


# draws vehicle boxes and labels on frame kept unchanged
def draw_vehicle_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class"]} #{det["track_id"]}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 140, 0), 2)
    return frame


# draws all parking spaces with color based on occupied state kept unchanged
def draw_parking_spaces(frame, spaces, style="simple"):
    for space in spaces:
        color = (0, 200, 0) if not space.occupied else (0, 0, 220)
        if style == "simple":
            overlay = frame.copy()
            cv2.fillPoly(overlay, [space.points], color)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            cv2.polylines(frame, [space.points], True, color, 2)
        else:
            cv2.polylines(frame, [space.points], True, color, 2)

        center = space.get_center()
        cv2.putText(frame, str(space.space_id), (center[0] - 8, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame