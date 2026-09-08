import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2

from tracker import Track


VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorbike", "motorcycle"}


class VehicleDetector:

    # loads yolo model with given confidence
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = confidence

    # runs detection on a frame and returns only vehicle boxes
    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], str, float]]:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, str(cls_id)).lower()
                if class_name not in VEHICLE_CLASS_NAMES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(((x1, y1, x2, y2), class_name, conf))

        return detections


# gets angle of movement vector in degrees
def angle_of_vector(vec: Tuple[float, float]) -> Optional[float]:
    dx, dy = vec
    magnitude = math.hypot(dx, dy)
    if magnitude < 3.0:
        return None
    return math.degrees(math.atan2(dy, dx))


# gets shortest difference between two angles
def angle_difference(a: float, b: float) -> float:
    diff = abs(a - b) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


class DirectionAnalyzer:

    # stores normal direction angle and wrong way threshold
    def __init__(self, normal_direction_angle: float, wrong_way_threshold: float = 100.0):
        self.normal_angle = normal_direction_angle
        self.wrong_way_threshold = wrong_way_threshold

    # checks if vehicle is moving in wrong direction
    def is_wrong_way(self, track: Track) -> Tuple[bool, Optional[float]]:
        vec = track.movement_vector(lookback=12)
        angle = angle_of_vector(vec)
        if angle is None:
            return False, None
        diff = angle_difference(angle, self.normal_angle)
        return diff > self.wrong_way_threshold, angle


class RestrictedZone:

    # stores zone polygon points
    def __init__(self, polygon_points: List[Tuple[int, int]]):
        self.polygon = np.array(polygon_points, dtype=np.int32)

    # checks if given point is inside the zone
    def contains_point(self, point: Tuple[float, float]) -> bool:
        result = cv2.pointPolygonTest(self.polygon, point, False)
        return result >= 0

    # draws the zone polygon on the frame
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [self.polygon], isClosed=True, color=color, thickness=4)
        return frame


@dataclass
class ViolationEvent:
    track_id: int
    vehicle_type: str
    violation_type: str
    frame_number: int
    timestamp_sec: float


class ViolationRecorder:

    # sets up empty events list and flagged set
    def __init__(self):
        self.events: List[ViolationEvent] = []
        self._already_flagged = set()

    # records a violation once per track and type
    def record(self, track_id: int, vehicle_type: str, violation_type: str,
               frame_number: int, timestamp_sec: float) -> bool:
        key = (track_id, violation_type)
        if key in self._already_flagged:
            return False

        self._already_flagged.add(key)
        self.events.append(ViolationEvent(
            track_id=track_id,
            vehicle_type=vehicle_type,
            violation_type=violation_type,
            frame_number=frame_number,
            timestamp_sec=timestamp_sec,
        ))
        return True

    # counts violations of a given type
    def count_by_type(self, violation_type: str) -> int:
        return sum(1 for e in self.events if e.violation_type == violation_type)

    # returns total number of recorded violations
    def total_violations(self) -> int:
        return len(self.events)


# draws bounding box and label for a tracked vehicle
def draw_track_box(frame: np.ndarray, track: Track, is_violation: bool) -> None:
    x1, y1, x2, y2 = [int(v) for v in track.box]
    color = (0, 0, 255) if is_violation else (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

    label = f"ID {track.track_id} | {track.class_name}"
    if is_violation:
        label += " | VIOLATION"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    label_y = max(text_h + baseline + 6, y1)
    cv2.rectangle(frame, (x1, label_y - text_h - baseline - 6),
                  (x1 + text_w + 10, label_y), color, -1)
    cv2.putText(frame, label, (x1 + 5, label_y - baseline - 3), font,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# draws direction arrow for a tracked vehicle
def draw_direction_arrow(frame: np.ndarray, track: Track, angle: Optional[float]) -> None:
    if angle is None:
        return
    cx, cy = track.current_centroid
    length = 55
    end_x = cx + length * math.cos(math.radians(angle))
    end_y = cy + length * math.sin(math.radians(angle))

    start_point = (int(cx), int(cy))
    end_point = (int(end_x), int(end_y))

    cv2.arrowedLine(frame, start_point, end_point, (0, 0, 0), 6, tipLength=0.45)
    cv2.arrowedLine(frame, start_point, end_point, (255, 0, 255), 3, tipLength=0.45)


# draws reference arrow showing normal traffic direction
def draw_normal_direction_indicator(frame: np.ndarray, angle: float) -> None:
    origin = (70, 70)
    length = 70
    end_x = origin[0] + length * math.cos(math.radians(angle))
    end_y = origin[1] + length * math.sin(math.radians(angle))
    end_point = (int(end_x), int(end_y))

    cv2.arrowedLine(frame, origin, end_point, (0, 0, 0), 8, tipLength=0.4)
    cv2.arrowedLine(frame, origin, end_point, (0, 255, 255), 4, tipLength=0.4)

    label = "Normal Dir"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_origin = (origin[0] - text_w // 2, origin[1] + length + 30)

    cv2.rectangle(frame, (text_origin[0] - 6, text_origin[1] - text_h - 6),
                  (text_origin[0] + text_w + 6, text_origin[1] + baseline + 4),
                  (0, 0, 0), -1)
    cv2.putText(frame, label, text_origin, font, font_scale, (0, 255, 255),
                thickness, cv2.LINE_AA)


# draws total violation counter box on the frame
def draw_violation_counter(frame: np.ndarray, count: int) -> None:
    cv2.rectangle(frame, (frame.shape[1] - 260, 10), (frame.shape[1] - 10, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Violations: {count}", (frame.shape[1] - 245, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)