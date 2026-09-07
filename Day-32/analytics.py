# Day 32 Analytics Module occupancy calc utilization level and stats panel for Variant 2

import cv2
import numpy as np
from collections import deque

# how many frames we keep for the mini graph
HISTORY_LENGTH = 60


# class to track occupancy history and unique vehicles seen so far
class OccupancyTracker:
    def __init__(self):
        self.history = deque(maxlen=HISTORY_LENGTH)
        self.seen_track_ids = set()

    # updates history with new pct and adds new track ids to the set
    def update(self, occupied, total, detections):
        pct = calculate_occupancy_percentage(occupied, total)
        self.history.append(pct)
        for det in detections:
            if det["track_id"] != -1:
                self.seen_track_ids.add(det["track_id"])
        return pct

    # returns count of unique vehicles seen till now
    def unique_vehicle_count(self):
        return len(self.seen_track_ids)


# calculates occupancy percent from occupied and total spaces
def calculate_occupancy_percentage(occupied, total):
    if total == 0:
        return 0.0
    return round((occupied / total) * 100, 1)


# gives utilization level and color based on simple threshold values
def get_utilization_level(pct):
    if pct < 40:
        return "Low", (0, 200, 0)
    elif pct < 75:
        return "Medium", (0, 165, 255)
    else:
        return "High", (0, 0, 220)


# draws small stats box in corner for Variant 1 kept minimal
def draw_basic_stats(frame, total, occupied, available):
    h, w = frame.shape[:2]
    box_w, box_h = 230, 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"Total: {total}", (25, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Occupied: {occupied}", (25, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 220), 2)
    cv2.putText(frame, f"Available: {available}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 200, 0), 2)
    return frame


# draws full side panel with pct level graph and time for Variant 2
def draw_analytics_panel(frame, total, occupied, available, pct, history,
                          frame_number, fps, unique_vehicles):
    h, w = frame.shape[:2]
    panel_w = 260
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    panel = canvas[:, w:]
    panel[:] = (25, 25, 25)

    level, level_color = get_utilization_level(pct)
    seconds = frame_number / fps if fps > 0 else 0
    timestamp = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    y = 30
    cv2.putText(panel, "PARKING ANALYTICS", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)
    y += 40
    cv2.putText(panel, f"Occupancy: {pct}%", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)
    y += 30
    cv2.putText(panel, f"Utilization: {level}", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, level_color, 2)
    y += 35
    cv2.putText(panel, f"Total spaces: {total}", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)
    y += 25
    cv2.putText(panel, f"Occupied: {occupied}", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 220), 1)
    y += 25
    cv2.putText(panel, f"Available: {available}", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 200, 0), 1)
    y += 25
    cv2.putText(panel, f"Unique vehicles: {unique_vehicles}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 25
    cv2.putText(panel, f"Frame: {frame_number}  Time: {timestamp}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    graph_top = y + 30
    graph_h = 80
    graph_w = panel_w - 24
    cv2.rectangle(panel, (12, graph_top), (12 + graph_w, graph_top + graph_h),
                  (60, 60, 60), 1)
    if len(history) > 1:
        pts = []
        for i, val in enumerate(history):
            px = 12 + int(i * graph_w / max(len(history) - 1, 1))
            py = graph_top + graph_h - int((val / 100) * graph_h)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i - 1], pts[i], (0, 165, 255), 2)

    canvas[:, :w] = frame
    canvas[:, w:] = panel
    return canvas