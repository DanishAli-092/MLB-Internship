import time
from datetime import datetime
import numpy as np


# rolling fps counter using a small time window
class FPSCounter:

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = []

    def tick(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


# checks if a point lies inside the roi polygon
def point_in_roi(point, roi_polygon):
    import cv2
    polygon = np.array(roi_polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0


# returns center point of a bounding box
def get_box_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


# tracks entry/exit events for each track_id relative to the roi
class EntryExitTracker:

    def __init__(self):
        self.inside_state = {}
        self.events = []

    def update(self, track_id, is_inside, frame_number):
        was_inside = self.inside_state.get(track_id, False)

        if is_inside and not was_inside:
            self.events.append({
                "track_id": track_id,
                "event": "entry",
                "frame": frame_number,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        elif not is_inside and was_inside:
            self.events.append({
                "track_id": track_id,
                "event": "exit",
                "frame": frame_number,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        self.inside_state[track_id] = is_inside

    def get_totals(self):
        entries = sum(1 for e in self.events if e["event"] == "entry")
        exits = sum(1 for e in self.events if e["event"] == "exit")
        return entries, exits