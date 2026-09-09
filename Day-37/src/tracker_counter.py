"""
tracker_counter.py
Counting logic built on top of YOLO's tracking output.
Has region based counting line crossing counting and peak tracking.
"""

import cv2
import numpy as np

# stores last known center point for each track id
        # needed to know which side of line they were on before

class CrowdCounter:
    def __init__(self):
        
        self.track_history = {}

        self.entry_count = 0
        self.exit_count = 0
        self.peak_count = 0

    # gets center point of a box
    @staticmethod
    def get_center(box):
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy

    # checks if point is inside region polygon
    def point_in_region(self, point, region_polygon):
        result = cv2.pointPolygonTest(np.array(region_polygon, dtype=np.int32), point, False)
        return result >= 0

    # counts how many tracked people are inside region right now
    def update_region_count(self, boxes, track_ids, region_polygon):
        inside_count = 0
        for box in boxes:
            center = self.get_center(box)
            if self.point_in_region(center, region_polygon):
                inside_count += 1

        self.peak_count = max(self.peak_count, inside_count)
        return inside_count

    # checks movement of each track and detects line crossing as entry or exit
    def update_line_count(self, boxes, track_ids, line_start, line_end):
        live_count = len(boxes)
        self.peak_count = max(self.peak_count, live_count)

        for box, track_id in zip(boxes, track_ids):
            center = self.get_center(box)

            if track_id in self.track_history:
                prev_center = self.track_history[track_id]
                side_prev = self._line_side(prev_center, line_start, line_end)
                side_curr = self._line_side(center, line_start, line_end)

                # side changed means line was crossed
                if side_prev != side_curr and side_prev != 0:
                    if side_curr > 0:
                        self.entry_count += 1
                    else:
                        self.exit_count += 1

            self.track_history[track_id] = center

        return live_count

    # cross product tells which side of line a point is on
    # positive is one side negative is other side zero is on the line
    @staticmethod
    def _line_side(point, line_start, line_end):
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross > 0:
            return 1
        elif cross < 0:
            return -1
        return 0