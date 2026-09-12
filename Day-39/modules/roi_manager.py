import cv2
import numpy as np

#roi_manager.py
# This class handles roi polygon and checks if a point lies inside it
class ROIManager:
    def __init__(self, roi_points):
        # roi_points is list of x y tuples defining the polygon
        self.roi_points = np.array(roi_points, dtype=np.int32)

    # this checks if given point lies inside the roi polygon
    def is_inside(self, point):
        result = cv2.pointPolygonTest(self.roi_points, point, False)
        # positive value means inside zero means on edge negative means outside
        return result >= 0

    # this draws the roi polygon on frame for visualization
    def draw(self, frame, color=(0, 255, 255), thickness=2):
        cv2.polylines(frame, [self.roi_points], isClosed=True,
                       color=color, thickness=thickness)
        return frame

    # this helper converts simple rectangle into 4 corner points
    @staticmethod
    def rectangle_to_points(x1, y1, x2, y2):
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]