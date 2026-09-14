import cv2
import numpy as np


# converts a canvas rectangle object into a 4-point polygon
def canvas_rect_to_polygon(canvas_object):
    left = canvas_object["left"]
    top = canvas_object["top"]
    width = canvas_object["width"] * canvas_object.get("scaleX", 1)
    height = canvas_object["height"] * canvas_object.get("scaleY", 1)

    polygon = [
        (int(left), int(top)),
        (int(left + width), int(top)),
        (int(left + width), int(top + height)),
        (int(left), int(top + height)),
    ]
    return polygon


# checks if a drawn ROI is valid before processing starts
def validate_roi(roi_polygon, frame_width, frame_height):
    if roi_polygon is None or len(roi_polygon) < 3:
        return False, "ROI needs at least 3 points."

    xs = [p[0] for p in roi_polygon]
    ys = [p[1] for p in roi_polygon]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    if width <= 0 or height <= 0:
        return False, "ROI area is zero, please draw it again."

    if max(xs) < 0 or min(xs) > frame_width or max(ys) < 0 or min(ys) > frame_height:
        return False, "ROI is outside the frame."

    return True, "ROI is valid."


# returns pixel area of the ROI polygon
def roi_area(roi_polygon):
    polygon = np.array(roi_polygon, dtype=np.int32)
    return cv2.contourArea(polygon)


# draws the ROI outline on the frame
def draw_roi(frame, roi_polygon, color=(0, 0, 255), thickness=4, label="ROI"):
    polygon = np.array(roi_polygon, dtype=np.int32)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)

    if label:
        x, y = roi_polygon[0]
        cv2.putText(frame, label, (x, max(y - 12, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


# draws a semi-transparent fill over the ROI
def highlight_roi_fill(frame, roi_polygon, color=(0, 0, 255), alpha=0.30):
    overlay = frame.copy()
    polygon = np.array(roi_polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame