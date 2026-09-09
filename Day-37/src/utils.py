"""
utils.py
Drawing helpers so app.py stays clean and readable.
"""

import cv2
import numpy as np


# draws box track id and confidence label for each person
def draw_boxes(frame, boxes, track_ids, confidences):
    # scale thickness and font with frame width so it stays crisp after stretch
    h, w = frame.shape[:2]
    box_thickness = max(2, w // 400)
    font_scale = max(0.5, w / 1600)
    font_thickness = max(1, w // 500)

    for box, track_id, conf in zip(boxes, track_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

        label = f"ID {track_id} | {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 6, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return frame


# draws stats panel in top left corner with live and peak count
def draw_info_panel(frame, live_count, peak_count, entry_count=None, exit_count=None):
    h, w = frame.shape[:2]
    font_scale = max(0.6, w / 1400)
    font_thickness = max(1, w // 500)

    panel_width = int(300 * font_scale)
    line_height = int(30 * font_scale)
    panel_height = line_height * (4 if entry_count is not None else 2) + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    y = line_height
    cv2.putText(frame, f"Live Count: {live_count}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    y += line_height
    cv2.putText(frame, f"Peak Count: {peak_count}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    if entry_count is not None:
        y += line_height
        cv2.putText(frame, f"Entries: {entry_count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
        y += line_height
        cv2.putText(frame, f"Exits: {exit_count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    return frame


# draws the counting line on frame
def draw_line(frame, line_start, line_end):
    h, w = frame.shape[:2]
    thickness = max(2, w // 350)
    cv2.line(frame, line_start, line_end, (255, 0, 0), thickness)
    return frame


# draws the counting region polygon on frame
def draw_region(frame, polygon_points):
    h, w = frame.shape[:2]
    thickness = max(2, w // 350)
    pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 255), thickness=thickness)
    return frame