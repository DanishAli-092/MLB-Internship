import cv2
import numpy as np
from PIL import Image

#helper.py
# This function converts pil image to opencv image
def pil_to_cv2(pil_image):
    rgb_array = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


# This function converts opencv image to pil image
def cv2_to_pil(cv2_image):
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image)
    rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


# This function draws text with a background box for readability
def draw_text_with_bg(frame, text, pos, font_scale=0.8, color=(255, 255, 255),
                       bg_color=(0, 0, 0), thickness=2, alpha=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - text_h - 8), (x + text_w + 5, y + baseline + 5),
                  bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


# This checks how much two boxes overlap on the x-axis (used to detect "close" boxes
# whose ID labels would otherwise be drawn on top of each other)
def _x_overlap(box_a, box_b):
    ax1, _, ax2, _ = box_a
    bx1, _, bx2, _ = box_b
    overlap = min(ax2, bx2) - max(ax1, bx1)
    return max(overlap, 0)


# This function draws boxes and ids and updates event logger for each roi
def draw_detections(frame, detections, roi_managers, event_logger, frame_number):
    # Sort left-to-right so label staggering below is stable/predictable frame-to-frame
    detections = sorted(detections, key=lambda d: d["bbox"][0])

    label_row_used = []  # tracks the vertical "row" each already-placed label used

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        track_id = det["track_id"]
        center = det["center"]

        inside_any = False
        for roi_id, roi_manager in roi_managers.items():
            inside = roi_manager.is_inside(center)
            event_logger.update(track_id, inside, frame_number, roi_id=roi_id)
            inside_any = inside_any or inside

        box_color = (0, 200, 0) if inside_any else (0, 0, 200)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)

        # Stagger the ID label vertically if this box's ID label would overlap a
        # nearby, already-placed label (i.e. two people standing close together)
        row = 0
        for prev_box, prev_row in label_row_used:
            if _x_overlap((x1, y1, x2, y2), prev_box) > 0:
                row = max(row, prev_row + 1)
        label_row_used.append(((x1, y1, x2, y2), row))

        label_y = max(y1 - 10 - (row * 28), 20)
        cv2.putText(frame, f"ID {track_id}", (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 3)

        cv2.circle(frame, center, 5, box_color, -1)

    total_active = 0
    y_offset = 90
    for roi_id, roi_manager in roi_managers.items():
        roi_manager.draw(frame)
        count = event_logger.active_count(roi_id=roi_id)
        total_active += count
        draw_text_with_bg(frame, f"{roi_id}: {count} active", (15, y_offset),
                           font_scale=0.9, color=(255, 255, 0))
        y_offset += 35

    draw_text_with_bg(frame, f"Total active (all ROIs): {total_active}", (15, 40),
                       font_scale=1.0, color=(255, 255, 255))

    return frame