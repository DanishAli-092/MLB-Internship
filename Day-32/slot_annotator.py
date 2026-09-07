# slot_annotator.py
import cv2, json, sys

if len(sys.argv) < 2:
    print("Usage: python slot_annotator.py <video_path> [output_json]")
    sys.exit(1)

video_path = sys.argv[1]
output_json = sys.argv[2] if len(sys.argv) > 2 else "parking_layout.json"

MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

cap = cv2.VideoCapture(video_path)
ok, frame = cap.read()
cap.release()
if not ok:
    print("Could not read frame from video")
    sys.exit(1)

orig_h, orig_w = frame.shape[:2]

# scale is same on both axes so boxes do not get distorted
scale = min(MAX_DISPLAY_WIDTH / orig_w, MAX_DISPLAY_HEIGHT / orig_h, 1.0)
disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)

clone = cv2.resize(frame, (disp_w, disp_h)) if scale < 1.0 else frame.copy()
spaces = []  # points stored here are always in original frame coordinates

drawing = False
start_point = None   # display coords for live preview only
current_point = None


# converts display coords back to original frame coords
def to_original(pt):
    return [int(pt[0] / scale), int(pt[1] / scale)]


# converts original coords to display coords
def to_display(pt):
    return (int(pt[0] * scale), int(pt[1] * scale))


# redraws window with all saved spaces and current drag box
def redraw():
    display = clone.copy()
    for s in spaces:
        pts = [to_display(p) for p in s["points"]]
        for i in range(len(pts)):
            cv2.line(display, pts[i], pts[(i+1) % len(pts)], (0, 255, 0), 2)
        cv2.putText(display, str(s["id"]), (pts[0][0] + 4, pts[0][1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if drawing and start_point and current_point:
        cv2.rectangle(display, start_point, current_point, (0, 0, 255), 2)
    cv2.imshow("Annotate Parking Slots", display)


# converts two corner points into a rectangle of four points
def rect_to_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])
    return [[xa, ya], [xb, ya], [xb, yb], [xa, yb]]


# handles mouse drag to draw and save a new parking slot
def click_event(event, x, y, flags, param):
    global drawing, start_point, current_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_point = (x, y)
        redraw()

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_point = (x, y)
        if abs(current_point[0] - start_point[0]) > 5 and abs(current_point[1] - start_point[1]) > 5:
            orig_p1 = to_original(start_point)
            orig_p2 = to_original(current_point)
            points = rect_to_points(orig_p1, orig_p2)
            spaces.append({"id": len(spaces) + 1, "points": points})
        redraw()

cv2.namedWindow("Annotate Parking Slots")
cv2.setMouseCallback("Annotate Parking Slots", click_event)
redraw()

print(f"Original frame: {orig_w}x{orig_h}, displayed at scale {scale:.2f}")
print("Click drag and release on each slot to draw a box. Press u to undo last s to save and quit q to quit without saving.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u') and spaces:
        spaces.pop()
        redraw()
    elif key == ord('s'):
        with open(output_json, "w") as f:
            json.dump(spaces, f, indent=2)
        print(f"Saved {len(spaces)} slots to {output_json}")
        break
    elif key == ord('q'):
        print("Quit without saving")
        break

cv2.destroyAllWindows()