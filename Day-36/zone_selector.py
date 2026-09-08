import argparse
import json
import sys

import cv2
import numpy as np


class ZoneSelector:
    def __init__(self, frame: np.ndarray):
        self.original_frame = frame
        self.points = []
        self.confirmed = False

    # handles mouse clicks to add or remove points
    def mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and not self.confirmed:
            self.points.append((x, y))
            print(f"Point added: ({x}, {y}) | total points: {len(self.points)}")
        elif event == cv2.EVENT_RBUTTONDOWN and not self.confirmed:
            if self.points:
                removed = self.points.pop()
                print(f"Point removed: {removed} | total points: {len(self.points)}")

    # draws points polygon and status text on the frame
    def render(self) -> np.ndarray:
        frame = self.original_frame.copy()

        instructions = [
            "Left click: add point | Right click: undo",
            "'c': confirm polygon | 'r': reset | 's': save | 'q': quit",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)

        for point in self.points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        if len(self.points) > 1:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=self.confirmed,
                          color=(0, 255, 0) if self.confirmed else (0, 200, 255),
                          thickness=2)

        if self.confirmed and len(self.points) >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(self.points, dtype=np.int32)], (0, 0, 255))
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            status = "CONFIRMED - press 's' to save"
            cv2.putText(frame, status, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            status = f"Points: {len(self.points)} (need >= 3 to confirm)"
            cv2.putText(frame, status, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        return frame


# reads a specific frame from the video file
def load_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not read frame {frame_index} from video. "
            "Try a smaller --frame_index."
        )
    return frame


# runs the click to draw zone tool and saves result to json
def main() -> None:
    parser = argparse.ArgumentParser(description="Click-to-draw restricted zone selector")
    parser.add_argument("--video", required=True, help="Path to a traffic video")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Which frame to use as the reference image (default: 0)")
    parser.add_argument("--output", default="zone.json",
                        help="Where to save the polygon points (default: zone.json)")
    args = parser.parse_args()

    frame = load_frame(args.video, args.frame_index)
    selector = ZoneSelector(frame)

    window_name = "Draw Restricted Zone - Day 36"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, selector.mouse_callback)

    print("\nDraw your restricted zone by clicking points on the video frame.")
    print("Press 'c' once you have at least 3 points to confirm the shape.\n")

    while True:
        cv2.imshow(window_name, selector.render())
        key = cv2.waitKey(20) & 0xFF

        if key == ord("c"):
            if len(selector.points) >= 3:
                selector.confirmed = True
                print("Polygon confirmed. Press 's' to save, or 'r' to redo.")
            else:
                print("Need at least 3 points before confirming.")

        elif key == ord("r"):
            selector.points = []
            selector.confirmed = False
            print("Reset. Start clicking again.")

        elif key == ord("s"):
            if selector.confirmed and len(selector.points) >= 3:
                with open(args.output, "w") as f:
                    json.dump({"zone_points": selector.points,
                               "frame_width": frame.shape[1],
                               "frame_height": frame.shape[0]}, f, indent=2)
                print(f"Saved zone to {args.output}: {selector.points}")
            else:
                print("Confirm the polygon with 'c' before saving.")

        elif key == ord("q") or key == 27:
            print("Exiting without saving (unless you already pressed 's').")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()