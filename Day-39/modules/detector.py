# Person Detector + Tracker
# detector.py
# Wraps YOLO model to detect and track people (class 0 = person in COCO)


import cv2
from ultralytics import YOLO


# confidence is min score to keep a detection
        # iou is threshold for NMS to remove duplicate boxes
        # infer_width shrinks big frames before inference then boxes scaled back up for speed

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.4, iou=0.45, infer_width=640,
                 tracker_config="botsort_reid.yaml"):
        
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{model_path}': {e}")

        self.confidence = confidence
        self.iou = iou
        self.infer_width = infer_width
        self.tracker_config = tracker_config

    # shrinks frame if too wide and returns resized frame with scale used
    def _resize_for_inference(self, frame):
        h, w = frame.shape[:2]
        if w <= self.infer_width:
            return frame, 1.0
        scale = self.infer_width / w
        resized = cv2.resize(frame, (self.infer_width, int(h * scale)))
        return resized, scale

    # runs detection plus tracking on one frame and returns list of person detections
    def track_frame(self, frame):
        if frame is None:
            return []

        infer_frame, scale = self._resize_for_inference(frame)

        try:
            results = self.model.track(
                infer_frame,
                persist=True,
                classes=[0],  # person class only
                conf=self.confidence,
                iou=self.iou,
                tracker=self.tracker_config,
                verbose=False
            )
        except Exception as e:
            print(f"[PersonDetector] track_frame inference failed: {e}")
            return []

        detections = []
        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return detections  # nothing tracked this frame

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            # scale coords back up to original frame size
            x1, y1, x2, y2 = box / scale
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append({
                "track_id": int(track_id),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": (cx, cy),
                "conf": float(conf)
            })

        return detections

    # runs plain detection with no track id on single standalone image
    def detect_image(self, frame):
        if frame is None:
            return []

        infer_frame, scale = self._resize_for_inference(frame)

        try:
            results = self.model.predict(
                infer_frame,
                classes=[0],
                conf=self.confidence,
                iou=self.iou,
                verbose=False
            )
        except Exception as e:
            print(f"[PersonDetector] detect_image inference failed: {e}")
            return []

        detections = []
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box / scale
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append({
                "track_id": i + 1,  # stable numbering for display only
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": (cx, cy),
                "conf": float(conf)
            })

        return detections