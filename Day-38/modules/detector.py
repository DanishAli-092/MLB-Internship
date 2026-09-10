
# Person Detector + Tracker

# Wraps a YOLO model to detect and track people (class 0 = person in COCO).
# Uses Ultralytics' built-in ByteTrack tracker so each detected person keeps
# a consistent track_id across frames.


from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def track_frame(self, frame):
        """
        Run detection + tracking on a single frame.
        Returns a list of dicts: {track_id, bbox (x1,y1,x2,y2), center (cx,cy), conf}
        Only 'person' class detections are kept.
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],  # person class only
            conf=self.confidence,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = []
        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return detections  # nothing tracked this frame

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append({
                "track_id": int(track_id),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": (cx, cy),
                "conf": float(conf)
            })

        return detections
