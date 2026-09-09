"""
detector.py
YOLO based person detector.
Wraps the model so rest of app only deals with clean boxes ids and confidences.
"""

from ultralytics import YOLO
import numpy as np
import os

# wraps yolo model to detect and track only person class
class PersonDetector:

    PERSON_CLASS_ID = 0  # coco class 0 is person

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        self.confidence = confidence
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{model_path}': {e}")

    # runs detection and tracking on single frame for person class only
    # uses botsort instead of bytetrack since it also matches people using
    def track(self, frame: np.ndarray, tracker: str = None):
        
        if tracker is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            tracker = os.path.join(current_dir, "botsort_reid.yaml")
            
        results = self.model.track(
            frame,
            classes=[self.PERSON_CLASS_ID],
            conf=self.confidence,
            tracker=tracker,
            persist=True,
            verbose=False,
        )
        return results[0]

    # plain detection without tracking used for single images
    def detect(self, frame: np.ndarray):
        results = self.model.predict(
            frame,
            classes=[self.PERSON_CLASS_ID],
            conf=self.confidence,
            verbose=False,
        )
        return results[0]