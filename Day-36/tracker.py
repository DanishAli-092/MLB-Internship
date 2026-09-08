from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# calculates overlap ratio between two boxes
def compute_iou(box_a: Tuple[float, float, float, float],
                 box_b: Tuple[float, float, float, float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# finds center point of a box
def centroid_of(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# finds distance between two points
def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


@dataclass
class Track:
    track_id: int
    box: Tuple[float, float, float, float]
    class_name: str
    centroid_history: List[Tuple[float, float]] = field(default_factory=list)
    missed_frames: int = 0
    age: int = 0

    # updates track with new box and resets missed counter
    def update(self, box: Tuple[float, float, float, float], class_name: str) -> None:
        self.box = box
        self.class_name = class_name
        self.centroid_history.append(centroid_of(box))
        if len(self.centroid_history) > 60:
            self.centroid_history.pop(0)
        self.missed_frames = 0
        self.age += 1

    @property
    def current_centroid(self) -> Tuple[float, float]:
        return centroid_of(self.box)

    # gets movement direction from recent centroid history
    def movement_vector(self, lookback: int = 10) -> Tuple[float, float]:
        if len(self.centroid_history) < 2:
            return (0.0, 0.0)
        window = self.centroid_history[-lookback:]
        start = window[0]
        end = window[-1]
        return (end[0] - start[0], end[1] - start[1])


class CentroidTracker:

    def __init__(self, max_missed: int = 15, max_distance: float = 120.0):
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.max_missed = max_missed
        self.max_distance = max_distance

    # matches new detections to existing tracks and returns active tracks
    def update(self, detections: List[Tuple[Tuple[float, float, float, float], str]]) -> Dict[int, Track]:
        unmatched_detections = list(range(len(detections)))
        matched_track_ids = set()

        for track_id, track in list(self.tracks.items()):
            best_match_idx = -1
            best_score = float("inf")

            for det_idx in unmatched_detections:
                box, _ = detections[det_idx]
                dist = euclidean_distance(track.current_centroid, centroid_of(box))
                iou = compute_iou(track.box, box)

                score = dist * (1.0 - iou)

                if dist < self.max_distance and score < best_score:
                    best_score = score
                    best_match_idx = det_idx

            if best_match_idx != -1:
                box, class_name = detections[best_match_idx]
                track.update(box, class_name)
                matched_track_ids.add(track_id)
                unmatched_detections.remove(best_match_idx)
            else:
                track.missed_frames += 1

        for det_idx in unmatched_detections:
            box, class_name = detections[det_idx]
            new_track = Track(track_id=self.next_id, box=box, class_name=class_name)
            new_track.update(box, class_name)
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].missed_frames > self.max_missed:
                del self.tracks[track_id]

        return self.tracks