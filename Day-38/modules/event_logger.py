import csv
import os
from datetime import datetime


# This class logs entry and exit events of people in roi and writes to csv
class EventLogger:
    def __init__(self, log_path="data/logs/event_log.csv", sessions_path="data/logs/sessions.csv"):
        self.log_path = log_path
        self.sessions_path = sessions_path

        # store current state of each track in each roi
        self.track_states = {}
        # store entry time of each track to calculate duration later
        self.entry_times = {}
        # store highest active count seen so far for each roi
        self.peak_counts = {}
        # store all track ids seen so far for unique visitor count
        self.unique_visitors = {}

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["track_id", "roi_id", "event_type", "timestamp", "frame_number"])

        if not os.path.exists(self.sessions_path):
            with open(self.sessions_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["track_id", "roi_id", "entry_time", "exit_time", "duration_seconds"])

    # this function runs every frame and checks if state changed then logs entry or exit
    def update(self, track_id, is_inside, frame_number, roi_id="ROI_1"):
        key = (track_id, roi_id)
        previous_state = self.track_states.get(key, "outside")
        current_state = "inside" if is_inside else "outside"

        if previous_state == current_state:
            return None  # no change so nothing to log

        self.track_states[key] = current_state
        now = datetime.now()

        if current_state == "inside":
            event_type = "ENTRY"
            self.entry_times[key] = now
            self.unique_visitors.setdefault(roi_id, set()).add(track_id)
        else:
            event_type = "EXIT"
            entry_time = self.entry_times.pop(key, now)
            self._write_session(track_id, roi_id, entry_time, now)

        self._write_event(track_id, roi_id, event_type, now, frame_number)

        # update peak count after any state change
        current_active = self.active_count(roi_id=roi_id)
        self.peak_counts[roi_id] = max(self.peak_counts.get(roi_id, 0), current_active)

        return event_type

    # this writes one row in event log file
    def _write_event(self, track_id, roi_id, event_type, timestamp, frame_number):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([track_id, roi_id, event_type,
                              timestamp.strftime("%Y-%m-%d %H:%M:%S"), frame_number])

    # this writes one row in session file with entry and exit time and duration
    def _write_session(self, track_id, roi_id, entry_time, exit_time):
        duration = (exit_time - entry_time).total_seconds()
        with open(self.sessions_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                track_id, roi_id,
                entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                round(duration, 2)
            ])

    # this returns number of people currently inside for given roi or all roi
    def active_count(self, roi_id=None):
        if roi_id is None:
            return sum(1 for state in self.track_states.values() if state == "inside")
        return sum(1 for (tid, rid), state in self.track_states.items()
                    if rid == roi_id and state == "inside")

    # this builds final summary report using sessions file
    def generate_summary(self):
        all_roi_ids = sorted(set(self.peak_counts.keys()) | set(self.unique_visitors.keys()))

        if not os.path.exists(self.sessions_path):
            return "No sessions recorded yet."

        sessions = []
        with open(self.sessions_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            sessions = list(reader)

        lines = ["=== Security Monitoring Summary Report ===", ""]

        for roi_id in all_roi_ids:
            roi_sessions = [s for s in sessions if s["roi_id"] == roi_id]
            durations = [float(s["duration_seconds"]) for s in roi_sessions]
            avg_duration = sum(durations) / len(durations) if durations else 0.0
            unique_count = len(self.unique_visitors.get(roi_id, set()))
            peak = self.peak_counts.get(roi_id, 0)
            still_inside = self.active_count(roi_id=roi_id)

            lines.append(f"[{roi_id}]")
            lines.append(f"  Unique visitors        : {unique_count}")
            lines.append(f"  Completed visits        : {len(roi_sessions)}")
            lines.append(f"  Currently still inside  : {still_inside}")
            lines.append(f"  Peak occupancy          : {peak}")
            lines.append(f"  Average visit duration  : {avg_duration:.2f} sec")
            lines.append("")

        return "\n".join(lines)