from dataclasses import dataclass, field
from typing import Dict, List
from collections import Counter

from traffic_violation import ViolationEvent


@dataclass
class TrafficAnalytics:

    total_unique_vehicles: int = 0
    vehicle_type_counts: Counter = field(default_factory=Counter)
    violation_events: List[ViolationEvent] = field(default_factory=list)

    # adds a new vehicle type count when a new track shows up
    def register_vehicle(self, vehicle_type: str) -> None:
        self.total_unique_vehicles += 1
        self.vehicle_type_counts[vehicle_type] += 1

    # adds one violation event to the list
    def add_violation(self, event: ViolationEvent) -> None:
        self.violation_events.append(event)

    # returns total number of violations
    def total_violations(self) -> int:
        return len(self.violation_events)

    # returns violation counts grouped by violation type
    def violations_by_type(self) -> Dict[str, int]:
        counts: Counter = Counter()
        for event in self.violation_events:
            counts[event.violation_type] += 1
        return dict(counts)

    # returns violation counts grouped by vehicle type
    def violations_by_vehicle_type(self) -> Dict[str, int]:
        counts: Counter = Counter()
        for event in self.violation_events:
            counts[event.vehicle_type] += 1
        return dict(counts)

    # returns count of wrong way violations
    def wrong_way_count(self) -> int:
        return sum(1 for e in self.violation_events if e.violation_type == "wrong_way")

    # returns count of restricted zone violations
    def restricted_zone_count(self) -> int:
        return sum(1 for e in self.violation_events if e.violation_type == "restricted_zone")

    # builds rows ready for streamlit dataframe
    def events_table(self) -> List[Dict]:
        rows = []
        for e in sorted(self.violation_events, key=lambda x: x.frame_number):
            rows.append({
                "Vehicle ID": e.track_id,
                "Vehicle Type": e.vehicle_type,
                "Violation Type": e.violation_type.replace("_", " ").title(),
                "Frame": e.frame_number,
                "Timestamp (s)": round(e.timestamp_sec, 2),
            })
        return rows

    # generates a simple overall status line for the dashboard
    def traffic_status_summary(self) -> str:
        if self.total_violations() == 0:
            return "Traffic is flowing normally. No violations detected."

        violation_rate = self.total_violations() / max(1, self.total_unique_vehicles)
        if violation_rate > 0.3:
            return "High violation rate detected. Traffic monitoring should be increased."
        elif violation_rate > 0.1:
            return "Moderate violations detected. Some vehicles are not following the rules."
        else:
            return "Traffic is mostly compliant, with a few isolated violations."

    # builds full summary as a dictionary for metrics or export
    def as_summary_dict(self) -> Dict:
        return {
            "total_vehicles": self.total_unique_vehicles,
            "vehicle_type_counts": dict(self.vehicle_type_counts),
            "total_violations": self.total_violations(),
            "wrong_way_violations": self.wrong_way_count(),
            "restricted_zone_violations": self.restricted_zone_count(),
            "violations_by_vehicle_type": self.violations_by_vehicle_type(),
            "status_summary": self.traffic_status_summary(),
        }