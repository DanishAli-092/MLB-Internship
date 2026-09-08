# Day 36 — Traffic Violation Monitoring System

**MLB Summer Internship | Danish Ali**

A YOLO-based vehicle detection, tracking, and rule-based traffic violation
system with two Streamlit-deployable output variants: **Wrong-Way Detection**
and **Traffic Violation Dashboard**.

## Project Structure

```
Day-36/
├── app.py                  # Streamlit deployment (both variants)
├── traffic_violation.py    # Detection, direction analysis, restricted zone, violation recorder
├── tracker.py               # Custom centroid + IOU multi-object tracker
├── analytics.py             # Violation statistics / analytics engine
├── zone_selector.py         # Local helper: click-to-draw a restricted zone, saves zone.json
├── requirements.txt
├── sample_videos/           # Place 3-5 traffic clips here
├── Screenshots/             # App/demo screenshots for submission
└── outputs/
    ├── variant-1/            # Wrong-Way Detection demo outputs
    └── variant-2/            # Traffic Violation Dashboard demo outputs
```

## How Vehicle Tracking Works

YOLO gives a fresh set of bounding boxes on every frame with **no memory**
of previous frames — it doesn't know that "car in box 5" is the same car
that was "box 3" a moment ago. `tracker.py` solves this with a lightweight
**centroid + IOU tracker**:

1. For every existing track, find the best matching new detection using a
   combined score of centroid distance and IOU overlap.
2. If a good match is found, the existing ID is reused and its box/history
   updated.
3. Unmatched detections become brand-new tracks with a new ID.
4. A track that goes unmatched for more than `max_missed` frames is dropped.

## How Direction Is Calculated

Each track keeps a short history of its centroid positions. The vector
between the centroid ~12 frames ago and now gives a `(dx, dy)` movement
vector, converted into an angle. This is intentionally windowed (not
frame-to-frame) so that small detection jitter doesn't cause false
direction flips.

## How Wrong-Way Vehicles Are Identified

The user (or app slider) sets a **normal direction angle** for the road.
For each vehicle, its movement angle is compared to the normal angle. If
the difference exceeds a threshold (default 100°), the vehicle is flagged
as moving against traffic.

## How Restricted Zones Are Defined

A restricted zone is ultimately just a polygon (list of `(x, y)` points)
drawn over the frame. Every frame, each tracked vehicle's centroid is
tested against the polygon using OpenCV's `pointPolygonTest`. If the
centroid falls inside, that vehicle is in violation.

Since the app runs on Streamlit Cloud (no display available for an
OpenCV click-to-draw window), the zone is built entirely from sidebar
controls, with a live preview drawn on the first frame before running
the analysis. Four shape modes are available from the **Zone Shape**
dropdown, all resolution-independent since they're defined in percentages:

- **Rectangle** — position (X/Y) and size (width/height) sliders build a
  4-point rectangular zone. This is the simplest option and the default.
- **Polygon** — a slider sets how many corners the shape has (3–8), and
  each corner gets its own X/Y percentage sliders, starting out arranged
  in a circle so the shape is valid immediately. A pair of "Shift X/Y"
  sliders lets the whole polygon be nudged as one unit instead of
  repositioning every point individually.
- **Line** — two endpoints (start/end X/Y) plus a thickness percentage.
  Since a line has no "inside" for `pointPolygonTest` to check, it's
  built as a thin rectangle rotated to match the line's direction, so it
  behaves visually like a line while still being a normal polygon
  underneath. Its own "Shift X/Y" sliders move both endpoints together.
- **Custom (upload zone.json)** — for shapes the sliders can't express,
  falls back to a `zone.json` produced locally by `zone_selector.py`
  (see below).

All four modes are resolved through a single `compute_zone_points()`
function, which both the live preview and the actual analysis run call
with the same inputs — so the zone drawn before clicking **Run Analysis**
is guaranteed to match the zone actually used for detection.

If no zone is configured at all, `app.py`'s `get_default_zone()` falls
back to a fixed rectangle in the lower-right of the frame.

## How Duplicate Violations Are Avoided

A vehicle that lingers in a restricted zone (or keeps moving wrong-way)
for several seconds would otherwise generate dozens of duplicate
violation records. `ViolationRecorder` keeps a set of
`(track_id, violation_type)` pairs already flagged — each vehicle can only
be recorded **once per violation type** for its entire time in the video.

## How Violation Statistics Are Generated

`analytics.py`'s `TrafficAnalytics` class collects:
- every unique vehicle ID seen (and its type), and
- every recorded `ViolationEvent`

and derives totals, per-type breakdowns, and a plain-language traffic
status summary from that raw data.

## Difference Between the Two Demo Variants

| | Variant 1 — Wrong-Way Detection | Variant 2 — Traffic Violation Dashboard |
|---|---|---|
| Focus | Vehicle movement | Analytics |
| Overlay | Bounding boxes, tracking IDs, direction arrows, normal-direction reference arrow, violation counter | Dark analytics panel below the video with live totals, per-type violation counts, vehicle-type breakdown |
| Layout | Minimal — just the annotated video | Video + stats panel stitched together (taller frame) |
| Streamlit output | Video + violation counter | Video + metrics, bar chart, events table, status summary |

Both variants share the exact same detection/tracking/violation engine —
only the rendering (`draw_track_box` + arrows vs. `draw_dashboard_overlay`)
differs, so the two outputs are always consistent with each other.

## Drawing a Custom Restricted Zone (zone_selector.py)

For a shape none of the built-in Rectangle/Polygon/Line slider modes can
express, `zone_selector.py` lets you draw an arbitrary polygon by hand.
Run this **locally** (it opens an OpenCV GUI window, so it won't run on a
headless server):

```bash
python zone_selector.py --video sample_videos/traffic1.mp4
```

- **Left click** to add a polygon point, **right click** to undo the last one.
- Press **`c`** once you have 3+ points to confirm the shape.
- Press **`s`** to save it to `zone.json` (or pass `--output my_zone.json`).
- Press **`q`** / `Esc` to quit without saving.

Then, in the Streamlit app's sidebar, set **Zone Shape** to
**Custom (upload zone.json)** and upload that file — the app automatically
rescales the saved points if the uploaded video has a different
resolution than the frame you drew on.

## Challenges and Limitations

- **Direction ambiguity at low speed:** vehicles that are nearly
  stationary (e.g., stopped at a light) don't produce a reliable movement
  vector, so direction checks are skipped for very small movements.
- **ID switches:** in heavy occlusion (vehicles overlapping), the centroid
  tracker can occasionally swap IDs. A production system would use
  YOLO's built-in ByteTrack/BoT-SORT or a Kalman-filter tracker for more
  robustness.
- **Fixed restricted zone:** the zone is defined once per video; it does
  not adapt automatically to camera angle changes.
- **Polygon precision:** corner points for the Polygon shape are placed
  with sliders rather than a click-and-draw interface, so it's possible
  to place corners out of order and get a self-intersecting (bow-tie)
  shape, which makes the containment check behave unpredictably.
- **Line is an approximation:** the Line zone isn't a true zero-width
  line — it's a thin rectangle rotated to match the line's direction —
  so at very small thickness percentages or steep angles it can be
  slightly off from an ideal geometric line.
- **Not a legal violation detector:** this is a rule-based project for
  learning purposes, not a certified traffic-enforcement system.

## Expected Usage

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a traffic video, pick a mode, choose a restricted zone shape, and
run the analysis. The processed video and statistics can be viewed
in-app and downloaded.