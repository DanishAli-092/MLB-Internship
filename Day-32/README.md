# Day 32 — Smart Parking Occupancy Detection

**MLB Summer Internship | Danish Ali**

A small AI system that takes a parking lot video and produces two different
demo outputs: a live parking monitor, and a fuller analytics view.

## How parking spaces are defined

Each parking space is a **polygon** (`parking_detection.ParkingSpace`), not
just a rectangle. Polygons handle angled or irregular real-world parking
spots better than fixed boxes. Three ways to generate that set of polygons
were tried, in this order:

### 1. Auto-generated equal grid (`generate_grid_spaces`)

Splits a chosen region of the frame into a simple rows × columns grid of
equal-sized rectangles no image analysis at all, purely geometric slicing.
Fast to try, but only lines up with the real painted spaces by coincidence:
real parking rows are rarely perfectly equal-width, rarely start exactly at
a frame-fraction boundary, and any camera tilt/perspective throws the grid
off further from one side of the lot to the other. Still available in the
app as **"Manual equal-width grid"** for quick experiments or genuinely
uniform layouts, with row/column counts, a region selector, and pixel-nudge
sliders to hand-align it.

### 2. Auto-detect from the image (Canny/Hough edges, brightness-column
   projection — `detect_columns_in_region`)

Tried to find the real divider lines automatically: first with Canny edge
detection and Hough line transforms, later with a brightness-based column
projection (scanning for vertical bright bands = painted lines). Neither was
reliable enough to use as the actual slot source, for the same underlying
reasons:

- **Lighting and shadows** shift which pixels count as "line-bright" from
  frame to frame and clip to clip, so a threshold tuned on one video misses
  lines or invents false ones on another.
- **Worn/faded paint** on real parking lots often doesn't cross the
  brightness threshold at all, so real dividers go undetected.
- **Cars parked over or right against a line** partially or fully occlude
  the exact pixels the detector is looking for, so a full lot (the case that
  matters most) is also the case where line detection degrades most.
- **Aerial/drone perspective** distortion means dividers aren't perfectly
  vertical/horizontal across the whole frame, which breaks the "mostly
  vertical" / "mostly horizontal" angle filtering the line-based version
  used.
- Cracks and non-parking markings (crosswalk arrows, disabled-parking icons)
  can look like real lines to a brightness/edge detector, inflating or
  shifting the detected slot count.

In short: both variants depend on the paint being clean, evenly lit, and
unobstructed at the exact moment the reference frame is grabbed  a
condition real parking-lot footage rarely satisfies. It's kept in the app
(**"Auto-detect columns (per row/band)"**) as a faster starting point to
manually correct, not as the primary method.

### 3. Manual, one-time slot annotation (recommended)

The reliable approach used for actual analysis:

- `slot_annotator.py` opens the video's first frame in an OpenCV window and
  lets you **click-drag-release** to draw a box over every physical parking
  space (occupied or empty at that moment — the box marks the space, not its
  current status). Large frames are automatically scaled down to fit the
  screen (max 1280×720) so no part of the frame is cut off, while saved
  coordinates are converted back to the original video resolution so they
  line up correctly during analysis.
- Press `u` to undo the last box, `s` to save all boxes to a JSON file
  (`{"id": ..., "points": [[x,y], [x,y], [x,y], [x,y]]}` per slot), or `q` to
  quit without saving.
- This only needs to be done **once per camera angle**  the same JSON is
  reused for every future video from that camera. It only needs to be redone
  if the camera position changes or the physical layout of the lot changes.
- The Streamlit app's sidebar also has "Manual equal-width grid" and
  "Auto-detect columns" modes for quick experiments, but **"Upload saved
  layout JSON"** (the file produced by `slot_annotator.py`) is the
  recommended, accurate option for real analysis.

## How vehicle detection works

Two detection models are supported:

- **General purpose (fast)** — pretrained YOLOv8n (`yolov8n.pt`), filtered to
  the COCO classes for car, motorcycle, bus, and truck.
- **Aerial/drone specialized (Geo-Trax)** — `rfonod/geo-trax` weights
  (YOLOv8s trained on 19k+ real drone/bird's-eye-view images). This is the
  one actually used for the project's aerial/top-down sample videos, since
  the general-purpose model missed vehicles from that viewing angle.

Detection is run through `model.track()` instead of plain `model.predict()`,
which uses ByteTrack under the hood to assign a persistent `track_id` to each
vehicle across frames.

## How occupied/free status is calculated

Occupancy is no longer just "does this vehicle overlap this slot" checked
independently per slot — that allowed one vehicle to mark multiple
overlapping slots occupied. Instead:

1. **One-to-one matching** (`match_vehicles_to_spaces`) builds an overlap
   matrix between every detected vehicle and every slot
   (`intersection_area / vehicle_box_area`, via `box_overlap_ratio`), then
   uses the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`)
   to find the best one-to-one assignment. A vehicle can claim at most one
   slot, and a slot can be claimed by at most one vehicle per frame.
2. **Hysteresis / debounce** (`update_slot_states`) — a slot only flips its
   displayed occupied/free state after the new raw reading has held steady
   for several consecutive frames (`confirm_frames`, roughly half a second
   of video). This stops a single missed detection or a borderline-overlap
   frame from instantly flickering a slot's status.

Using overlap ratio (rather than "is the center point inside the polygon")
still means a vehicle that is only partially visible or partially parked can
correctly mark a space occupied.

## How duplicate vehicle detection is handled

Because tracking assigns each vehicle a stable `track_id`, the same physical
car keeps the same ID across frames instead of being re-detected as a new
vehicle each frame. `OccupancyTracker` keeps a set of all track IDs seen so
far, so the "unique vehicles" count in Variant 2 does not inflate just
because a car sits in frame for hundreds of frames.

## How occupancy percentage is calculated

`occupancy_percentage = (occupied_spaces / total_spaces) * 100`, rounded to
one decimal place. Utilization level is bucketed from that percentage:

- **Low**: under 40%
- **Medium**: 40%–75%
- **High**: above 75%

## Difference between the two demo variants

| | Parking Monitor (Variant 1) | Parking Analytics (Variant 2) |
|---|---|---|
| Layout | Overlays only, same frame size | Frame + a dedicated side panel (wider output) |
| Space rendering | Filled color overlay (green/red) | Thin outline only |
| Info shown | Total / occupied / available counts | Occupancy %, utilization level, mini trend graph, unique vehicle count, frame/timestamp |
| Purpose | Quick at-a-glance monitoring | Deeper reporting/analytics view |

The two variants use genuinely different layouts (a corner box vs. a full
side panel with a graph), not just different colors on the same overlay.

## Challenges and limitations

- Automatic slot detection (edge/line/brightness-based) is not reliable
  enough for real footage; manual one-time annotation per camera angle is
  the approach that actually works, at the cost of a short manual step.
- Occupancy is based on a single overlap threshold, so a vehicle waiting to
  park at the edge of a space can occasionally be flagged as occupying it.
- A general-purpose COCO-trained model can miss vehicles in aerial/top-down
  footage; the Geo-Trax aerial-specialized model is needed for that camera
  angle.
- Tracking IDs can occasionally switch if a vehicle is fully occluded for
  several frames, which can slightly affect the unique-vehicle count.
- The system assumes a **static camera**: parking space polygons are
  defined once per video and reused for every frame. If the source footage
  has a drone that pans or drifts during the clip, the fixed boxes slowly
  fall out of alignment with the real spaces as the frame content shifts.
  Best results come from either a genuinely stationary camera, or a short
  trimmed clip from the steadiest part of a longer drone shot.

## Project structure

```
Day-32/
├── app.py                 # Streamlit app (upload, run, preview, download)
├── parking_detection.py   # Parking space + vehicle detection + occupancy logic
├── analytics.py           # Occupancy stats, utilization level, analytics panel
├── slot_annotator.py      # One-time manual slot annotation tool (drag-to-draw)
├── requirements.txt
├── README.md
├── sample_videos/         # Place 3-5 public parking lot videos here
├── layouts/               # Saved per-camera-angle slot layout JSON files
└── outputs/
    ├── variant-1/          # Parking Monitor demo outputs
    └── variant-2/          # Parking Analytics demo outputs
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Defining parking slots (one-time, per camera angle)

```bash
python slot_annotator.py sample_videos/your_video.mp4 layouts/camera1_layout.json
```

Click-drag-release over every parking space, `u` to undo, `s` to save. Then
in the Streamlit sidebar, choose **"Upload saved layout JSON"** and select
the generated file.

