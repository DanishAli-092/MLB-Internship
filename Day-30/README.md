# Day 30 - Smart Vehicle Counting System

**MLB Summer Internship | Danish Ali**

**Streamlit App URL:** [View Live Application Here](https://mlb-internship-danish-ali-day-29-object-tracking.streamlit.app/)

A computer vision application that detects, tracks, and counts vehicles (cars, buses,
trucks, motorcycles) crossing a defined line in traffic videos, using YOLO for
detection and BoT-SORT / ByteTrack for multi-object tracking. The standalone
script (`vehicle_counting.py`) uses BoT-SORT by default, while the Streamlit app
(`app.py`) lets you switch between BoT-SORT and ByteTrack from a dropdown.

## How Vehicle Counting Works

1. Every frame of the video is passed through a YOLO object detector, which finds
   vehicles and classifies them into car, truck, bus, or motorcycle.
2. A virtual counting line is drawn across the frame (position is configurable).
3. For every detected vehicle, we track the center point (centroid) of its bounding
   box across frames.
4. When a vehicle's centroid moves from one side of the line to the other between
   two consecutive frames, it is counted as "crossed."
5. The direction of crossing is also recorded — top-to-bottom is counted as
   "Down (entering)" and bottom-to-top as "Up (exiting)" — so the app can tell how
   many vehicles moved in each direction, not just a single combined total.
6. The running total, per-direction counts, and a per-class breakdown are all
   displayed live on the video.

## How Tracking IDs Prevent Duplicate Counting

A single vehicle produces a new detection in every frame it appears in, so without
tracking the same car could accidentally get counted multiple times as it passes
near the line. To solve this, we use a tracker on top of YOLO (via
`model.track(persist=True)`), which assigns a consistent ID to each vehicle for
as long as it stays visible. The script defaults to BoT-SORT (`botsort.yaml`),
and the Streamlit app additionally lets you pick ByteTrack (`bytetrack.yaml`)
from the sidebar dropdown if you want faster, motion-only tracking.

We keep a `counted_ids` set — once a track ID has been counted, it is added to this
set and skipped in all future frames, even if it lingers near the line or its
detection flickers for a frame or two. This guarantees each physical vehicle is
counted exactly once.

## Vehicle Types Counted

- Car
- Truck
- Bus
- Motorcycle

These map to COCO class IDs 2, 7, 5, and 3 respectively, which is what the
pretrained YOLO model already recognizes out of the box.

## Streamlit App Features

- Upload your own traffic video or pick from bundled sample videos.
- Choose the YOLO model, tracking algorithm, counting line position, and
  confidence threshold from the sidebar.
- Live progress display while the video is processing, showing the current
  frame number against the total frame count.
- Once processing finishes, the app shows how many frames were processed and
  the total time taken to process the video.
- Final results panel with total count, per-class counts, and direction-wise
  crossing counts.
- Preview and download the processed video with counts drawn on it.

## Challenges Faced

- **ID switching**: occasionally the tracker would drop and reassign a new ID to
  the same vehicle (e.g. after occlusion by another vehicle), which could cause a
  double count. Using BoT-SORT instead of simple IoU-based tracking, and keeping
  a slightly wider counting line region, reduced this significantly. BoT-SORT's
  appearance-based re-identification specifically helped when vehicles were
  briefly hidden behind each other near the line.
- **Line placement**: a line placed too close to the frame edge missed fast-moving
  vehicles between frames since the centroid could jump past the line in a single
  step. Placing the line in the middle-lower part of the frame and checking both
  crossing directions fixed most of it.
- **False classifications**: small or distant vehicles were sometimes misclassified
  (e.g. a car identified as a truck). Increasing the confidence threshold slightly
  and using a marginally larger model (`yolov8s.pt`) improved accuracy at the cost
  of speed.
- **Validating true crossings**: verified that the system only counts vehicles
  that genuinely cross the defined line, avoiding inflated counts from vehicles
  that pass nearby without crossing.

## Project Structure

```
Day-30/
├── vehicle_counting.py     # Standalone script (CLI) for detection + tracking + counting
├── app.py                  # Streamlit deployment app
├── requirements.txt        # Python dependencies
├── README.md
├── sample_input_videos/    # Traffic videos used for testing
└── output_videos/          # Processed videos with counts drawn
```

## Running Locally

```bash
# run on a single video from the command line
python vehicle_counting.py --input sample_input_videos/traffic1.mp4 --output output_videos/result1.mp4

# run the streamlit app
streamlit run app.py
```

