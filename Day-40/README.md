#  Day 40 - 🎥 Smart Video Analytics System

**ML Bench Summer Internship | Danish Ali**

**Real-Time Video Analytics**

A Streamlit-based application that performs YOLO object detection, multi-object tracking, ROI-based entry/exit detection, and generates a full analytics report (live counts, unique object counts, FPS, and event logs) from an uploaded video.

---

## 📌 Overview

This project processes a recorded video frame-by-frame using YOLOv8, tracks detected objects across frames with consistent tracking IDs, lets the user draw one or more Regions of Interest (ROI) directly on the video's first frame, and logs every entry/exit event into a CSV file. At the end of processing, it produces a browser-playable annotated output video along with a full analytics summary.

---

## ✨ Features

- 📤 Video upload with live preview
- ✏️ Interactive ROI drawing (supports multiple ROIs on the same frame)
- 🎯 YOLOv8 object detection + multi-object tracking
- 🔀 User-selectable tracking algorithm — **ByteTrack** or **BoT-SORT**
- 📊 Real-time overlay: FPS, current object count, current-in-ROI count, unique object count, entries/exits
- 🚪 Automatic entry/exit event detection and logging to `events.csv`
- 🎬 Processed output video (re-encoded to H.264 for browser playback)
- ⚙️ Adjustable inference image size (640px / 480px), confidence threshold, and frame-skipping rate
- 📈 Standalone performance benchmarking script across configurations

---

## 🗂️ Project Structure

```
Day-40/
├── app.py                    # Main Streamlit application
├── test_tracking.py          # Standalone script to test tracking on sample videos
├── performance_test.py       # Automated FPS/config benchmarking script
├── requirements.txt
├── README.md
├── events.csv                # Generated after processing (entry/exit log)
├── performance_results.csv   # Generated after running performance_test.py
├── config/
│   ├── bytetrack_custom.yaml
│   └── botsort_custom.yaml
├── src/
│   ├── __init__.py
│   ├── tracker_utils.py      # FPS counter, ROI point check, entry/exit tracker
│   └── roi_utils.py          # ROI drawing, fill overlay, validation helpers
├── sample_videos/            # 3 test videos (traffic/people) go here
└── outputs/
    └── processed/            # Processed output videos saved here
```

---

## ⚙️ Setup

```powershell
pip install -r requirements.txt
```

**requirements.txt:**
```
ultralytics==8.4.115
lap
streamlit>=1.53.0
streamlit-drawable-canvas==0.10.0
opencv-python-headless==4.10.0.84
numpy==2.3.5
Pillow>=10.0.0
pandas>=2.0.0
imageio-ffmpeg>=0.5.1
```

Place your 3 test videos (15–30 seconds each, people/vehicles/moving objects) inside `sample_videos/`.

---

## ▶️ Running the App

```powershell
streamlit run app.py
```

**Workflow:**
1. Upload a video (preview is shown immediately).
2. Draw one or more ROI rectangles on the displayed first frame.
3. Choose tracking algorithm, image size, confidence threshold, and frame-skip rate from the sidebar.
4. Click **Start Processing**.
5. View the live processing preview, then the final summary, processed video, and event log.
6. Download `events.csv` directly from the app.

---

## 🧠 How It Works

**Pipeline:**
```
Video → Frame Read → YOLO Detection → Tracking (ByteTrack/BoT-SORT) → ROI Check → Entry/Exit Event → Overlay Draw → Output Video → Summary
```

### Tracking IDs
Ultralytics' built-in tracker (`model.track()`) assigns each detected object a persistent numeric ID across frames, matching detections frame-to-frame based on motion (and appearance, for BoT-SORT). This is what allows the same object to be counted once rather than repeatedly across frames.

### Entry/Exit Detection
For every tracked object, the center point of its bounding box is checked against each drawn ROI polygon using `cv2.pointPolygonTest`. The system keeps a per-object "was it inside the ROI last frame?" state; a transition from outside → inside logs an **entry** event, and inside → outside logs an **exit** event. All events are timestamped and written to `events.csv`.

### Frame Skipping
To improve throughput, detection can be run only on every Nth frame (user-configurable). On skipped frames, the last known detections are reused for overlay drawing, keeping the output video visually smooth while reducing the number of YOLO inference calls.

---

## 🔀 Tracker Choice: ByteTrack vs BoT-SORT

| | ByteTrack | BoT-SORT |
|---|---|---|
| Matching basis | Motion / IOU only | Motion + appearance (ReID) |
| Speed | Faster | Slower |
| Best for | Real-time, high-FPS needs | Scenes with more crossing/occlusion |

Both are exposed as a sidebar dropdown so results can be compared directly on the same video. Custom tracker configs (`config/bytetrack_custom.yaml`, `config/botsort_custom.yaml`) increase `track_buffer` to 60 frames to improve ID persistence through brief occlusions.

---

## 📊 Performance Test Results

Tested using `performance_test.py` across six configurations on the sample videos:

| Configuration | Avg FPS |
|---|---|
| 640px, frame skip = 2 (BoT-SORT) | ~24.45 FPS |
| 640px, frame skip = 2 (ByteTrack) | ~21.76 FPS |
| 480px, no frame skip (ByteTrack) | ~16.99 FPS |
| 480px, no frame skip (BoT-SORT) | ~15.05 FPS |
| 640px, no frame skip (BoT-SORT) | ~12.81 FPS |
| 640px, no frame skip (ByteTrack) | ~12.14 FPS |

**Best performing configuration:** `640px, frame skip = 2 (BoT-SORT)`

> **📌 Tracker Comparison Insight:** On average across all configurations, **BoT-SORT (~17.4 FPS)** slightly outperformed **ByteTrack (~16.9 FPS)** in terms of overall speed for this specific video set.

**Observation:** Lowering image size from 640px to 480px consistently improved FPS, as expected — smaller input means less inference compute per frame. Frame skipping reduces the number of YOLO inference calls, but overall video throughput does not scale proportionally, since frame reading, overlay drawing, and video writing still happen on every frame regardless of whether detection runs. This indicates that video I/O and rendering overhead — not just inference — is a meaningful part of total processing cost in this pipeline.

---

## 🧾 Sample Summary Output

```
Total Objects: 18
Total Entries: 12
Total Exits: 9
Maximum Objects in ROI: 7
Average FPS: 24.5
```

*(Actual numbers vary per video, tracker, and configuration — see `performance_results.csv` for full comparison data.)*

---

## ⚠️ Known Limitation: ID Switching

Occasionally, tracking IDs switch when two objects cross paths or briefly occlude one another. This is a known limitation of motion-based multi-object trackers (ByteTrack) — and to a lesser extent, appearance-based ones (BoT-SORT) — particularly when paired with a lightweight detection model like YOLOv8n. Increasing `track_buffer` helps with short occlusions but does not eliminate ID switches caused by close-proximity crossing, since the underlying association is still confidence/overlap-based rather than a dedicated re-identification model. A more robust fix would involve a heavier backbone (YOLOv8m/l) or a dedicated ReID-based tracker.

---

## 🐛 Problems Faced & Solutions

- **ROI canvas coordinates mismatch:** Canvas coordinates are in display (resized) space, not original frame space — fixed by scaling coordinates back up using the display-to-original scale factor.
- **Only one ROI was being captured:** Initial implementation read only the last drawn rectangle from the canvas object list — fixed by looping over all drawn objects and checking membership against any of them.
- **Processed video not playable in browser:** OpenCV's `mp4v`-encoded output isn't broadly browser-compatible — fixed by re-encoding the output to H.264 (`libx264`) using `imageio-ffmpeg` before serving it via `st.video()`.
- **ROI overlay disappearing after compression:** Low alpha transparency faded out during video re-encoding — fixed by increasing fill opacity and outline thickness/contrast.
- **Large/portrait video frames overwhelming the UI:** Fixed by computing a display scale that fits any frame within a fixed max width/height while preserving aspect ratio.
- **Misleading FPS trend with frame skipping:** FPS was only measured on frames where detection ran, making skipped configurations appear slower rather than faster — fixed by measuring FPS against total frames processed (wall-clock throughput) rather than detection-call count alone.

---

## 🚀 Deployment

- **Live App URL (Streamlit Cloud):** [Streamlit App link Day-40 Real Time Video Analytics ](https://mlb-internship-danish-day-40-real-time-video-analytics.streamlit.app/)

- **Demo Recording (3–5 min):** [Watch Demo Here](https://drive.google.com/file/d/1i2x73xgJw6nwyT6SU7VAegBJ1ObJ1Hat/view?usp=drive_link)

---

## 👤 Author

**Danish Ali**

BS Computer Science Final Year Student — AI/ML & Computer Vision
ML Bench Summer Internship — Day 40