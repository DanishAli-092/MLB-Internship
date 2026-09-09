# Day 37 — Smart People Counting & Crowd Analysis System

**MLB Summer Internship | Danish Ali**

A computer vision application built with **YOLO** and **Streamlit** that detects, tracks, and counts people in images and videos. Built as part of the MLB (ML Bench) Summer Internship.

---

## 📌 Overview

People counting is widely used in shopping malls, offices, airports, retail stores, and public spaces to monitor occupancy, analyze customer flow, and improve security. This project implements a complete pipeline for detecting people, tracking them with consistent IDs across frames, and generating crowd analytics.

---

## ✨ Features

- **People Detection** — YOLO-based detection filtered to the `person` class only
- **Consistent Tracking** — Each person keeps the same ID across frames using BoT-SORT (motion + appearance based tracking, more resilient to occlusion than plain ByteTrack)
- **Live People Count** — Real-time count of people visible in each frame
- **Bounding Boxes + Confidence Scores** — Drawn on every detected person
- **Peak Occupancy Tracking** — Maximum number of people seen at any point during the video
- **Two Counting Modes**:
  - **Line Crossing (Entry/Exit)** — A virtual line; crossing direction determines entry vs exit
  - **Region Based Counting** — A movable, resizable zone; counts how many people are inside it at any moment
- **Interactive Setup** — Line/region position is set visually on the video's actual first frame before processing (no guessing coordinates blind)
- **Image & Video Support** — Upload either, get processed output with boxes and stats
- **Playable Previews** — Original and processed video both playable directly in the app
- **Downloadable Output** — Processed image/video can be downloaded
- **Deployed on Streamlit**

---

## 🗂️ Project Structure

```
Day-37/
├── app.py                     # Streamlit application (main entry point)
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── detector.py            # YOLO person detector + tracker wrapper
│   ├── tracker_counter.py     # Line crossing & region counting logic
│   └── utils.py                # Drawing helpers (boxes, labels, info panel, line, region)
├── data/
│   ├── videos/                 # Sample test videos
│   └── images/                 # Sample test images
├── outputs/
│   ├── videos/                  # Processed video outputs
│   └── images/                  # Processed image outputs
└── models/                       # (optional) local YOLO weights
```

---

## 🧠 Key Concepts

| Concept | Summary |
|---|---|
| **People Detection** | YOLO detects objects and filters to COCO class `0` (person) |
| **ROI (Region of Interest)** | A defined area used to limit counting to a specific zone |
| **Line Crossing vs Region Counting** | Line crossing tracks directional entry/exit; region counting tracks live occupancy inside a zone |
| **Occlusion Handling** | BoT-SORT uses appearance (ReID) + motion to reduce ID switches when people overlap |
| **Peak Occupancy** | Running maximum of the live count across the whole video |

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
```
ultralytics==8.4.115
lap
streamlit>=1.30.0
opencv-python-headless==4.10.0.84
numpy==2.3.5
Pillow>=10.0.0
pandas>=2.0.0
imageio>=2.31.0
imageio-ffmpeg>=0.5.1

```

---

## ▶️ Usage

Run the app locally:

```bash
streamlit run app.py
```

1. Choose **Image** or **Video** input from the sidebar
2. For video, choose a counting mode: Simple Count, Line Crossing, or Region Based
3. If a counting mode is selected, position the line/region using the sliders — a live     preview on the actual first frame updates as you adjust
4. Click **Process Video** (or upload an image, which processes instantly)
5. View live stats (Live Count, Peak Count, Entries/Exits) during processing
6. Watch the processed output and download it

---

## 📊 Dataset

Tested on 5+ videos covering different crowd scenarios:
- Shopping mall footage
- Office / hallway footage
- Campus / classroom footage
- Public street footage

Sourced from Pexels/Pixabay (free stock footage) and personal recordings.

---

## ⚠️ Known Limitations

- **Full occlusion** (e.g. a person completely hidden behind a static object like a tree/pillar for several seconds) can still cause an ID switch — this is an inherent limitation of any motion+appearance tracker when no visible pixels remain to match against.
- Heavy crowd density can reduce detection accuracy due to overlapping bounding boxes.
- Tracking accuracy depends on video quality, frame rate, and camera angle.

---

## 🚀 Deployment

Deployed on **Streamlit Community Cloud**. Live app link: _[add your deployed URL here]_