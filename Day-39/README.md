# Day - 39 🛡️ Intelligent Security Monitoring and Image Segmentation System

**MLB Summer Internship | Danish Ali**

An improved, production-style version of the Day-38 Security Monitoring project built with **YOLOv8 + ByteTrack** for person detection/tracking, custom ROI-based entry/exit event logging, and an image segmentation module, all wrapped in a polished Streamlit app.

This is an optimization pass on the original project: faster, more configurable, more robust, and easier to use following the Day-39 focus on refining an existing application rather than building from scratch.

## ✨ Features

- **Confidence & IoU threshold sliders** — tune detection sensitivity and duplicate-box suppression live from the sidebar.
- **Interactive ROI selection** — draw one or more regions of interest directly on the video's first frame or an uploaded image.
- **Selectable tracker (ByteTrack / BoT-SORT+ReID)** — choose the tracking algorithm live from the sidebar: ByteTrack (fast, motion-only) or BoT-SORT+ReID (also matches appearance, holds IDs more reliably through brief occlusions but slower).
- **Image + video support** — run the same detection/ROI pipeline on a single image or a full video.
- **Processing progress bar** — live frame-by-frame progress while a video is processed.
- **CSV report generation** — raw event log, entry/exit sessions with duration, and a text summary report (unique visitors, peak occupancy, average visit time).
- **Downloadable results** — processed video, annotated image, and all CSV/report files.
- **Error handling** — corrupted files, unreadable frames, failed model loads, and failed inference are caught and surfaced as clear messages instead of crashing the app.
- **Image Segmentation module** — Binary / Adaptive / Otsu thresholding on any uploaded image, with side-by-side comparison and download.

## 🖥️ Deployment

- **Live app:** Deployed on **Streamlit Community Cloud**. Live app link: [Click here to view improved Intelligent Security Monitoring System and Image Segmenation](https://mlb-internship-dani-day-39-security-monitor-img-segmentation.streamlit.app/)

## 📁 Project Structure

```
Day-39/
├── app.py                     # Main Streamlit app
├── modules/
│   ├── detector.py            # YOLO detection + ByteTrack tracking
│   ├── roi_manager.py         # ROI polygon logic (point-in-polygon)
│   ├── event_logger.py        # Entry/exit event + session CSV logging
│   └── segmentation.py        # Binary/Adaptive/Otsu thresholding
├── utils/
│   └── helpers.py             # Image conversions + drawing utilities
├── data/
│   ├── sample_videos/         # Sample input videos (incl. ID-crossing demo pair)
│   ├── sample_images/         # Sample input images
│   └── logs/                  # Generated event_log.csv / sessions.csv
├── outputs/
│   ├── processed_videos/      # Annotated output videos
│   ├── monitored_images/      # Annotated output images
│   └── segmented_images/      # Segmentation outputs
├── yolov8n.pt                 # YOLO weights
├── byte_track.yaml            # ByteTrack tracker config (fast, motion-only)
├── botsort_reid.yaml          # BoT-SORT + ReID tracker config (appearance-aware, more consistent)
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd Day-39
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Usage

```bash
streamlit run app.py
```

Then in the app:

1. Open the **🎥 Security Monitoring** tab.
2. In the sidebar, choose the **tracking algorithm** (ByteTrack for speed, BoT-SORT+ReID for more consistent IDs through occlusion) and adjust confidence/IoU/resize-width as needed.
3. Upload a **video or image**.
4. Draw one or more **ROIs** on the preview frame.
5. Click **Process** — watch the progress bar, then view/download the annotated result and CSV reports.
6. Switch to the **🧩 Image Segmentation** tab to try thresholding on any image.

## 🧠 Tech Stack

- **Detection & Tracking:** Ultralytics YOLOv8 (`yolov8n.pt`), ByteTrack , Botsort
- **Computer Vision:** OpenCV
- **UI:** Streamlit, `streamlit-drawable-canvas`
- **Data:** Pandas (CSV reports)
- **Video encoding:** `imageio-ffmpeg`

## 🚀 Performance Optimizations (Day 39 focus)

- **Inference resize** — frames wider than the selected width (default 640px) are downscaled before running YOLO and boxes are scaled back up; this is the main inference-speed lever, adjustable live from the sidebar.
- **Confidence & IoU tuning** — both exposed as sliders instead of hardcoded, so precision/recall and duplicate-box suppression can be tuned per video without touching code.
- **Cached model loading** — `st.cache_resource` ensures the YOLO model is loaded into memory once per session, not on every rerun.
- **Nano model (`yolov8n.pt`)** — smallest/fastest YOLOv8 variant, chosen deliberately for near real-time performance over a larger, slower model.
- **Graceful degradation** — a single frame that fails inference no longer crashes the whole run; it's logged and skipped so processing continues.
- **Tracker: user-selectable** — a sidebar dropdown lets you switch between ByteTrack (`byte_track.yaml`) and BoT-SORT+ReID (`botsort_reid.yaml`) per run. Testing across occlusion cases showed BoT-SORT+ReID retains IDs more reliably when one person briefly blocks another, at the cost of extra processing time — so both are exposed and the user can pick based on their speed/consistency priority.
- **Observed accuracy/speed trade-off** — lowering the inference resize width increases speed but produces blurrier detections, which causes more track ID switches and an inflated occupancy/visitor count; higher resize widths give fewer, more accurate counts at the cost of processing speed.

## 📌 Notes

- Track-ID consistency was evaluated across three tracker configurations (default ByteTrack, tuned ByteTrack, BoT-SORT+ReID). BoT-SORT+ReID performed best on partial/close-range occlusion. Full occlusion recovery in far-field, low-resolution footage remains a known open limitation of real-time trackers in general, not specific to this implementation.

- The two crossing-path sample videos in `data/sample_videos/` demonstrate that track IDs remain stable when two people's paths intersect — this was verified manually by inspecting the annotated output frame-by-frame.
- Model weights (`yolov8n.pt`) are pre-trained on COCO and used out of the box for the `person` class (class 0).

## 👤 Author

**Danish Ali** — Final-year CS student (AI/ML & Computer Vision), ML Bench Summer Internship
