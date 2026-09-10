# Day 38 — Intelligent Security Monitoring System and Image Segmentation

**MLB Summer Internship | Danish Ali**

An industry-style AI application combining **event-based video analytics**
(person detection, tracking, and ROI entry/exit logging) with an
**image segmentation** module (Binary / Adaptive / Otsu thresholding).

## Features

### 1. Security Monitoring
- Detects and tracks people in a video using YOLOv8 + ByteTrack.
- Lets the user define a custom rectangular Region of Interest (ROI).
- Detects when a tracked person enters or leaves the ROI.
- Logs every entry/exit event (track ID, event type, timestamp, frame number) to `data/logs/event_log.csv`.
- Displays the number of people currently active inside the ROI in real time.

### 2. Image Segmentation
- Upload any image.
- Choose between Binary, Adaptive, or Otsu thresholding.
- Preview the original vs. segmented result side by side.
- Download the segmented output as a PNG.

## Project Structure

```
Day-38/
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── detector.py
│   ├── roi_manager.py
│   ├── event_logger.py
│   └── segmentation.py
├── utils/
│   └── helpers.py
├── data/
│   ├── sample_videos/
│   ├── sample_images/
│   └── logs/          # event_log.csv, sessions.csv, summary_report.txt
└── outputs/
    └── segmented_images/
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To expose the app publicly:

```bash
ngrok http 8501
```



**What is image segmentation?**
Segmentation is the process of dividing an image into meaningful regions
at the pixel level — separating foreground objects from the background —
rather than just drawing a bounding box around them.

**Difference between Binary, Adaptive, and Otsu Thresholding**
- *Binary*: uses one fixed threshold value for the whole image; simple but
  breaks down under uneven lighting.
- *Adaptive*: calculates a local threshold for small regions of the image,
  handling uneven lighting much better than a single global value.
- *Otsu*: automatically computes the optimal global threshold from the
  image's histogram instead of the user guessing a value; works best when
  the image has a clear bimodal (two-peak) histogram.

**Which method worked best for the dataset and why**
Otsu generally gave the cleanest separation on evenly-lit images since it
picks the statistically optimal threshold automatically. Adaptive performed
better on images with shadows or uneven lighting across the frame. Binary
was the least reliable unless the image had already-uniform lighting.

**Challenges faced during implementation**
- Reducing duplicate ENTRY/EXIT events caused by detection shaking at the
  ROI boundary — solved by logging only on actual state transitions.
- Keeping track IDs consistent across frames required ByteTrack instead of
  per-frame detection alone.
- Choosing a sensible adaptive threshold block size that works across
  differently-sized images.


## 🚀 Deployment

Deployed on **Streamlit Community Cloud**. Live app link: [Click here to view Intelligent Security Monitoring System and Image Segmenation](https://mlb-internship-danish-day-37-people-counting-system.streamlit.app/)