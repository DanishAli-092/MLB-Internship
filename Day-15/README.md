# Day 15 — Object Detection with YOLO

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLO](https://img.shields.io/badge/YOLO-v11-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

Part of the **MLB Summer Internship** program. This module covers the
fundamentals of object detection, running inference with pretrained YOLO
models, and building an interactive detection dashboard.

---

## 📌 Overview

| | |
|---|---|
| **Task** | Object Detection using pretrained YOLO models |
| **Dataset** | [PPE Detection](https://universe.roboflow.com/sdp-lfigk/ppe-detection-ozhfb) — Roboflow Universe |
| **Models Used** | YOLO11n, YOLO11m (Ultralytics) |
| **Deliverable** | Streamlit detection dashboard + inference scripts |

---

## 🧠 Concepts Covered

### Object Detection vs Image Classification

| | Image Classification | Object Detection |
|---|---|---|
| Output | Single label for the whole image | Multiple labels + locations |
| Multiple objects | Not handled | Handled |
| Output format | Class + confidence | Bounding box + class + confidence |

### What is Object Detection?
Object detection identifies **what** objects are present in an image and
**where** they are located, by drawing a bounding box around each object
and assigning it a class label with a confidence score.

### What is YOLO?
**YOLO (You Only Look Once)** is a real-time object detection algorithm
that processes an entire image in a single forward pass through the neural
network, predicting all bounding boxes and class probabilities
simultaneously. This makes it significantly faster than older two-stage
detectors, while remaining accurate enough for real-time use cases like
video and webcam detection.

---

## 📂 Dataset

**PPE Detection** — by SDP, via Roboflow Universe

- 2,118 total images, exported in YOLOv11 format
- Split into `train` / `valid` / `test`
- Used the **test** split (213 images) for inference — training was not
  required for this task
- 🔗 [Dataset link](https://universe.roboflow.com/sdp-lfigk/ppe-detection-ozhfb)

---

## 🤖 Models Compared

| Model | Size | Speed | Notes |
|---|---|---|---|
| `yolo11n.pt` | ~5 MB | Fastest | Missed smaller/background objects |
| `yolo11m.pt` | ~40 MB | Slower | Caught more objects, higher confidence overall |

Both models are pretrained on the **COCO dataset (80 general classes)** —
not on PPE-specific classes like helmet or vest.

---

## 🔍 Objects Detected

Since the pretrained models were trained on COCO rather than this
dataset's actual PPE labels, they detected general COCO classes instead
of PPE items:

- ✅ **person** — consistently detected across nearly all images
- ⚠️ **suitcase**, **potted plant** — occasional false/irrelevant detections
- ⚠️ **chair**, **couch** — detected in some sample images, especially with `yolo11m`
- ❌ **helmet / vest / gloves** — never detected (not part of COCO's 80 classes)

---

## 📝 Observations

- The model can only detect classes it was trained on. Since COCO doesn't
  include PPE-specific labels, "person" was detected reliably, but no PPE
  item was ever recognized as such.
- Switching from `yolo11n` → `yolo11m` improved detection quality —
  it caught background objects the nano model missed and produced higher
  confidence scores overall, at the cost of slightly slower inference.
- Confidence scores dropped for small, blurry, or partially visible
  objects, and rose for clear, well-lit, centered ones.
- Running the same image individually vs. in a batch produced slightly
  different confidence scores, due to YOLO's resizing/letterboxing during
  batch preprocessing — this can push a borderline detection above or
  below the confidence threshold between runs.

> **Conclusion:** To actually detect PPE items (helmet, vest, no-helmet,
> no-vest, etc.), the model needs to be fine-tuned on this dataset's real
> PPE labels — covered in the next session.

---

## 🗂️ Project Structure

```
Day-15/
├── practice1_yolo_basics/
│   └── yolo_basics.py
├── practice2_own_images/
│   └── test_own_images.py
├── project_ppe_detection/
│   ├── outputs/
│   ├── download_dataset.py
│   └── ppe_detection.py
├── sample_images/
├── streamlit_app/
│   └── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the practice scripts**
```bash
python practice1_yolo_basics/yolo_basics.py
python practice2_own_images/test_own_images.py
```

**3. Download the dataset and run PPE inference**
```bash
python project_ppe_detection/download_dataset.py
python project_ppe_detection/ppe_detection.py
```

**4. Launch the detection dashboard**
```bash
cd streamlit_app
streamlit run app.py
```

> Note: Roboflow dataset download requires a `.env` file with
> `ROBOFLOW_API_KEY=your_key` — this is not included in the repo for
> security reasons.

---

## 📊 Streamlit Dashboard Features

- Upload an image or video for detection
- Switch between `yolo11n` / `yolo11s` / `yolo11m` on the fly
- Adjustable confidence threshold slider
- Side-by-side original vs. detected image view
- Summary metrics — total objects, unique classes, average confidence
- Interactive Plotly charts — objects by class, confidence per detection,
  class distribution
- Download the annotated result image/video

---

