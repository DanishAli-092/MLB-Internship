# 🎯 Day 27 — Object Detection with YOLOv8

**MLB Summer Internship | Danish Ali**

A Streamlit-based object detection application built on top of a pre-trained
YOLOv8 model, capable of detecting objects in both images and videos with
adjustable confidence thresholds, per-class colored bounding boxes, and
downloadable results.

---

## 📖 What is Object Detection?

Object Detection is a Computer Vision task where the goal is not just to say
what is in an image, but also *where* it is. The model draws a bounding box
around every object it finds and assigns a class label plus a confidence
score to each one. Unlike simple classification, it can find and label
multiple objects in a single image at the same time.

## 🆚 How is YOLO different from Image Classification?

Image classification looks at the whole image and outputs a single label
("this is a dog photo"). It has no idea where the dog actually is in the
frame, and it breaks down completely if there is more than one object in
the picture.

YOLO (You Only Look Once) instead scans the entire image in a single pass
of the network and directly predicts bounding boxes, class probabilities,
and confidence scores for every object it detects — all at once, which is
what makes it fast enough for real-time video.

## 🧠 Which YOLO model did I use?

`yolov8n.pt` (YOLOv8 Nano) from the Ultralytics library. It's the smallest
and fastest variant, pre-trained on the COCO dataset (80 object classes),
which makes it a good fit for quick inference on a normal laptop without a
dedicated GPU.

## 🔍 What objects did the application detect?

Depends on the sample images/videos used, but since the model is trained on
COCO, common detections included things like: person, car, dog, cat,
bicycle, chair, bottle, laptop, etc. (Add your actual detected classes here
after running the app on your own samples.)

---

## ⚠️ Edge Cases & AI Limitations Discovered

While testing the application on real-world images, a few interesting model
limitations came up that are worth documenting — they say a lot about how a
COCO-pretrained Nano model actually "sees" the world.

### 1. Geometric Similarity (iPhone vs. TV Remote)
The YOLOv8n model misclassified the camera module of an iPhone as a
`remote` (0.69 confidence). This highlights the speed-vs-accuracy trade-off
of the Nano model and how it gets confused by geometric similarities —
camera lenses arranged in a row visually resemble the buttons on a remote
control, and the model latched onto that shape pattern rather than true
object identity.

### 2. Missing Dataset Classes (The "Tree" Issue)
The model failed to draw a bounding box around a prominent tree in one of
the test images. This is **not a code bug** — it's a limitation of the base
COCO dataset, which contains 80 classes but does not include a generic
`tree` class at all. The model literally has no concept of "tree" to
predict. This is a good real-world example of why domain-specific
applications need **custom-trained YOLO models** rather than relying on the
default COCO weights.

### 3. Dataset Bias & Shape Bias (Classic Car vs. Truck)
A classic, boxy BMW was misclassified as a `truck` (0.56 confidence), while
modern aerodynamic cars parked next to it were correctly labeled as `car`.
This exposes two related issues:
- **Dataset bias** — COCO's `car` examples are heavily weighted toward
  modern, curvy vehicle designs, so older body styles are underrepresented.
- **Shape bias** — the flat, boxy grille and upright silhouette of the
  vintage car geometrically resembles a small truck more than it resembles
  the curved cars the model is used to seeing, so the model followed shape
  over actual vehicle type.

---

## 🛠️ Challenges Faced & Solutions Implemented

### 1. Streamlit Video Codec Playback
OpenCV natively writes videos using the `mp4v` codec, which modern browsers
refuse to play directly inside a Streamlit `st.video()` component (the
player showed up blank, even though the file itself processed correctly
and could be downloaded). I solved this by integrating the
`imageio_ffmpeg` library to dynamically re-encode the processed output into
an `h264` codec with a `+faststart` flag, allowing seamless in-browser
playback without requiring the user to have ffmpeg installed separately.

### 2. Portrait Video UI Stretching
When uploading tall portrait-orientation videos, Streamlit's default
full-width behavior stretched them massively across the screen, making the
UI look broken and unbalanced. I fixed this UI/UX issue by confining the
video player within a centered, proportional column layout
(`st.columns([1, 2, 1])`), keeping the video a manageable, aesthetically
pleasing size regardless of its aspect ratio.

### 3. Consistent Bounding Box Colors
Getting bounding box colors to stay consistent per class — instead of
changing randomly on every run — was solved by seeding the color generator
with the class ID, so the same class always renders in the same color
across every image and video.

### 4. Handling In-Memory Uploads with OpenCV
Streamlit gives uploaded files as in-memory objects, but OpenCV's video
functions need a real file path on disk. This was solved by writing
uploads to a temporary file before processing.

### 5. Label Readability on Busy Images
Bounding box labels became hard to read on cluttered/busy images. This was
fixed by drawing a filled background rectangle behind each label before
rendering the text, guaranteeing readability regardless of what's behind it.

### 6. Choosing a Sensible Confidence Threshold
Picking a single confidence threshold that filters out noise without
hiding real detections is tricky, since it varies by image. Instead of
hardcoding one value, it's exposed as an adjustable slider in the sidebar
so the threshold can be tuned per input.

---

## 📁 Project Structure
```
Day-27/
├── app.py                     # Streamlit mini project (image + video detection)
├── scripts/yolo_practice.py   # practice script for batch image/video inference
├── requirements.txt
├── README.md
├── sample_images/             # 10+ test images
├── sample_videos/             # 2 test videos
└── outputs/                   # annotated results saved here
```

## ▶️ How to Run
```powershell
pip install -r requirements.txt
streamlit run app.py
```

