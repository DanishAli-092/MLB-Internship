# Day 20 – Video Processing with OpenCV

## Overview

This project covers video processing using OpenCV — reading videos frame by frame, extracting video properties, applying image processing techniques (grayscale, Gaussian Blur, Canny Edge Detection) to each frame, and saving the processed output. It also includes real-time webcam processing, a combined Mini Project tool, a mandatory 3-video comparison challenge, and a Streamlit web app deployed publicly on HuggingFace Spaces.

Since a video is simply a sequence of frames, the core idea throughout this task is: **read one frame at a time → process it like a normal image → write it back out (or display it live)**.

---

## Folder and File Structure

```
Day-20/
├── data/
│   ├── sample_video.mp4          # input video for Coding Practice + Mini Project
│   ├── challenge_video1.mp4      # input video 1 for Challenge Task
│   ├── challenge_video2.mp4      # input video 2 for Challenge Task
│   └── challenge_video3.mp4      # input video 3 for Challenge Task
│
├── output/
│   ├── processed_video.mp4       # output of video_processing.py
│   ├── webcam_output.mp4         # output of webcam_processing.py
│   ├── mini_project_video.mp4    # output of mini_project.py (file mode)
│   ├── mini_project_webcam.mp4   # output of mini_project.py (webcam mode)
│   └── challenge/
│       ├── original_video1.mp4
│       ├── processed_video1.mp4
│       ├── original_video2.mp4
│       ├── processed_video2.mp4
│       ├── original_video3.mp4
│       ├── processed_video3.mp4
│       └── comparison_summary.txt
│
├── examples/
│   ├── original_sample.jpg       # sample frame shown in the HuggingFace app
│   └── processed_sample.jpg      # sample processed frame shown in the HuggingFace app
│
├── screen_recording/
│   └── day20_explanation.mp4     # 3–5 min walkthrough recording
│
├── video_processing.py           # Coding Practice: read, display, print properties,
│                                  # grayscale + Canny, save
├── webcam_processing.py          # Real-time webcam capture, grayscale + Gaussian
│                                  # Blur + Canny, save
├── mini_project.py               # Real-Time Video Processing Tool (VideoProcessor
│                                  # class) — handles both file and webcam input
├── challenge_task.py             # Challenge Task: processes 3 videos, saves
│                                  # originals + processed versions, generates
│                                  # comparison metrics
├── app.py                        # Streamlit app deployed on HuggingFace Spaces
├── requirements.txt
└── README.md
```

---

## How OpenCV Reads Videos

OpenCV treats a video as a stream of individual image frames rather than one single file to load into memory all at once. Access to that stream happens through the `cv2.VideoCapture` object, which can point either to a video file path or to a live camera index (`0` for the default webcam).

```python
cap = cv2.VideoCapture("data/sample_video.mp4")   # from a file
cap = cv2.VideoCapture(0)                          # from a webcam
```

Frames are then pulled out one at a time in a loop using `.read()`, which returns two values:

- `ret` — a boolean indicating whether a frame was successfully retrieved
- `frame` — the actual frame, returned as a NumPy array, exactly like a normal image

```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # process frame here
```

`ret` becomes `False` once the video ends (or a frame can't be read), which is the natural signal to break out of the loop. This is why every processing script in this project follows the same read → process → write pattern, whether the source is a video file or a live webcam feed.

Once done, `cap.release()` is called to free the video/camera resource, and `cv2.destroyAllWindows()` closes any OpenCV display windows.

---

## What FPS Means

FPS (Frames Per Second) is the number of individual frames displayed every second during playback. It determines how smooth the motion looks — a higher FPS means more frames are shown per second, producing smoother motion, while a lower FPS can look choppy.

FPS is read directly from the video's metadata:

```python
fps = cap.get(cv2.CAP_PROP_FPS)
```

FPS matters in two places in this project:

1. **Saving output** — `cv2.VideoWriter` needs the correct FPS value, otherwise the saved video plays back faster or slower than the original.
2. **Live display timing** — `cv2.waitKey(delay)` controls how long each frame is held on screen. Using a dynamic delay calculated as `int(1000 / fps)` makes the preview play at the video's natural speed instead of an arbitrary fixed speed.

Other useful video properties extracted the same way:

```python
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
```

---

## Processing Techniques Applied

| Technique | Purpose |
|---|---|
| **Grayscale Conversion** (`cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`) | Reduces each frame from 3 color channels to 1 intensity channel — required before edge detection. |
| **Gaussian Blur** (`cv2.GaussianBlur(gray, (5,5), sigmaX=0)`) | Smooths out sensor noise before edge detection. Without this, Canny picks up small pixel-level noise as false edges, especially noticeable on webcam feeds in normal room lighting. |
| **Canny Edge Detection** (`cv2.Canny(blurred, low, high)`) | Detects edges based on intensity gradients, producing a clean outline of shapes/objects in the frame. |
| **Grayscale → BGR Conversion** (`cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)`) | Canny output is single-channel; `cv2.VideoWriter` expects 3-channel frames, so this conversion is required before writing/saving, otherwise the output video is corrupted. |

**Where each technique was applied:**
- `video_processing.py` (Coding Practice) — Grayscale + Canny (as specified in the base requirement)
- `webcam_processing.py` — Grayscale + **Gaussian Blur** + Canny (Blur added beyond the base spec because webcam sensors introduce visible noise in normal lighting; this matches the pipeline required in the Mini Project and produces cleaner, more stable edges)
- `mini_project.py` and `challenge_task.py` — Grayscale + Gaussian Blur + Canny (as required)
- `app.py` (Streamlit) — same Grayscale + Gaussian Blur + Canny pipeline, with adjustable Canny thresholds via sliders

---

## Challenge Task: 3-Video Comparison

Run with:
```powershell
python challenge_task.py
```

### Auto-Generated Metrics

```
=== Challenge Task: Comparison of Results ===

video1 (data/challenge_video1.mp4): 1662 frames, 30.0 FPS, 1280x720, avg edge density = 0.0162
video2 (data/challenge_video2.mp4): 1440 frames, 24.0 FPS, 1280x720, avg edge density = 0.0575
video3 (data/challenge_video3.mp4): 721 frames, 29.97 FPS, 1280x720, avg edge density = 0.0431

--- Comparison ---
video2 had the HIGHEST average edge density (0.0575), meaning it likely contains more texture, detail, or motion between frames.
video1 had the LOWEST average edge density (0.0162), suggesting a simpler scene, less motion, or smoother/flatter surfaces.

Resolutions across videos: {'video1': '1280x720', 'video2': '1280x720', 'video3': '1280x720'}
FPS across videos: {'video1': 30.0, 'video2': 24.0, 'video3': 29.97}
```

*Edge density = the fraction of pixels detected as edges in the processed (Canny) frame, averaged across all frames. It's used here as an automated, quantitative proxy for how visually complex/detailed a video is.*

### My Observations (After Watching the Processed Videos)

| Video | What I Observed |
|---|---|
| Video 1 (edge density: 0.0162) | Balcony & Ocean:A static, calm scene with minimal movement. Edges are clean, sharp, and highly stable, mostly capturing architectural lines and the horizon. Lowest edge density. |
| Video 2 (edge density: 0.0575) | Daytime Traffic: A timelapse with intense motion and complex details (road lines, vehicles). Fast-moving traffic results in highly dense, flickering edges. Highest edge density. |
| Video 3 (edge density: 0.0431) | Nighttime Street: Low lighting hides background details, but moving pedestrians and bright streetlights create sharp, high-contrast edges. Shows noticeable camera sensor noise. |

---

## Challenges / Blockers Faced While Working With Video Frames

- **Grayscale-to-BGR conversion before saving:** Canny output is a single-channel (2D) array, but `cv2.VideoWriter` expects 3-channel frames. Writing single-channel frames directly resulted in a corrupted/broken output video until `cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)` was added before every `.write()` call.
- **Webcam FPS unreliability:** `cap.get(cv2.CAP_PROP_FPS)` sometimes returned `0` or `NaN` for the webcam instead of a real value, which would break FPS-dependent calculations (like `1000 / fps` for playback delay). Fixed with a fallback: if FPS is `0` or `NaN`, default to `20.0`.
- **Hardcoded vs. dynamic playback delay:** Initially used a fixed `cv2.waitKey(25)` for preview playback, which forced every video to display at roughly the same speed regardless of its actual FPS. Switched to a dynamic delay (`int(1000 / fps)`) with a safe fallback so the preview matches each video's natural speed.
- **H.264 (avc1) codec missing on Windows:** When writing videos with the `avc1` (H.264) codec for browser-compatible playback in the Streamlit app, Windows threw `Failed to load OpenH264 library` errors because the required DLL isn't bundled by default. Solved with a fallback function that tries `avc1` first and automatically switches to `mp4v` if the writer fails to open — this keeps the app working on any machine, though `mp4v` output doesn't preview inline in all browsers (the download button always works regardless). On HuggingFace's Linux-based deployment environment, H.264 encoding is available out of the box, so the app preview works correctly there.
- **Webcam noise affecting edge detection:** Webcam frames, especially in normal room lighting, contain visible sensor noise. Running Canny directly on grayscale (without blurring first) produced flickery, noisy edges. Adding a Gaussian Blur step before Canny fixed this.
- **opencv-python vs. opencv-python-headless for deployment:** The standard `opencv-python` package depends on GUI libraries that aren't available on HuggingFace's server environment, which breaks the build. Switched to `opencv-python-headless` in `requirements.txt`, which has no GUI dependency and works correctly in a hosted/server context.

---

## How to Run

```powershell
# Coding Practice
python video_processing.py

# Webcam script
python webcam_processing.py

# Mini Project (file + webcam)
python mini_project.py

# Challenge Task (3-video comparison)
python challenge_task.py

# Streamlit app (local)
python -m streamlit run app.py
```

---

