# 🖼️ Day 16 — OpenCV Fundamentals & Image Processing Toolkit

**Machine Learning  Internship ML BENCH — Day 16**
**Author:** Danish Ali


---

## 📌 Overview

This project covers the fundamentals of Computer Vision using **OpenCV** — how images are read, represented, and manipulated at the pixel level. It includes standalone practice programs, a reusable object-oriented **Image Processing Toolkit**, a **Challenge Task** that applies every operation across five different image categories, and a fully interactive **Streamlit web app** version of the toolkit.

This forms the foundation before moving on to advanced topics like YOLO and Object Detection later.

---

## 📂 Folder Structure

```
Day-16/
│
├── image_toolkit/
│   ├── __pycache__/
│   └── toolkit.py                  # Core ImageToolkit class (menu-driven CLI app)
│
├── opencv_practice/
│   ├── challenge_task.py           # Challenge Task — runs all ops on 5 category images
│   └── practice_programs.py        # Day 16 coding practice programs
│
├── output_images/
│   ├── challenge_task/
│   │   ├── document/
│   │   ├── landscape/
│   │   ├── object/
│   │   ├── person/
│   │   └── vehicle/
│   ├── practice/                   # Outputs from practice_programs.py
│   └── toolkit/
│       └── processed_vehicle.jpg
│
├── sample_images/
│   ├── document.jpg
│   ├── landscape.jpg
│   ├── object.jpg
│   ├── person.jpg
│   ├── sample1.jpg
│   └── vehicle.jpg
│
├── screen_recording/                # Demo video (3–5 min walkthrough)
│
├── streamlit_app/
│   ├── app.py                      # Streamlit web version of the toolkit
│   ├── README.md
│   └── requirements.txt
│
└── README.md                       # This file
```

---

## 🎯 Objectives

- Understand how images are represented in OpenCV (arrays, channels, color spaces).
- Perform common image processing operations confidently.
- Build a reusable, menu-driven image processing application.
- Deploy an interactive web version using Streamlit.
- Organize code into clean, reusable functions and classes.

---

## 🧠 Core Concepts

### 🔴🟢🔵 BGR vs RGB

OpenCV reads and stores color images in **BGR (Blue-Green-Red)** channel order by default, instead of the more commonly used **RGB (Red-Green-Blue)** order used by most image libraries (like PIL, Matplotlib, and web browsers).

| | BGR (OpenCV default) | RGB (Standard) |
|---|---|---|
| Channel order | Blue → Green → Red | Red → Green → Blue |
| Used by | OpenCV (`cv2.imread`, `cv2.imshow`) | PIL, Matplotlib, Streamlit, most displays |

**Why it matters:** If a BGR image is displayed using a tool that expects RGB (e.g. Streamlit's `st.image()` or Matplotlib), the red and blue channels appear swapped — producing an unnatural, bluish-tinted result often called the **"Smurf Effect."**

**Fix:** Convert before displaying:
```python
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
```

This project includes a dedicated **"Compare BGR vs RGB"** feature in both the CLI toolkit and the Streamlit app to visually demonstrate this difference side by side.

### ⚫⚪ Grayscale Images

A grayscale image stores only **light intensity** (brightness) per pixel instead of three separate color channels — reducing a `(H, W, 3)` array down to `(H, W)`.

**Why grayscale is used:**
- Reduces memory and computation (1 channel instead of 3).
- Removes color as a variable, useful when only shape, edges, or structure matter.
- Standard preprocessing step for many classical CV algorithms (edge detection, thresholding, feature matching) and even some deep learning pipelines.
- Simplifies operations like contour detection and Canny edge detection.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

## 🛠️ OpenCV Functions Used

| Function | Purpose |
|---|---|
| `cv2.imread()` | Load an image from disk (as BGR) |
| `cv2.imwrite()` | Save an image to disk |
| `cv2.imshow()` / `cv2.waitKey()` / `cv2.destroyAllWindows()` | Display images in a window (CLI toolkit) |
| `cv2.cvtColor()` | Convert between color spaces (BGR ↔ RGB, BGR ↔ Grayscale) |
| `cv2.resize()` | Resize image to a target width/height |
| `cv2.rotate()` | Rotate image by 90°, 180°, 270° |
| `cv2.flip()` | Flip image horizontally, vertically, or both |
| NumPy slicing `img[y1:y2, x1:x2]` | Crop a region of interest |
| `cv2.rectangle()` | Draw a rectangle |
| `cv2.circle()` | Draw a circle |
| `cv2.line()` | Draw a straight line |
| `cv2.polylines()` | Draw a closed/open polygon |
| `cv2.putText()` | Overlay custom text on an image |
| `cv2.convertScaleAbs()` | Adjust brightness (`beta`) and contrast (`alpha`) |
| `np.hstack()` | Stack images side by side for comparison |
| `cv2.imencode()` | Encode processed image into bytes for download (Streamlit app) |

---

## 🚀 Project Components

### 1️⃣ Practice Programs (`opencv_practice/practice_programs.py`)
Standalone functions covering all Day 16 fundamentals:
- Image info (dimensions, channels, file size)
- Grayscale conversion
- Resizing to multiple resolutions
- Cropping into 5 regions (corners + center)
- Rotation (90°, 180°, 270°)
- Horizontal & vertical flipping
- Drawing shapes (rectangle, circle, line, polygon) + custom text

### 2️⃣ Image Processing Toolkit (`image_toolkit/toolkit.py`)
An object-oriented, **menu-driven CLI application** (`ImageToolkit` class) supporting:
- Load, reset, save, and display images
- Grayscale, resize, rotate, flip, crop
- Draw shapes and add custom text
- **Bonus:** Brightness/contrast adjustment, BGR vs RGB comparison, side-by-side original vs processed view

### 3️⃣ Challenge Task (`opencv_practice/challenge_task.py`)
Reuses the `ImageToolkit` class to apply **every operation** (grayscale, resize, crop, rotate ×3, flip ×2, shapes + text, brightness/contrast) across **5 different image categories**:

| Category | Sample Image |
|---|---|
| 🏞️ Landscape | `landscape.jpg` |
| 👤 Person | `person.jpg` |
| 🚗 Vehicle | `vehicle.jpg` |
| 📄 Document | `document.jpg` |
| 📦 Object | `object.jpg` |

Each category's results are saved into its own labeled subfolder under `output_images/challenge_task/`.

### 4️⃣ Streamlit Web App (`streamlit_app/app.py`)
A fully interactive browser-based version of the toolkit — **VisionKit** — featuring:
- Sidebar image upload with instant Original vs Processed comparison
- Live image metadata panel (width, height, channels, memory size)
- Undo & Reset history stack
- All 9 core + bonus operations available through a dropdown
- One-click PNG download of the processed image

---

## ▶️ How to Run

### CLI Toolkit
```bash
cd image_toolkit
python toolkit.py
```

### Practice Programs
```bash
cd opencv_practice
python practice_programs.py
```

### Challenge Task
```bash
cd opencv_practice
python challenge_task.py
```

### Streamlit App
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Challenges Faced & Solutions

| Challenge | Solution |
|---|---|
| Displayed images looked blue/wrong-colored in Streamlit | Root cause traced to BGR vs RGB mismatch; added a `to_display_format()` helper that converts BGR → RGB before every `st.image()` call, plus a dedicated comparison feature to visualize the bug directly. |
| Repeated crop/resize/rotate on the same image compounded errors and made undo impossible | Introduced a `history` stack in session state — every operation pushes the previous processed image before applying a new one, enabling a proper Undo button. |
| Grayscale output couldn't be combined with color images for side-by-side display (`np.hstack` shape mismatch) | Converted single-channel grayscale results back to 3-channel BGR (`cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)`) before stacking/displaying. |
| Shapes/text drawn with fixed pixel sizes looked too small on large images and too large on small ones | Calculated font scale, text thickness, and circle radius dynamically relative to image width/height instead of using constants. |
| Invalid crop coordinates (x1 ≥ x2 or y1 ≥ y2) caused silent errors or empty arrays | Added explicit validation with clear error messages before slicing the image array. |
| Downloading the processed image from Streamlit required raw bytes, not a file path | Used `cv2.imencode()` to encode the OpenCV array into PNG bytes wrapped in a `BytesIO` buffer for `st.download_button()`. |

---

## ✅ Deliverables Checklist

- [x] OpenCV Practice Programs
- [x] Image Processing Toolkit (menu-driven CLI)
- [x] Sample Input Images (5 categories)
- [x] Processed Output Images (organized by category)
- [x] Streamlit Web App (public link below)
- [x] README.md (this file)
- [x] Screen recording demo (3–5 minutes)

---


## 🧰 Tech Stack

`Python` · `OpenCV (cv2)` · `NumPy` · `Streamlit` · `Pillow (PIL)`

---

*Day 16 — Machine Learning  Internship | OpenCV Fundamentals & Basic Image Processing*