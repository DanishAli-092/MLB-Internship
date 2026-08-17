# Day 21 — Computer Vision Image Processing Studio

**MLB Summer Internship | Danish Ali**


---

## 1. Overview

Up until now (Day 16–20) we learned individual OpenCV concepts — transformations, filters, edge detection, contours, video processing, etc. The goal for Day 21 was to combine all of these into **one single, complete, deployable application**, exactly the way a real AI developer would build it: the user uploads an image, chooses an operation, sees the result, and downloads it.

This project is a **Streamlit-based Computer Vision Image Processing Studio** where:
- The user can upload any image
- Choose from 11 different OpenCV operations (via a dropdown or a multi-step pipeline)
- View the original and processed images side by side
- Download the result

---

## 2. Folder & File Structure

```
Day-21/
├── app.py                 # The entire application - UI + all image processing logic
├── requirements.txt        # Dependencies (opencv-python-headless for deployment)
├── README.md               # This file
├── sample_inputs/          # Input images used for testing
├── sample_outputs/         # Corresponding processed output images
└── screen_recording/
    └── link.md             # Google Drive link to the demo recording
```

**PowerShell commands (to create the structure):**
```powershell
mkdir Day-21
cd Day-21
mkdir sample_inputs
mkdir sample_outputs
mkdir screen_recording
New-Item app.py -ItemType File
New-Item requirements.txt -ItemType File
New-Item README.md -ItemType File
New-Item screen_recording/link.md -ItemType File
```

---

## 3. Topics for Today

### 3.1 Organizing your project

Instead of writing everything in one messy script, I clearly separated the code into:
- **UI layer** (Streamlit widgets: uploader, dropdown, sliders) separate
- **Logic layer** (OpenCV processing functions) separate

This is separation of concerns — if tomorrow someone wants to use this CV logic in another project (like a CLI tool or an API), only the logic functions need to be copied, no Streamlit code needs to be dragged along.

### 3.2 Writing clean and reusable code

- Each operation has its own function — `op_grayscale()`, `op_blur()`, `op_rotate()`, etc. Each function does one job (Single Responsibility Principle).
- All operations are mapped in a **dispatch dictionary** (`OPERATIONS`) — to add a new operation, you just write a function and add one entry to the dictionary, no long if-elif jungle.
- Consistent function signature: every function takes a BGR numpy array and returns a numpy array — this makes it easy to chain functions together (pipeline).

### 3.3 Creating a simple user interface with Streamlit

Building the UI in Streamlit means calling Python functions:
- `st.file_uploader()` — image upload
- `st.selectbox()` / `st.multiselect()` — choosing an operation (single dropdown for the base requirement, multiselect for the challenge's "pipeline mode")
- `st.columns()` — original and processed image side by side
- `st.slider()` / `st.radio()` — operation-specific parameters (blur kernel size, rotation angle, etc.)
- `st.download_button()` — downloading the processed result

Important technical detail learned: OpenCV uses the BGR format, while PIL/Streamlit expect RGB. Conversion is needed on every upload and display (that's why I built the `pil_to_cv()` and `cv_to_pil()` helper functions), otherwise the colors show up inverted (bluish).

### 3.4 Deploying AI Applications

**What is Hugging Face Spaces?**
A free hosting platform where you can upload your ML/CV app and get a live public link. Every Space is actually a Git repository — push `app.py` and `requirements.txt`, and HF automatically builds and deploys it.

**Gradio vs Streamlit?**
- **Streamlit**: reactive re-execution model — whenever any widget changes, the entire script re-runs top-to-bottom. Layout control (columns, sidebar) is more flexible.
- **Gradio**: function-based — you wrap a function in `gr.Interface()`. It's HF Spaces' default/native choice, but Streamlit is also fully supported.
- Since Streamlit was already being used on Days 15–20, I chose Streamlit again on Day 21 for consistency.

**How do you deploy?**
1. Create an HF account, "New Space" → select SDK = Streamlit
2. README.md needs a YAML metadata block at the top (`sdk`, `app_file`, etc. — already in this file)
3. `git remote add space <space-url>` then `git push space main`

**How do you update a deployed app?**
Just change the code, commit it, and `git push` again — HF automatically rebuilds it. Same thing on Streamlit Community Cloud — as soon as you push to GitHub, it auto-redeploys.

---

## 4. What We Delivered (Deliverables)

| Requirement | Status |
|---|---|
| Image upload | ✅ |
| Select a single operation from dropdown | ✅ |
| Process + display output | ✅ |
| Download processed result | ✅ |
| 7 required operations (Grayscale, Blur, Edge Detection, Rotation, Enhancement, Contour Detection, Shape Detection) | ✅ all 7 |
| Original + processed side-by-side | ✅ |
| Clean, easy-to-use UI | ✅ |
| Challenge: custom feature | ✅ 4 extra ops (Brightness/Contrast, Sharpen, Flip, Threshold) + Pipeline Mode (sequential multi-filter chaining) |
| Deployment-ready files (app.py + requirements.txt) | ✅ |

**A total of 11 operations** were implemented (7 required + 4 challenge), plus a **Pipeline Mode** where the user can chain multiple operations in order — like Blur → Sharpen → Edge Detection in a single run.

---

## 5. Challenges Faced and Their Solutions

This is the most important section — because all the bugs encountered were genuinely real, and something was learned from each one:

### Challenge 1: Contour Detection was only highlighting the outer border
**Problem:** In the `THRESH_OTSU` + `RETR_EXTERNAL` approach, if the background was lighter than the object (e.g., black shapes on a white background), the entire background would become one giant "contour" — individual shapes weren't being detected at all.
**Solution:** Instead of threshold-based binarization, I used **Canny edge detection** (polarity-independent — works reliably whether the object is dark or light), and used `RETR_LIST` instead of `RETR_EXTERNAL` so that nested/internal shapes get discarded.

### Challenge 2: Duplicate contours and labels
**Problem:** When a shape's fill color and border/stroke color were different (like a yellow-filled triangle with a blue border), Canny would detect two parallel edges — one at the fill-boundary, one at the stroke-boundary. This caused the same shape to be labeled twice ("Circle" + "Triangle" overlapping).
**Solution:** I wrote a dedup logic that checks both the bounding-box **containment ratio** and **IOU (Intersection over Union)** — if two contours overlap in the same place, the smaller one is treated as a duplicate and skipped.

### Challenge 3: It was calling a hexagon a "Circle"
**Problem:** The shape classification logic only had categories up to Triangle/Square/Rectangle/Pentagon — any shape with 6+ vertices became "Circle".
**Solution:** Added a separate category for Hexagon (6 vertices), and used the **circularity formula** (`4π×Area / Perimeter²`) for shapes with more than 6 vertices, to distinguish an actual circle from a higher-sided polygon (a circle's circularity is close to ~1.0).

### Challenge 4: The label position was overlapping the shape
**Problem:** The label position was taken from `approx[0][0]` (the contour's first point), which depending on the shape's orientation could end up at the top, bottom, or side — causing the text to sometimes land right inside the shape.
**Solution:** Consistently placed the label at the **top-center** of the bounding box — independent of orientation, always readable.

### Challenge 5: Total chaos on dense/text-heavy images (like document photos)
**Problem:** If a user uploaded a photo of a text-heavy document, every single letter/word would be treated as a "shape" and get labeled — making the output completely unreadable.
**Solution:** Understood that this is a genuine limitation of classical contour-based shape detection (distinguishing text from geometric shapes needs OCR or deep learning, not just contour geometry). As a practical fix, area-based filtering and showing only the top-N largest shapes was considered, so the operation performs best for its intended use case (clean, separated geometric shapes).

---

## 6. Learning Outcomes

This project gave a deep understanding of:

1. **How classical CV algorithms fail on edge cases** — testing only the "happy path" isn't enough; testing on real-world images (different backgrounds, colors, density) is necessary.
2. **The polarity sensitivity of contour detection** — thresholding approaches depend on a background/foreground assumption, while Canny edge-based approaches are more robust.
3. **A systematic approach to debugging** — reproducing every bug, finding the root cause, fixing it, then regression testing (so the new fix doesn't break old functionality) — this is a proper engineering habit.
4. **The dispatch-table pattern** — instead of large if-elif chains, dictionary-based dispatch makes code readable and extensible.
5. **The deployment workflow** — for Streamlit/HF Spaces, deployment-specific choices in `requirements.txt` (like `opencv-python-headless`) matter, otherwise the app can crash in the cloud.
6. **The limitations of classical CV** — some problems (like distinguishing text from shapes) can't be solved with contour geometry alone; they need deep learning/OCR. Every technique has its own intended use case.

---

## 7. How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

