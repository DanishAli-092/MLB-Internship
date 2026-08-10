# Document Image Enhancer Pro — Day 17

A Streamlit application that applies OpenCV-based image transformation and enhancement techniques to clean up, straighten, and sharpen document photos before they are used for OCR or other Computer Vision tasks.

---

## 📁 Folder Structure

```
document_tool/
├── app.py                          # Streamlit application (Single Image + Batch modes)
├── document_enhancer.py            # Core pipeline used by the Streamlit app
├── make_challenge_comparison.py    # Script that builds the before/after challenge comparison grid
├── requirements.txt                # Python dependencies
│
├── input_images/                   # Raw/tilted document images used for testing (10+)
│
├── output_images/
│   ├── challenge_comparison/       # Original vs. Perspective Corrected vs. Enhanced (5 tilted docs)
│   ├── enhancement/                # Output of enhancement.py experiments
│   └── transformations/            # Output of transformations.py experiments
│
├── scripts/
│   ├── transformations.py          # Translation, Rotation, Scaling, Affine, Perspective
│   └── enhancement.py              # Brightness, Contrast, Gaussian/Median Blur, Bilateral Filter, Sharpen
│
└── README.md
```

---

## 🎯 Objective

Today's task builds on the OpenCV fundamentals covered on Day 16 by focusing on **image transformations** and **image enhancement** — the standard preprocessing steps used before feeding document images into AI/Computer Vision models (OCR, layout detection, classification, etc.).

---

## 🔄 Image Transformations (`scripts/transformations.py`)

| Transformation | What it does | Typical real-world use |
|---|---|---|
| **Translation** | Shifts the image along the X and Y axes | Repositioning/aligning content within a frame |
| **Rotation** | Rotates the image by a given angle around a center point | Correcting camera tilt, orientation normalization |
| **Scaling** | Resizes the image up or down | Standardizing input size for ML models, thumbnails |
| **Affine Transformation** | Maps a triangle of points to another, preserving parallel lines | Correcting shear/skew while keeping proportions |
| **Perspective Transformation** | Maps a quadrilateral of points to a rectangle using a 4-point transform | Straightening a document photographed at an angle |

Each transformation is implemented as an independent, reusable function and demonstrated on sample images in `output_images/transformations/`.

---

## ✨ Image Enhancement (`scripts/enhancement.py`)

| Technique | Purpose |
|---|---|
| **Brightness Adjustment** | Increases/decreases overall pixel intensity to correct under/over-exposed scans |
| **Contrast Adjustment** | Stretches the intensity range so text stands out more clearly from the background |
| **Gaussian Blur** | Smooths the image using a Gaussian kernel — reduces fine-grained noise before further processing |
| **Median Blur** | Replaces each pixel with the median of its neighborhood — effective against salt-and-pepper noise |
| **Bilateral Filter** | Reduces noise while preserving edges — ideal for cleaning document backgrounds without blurring text |
| **Image Sharpening** | Applies a sharpening kernel to enhance edges and make text more legible after denoising |

Results for each technique are saved in `output_images/enhancement/`.

---

## 🛠️ The Document Image Enhancement Tool (`app.py` + `document_enhancer.py`)

The mini-project combines the above into a single, configurable Streamlit pipeline:

1. **Load** a document image (JPG/PNG)
2. **Perspective Correction** — detects the document edges (Canny edge detection + contour analysis) and warps it into a straightened, rectangular view
3. **Grayscale Conversion** — removes color information to simplify further processing
4. **Noise Reduction** — bilateral filtering to clean the scan while preserving text edges
5. **Brightness/Contrast Correction** — user-adjustable via sliders
6. **Sharpening** — applies a sharpening kernel to make text crisp and readable
7. **Save/Download** the enhanced image

### App Features

- **Single Image tab** — upload one document photo and see a full step-by-step breakdown (Original → Perspective Corrected → Grayscale → Denoised → Brightness/Contrast → Sharpened), plus live metrics (input/output size, processing time, whether perspective correction was applied) and a one-click download.
- **Batch tab** — upload 10+ document images at once, process all of them through the same pipeline, and download every enhanced image bundled into a single ZIP.
- **Sidebar controls** — toggle any pipeline step on/off and fine-tune brightness/contrast values and output dimensions.

---

## 🚀 Running the App

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd document_tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

**Live App:** `<add your Streamlit Cloud / Hugging Face Spaces link here>`

---

## 🧪 Dataset

A total of **10+ document images** were used for testing, sourced from personal photos of tilted/angled documents. All raw images are available in `input_images/`.

---

## 🏆 Challenge Task — Before vs. After Comparison

Five deliberately tilted document images were processed end-to-end through the pipeline. For each one, the following three stages are saved side-by-side in `output_images/challenge_comparison/`:

1. **Original** (tilted, unprocessed) image
2. **Perspective Corrected** image
3. **Final Enhanced** image (grayscale + denoised + brightness/contrast + sharpened)

The comparison grids were generated using `make_challenge_comparison.py`.

### Which transformation had the biggest impact?

**Perspective correction** had by far the biggest visible impact on document quality. Tilted or angled photos are hard to read and unusable for OCR — once the document is warped back into a straight, rectangular view, every downstream step (denoising, contrast, sharpening) becomes noticeably more effective, since the text is now properly aligned instead of skewed.

---

## ⚠️ Challenges Faced

- **Detecting the correct document contour**: Canny edge detection sometimes picked up background clutter instead of the document's actual boundary, requiring careful contour filtering (largest 4-point contour by area).
- **Balancing noise reduction vs. detail loss**: Standard Gaussian/median blurring was too aggressive on text edges; switching to a **bilateral filter** solved this by smoothing flat regions while preserving edges.
- **Streamlit Cloud deployment**: `opencv-python` failed to build in the cloud environment due to missing system GUI libraries — resolved by switching to `opencv-python-headless` in `requirements.txt`.
- **Session/state handling in the Streamlit UI**: Ensuring the step-by-step breakdown and metrics stayed in sync with the currently selected pipeline options (checkboxes toggled on/off) required restructuring the pipeline into a single reusable `process_one_image()` function.

---

## 📦 Tech Stack

- **Python 3**
- **OpenCV** — image transformations and enhancement
- **Streamlit** — interactive web app
- **NumPy / Pillow** — image array handling
- **Canny Edge Detection + Contour Analysis** — perspective correction
- **Bilateral Filtering** — edge-preserving noise reduction

---

