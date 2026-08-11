# Day 18 — Edge Detection & Morphological Operations

**ML Bench (MLB) Summer Internship**

## Overview

This project covers two foundational OpenCV techniques — **Edge Detection** (Sobel, Laplacian, Canny) and **Morphological Operations** (Erosion, Dilation, Opening, Closing, Gradient, Top Hat, Black Hat) — and applies them together in a complete preprocessing pipeline for a mini project: a **Document Boundary Detection Tool**.

The tool takes a photo or scan of a document, detects its outer boundary using contour analysis, and draws it on the original image — a pipeline commonly used before OCR, document scanning, or automated data extraction. All techniques were tested and compared across 11 real document images captured under a range of conditions (straight scans, tilted photos, uneven lighting, shadows, blur, and noise) to evaluate robustness.

An interactive **Streamlit app** was also built so the boundary detection tool can be tested on any uploaded image directly in the browser.

---

## Folder & File Structure

```
Day-18/
├── document_boundary_tool/
│   ├── app.py                    # Streamlit app for interactive boundary detection
│   ├── boundary_detector.py      # Mini project: full document boundary detection pipeline
│   ├── challenge_task.py         # Mandatory challenge: batch processing + comparison grid
│   └── requirements.txt          # Dependencies for Streamlit Cloud deployment
│
├── edge_detection/
│   └── edge_detection.py         # Sobel, Laplacian, Canny implementation + comparison
│
├── morphological_ops/
│   └── morphological_ops.py      # All 7 morphological operations + before/after comparison
│
├── input_images/                 # 11 test document images (varied conditions)
│   ├── doc1_straight.jpeg
│   ├── doc2.jpeg
│   ├── doc3_uneven_lighting.jpeg
│   ├── doc4_normal.jpg
│   ├── doc5_blurred.jpg
│   ├── doc6_perspective_distortion.jpg
│   ├── doc6_tilt.jpg
│   ├── doc7_low_light.jpg
│   ├── doc8_slightly_blurred.jpg
│   ├── doc9_Clean.jpg
│   └── doc10_noisy.jpg
│
├── output_images/
│   ├── edge_detection/            # Sobel/Laplacian/Canny outputs + comparison.png
│   ├── morphological_ops/         # All 7 operation outputs + comparison.png
│   ├── boundary_detection/        # Per-image original/edges/morphology/boundary outputs
│   └── challenge_task/            # Batch outputs + challenge_comparison_grid.png
│
├── screen_recording/
│   └── recording_link.md          # Google Drive link to walkthrough video (3–5 min)
│
└── README.md
```

---

## 1. Edge Detection

### What is Edge Detection?

Edge detection identifies pixels where image intensity changes sharply — these transitions typically mark the boundary between an object and its background, or between two distinct regions. It is a foundational preprocessing step for object detection, OCR, image segmentation, and document analysis.

### Difference Between Sobel, Laplacian, and Canny

| Method | Approach | Characteristics | Best Used For |
|---|---|---|---|
| **Sobel** | First-order derivative; computes gradients separately in the X and Y direction using two 3×3 kernels, then combines them into a gradient magnitude. | Directional, moderately noise-resistant due to built-in smoothing in the kernel. | Detecting edges with a known orientation (horizontal/vertical emphasis). |
| **Laplacian** | Second-order derivative; measures the rate of change of the gradient itself using a single kernel across all directions. | Highly sensitive to noise since it amplifies high-frequency components; must always be paired with blurring. | Detecting fine detail and edges in all directions simultaneously. |
| **Canny** | Multi-stage algorithm: noise reduction → gradient calculation → non-maximum suppression → double thresholding → edge tracking by hysteresis. | Produces the cleanest, thinnest, and most accurate edges of the three; most computationally involved. | Production-grade edge detection, including this project's boundary detection pipeline. |

**Choosing threshold values (Canny):** A fixed pair of thresholds does not generalize well across images with different lighting conditions. This project used both a fixed threshold (50, 150) for the core pipeline and a median-based adaptive threshold (`low = 0.66 × median`, `high = 1.33 × median`) during experimentation — see *Challenges* below for what was learned from comparing the two.

### Real-World Applications
- Optical Character Recognition (OCR) preprocessing
- Object detection and contour-based segmentation
- Medical imaging (organ/tumor boundary detection)
- Lane detection in autonomous vehicles
- Document boundary detection (this project)

---

## 2. Morphological Operations

Morphological operations process binary/grayscale images using a structuring element (kernel) to remove noise, fill gaps, or refine shapes.

### Purpose of Each Operation

| Operation | Purpose |
|---|---|
| **Erosion** | Shrinks white (foreground) regions; removes small noise specks and thin protrusions. |
| **Dilation** | Expands white regions; fills small gaps and connects broken edges. |
| **Opening** (Erosion → Dilation) | Removes small noise while preserving the overall size/shape of larger objects. |
| **Closing** (Dilation → Erosion) | Fills small holes and gaps inside objects without significantly changing their size. |
| **Morphological Gradient** | Highlights the outline/boundary of objects (Dilation − Erosion). |
| **Top Hat** | Highlights small bright regions relative to the background (Original − Opening). |
| **Black Hat** | Highlights small dark regions relative to the background (Closing − Original). |

---

## 3. Mini Project: Document Boundary Detection Tool

**Pipeline:** Grayscale → Gaussian Blur → Canny Edge Detection → Morphological Closing + Dilation → Largest External Contour → Polygon Approximation → Draw Boundary

For each input image, the tool saves the **original**, the **Canny edge map**, the **morphological result**, and the **final image with the detected boundary drawn**.

### Which Combination of Techniques Gave the Best Results

The combination of **Gaussian Blur → Canny Edge Detection → Morphological Closing (to bridge broken edge segments) → largest external contour → 4-point polygon approximation** gave the most consistent results across straight scans and moderately tilted or perspective-distorted photos (`doc6_tilt`, `doc6_perspective_distortion`, `doc7_low_light`, `doc5_blurred`, `doc8_slightly_blurred`, `doc11_tilt`).

Morphological **closing** specifically was the operation that mattered most  without it, edges broken by shadows or lighting variation produced incomplete contours that `findContours` could not close into a proper quadrilateral.

### Challenges Faced While Detecting Document Boundaries

- **Degenerate contours on edge-to-edge documents:** For `doc4_normal.jpg`, the document filled almost the entire frame with no background margin. Its contour touched the image border, causing `approxPolyDP` to collapse to a 2-point degenerate shape instead of a proper rectangle. **Fix:** added a fallback — when fewer than 4 polygon points are detected, the bounding rectangle of the largest contour is used instead.

- **Uneven lighting and shadows** (`doc2`, `doc3_uneven_lighting`): Photos taken by hand had inconsistent contrast between the page and the background due to shadows and a notebook spiral binding. This caused Canny to detect a broken edge loop on the shadowed side, so the largest closed contour found was only a partial region of the page rather than its full outline — producing a triangular, incorrect boundary rather than a clean rectangle.

- **Salt-and-pepper noise** (`doc10_noisy.jpg`): A standard Gaussian Blur was not sufficient to suppress the dense speckle noise across the background. Canny picked up large amounts of noise as false edges, and after morphological closing merged them with the real text edges, the "largest contour" traced an irregular zigzag shape instead of the page boundary.

- **Threshold tuning trade-off:** An adaptive, median-based Canny threshold combined with a larger closing kernel improved results on the noisy and shadowed images, but degraded results on otherwise clean images (e.g. `doc1_straight`, `doc5_blurred`) by over-merging their edges. This confirmed that a single fixed pipeline configuration does not generalize perfectly across all real-world capture conditions  a genuine, expected limitation of a simple rule-based (non-learning) approach, and one that would require per-image adaptive tuning or a learned model to fully resolve.

- **File naming consistency:** Some images used a `.jpeg` extension instead of `.jpg`, which was initially missed by the file-matching pattern and silently excluded a few images from processing until the pattern was corrected to include `.jpeg`.

---

## 4. Challenge Task (Mandatory)

`challenge_task.py` processes all document images in `input_images/` and, for each one, saves:
1. Original Image
2. Edge Detection Result
3. Morphological Operation Result
4. Final Image with Detected Document Boundary

It also generates a single combined **comparison grid** (`output_images/challenge_task/challenge_comparison_grid.png`) with one row per image and one column per stage, making it easy to visually compare results across all documents at once.

---

## 5. How to Run

```powershell
# Edge detection comparison
cd edge_detection
python edge_detection.py

# Morphological operations comparison
cd ../morphological_ops
python morphological_ops.py

# Document boundary detection (single pass, all images)
cd ../document_boundary_tool
python boundary_detector.py

# Mandatory challenge task (batch + comparison grid)
python challenge_task.py

# Streamlit app (interactive)
streamlit run app.py
```

**Dependencies** (already available in the shared internship virtual environment):
```
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
streamlit>=1.30.0
Pillow>=10.0.0
```

For Streamlit Cloud deployment, `requirements.txt` uses `opencv-python-headless` instead of `opencv-python`, since the standard package requires GUI libraries unavailable on the deployment server.

---


## Learning Outcomes

By completing Day 18, the following skills and concepts were developed:

- **Edge detection fundamentals** — understood how Sobel, Laplacian, and Canny differ mathematically (first-order vs. second-order derivatives vs. a multi-stage algorithm) and how that difference shows up in practice: Sobel and Laplacian are quick but noisier, while Canny consistently produces the cleanest, thinnest edges.

- **Threshold selection matters as much as the algorithm** — a fixed Canny threshold that works well on one image can fail on another with different lighting. Experimenting with adaptive, median-based thresholding showed both its benefits (better on noisy/shadowed images) and its trade-offs (can over-merge edges on already-clean images), reinforcing that there is no single "correct" setting — only the right setting for a given image distribution.

- **Morphological operations as targeted tools, not a one-size-fits-all step** — erosion, dilation, opening, and closing each solve a specific structural problem (removing noise vs. filling gaps vs. connecting broken edges), and picking the wrong one (or the wrong kernel size) can erase the very detail you're trying to preserve, as seen when a 5×5 kernel erased thin text strokes during dilation and closing.

- **Building a real preprocessing pipeline end-to-end** — connecting grayscale conversion, blurring, edge detection, morphology, and contour analysis into a single working pipeline that solves a concrete, practical problem (document boundary detection), rather than treating each technique as an isolated exercise.

- **Debugging with real, imperfect data** — using a self-collected dataset (11 images with straight scans, tilted photos, shadows, blur, and noise) instead of a single clean sample surfaced real failure modes: degenerate contours when a document touches the image border, broken edge loops from uneven lighting, and false edges from sensor-like noise. Diagnosing *why* a result was wrong (not just that it was wrong) was as important as writing the code itself.

- **Recognizing the limits of a rule-based approach** — no single fixed combination of blur, threshold, and kernel size performed best across every image. This is an expected limitation of classical, non-learning computer vision methods, and understanding *why* (rather than chasing a single perfect configuration) is a more valuable takeaway than the pipeline itself.

- **Practical engineering hygiene** — consistent file naming, checking for missing file extensions (`.jpeg` vs `.jpg`), and adding fallback logic for edge cases are small details that determine whether a pipeline is genuinely production-ready or only works on the one image it was tested with.