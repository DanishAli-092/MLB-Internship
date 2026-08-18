# Day 22 - Introduction to OCR (Optical Character Recognition)

**MLB Summer Internship | Danish Ali**

---

## Folder Structure

```
Day-22/
├── sample_inputs/             # Test images (15 total - 5 categories x 3 each)
├── extracted_texts/           # Extracted .txt output files + comparison summary
├── screen_recording/
│   └── link.md                 # Link to the demo screen recording
├── ocr_practice.py             # OCR Practice Scripts - EasyOCR setup, batch testing, raw vs preprocessed comparison
├── ocr_document_reader.py      # Mini Project Source Code - standalone OCRDocumentReader class + CLI
├── app.py                      # Deployment wrapper - imports OCRDocumentReader, converts it into a Streamlit app
├── requirements.txt
└── README.md
```

**Why two files for the mini project?**
`ocr_document_reader.py` is the actual mini-project application. It works standalone from the command line (no Streamlit required) and independently satisfies the mini-project spec: accept an image, extract text, display the original image and text together, and save the result to a `.txt` file. `app.py` imports the same `OCRDocumentReader` class and wraps it in a Streamlit UI, satisfying the separate deployment requirement ("convert your OCR application into a Streamlit app"). This keeps the core logic in one place instead of duplicating it across two files.

```bash
# Run the mini project standalone (no Streamlit)
python ocr_document_reader.py --image sample_inputs/receipt_01_grocery.png --smart

# Run the deployed Streamlit app
streamlit run app.py
```

---

## What is OCR?

Optical Character Recognition (OCR) is a computer vision technique that converts text present in images — scanned documents, photos, screenshots — into machine-readable, editable text. It combines two sub-tasks: **text detection** (locating where text exists in an image, usually as bounding boxes) and **text recognition** (identifying what the characters and words inside those regions actually are). Modern OCR engines rely on deep learning — CNN-based detectors and CRNN/Transformer-based recognizers — rather than the older template-matching approaches used in early OCR systems.

### OCR Applications
- Document digitization (books, scanned PDFs)
- Invoice / receipt processing (automated bookkeeping)
- ID card and form data extraction (KYC systems)
- Number plate recognition (ANPR)
- Accessibility tools (screen readers)

### General Challenges in OCR
- Poor image quality (blur, low resolution)
- Font variation and stylized text
- Handwritten text (much harder than printed)
- Noisy or busy backgrounds
- Skewed or rotated text
- Low contrast between text and background

### Why Preprocessing Matters (in theory)
Grayscale conversion, contrast enhancement (CLAHE), and denoising can improve OCR accuracy on low-quality images, since OCR models are generally trained on relatively clean text patterns. However, as shown in the Findings section below, preprocessing is **not universally beneficial** — its effect depends heavily on the original image quality.

### Multithreading Support Across OCR Libraries

| Library | Multithreading Support |
|---|---|
| **Tesseract OCR** | Native multithreading at the C++ engine level (OpenMP). The Python wrapper (`pytesseract`) is single-threaded per call — parallelize multiple images via `multiprocessing`. |
| **EasyOCR** | PyTorch-based. GPU inference batches naturally; CPU inference requires manual `multiprocessing` for parallel calls. |
| **PaddleOCR** | Supports multi-thread and multi-process inference natively through PaddlePaddle's inference engine — the most production-ready option out of the box. |
| **DocTR** | TensorFlow/PyTorch backend, supports GPU batch processing; CPU threading must be handled manually. |

---

## Which OCR Library Was Used, and Why

### Library Comparison

| Library | Advantages | Limitations | Commonly Used When |
|---|---|---|---|
| **Tesseract OCR** | Free, mature, lightweight, huge language support | Weaker on noisy/natural scene images, needs good preprocessing | Simple, clean scanned documents |
| **EasyOCR** | Easy Python API, decent accuracy out-of-the-box, 80+ languages, works well on natural scene text (signboards, receipts) | Slower on CPU, larger model size | Quick prototyping, mixed document types |
| **PaddleOCR** | Very high accuracy, fast, strong multithreading/production support | Larger setup, PaddlePaddle dependency can be heavier to configure | Production-scale document pipelines |
| **DocTR** | Modern deep learning architecture, good structured-document performance | Smaller community, less beginner documentation | Structured document analysis (forms, tables) |

**Library chosen: EasyOCR.** It was selected for its simple Python API, solid out-of-the-box accuracy across varied document types (printed text, receipts, signboards, certificates), and the absence of complex native dependencies — which matters for a straightforward Hugging Face Spaces / Streamlit Cloud deployment.

---

## What Preprocessing Techniques Improved the Results

The preprocessing pipeline applies: **grayscale conversion → CLAHE contrast enhancement → Non-Local Means denoising**.

Rather than assuming preprocessing always helps, it was tested directly — raw vs. preprocessed — across all 15 sample images, and the results were counter-intuitive in places.

### Full confidence data from testing (all 15 images)

```
OCR Comparison Summary - Raw vs Preprocessed
==================================================

book_01_page.png                    | raw: 0.934 | preprocessed: 0.935 | diff: +0.001
book_02_page.jpg                    | raw: 0.814 | preprocessed: 0.782 | diff: -0.031
book_03_page_scanned.png            | raw: 0.778 | preprocessed: 0.727 | diff: -0.051
doc_01_printed_report.png           | raw: 0.956 | preprocessed: 0.862 | diff: -0.094
doc_02_printed_letter.png           | raw: 0.872 | preprocessed: 0.853 | diff: -0.019
doc_03_printed_lowlight.png         | raw: 0.947 | preprocessed: 0.932 | diff: -0.015
handwritten_01_note.jpg             | raw: 0.944 | preprocessed: 0.856 | diff: -0.088
handwritten_02_note.png             | raw: 0.982 | preprocessed: 0.974 | diff: -0.008
handwritten_03_note.png             | raw: 0.743 | preprocessed: 0.820 | diff: +0.077
receipt_01.jpg                      | raw: 0.907 | preprocessed: 0.906 | diff: -0.001
receipt_02_invoice.png              | raw: 0.653 | preprocessed: 0.707 | diff: +0.053
receipt_03_faded.png                | raw: 0.642 | preprocessed: 0.687 | diff: +0.045
signboard_01_warning.jpg            | raw: 0.924 | preprocessed: 0.986 | diff: +0.062
signboard_02_direction.png          | raw: 0.793 | preprocessed: 0.980 | diff: +0.187
signboard_03_warning.png            | raw: 0.998 | preprocessed: 1.000 | diff: +0.001
```

**Key finding:** Out of 15 images, preprocessing helped 7 and hurt 8. The clearest wins are on signboards and already-degraded images: `signboard_02_direction.png` (low light, +0.187), `signboard_01_warning.jpg` (+0.062), and the noisy/faded receipts and blurred handwritten note (+0.045 to +0.077). The clearest losses are on already-clean, high-contrast printed material: `doc_01_printed_report.png` (-0.094), `handwritten_01_note.jpg` (-0.088), and `book_03_page_scanned.png` (-0.051). On images that are already clean and high-contrast, CLAHE over-enhances the image — amplifying paper texture, JPEG artifacts, and subtle background patterns into false edges that confuse the text detector — which lowers accuracy. On images that are near-perfect already (`signboard_03_warning.png`, `receipt_01.jpg`, `book_01_page.png`), the effect is negligible either way (±0.001).

### Solution implemented: Smart Auto-Enhance

Because the correct choice depends on the individual image and cannot be predicted in advance, a manual toggle was replaced with an automatic decision system in `ocr_document_reader.py` and `app.py`:

1. Run OCR on the raw image and record its average confidence.
2. Run OCR on the preprocessed image and record its average confidence.
3. Automatically keep whichever result has the higher confidence score.

This removes the guesswork for the user entirely, at the cost of running OCR twice per image (a deliberate accuracy-vs-latency trade-off, acceptable for this project's scale).

---

## Challenges Faced While Extracting Text

These are challenges discovered through direct testing on real images (a resume, an AWS certificate, road signs, a modern book page, and a vintage scanned page) rather than assumptions made in advance.

### 1. URLs and hyperlinks are the most error-prone content
Across multiple different images (a resume with LinkedIn/GitHub links, an AWS certificate with a Credly link), URLs consistently came out corrupted regardless of preprocessing:
- `linkedin.com/in/danish-ali092` → `linkedin comlin/danish-ali092`
- `github.com/Danish-ali092` → `githubcom/Danish-ali0g2`
- `https://www.credly.com/go/e1znzlxp` → `https:I WWW credly com/golelznzlxp`

Meanwhile, plain paragraph text in the *same images* extracted at near-100% accuracy. This indicates the issue is a genuine EasyOCR limitation with dense special-character sequences (`/`, `.`, mixed-case domains), not a general image-quality problem.

### 2. Confidence score is not a reliable proxy for correctness
A "CAUTION" road sign image produced **100% correct** extracted text but only **36.80%** average confidence — while a resume with several character-level errors scored **83%+** confidence. Confidence reflects the model's internal certainty, not actual correctness, so extracted text should always be spot-checked rather than trusted purely based on a high confidence score.

### 3. Reading order breaks down on dense paragraphs
On a densely printed book page, EasyOCR fragmented full sentences into individual word-level detections and returned them out of natural reading order (e.g. "Does school prepare children..." came back as scattered single words: `There / is / Need / Does / school / prepare / children`). This is a **detection-stage** failure, not a recognition error — the underlying text was read correctly, but the pieces were reassembled in the wrong order.

**Mitigation implemented:** A `sort_reading_order()` function was added that groups detected bounding boxes into horizontal "line bands" using their top-Y coordinate, then sorts left-to-right within each band by X coordinate. This restores natural line-by-line reading order and is applied automatically inside `OCRDocumentReader.extract_text()`.

### 4. Vintage/degraded documents cause character-level substitution
On an aged, yellowed scanned page with an old serif typeface, character-level misreads appeared that were not present on clean modern documents: `while` → `wilh`, `not` → `noL`, `afford` → `alford`, and even `a house` → `& house` (the letter "a" misread as an ampersand). Errors were also concentrated near visible paper stains rather than spread evenly across the page — suggesting physical document degradation directly affects recognition accuracy in a localized way, which no amount of code-level preprocessing can fully correct.

### 5. A genuine implementation bug surfaced during Smart Auto-Enhance testing
While building the Smart Auto-Enhance feature, an early version of the code discarded the preprocessed-image confidence score instead of returning it, causing the Streamlit UI to accidentally display the raw score twice (once labeled "Raw," once mislabeled "Enhanced") whenever the raw image was chosen as the better result. This was caught by directly testing the UI against a real image and comparing the displayed numbers against the underlying computed values, then fixed by explicitly returning both `raw_confidence` and `proc_confidence` as separate fields from `extract_text()`.

---

## Expected Outcome

By the end of this task:
- Understood the fundamentals of OCR and its detection/recognition pipeline.
- Extracted text from 15 real and synthetic images across 5 categories (printed documents, receipts/invoices, signboards, book pages, handwritten notes).
- Measured, rather than assumed, the effect of preprocessing — and built an automatic system (Smart Auto-Enhance) to handle the fact that its benefit is image-dependent.
- Identified and partially mitigated a real failure mode (reading-order scrambling) with a concrete code fix.
- Built and deployed a working OCR application usable by others online.

---

## How to Run Locally

```powershell
pip install -r requirements.txt

# OCR Practice Scripts - batch test on sample_inputs/, raw vs preprocessed comparison
python ocr_practice.py

# Mini Project Source Code - standalone CLI
python ocr_document_reader.py --image sample_inputs/receipt_01_grocery.png --smart

# Deployment - Streamlit app
streamlit run app.py
```



