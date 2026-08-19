# Day 23 - Document OCR Web Application

**MLB Summer Internship | Danish Ali**

A Streamlit web app that lets you upload a document, receipt, invoice or form
image, applies preprocessing, and extracts readable text using either
**EasyOCR** or **PaddleOCR** — selectable from the sidebar.

## OCR Libraries Used
This project supports two OCR engines, selectable from the sidebar:

- **EasyOCR** - runs entirely in Python without needing extra system
  dependencies (unlike Tesseract, which needs a separate binary install),
  and works well on Streamlit Cloud's environment.
- **PaddleOCR** - added as a second engine to compare accuracy. In testing,
  it consistently outperformed EasyOCR on both printed and handwritten
  samples (see Engine Comparison below).

Having both available lets the user pick whichever performs better for their
specific image.

## Preprocessing Techniques Applied
- **Grayscale conversion** - removes color channels so OCR focuses on shape/intensity
- **Denoising** (`fastNlMeansDenoising`) - removes scan/photo grain
- **CLAHE contrast enhancement** - boosts local contrast for faded or low-light text
- **Adaptive thresholding** - binarizes the image, handles uneven lighting better
  than a single fixed threshold value
- **Deskew (currently disabled)** - straightens tilted photos using
  `minAreaRect` on detected text pixels. Disabled by default for now since it
  misfires on photos with visible background (see Challenges below).

Three preprocessing modes are available in the sidebar:
- `standard` - for regular clean printed documents
- `receipt` - for thermal-printed receipts with low contrast
- `low_light` - for photos taken in poor lighting conditions

## How It Works
1. User uploads an image through the Streamlit interface
2. Image is converted from PIL to OpenCV format (BGR)
3. Selected preprocessing pipeline is applied
4. User picks an OCR engine (EasyOCR or PaddleOCR) from the sidebar
5. Processed image is passed to the selected engine for text detection + recognition
6. Extracted text is filtered by a user-adjustable confidence threshold
7. Text is displayed on screen and made available for download as `.txt`

## Project Structure
```
Day-23/
├── app.py                   # main Streamlit app
├── requirements.txt
├── README.md
├── sample_images/           # test images used
├── sample_outputs/          # extracted text samples
└── utils/
    ├── preprocessing.py     # image preprocessing functions
    └── ocr_engine.py        # EasyOCR + PaddleOCR wrappers
```

## Running Locally
```powershell
pip install -r requirements.txt --upgrade
streamlit run app.py
```
First run downloads EasyOCR/PaddleOCR models (~100–300MB total), so it will
be slower once. After that, models are cached locally and load instantly.

## Challenges Faced

**From the original EasyOCR-only version:**
- EasyOCR's first-time model download made the app slow to start; solved
  this by caching the reader with `st.cache_resource` so the model loads
  only once per session.
- Receipts with thermal printing had very low contrast, plain grayscale
  thresholding was not enough, needed CLAHE contrast enhancement first.
- Uneven lighting in photographed documents caused fixed thresholding to
  fail on parts of the image; switched to adaptive thresholding to fix this.

**From adding PaddleOCR as a second engine:**
PaddleOCR's API has changed significantly across versions, so getting it
working involved several fixes:
1. **`show_log` argument removed** — Newer PaddleOCR versions no longer accept
   `show_log` in the `PaddleOCR()` constructor. Fixed by removing it and
   silencing PaddleOCR's own logger separately via `logging.getLogger("ppocr")`.
2. **`.ocr(image, cls=True)` deprecated** — PaddleOCR 3.x removed the `cls`
   keyword from `.ocr()`. Angle classification is now handled internally
   through `use_angle_cls` at init time instead.
3. **Output format changed entirely** — PaddleOCR 3.x's `.ocr()` no longer
   returns the old `[[bbox, (text, conf)], ...]` tuple format; it returns
   dict-like `OCRResult` objects. Switched to the documented `.predict()`
   method and read results via `res["rec_texts"]`, `res["rec_scores"]`,
   `res["rec_polys"]`.
4. **Grayscale image shape mismatch** — Our preprocessing pipeline outputs a
   2D grayscale/binary image, but PaddleOCR expects a 3-channel (H, W, C)
   color image internally. Passing a 2D array caused
   `not enough values to unpack (expected 3, got 2)` when PaddleOCR tried to
   unpack `image.shape`. Fixed by converting to BGR with
   `cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)` before passing it to PaddleOCR.
5. **oneDNN / PIR executor crash** — A low-level PaddlePaddle framework bug
   (`(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support [...]`)
   occurred on CPU, caused by the new PIR executor combined with oneDNN
   acceleration on this machine's PaddlePaddle build. Fixed by disabling
   oneDNN with `enable_mkldnn=False` in the `PaddleOCR()` constructor.
6. **Deskew step misrotating images** — Unrelated to PaddleOCR, but found
   during testing: the OpenCV-based `deskew_image()` function used a global
   Otsu threshold across the *entire* photo (including visible background)
   to estimate skew angle. On photos with visible table/background around
   the page, it picked up background contrast instead of actual text and
   rotated the image ~90°, producing garbage OCR output. Fixed by disabling
   the deskew step for now (input photos were already reasonably straight),
   with a note to re-enable once the page is properly cropped from the
   background first.
7. **Slow inference on CPU** — PaddleOCR 3.x by default runs extra pipeline
   stages (document orientation classification, document unwarping, textline
   orientation) that add noticeable latency and were unnecessary here since
   our own preprocessing already handles image orientation. Disabled via
   `use_doc_orientation_classify=False`, `use_doc_unwarping=False`,
   `use_textline_orientation=False`, which also improved speed.

## Engine Comparison (observed during testing)

For clean, printed serif-font documents, **PaddleOCR** produced near-perfect
output matching the source text exactly (punctuation, capitalization, line
breaks), while **EasyOCR** made small recognition errors — for example
misreading "so" as "80", and swapping periods for colons/semicolons in a few
places.

For **handwritten notes**, PaddleOCR again outperformed EasyOCR by a large
margin. EasyOCR's `paragraph=True` grouping (used to sort fragments into
reading order) relies on clustering fragments by vertical position, which
broke down on angled, multi-line handwriting and produced badly scrambled
word order. PaddleOCR's polygon-based line detection kept correct
line-by-line reading order with only minor word-level errors.

Overall in this testing, PaddleOCR was the stronger engine on both printed
and handwritten samples. Both engines remain selectable in the app so the
user can compare per image and pick whichever works best for their specific
case.

## Possible Improvements
- Re-enable automatic deskewing for tilted document photos, using a more
  robust method (e.g. cropping to the page contour first before estimating
  skew angle) so it doesn't misfire on backgrounds
- Batch processing support for multiple images at once
- Table structure detection for forms and invoices
- Side-by-side output comparison view (EasyOCR vs PaddleOCR results together)

## Testing
Tested on 15+ images including printed documents, receipts, invoices,
handwritten notes, and forms with varying lighting conditions and
orientations. Sample inputs and are included in `sample_images/`.