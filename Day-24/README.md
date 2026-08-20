# Day 24 - Document OCR Web Application 📄🔍

**MLB Summer Internship | Danish Ali**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Deployed-success)

A Streamlit application that runs and compares 5 OCR engines — Tesseract, PaddleOCR, EasyOCR, RapidOCR, and DocTR — for text extraction from document images.

## Features

### Supported OCR Engines
A unified interface to load, switch between, and benchmark **5 different OCR text-extraction models**:
| Engine | Backend | Best For |
|---|---|---|
| **EasyOCR** | PyTorch | General-purpose, multilingual text |
| **PaddleOCR** | PaddlePaddle (PP-OCRv5 mobile) | Fast, lightweight detection + recognition |
| **RapidOCR** | ONNX Runtime | Lightweight, CPU-friendly inference |
| **Tesseract** | Google Tesseract (LSTM) | Clean, high-contrast scanned documents |
| **DocTR** | PyTorch (DBNet + CRNN) | Structured document layout parsing |

### Parallel Multi-Document Processing
Implemented **thread-safe inference** using a global `threading.Lock()` around each engine's `predict()` / `readtext()` call. This prevents race conditions and memory crashes in C++-backed inference engines (PaddleOCR, RapidOCR, DocTR) when multiple images are processed concurrently through Streamlit's caching layer.

### Dynamic OS Path Routing
Smart environment detection via `sys.platform` automatically routes the Tesseract binary path:
- **Windows (local dev):** `C:\Program Files\Tesseract-OCR\tesseract.exe`
- **Linux (cloud deployment):** `/usr/bin/tesseract`

This means the exact same codebase runs without modification on both a developer's local machine and the Streamlit Cloud container.

### Output & Metrics
- **Confidence filtering** — results below a configurable `min_confidence` threshold are dropped
- **Extraction time tracking** — each engine's inference time is measured via `time.perf_counter()` and returned alongside the results, enabling direct speed comparison across engines
- **Consistent output format** — every engine returns `(full_text, filtered_results, elapsed_time)` for direct comparison

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Web Framework | Streamlit |
| Image Processing | OpenCV |
| OCR Engines | Tesseract (PyTesseract), PaddleOCR, EasyOCR, RapidOCR, DocTR |
| Deep Learning Backend | PyTorch, PaddlePaddle, ONNX Runtime |
| Server | Uvicorn |
| Version Control | Git / GitHub |
| Deployment | Streamlit Community Cloud |

## Folder Structure

```
Day-24/
├── screen_recording/
│   └── link.md
├── utils/
│   ├── __init__.py
│   ├── ocr_engine.py       # engine loading + inference for all 5 OCR engines
│   └── preprocessing.py    # OpenCV preprocessing
├── app.py
├── requirements.txt
└── README.md

packages.txt                # at repo root, not inside Day-24/
```

`packages.txt` must be at the repository root — Streamlit Cloud only reads it from root, not from subfolders.

## Issues Fixed

**1. Tesseract not found on Streamlit Cloud**

Error:
```
TesseractNotFoundError: /usr/bin/tesseract is not installed or it's not in your PATH.
```

`tesseract-ocr` was listed in `packages.txt`, but the build logs showed no `apt-get install` step running. Two causes:
- `packages.txt` had local edits that were never committed/pushed — Cloud was building from a stale version.
- The file was in a subdirectory, not the repo root, where Streamlit Cloud actually looks for it.

Fix: confirmed the file was at repo root, committed and pushed the update, then rebooted the app to trigger a fresh build.

**2. Invalid extra in `python-doctr[torch]`**

Build warning:
```
warning: The package `python-doctr==1.0.1` does not have an extra named `torch`
```

The `[torch]` extra was removed in `python-doctr` 1.0.x, so it silently resolved to nothing instead of installing PyTorch. Since `ocr_engine.py` sets `USE_TORCH=1` and depends on `torch`/`torchvision` for the DocTR predictor, these are now pinned directly.

## requirements.txt (corrected)

```txt
streamlit>=1.30.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0

easyocr>=1.7.1
paddleocr>=3.0.0
paddlepaddle==3.2.2
rapidocr_onnxruntime>=1.3.0
pytesseract>=0.3.10

python-doctr>=0.9.0
torch>=2.0.0
torchvision>=0.15.0
```

```txt
# packages.txt (repo root)
libgl1
libglib2.0-0t64
tesseract-ocr
```

## How to Run

```bash
git clone https://github.com/DanishAli-092/MLB_Summer_internship.git
cd MLB_Summer_internship/Day-24

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

**Windows only:** install Tesseract from [UB-Mannheim's build](https://github.com/UB-Mannheim/tesseract/wiki) to the default path `C:\Program Files\Tesseract-OCR\`. On Linux/Cloud, `packages.txt` installs it automatically.

