# Day 22 - OCR Practice Script
# This script uses EasyOCR to extract text from multiple images
# and checks if preprocessing (grayscale + contrast enhancement) improves OCR accuracy or not

# What this script does:
# 1. Loads all images from sample_inputs folder
# 2. Runs OCR twice on each image
#    - On raw image (without any preprocessing)
#    - On preprocessed image (grayscale + contrast boost + denoise)
# 3. Compares both results and prints them on terminal
# 4. Saves extracted text for each image inside extracted_texts folder
# 5. Also saves final comparison summary inside extracted_texts folder

import os
import time
import cv2
import easyocr


# Config
INPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_inputs")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "extracted_texts")
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")


# This function returns list of all supported image paths from given folder
def load_image_paths(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Sample images folder not found: {folder}")

    paths = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if len(paths) < 10:
        print(f"Warning: only {len(paths)} images found minimum 10 required")

    return paths


# This function preprocesses the image to make it better for OCR
# Step 1 Grayscale conversion removes color info and keeps only intensity
# Step 2 Contrast enhancement using CLAHE makes low light or faded text clearer
# Step 3 Denoising removes random noise pixels from image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE = Contrast Limited Adaptive Histogram Equalization
    # this improves local contrast especially for low light images
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    return denoised


# This function extracts text using EasyOCR and returns text confidence and time taken
def run_ocr(reader, image_input):
    start = time.time()
    results = reader.readtext(image_input)
    elapsed = time.time() - start

    extracted_lines = [text for (_bbox, text, _conf) in results]
    avg_confidence = (
        sum(conf for (_b, _t, conf) in results) / len(results) if results else 0.0
    )

    return "\n".join(extracted_lines), avg_confidence, elapsed


# Main function that runs the full OCR comparison process
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading EasyOCR reader this will take some time on first run")
    # gpu=False because GPU is not available on deployment like Hugging Face free tier
    reader = easyocr.Reader(["en"], gpu=False)

    image_paths = load_image_paths(INPUT_DIR)
    comparison_report = []

    for idx, path in enumerate(image_paths, start=1):
        filename = os.path.basename(path)
        print(f"\n[{idx}/{len(image_paths)}] Processing: {filename}")

        # OCR on raw image
        raw_text, raw_conf, raw_time = run_ocr(reader, path)

        # OCR on preprocessed image
        try:
            preprocessed = preprocess_image(path)
            pre_text, pre_conf, pre_time = run_ocr(reader, preprocessed)
        except Exception as e:
            print(f"Preprocessing failed for {filename}: {e}")
            pre_text, pre_conf, pre_time = "", 0.0, 0.0

        print(f"  Raw OCR          -> confidence: {raw_conf:.2f}, time: {raw_time:.2f}s")
        print(f"  Preprocessed OCR -> confidence: {pre_conf:.2f}, time: {pre_time:.2f}s")

        # Saving extracted text into individual txt file
        out_name = os.path.splitext(filename)[0] + ".txt"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("=== RAW IMAGE OCR ===\n")
            f.write(raw_text + "\n\n")
            f.write("=== PREPROCESSED IMAGE OCR ===\n")
            f.write(pre_text + "\n")

        comparison_report.append(
            {
                "file": filename,
                "raw_confidence": round(raw_conf, 3),
                "preprocessed_confidence": round(pre_conf, 3),
                "improvement": round(pre_conf - raw_conf, 3),
            }
        )

    # Saving summary report
    summary_path = os.path.join(OUTPUT_DIR, "_comparison_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("OCR Comparison Summary - Raw vs Preprocessed\n")
        f.write("=" * 50 + "\n\n")
        for row in comparison_report:
            f.write(
                f"{row['file']:35s} | raw: {row['raw_confidence']:.3f} | "
                f"preprocessed: {row['preprocessed_confidence']:.3f} | "
                f"diff: {row['improvement']:+.3f}\n"
            )

    print(f"\nDone! Extracted text files and summary saved in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()