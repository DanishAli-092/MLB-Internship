# Day 22 - Mini Project Source Code: Simple OCR Document Reader
# Standalone CLI app that extracts text from an image using EasyOCR
# displays it saves it to a txt file and shows image plus text together

# Usage:
#     python ocr_document_reader.py --image sample_inputs/receipt_01_grocery.png
#     python ocr_document_reader.py --image sample_inputs/book_01_page.png --preprocess

# Note: app.py imports OCRDocumentReader class from here for the Streamlit UI

import os
import argparse

import cv2
import matplotlib.pyplot as plt
import easyocr


# EasyOCR returns results by confidence not reading order which scrambles
# dense paragraphs so this groups words into line bands by top-Y then sorts
# each band left to right to restore natural reading order
def sort_reading_order(results, line_band_px=15):
    def sort_key(item):
        bbox, text, conf = item
        top_y = bbox[0][1]
        left_x = bbox[0][0]
        # group words of same line into one band
        return (round(top_y / line_band_px), left_x)

    return sorted(results, key=sort_key)


# Core class that extracts text from an image using EasyOCR
class OCRDocumentReader:

    # Loads the EasyOCR reader
    def __init__(self, languages=None, use_gpu=False):
        if languages is None:
            languages = ["en"]
        print("Loading EasyOCR reader...")
        self.reader = easyocr.Reader(languages, gpu=use_gpu)

    # Grayscale plus CLAHE contrast enhancement plus denoising
    def preprocess(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        return denoised

    # Extracts text with Smart Auto-Enhance
    # runs OCR on raw image then on preprocessed image if enabled
    # and returns whichever has higher confidence
    def extract_text(self, image_path, smart_enhance=True):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # OCR on raw image
        results_raw = self.reader.readtext(image_bgr)
        confidences_raw = [conf for (_bbox, _text, conf) in results_raw]
        conf_raw = sum(confidences_raw) / len(confidences_raw) if confidences_raw else 0.0

        best_results = results_raw
        best_conf = conf_raw
        was_enhanced = False
        proc_conf = None  # stays None if smart_enhance off

        # test preprocessed version too if Smart Enhance is on
        if smart_enhance:
            ocr_input_proc = self.preprocess(image_bgr)
            results_proc = self.reader.readtext(ocr_input_proc)
            confidences_proc = [conf for (_bbox, _text, conf) in results_proc]
            proc_conf = sum(confidences_proc) / len(confidences_proc) if confidences_proc else 0.0

            # pick preprocessed result if its score is better
            if proc_conf > conf_raw:
                best_results = results_proc
                best_conf = proc_conf
                was_enhanced = True

        best_results = sort_reading_order(best_results)

        lines = [text for (_bbox, text, _conf) in best_results]

        return {
            "text": "\n".join(lines),
            "avg_confidence": round(best_conf, 3),
            "raw_confidence": round(conf_raw, 3),
            # separate field, not the same as raw_confidence
            "proc_confidence": round(proc_conf, 3) if proc_conf is not None else None,
            "detections": len(best_results),
            "display_image": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            "was_enhanced": was_enhanced,
        }

    # Saves extracted text into a txt file
    @staticmethod
    def save_text_to_file(text, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return output_path

    # Shows original image and extracted text side by side using matplotlib
    @staticmethod
    def show_image_and_text(display_image, text, image_name):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(display_image)
        axes[0].set_title(f"Original Image: {image_name}")
        axes[0].axis("off")

        axes[1].axis("off")
        axes[1].set_title("Extracted Text")
        display_text = text if text.strip() else "(No text detected)"
        axes[1].text(0, 1, display_text, fontsize=10, va="top", wrap=True)

        plt.tight_layout()
        output_path = os.path.join(
            os.path.dirname(__file__), "extracted_texts",
            f"{os.path.splitext(image_name)[0]}_preview.png",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100)
        print(f"Preview image saved: {output_path}")
        plt.close(fig)


# Runs the CLI OCR document reader
def main():
    parser = argparse.ArgumentParser(description="Simple OCR Document Reader")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--smart", action="store_true", default=True,
        help="Enable Smart Auto-Enhance tests both raw and preprocessed and picks the better one",
    )
    parser.add_argument(
        "--no-smart", dest="smart", action="store_false",
        help="Disable Smart Auto-Enhance only raw image will be used",
    )
    args = parser.parse_args()

    reader = OCRDocumentReader(languages=["en"], use_gpu=False)
    result = reader.extract_text(args.image, smart_enhance=args.smart)

    print("\n" + "=" * 50)
    print("EXTRACTED TEXT")
    print("=" * 50)
    print(result["text"] if result["text"].strip() else "(No text detected)")
    print("=" * 50)
    if args.smart:
        status = "Enhanced version used" if result["was_enhanced"] else "Raw version used"
        print(f"Smart Auto-Enhance: {status}")
        print(f"  Raw confidence:       {result['raw_confidence']:.2%}")
        print(f"  Preprocessed confidence: {result['proc_confidence']:.2%}")
    print(f"Final confidence: {result['avg_confidence']:.2%}")
    print(f"Text regions detected: {result['detections']}")

    image_name = os.path.basename(args.image)
    output_txt_path = os.path.join(
        os.path.dirname(__file__), "extracted_texts",
        f"{os.path.splitext(image_name)[0]}_extracted.txt",
    )
    reader.save_text_to_file(result["text"], output_txt_path)
    print(f"\nExtracted text saved to: {output_txt_path}")

    reader.show_image_and_text(result["display_image"], result["text"], image_name)


if __name__ == "__main__":
    main()