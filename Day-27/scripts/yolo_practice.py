
# Day 27 - YOLO Practice Script
# Goal: load a pre-trained YOLO model and run it on a bunch of images + videos
# save the outputs and print out what got detected.


import os
from pathlib import Path
from ultralytics import YOLO

# folders we're working with (relative to project root run this from Day-27/)
IMAGES_DIR = Path("sample_images")
VIDEOS_DIR = Path("sample_videos")
OUTPUT_IMAGES_DIR = Path("outputs/images")
OUTPUT_VIDEOS_DIR = Path("outputs/videos")

# confidence threshold  anything below this gets ignored
CONF_THRESHOLD = 0.5


def load_model():
    # yolov8n = nano version small and fast
    
    print("Loading YOLOv8n pre-trained model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded. Classes it knows:", len(model.names))
    return model


def run_on_images(model):
    # grab every image file in the sample_images folder
    valid_ext = (".jpg", ".jpeg", ".png")
    image_paths = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in valid_ext]

    if len(image_paths) == 0:
        print(f"No images found in {IMAGES_DIR}. Drop at least 10 images there and re-run.")
        return

    if len(image_paths) < 10:
        print(f"Warning: task asks for 10+ images, only found {len(image_paths)}.")

    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        # verbose=False so the console doesn't get flooded per image
        results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        # save the annotated version (boxes + labels drawn already)
        save_path = OUTPUT_IMAGES_DIR / img_path.name
        result.save(filename=str(save_path))

        # print what we found so the confidence scores
        print(f"\n{img_path.name}:")
        if len(result.boxes) == 0:
            print("  -> nothing detected above threshold")
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            print(f"  - {class_name}: {confidence:.2f}")


def run_on_videos(model):
    valid_ext = (".mp4", ".avi", ".mov")
    video_paths = [p for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in valid_ext]

    if len(video_paths) == 0:
        print(f"No videos found in {VIDEOS_DIR}. Drop 2 short videos there and re-run.")
        return

    OUTPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    for vid_path in video_paths:
        print(f"\nProcessing video: {vid_path.name} (this can take a bit)...")
        model.predict(
            source=str(vid_path),
            conf=CONF_THRESHOLD,
            save=True,
            project=str(OUTPUT_VIDEOS_DIR),
            name=vid_path.stem,
            verbose=False,
        )
        print(f"Done. Check {OUTPUT_VIDEOS_DIR / vid_path.stem} for the result.")


def main():
    model = load_model()

    print("\n--- Running detection on images ---")
    run_on_images(model)

    print("\n--- Running detection on videos ---")
    run_on_videos(model)

    print("\nAll done. Annotated outputs are saved under outputs/")


if __name__ == "__main__":
    main()