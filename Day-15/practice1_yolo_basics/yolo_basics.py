
#Day 15 - Practice 1: Introduction to YOLO Object Detection
#Goal: Load a pretrained YOLO model and run detection on single/multiple images.

from ultralytics import YOLO
import os


# Step 1: Load a pretrained YOLO model

# "yolo11n.pt" = nano version -> lightweight and fast, good for testing.
# Ultralytics automatically downloads it the first time we run this.
model = YOLO("yolo11n.pt")

print("Model loaded successfully!")
print(f"Model classes: {model.names}")  # shows all 80 COCO classes it can detect


# Step 2: Run detection on a single image
def detect_single_image(image_path, save_dir="../sample_images/results"):
    """
    Runs YOLO inference on one image and saves the annotated result.
    """
    results = model.predict(
        source=image_path,
        conf=0.5,          # confidence threshold - ignore weak detections
        save=True,          # saves the output image with boxes drawn
        project=save_dir,
        name="single_image_run",
        exist_ok=True
    )

    # results is a list (one entry per image). Since we passed one image, take index 0.
    result = results[0]

    print(f"\nDetections for {image_path}:")
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]

        print(f"  -> {class_name} | confidence: {confidence:.2f} | box: {coords}")

    return result


# Step 3: Run detection on multiple images

def detect_multiple_images(folder_path, save_dir="../sample_images/results"):
    """
    Runs YOLO inference on every image inside a folder.
    """
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(valid_extensions)
    ]

    if not image_paths:
        print("No images found in folder:", folder_path)
        return

    results = model.predict(
        source=image_paths,
        conf=0.5,
        save=True,
        project=save_dir,
        name="batch_run",
        exist_ok=True
    )

    for img_path, result in zip(image_paths, results):
        print(f"\n{os.path.basename(img_path)} -> {len(result.boxes)} objects detected")



# Run the script

if __name__ == "__main__":
    # Single image test
    detect_single_image("../sample_images/test1.jpg")

    # Multiple images test
    detect_multiple_images("../sample_images")

    print("\nDone! Check the results folder for saved output images.")