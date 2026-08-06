
#Day 15 - Practice 2: Test YOLO on your own images
#Goal: Observe detected objects, confidence scores, and bounding boxes closely.


from ultralytics import YOLO
import os

model = YOLO("yolo11n.pt")


IMAGE_FOLDER = "../sample_images"
OUTPUT_FOLDER = "../sample_images/own_results"


def analyze_detections(image_folder, output_folder):
    results = model.predict(
        source=image_folder,
        conf=0.4,
        save=True,
        project=output_folder,
        name="my_images",
        exist_ok=True
    )

    for result in results:
        image_name = os.path.basename(result.path)
        print(f"\n===== {image_name} =====")

        if len(result.boxes) == 0:
            print("No objects detected.")
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

            print(
                f"Object: {class_name:15s} | "
                f"Confidence: {confidence:.2f} | "
                f"Box: ({x_min:.0f}, {y_min:.0f}) -> ({x_max:.0f}, {y_max:.0f})"
            )


if __name__ == "__main__":
    analyze_detections(IMAGE_FOLDER, OUTPUT_FOLDER)