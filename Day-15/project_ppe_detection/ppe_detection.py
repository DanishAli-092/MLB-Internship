from ultralytics import YOLO
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use("TkAgg")

model = YOLO("yolo11n.pt")

dataset_path = "PPE-DETECTION-14/test/images"
output_path = "outputs"

all_images = glob.glob(os.path.join(dataset_path, "*.jpg"))
print("total images in dataset:", len(all_images))

sample = all_images[:10]

results = model.predict(source=sample, conf=0.4, save=True, project=output_path, name="ppe_predictions", exist_ok=True)

for img_path, result in zip(sample, results):
    name = os.path.basename(img_path)
    classes_found = []
    for box in result.boxes:
        classes_found.append(model.names[int(box.cls[0])])
    print(name, "->", classes_found)

saved_images = glob.glob(os.path.join(output_path, "ppe_predictions", "*.jpg"))
if saved_images:
    img = mpimg.imread(saved_images[0])
    plt.imshow(img)
    plt.axis("off")
    plt.show()