# Day 14 — Transfer Learning: Cats vs Dogs Classifier

## Overview

This module covers **Transfer Learning**, a technique that reuses a
Convolutional Neural Network (CNN) pre-trained on a large dataset (ImageNet)
as a feature extractor for a new task, instead of training a network from
scratch. The deliverable is an image classifier that distinguishes cats from
dogs, built on top of **MobileNetV2**.

---

## 1. What is Transfer Learning?

Instead of training a CNN from zero, we take a model that's already been
trained on a huge dataset (ImageNet — 1.4M images, 1000 classes) and reuse
it for our own, much smaller task.

The reason this works: the early layers of a CNN mostly learn generic stuff
- edges, textures, shapes, color gradients - which is useful for almost any
image task, not just the one it was originally trained on. So there's no
point retraining that from scratch every time. We keep those layers as-is
and only train new layers on top for our specific problem (cats vs dogs
here).

**Why bother:**
- Way less training data needed.
- Trains faster, costs less compute.
- Usually beats a from-scratch CNN on smaller datasets like this one.

---

## 2. Why MobileNetV2?

| Model | Parameters | Best suited for |
|---|---|---|
| VGG16 | ~138M | Simple architecture, educational use |
| ResNet50 | ~25M | High-accuracy tasks with more compute available |
| **MobileNetV2** | **~3.4M** | **Lightweight, fast, ideal for limited compute / edge devices** |
| EfficientNetB0 | ~5.3M | Best accuracy-to-size ratio, but trickier to tune |

Went with MobileNetV2 mainly because it's light and fast to train, without
losing much accuracy on a binary task like this. VGG16 and ResNet50 would
probably also work fine here, but they're heavier than this problem
actually needs.

---

## 3. Approach

1. **Data pipeline** — Loaded the Cats vs Dogs dataset via TensorFlow
   Datasets (TFDS), resized all images to `224x224`, and split into an
   80/20 train/validation set.
2. **Feature extraction phase** — Loaded MobileNetV2 with `include_top=False`
   and `weights="imagenet"`, froze all its layers, and attached a custom
   classification head (`GlobalAveragePooling2D → Dense(128) → Dropout →
   Dense(1, sigmoid)`).
3. **Fine-tuning phase** — Unfroze the top layers of MobileNetV2 and
   continued training with a much lower learning rate (`1e-5`) to adapt
   the pre-trained features to this specific dataset.
4. **Data augmentation** — Applied `RandomFlip`, `RandomRotation`, and
   `RandomZoom` during training to reduce overfitting.

---

## 4. Experiments Performed

| Experiment | Configuration | Validation Accuracy |
|---|---|---|
| Baseline (feature extraction only) | Base frozen, `8` epochs | `98.90`% |
| + Data augmentation | Same as above + augmentation layers | `98.90`% |
| + Fine-tuning | Unfroze from layer `100` onward, lr = 1e-5 | `98.97`% |
| Final configuration | Batch size: `32`, Epochs: `13 (8 Initial + 5 Fine-tune)` | **`98.97`%** |

---

## 5. Final Results

**Final Validation Accuracy:** `98.97`%
**Final Validation Loss:** `0.0326`

Training/validation accuracy and loss curves: [`results/graphs/accuracy_loss_curves.png`](results/graphs/accuracy_loss_curves.png)
Sample predictions on the validation set: [`results/predictions/sample_predictions.png`](results/predictions/sample_predictions.png)

---

## 6. Key Challenges and Lessons Learned

- **TFDS Windows Path Bug (Critical Fix):** Faced a known `KeyError` crash during dataset extraction because the Windows file system uses backslashes (`\`) while the Zip archive uses forward slashes (`/`). Solved this by surgically editing the core TFDS library file (`cats_vs_dogs.py`) and applying `.replace("\\", "/")` to the path normalizer.
- **Hardware / GPU Limitations:** Discovered that TensorFlow 2.11+ no longer supports native Windows GPU acceleration. The training fell back to the CPU (Intel i7), increasing training time significantly. Learned that migrating to a WSL2 (Windows Subsystem for Linux) environment is the industry standard for utilizing NVIDIA GPUs on Windows.
- **Understanding Validation Spikes:** Initially worried about overfitting when validation accuracy surpassed training accuracy in early epochs. Learned that this is a normal effect of Dropout layers (which are active only during training, handicap the model, and turn off during validation) and the immense pre-trained knowledge base of MobileNetV2.
- **Feature Extraction vs. Fine-Tuning:** Grasped why the base model must stay frozen during the first phase and why fine-tuning requires a significantly lower learning rate (`1e-5`) to avoid destroying pre-trained ImageNet weights.



---

## 7. Project Structure

```
Day-14/
├── practice1_transfer_learning_basics.py   # Loading MobileNetV2, exploring architecture, freezing, custom head
├── practice2_dataset_loading.py            # Loading and preprocessing Cats vs Dogs dataset (TFDS)
├── cats_vs_dogs_classifier.py              # Full mini project: training + fine-tuning + evaluation
├── requirements.txt                        # Python dependencies
├── README.md
├── results/
│   ├── graphs/                             # Accuracy & loss curves
│   └── predictions/                        # Sample prediction images
└── screen_recording/
    └── LINK.txt                            # External link to implementation walkthrough recording
```

---

## 8. Setup & Usage

```bash
pip install -r requirements.txt

python practice1_transfer_learning_basics.py
python practice2_dataset_loading.py
python cats_vs_dogs_classifier.py
```

---

## 9. What I Learned Today

* **The Generalization Gap:** Mastered the critical distinction between training accuracy (memorization) and validation accuracy (real-world performance on unseen data).
* **The Dropout Effect:** Understood why validation accuracy can gracefully outpace training accuracy early on due to regularization techniques like Dropout and Data Augmentation being active exclusively during training.
* **Two-Phase Fine-Tuning Strategy:** Successfully executed a dual-phase training pipeline: locking base feature extractors first, then selectively unfreezing top layers with a micro-learning rate (`1e-5`) to specialize the model without triggering catastrophic forgetting.
* **Low-Level Library Patching:** Gained hands-on experience debugging third-party infrastructure by surgically fixing a Windows path-extraction bug (`\` vs `/`) directly inside the TensorFlow Datasets (`tfds`) source code.
* **Hardware Constraints & Scaling:** Experienced the heavy computational limits of CPU-bound deep learning training, validating the practical necessity of transitioning to WSL2/Linux environments for native GPU acceleration.